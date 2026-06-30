use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

use crate::config::Config;
use crate::document_processor::DocumentProcessor;
use crate::embeddings::EmbeddingClient;
use crate::rag_pipeline::RagPipeline;
use crate::vector_store::VectorStore;

/// Shared application state passed to all handlers.
pub struct AppState {
    pub doc_processor: Arc<DocumentProcessor>,
    pub embedding_client: Arc<EmbeddingClient>,
    pub vector_store: Arc<RwLock<VectorStore>>,
    pub rag: Arc<RagPipeline>,
    pub doc_names: Arc<RwLock<HashMap<String, String>>>,
    pub config: Config,
}

#[derive(Deserialize)]
pub struct QuestionRequest {
    pub question: String,
}

#[derive(Serialize)]
pub struct QuestionResponse {
    pub answer: String,
    pub sources: Vec<crate::vector_store::ScoredChunk>,
    pub elapsed_ms: u64,
}

#[derive(Serialize)]
pub struct DocsResponse {
    pub documents: Vec<DocInfo>,
    pub total_chunks: usize,
}

#[derive(Serialize)]
pub struct DocInfo {
    pub name: String,
    pub chunks: usize,
}

#[derive(Serialize)]
pub struct UploadResponse {
    pub success: bool,
    pub document: String,
    pub chunks: usize,
    pub elapsed_ms: u64,
}

pub fn build_router(
    doc_processor: Arc<DocumentProcessor>,
    embedding_client: Arc<EmbeddingClient>,
    vector_store: Arc<RwLock<VectorStore>>,
    rag: Arc<RagPipeline>,
    doc_names: Arc<RwLock<HashMap<String, String>>>,
    bind_addr: String,
) -> Router {
    let state = Arc::new(AppState {
        doc_processor,
        embedding_client,
        vector_store,
        rag,
        doc_names,
        config: Config {
            bind_addr,
            ..Config::from_env()
        },
    });

    Router::new()
        .route("/", get(index_handler))
        .route("/api/health", get(health_handler))
        .route("/api/docs", get(list_docs_handler))
        .route("/api/upload", post(upload_handler))
        .route("/api/ask", post(ask_handler))
        .route("/api/delete/:name", axum::routing::delete(delete_doc_handler))
        .layer(CorsLayer::very_permissive())
        .with_state(state)
}

// -- Handlers --

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let vs = state.vector_store.read().await;
    Json(serde_json::json!({
        "status": "ok",
        "total_chunks": vs.total_chunks(),
        "documents": vs.doc_names().len(),
        "embedding_model": state.config.embedding_model,
        "llm_model": state.config.llm_model,
    }))
}

async fn list_docs_handler(State(state): State<Arc<AppState>>) -> Json<DocsResponse> {
    let vs = state.vector_store.read().await;
    let docs: Vec<DocInfo> = vs
        .doc_info()
        .into_iter()
        .map(|(name, chunks)| DocInfo { name, chunks })
        .collect();
    Json(DocsResponse {
        documents: docs,
        total_chunks: vs.total_chunks(),
    })
}

async fn upload_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<UploadResponse>, (StatusCode, String)> {
    let start = std::time::Instant::now();
    let mut filename = String::new();
    let mut file_data: Option<Vec<u8>> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            filename = field.file_name().unwrap_or("upload.txt").to_string();
            file_data = Some(field.bytes().await.map_err(|e| {
                (StatusCode::BAD_REQUEST, format!("Failed to read file: {}", e))
            })?.to_vec());
        }
    }

    let data = file_data.ok_or((StatusCode::BAD_REQUEST, "No file uploaded".into()))?;
    let file_path = std::env::temp_dir().join(format!("frag_{}", filename));
    std::fs::write(&file_path, &data).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Write error: {}", e))
    })?;

    // Extract text
    let text = state
        .doc_processor
        .extract_text(&file_path)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Extraction error: {}", e)))?;

    let _ = std::fs::remove_file(&file_path);

    if text.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "No text could be extracted".into()));
    }

    // Chunk
    let doc_name = filename.clone();
    let chunks = state.doc_processor.chunk_text(&text, &doc_name);
    if chunks.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "No chunks generated".into()));
    }

    // Embed all chunks
    let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let vectors = state
        .embedding_client
        .embed_batch(&chunk_texts)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Embedding error: {}", e)))?;

    // Store
    state.vector_store.write().await.add(chunks, vectors);
    state
        .doc_names
        .write()
        .await
        .insert(doc_name.clone(), doc_name.clone());

    let elapsed = start.elapsed().as_millis() as u64;
    let vs = state.vector_store.read().await;
    let chunk_count = vs.doc_info()
        .iter()
        .find(|(n, _)| *n == doc_name)
        .map(|(_, c)| *c)
        .unwrap_or(0);

    Ok(Json(UploadResponse {
        success: true,
        document: doc_name,
        chunks: chunk_count,
        elapsed_ms: elapsed,
    }))
}

async fn ask_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QuestionRequest>,
) -> Result<Json<QuestionResponse>, (StatusCode, String)> {
    let start = std::time::Instant::now();

    // Embed the question
    let query_vec = state
        .embedding_client
        .embed(&req.question)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Embed error: {}", e)))?;

    // Retrieve top-K
    let top_k = state.config.top_k;
    let retrieved = state.vector_store.read().await.search(&query_vec, top_k);

    // Generate answer
    let result = state
        .rag
        .answer(&req.question, retrieved)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("LLM error: {}", e)))?;

    let elapsed = start.elapsed().as_millis() as u64;

    Ok(Json(QuestionResponse {
        answer: result.answer,
        sources: result.sources,
        elapsed_ms: elapsed,
    }))
}

async fn delete_doc_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Json<serde_json::Value> {
    let removed = state.vector_store.write().await.remove_document(&name);
    state.doc_names.write().await.remove(&name);
    Json(serde_json::json!({
        "success": true,
        "removed_chunks": removed,
    }))
}