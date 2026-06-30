use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Multipart, Path, Query, State},
    http::StatusCode,
    response::{Html, Json, Sse},
    routing::{delete as delete_route, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;
use tower_http::{
    cors::CorsLayer,
    limit::RequestBodyLimitLayer,
    timeout::TimeoutLayer,
    trace::TraceLayer,
};

use crate::auth::AuthService;
use crate::config::AppConfig;
use crate::document_processor::DocumentProcessor;
use crate::document_queue::{DocumentQueue, ProcessingJob};
use crate::embeddings::EmbeddingClient;
use crate::rag_pipeline::RagPipeline;
use crate::vector_store::VectorStore;

pub struct AppState {
    pub config: AppConfig,
    pub doc_processor: Arc<DocumentProcessor>,
    pub embedding_client: Arc<EmbeddingClient>,
    pub vector_store: Arc<VectorStore>,
    pub rag: Arc<RagPipeline>,
    pub doc_queue: Arc<DocumentQueue>,
    pub auth: Arc<AuthService>,
}

#[derive(Deserialize)]
pub struct QuestionRequest {
    pub question: String,
    pub source_filter: Option<String>,
    pub stream: Option<bool>,
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
    pub job_id: String,
    pub document: String,
    pub message: String,
}

#[derive(Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: Option<String>,
}

#[derive(Serialize)]
pub struct LoginResponse {
    pub token: String,
    pub expires_in_hours: i64,
}

#[derive(Serialize)]
pub struct JobStatusResponse {
    pub job_id: String,
    pub status: String,
    pub result: Option<crate::document_queue::ProcessingResult>,
}

pub fn build_router(state: Arc<AppState>) -> Router {
    let max_body = state.config.server.max_body_size_mb * 1024 * 1024;
    let timeout = std::time::Duration::from_secs(state.config.server.request_timeout_secs);

    let cors = if state.config.server.cors_origins.contains(&"*".to_string()) {
        CorsLayer::very_permissive()
    } else {
        let origins: Vec<_> = state
            .config
            .server
            .cors_origins
            .iter()
            .map(|s| s.as_str())
            .collect();
        CorsLayer::new()
            .allow_origin(tower_http::cors::Any)
            .allow_methods(tower_http::cors::Any)
            .allow_headers(tower_http::cors::Any)
    };

    let routes = Router::new()
        .route("/", get(index_handler))
        .route("/api/health", get(health_handler))
        .route("/api/docs", get(list_docs_handler))
        .route("/api/upload", post(upload_handler))
        .route("/api/ask", post(ask_handler))
        .route("/api/ask/stream", post(ask_stream_handler))
        .route("/api/delete/:name", delete_route(delete_doc_handler))
        .route("/api/job/:job_id", get(job_status_handler))
        .route("/api/login", post(login_handler));

    let protected = routes
        .layer(axum::middleware::from_fn_with_state(
            state.auth.clone(),
            crate::auth::auth_middleware,
        ));

    protected
        .layer(RequestBodyLimitLayer::new(max_body))
        .layer(TimeoutLayer::new(timeout))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let vs = &state.vector_store;
    Json(serde_json::json!({
        "status": "ok",
        "version": "2.0.0",
        "total_chunks": vs.total_chunks(),
        "documents": vs.doc_names().len(),
        "embedding_model": state.config.ollama.embedding_model,
        "llm_model": state.config.ollama.llm_model,
        "auth_enabled": state.config.auth.enabled,
        "storage": "sled (persistent)",
    }))
}

async fn list_docs_handler(State(state): State<Arc<AppState>>) -> Json<DocsResponse> {
    let docs: Vec<DocInfo> = state
        .vector_store
        .doc_info()
        .into_iter()
        .map(|(name, chunks)| DocInfo { name, chunks })
        .collect();
    Json(DocsResponse {
        documents: docs,
        total_chunks: state.vector_store.total_chunks(),
    })
}

async fn upload_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<UploadResponse>, (StatusCode, String)> {
    let mut filename = String::new();
    let mut file_data: Option<Vec<u8>> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            filename = field.file_name().unwrap_or("upload.txt").to_string();
            file_data = Some(
                field
                    .bytes()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Read error: {}", e)))?
                    .to_vec(),
            );
        }
    }

    let data =
        file_data.ok_or((StatusCode::BAD_REQUEST, "No file uploaded".into()))?;

    let temp_path = std::env::temp_dir().join(format!("rag_{}", filename));
    std::fs::write(&temp_path, &data)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Write error: {}", e)))?;

    let text = state
        .doc_processor
        .extract_text(&temp_path)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Extraction error: {}", e)))?;

    let _ = std::fs::remove_file(&temp_path);

    if text.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "No text could be extracted".into()));
    }

    let job_id = uuid::Uuid::new_v4().to_string();
    let job = ProcessingJob {
        job_id: job_id.clone(),
        source: filename.clone(),
        text,
    };

    state
        .doc_queue
        .submit(job)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Queue error: {}", e)))?;

    Ok(Json(UploadResponse {
        success: true,
        job_id: job_id.clone(),
        document: filename,
        message: "Document queued for processing. Use /api/job/{job_id} to check status.".into(),
    }))
}

async fn ask_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QuestionRequest>,
) -> Result<Json<QuestionResponse>, (StatusCode, String)> {
    let start = std::time::Instant::now();

    let query_vec = state
        .embedding_client
        .embed(&req.question)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Embed error: {}", e)))?;

    let top_k = state.config.rag.top_k;
    let min_score = state.config.rag.min_score;
    let retrieved = state
        .vector_store
        .search(&query_vec, top_k, min_score, req.source_filter.as_deref())
        .await;

    let result = state
        .rag
        .answer(&req.question, retrieved)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("LLM error: {}", e)))?;

    Ok(Json(QuestionResponse {
        answer: result.answer,
        sources: result.sources,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }))
}

async fn ask_stream_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QuestionRequest>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>>, (StatusCode, String)> {
    let query_vec = state
        .embedding_client
        .embed(&req.question)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Embed error: {}", e)))?;

    let top_k = state.config.rag.top_k;
    let min_score = state.config.rag.min_score;
    let retrieved = state
        .vector_store
        .search(&query_vec, top_k, min_score, req.source_filter.as_deref())
        .await;

    let sources_json = serde_json::to_string(&retrieved).unwrap_or_default();

    let stream = state
        .rag
        .answer_stream(&req.question, retrieved)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Stream error: {}", e)))?;

    let sources_event = axum::response::sse::Event::default()
        .event("sources")
        .data(sources_json);

    let token_stream = stream.map(|token_result| {
        let token = token_result.unwrap_or_default();
        Ok::<_, std::convert::Infallible>(
            axum::response::sse::Event::default().event("token").data(token),
        )
    });

    let combined = tokio_stream::iter(vec![Ok::<_, std::convert::Infallible>(sources_event)])
        .chain(token_stream);

    Ok(Sse::new(combined))
}

async fn delete_doc_handler(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Json<serde_json::Value> {
    let removed = state.vector_store.remove_document(&name);
    Json(serde_json::json!({
        "success": true,
        "removed_chunks": removed,
    }))
}

async fn job_status_handler(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Json<JobStatusResponse> {
    let status = state.doc_queue.check_result(&job_id).await;
    let (status_str, result) = match status {
        crate::document_queue::ProcessingStatus::Pending => ("pending".into(), None),
        crate::document_queue::ProcessingStatus::Done(r) => ("done".into(), Some(r)),
        crate::document_queue::ProcessingStatus::Failed(r) => ("failed".into(), Some(r)),
    };

    Json(JobStatusResponse {
        job_id,
        status: status_str,
        result,
    })
}

async fn login_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<LoginResponse>, (StatusCode, String)> {
    if !state.config.auth.enabled {
        let token = state
            .auth
            .issue_token(&req.username)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        return Ok(Json(LoginResponse {
            token,
            expires_in_hours: state.config.auth.jwt_expiry_hours,
        }));
    }

    let token = state
        .auth
        .issue_token(&req.username)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(LoginResponse {
        token,
        expires_in_hours: state.config.auth.jwt_expiry_hours,
    }))
}