mod auth;
mod config;
mod document_processor;
mod document_queue;
mod embeddings;
mod rag_pipeline;
mod server;
mod vector_store;

use std::sync::Arc;

use config::AppConfig;
use document_processor::DocumentProcessor;
use document_queue::DocumentQueue;
use embeddings::EmbeddingClient;
use rag_pipeline::RagPipeline;
use auth::AuthService;
use vector_store::VectorStore;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with_target(true)
        .init();

    let config = AppConfig::load();

    tracing::info!(
        "Starting Financial RAG Analyzer v2.0.0 on http://{} (auth={})",
        config.server.bind_addr,
        config.auth.enabled
    );

    let doc_processor = Arc::new(DocumentProcessor::new(
        config.rag.chunk_size,
        config.rag.chunk_overlap,
    ));
    let embedding_client = Arc::new(EmbeddingClient::new(config.ollama.clone()));
    let vector_store = Arc::new(VectorStore::new(&config.storage.sled_path)?);
    let rag = Arc::new(RagPipeline::new(
        config.rag.clone(),
        config.ollama.url.clone(),
        config.ollama.llm_model.clone(),
    ));
    let auth = Arc::new(AuthService::new(config.auth.clone()));

    let doc_queue = Arc::new(DocumentQueue::new(
        doc_processor.clone(),
        embedding_client.clone(),
        vector_store.clone(),
        2,
    ));

    let state = Arc::new(server::AppState {
        config: config.clone(),
        doc_processor,
        embedding_client,
        vector_store,
        rag,
        doc_queue,
        auth,
    });

    let app = server::build_router(state);

    let listener = tokio::net::TcpListener::bind(&config.server.bind_addr).await?;
    tracing::info!("Server listening on http://{}", config.server.bind_addr);
    axum::serve(listener, app).await?;

    Ok(())
}