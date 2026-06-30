use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::Config;
use crate::document_processor::DocumentProcessor;
use crate::embeddings::EmbeddingClient;
use crate::vector_store::VectorStore;
use crate::rag_pipeline::RagPipeline;

pub mod config;
pub mod document_processor;
pub mod embeddings;
pub mod vector_store;
pub mod rag_pipeline;
pub mod server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let config = Config::from_env();
    tracing::info!("Starting Financial RAG Analyzer on http://{}", config.bind_addr);

    // Shared state
    let embedding_client = Arc::new(EmbeddingClient::new(config.ollama_url.clone(), config.embedding_model.clone()));
    let vector_store = Arc::new(RwLock::new(VectorStore::new()));
    let doc_processor = Arc::new(DocumentProcessor::new());
    let rag = Arc::new(RagPipeline::new(
        embedding_client.clone(),
        config.ollama_url.clone(),
        config.llm_model.clone(),
    ));

    // Track uploaded document names
    let doc_names: Arc<RwLock<HashMap<String, String>>> = Arc::new(RwLock::new(HashMap::new()));

    let app = server::build_router(
        doc_processor,
        embedding_client,
        vector_store,
        rag,
        doc_names,
        config.bind_addr.clone(),
    );

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}