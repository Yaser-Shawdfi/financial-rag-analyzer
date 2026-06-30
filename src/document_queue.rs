use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::document_processor::{Chunk, DocumentProcessor};
use crate::embeddings::EmbeddingClient;
use crate::vector_store::VectorStore;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessingJob {
    pub job_id: String,
    pub source: String,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub job_id: String,
    pub success: bool,
    pub source: String,
    pub chunk_count: usize,
    pub error: Option<String>,
}

pub struct DocumentQueue {
    tx: mpsc::Sender<ProcessingJob>,
    results: Arc<tokio::sync::RwLock<std::collections::HashMap<String, ProcessingResult>>>,
}

impl DocumentQueue {
    pub fn new(
        doc_processor: Arc<DocumentProcessor>,
        embedding_client: Arc<EmbeddingClient>,
        vector_store: Arc<VectorStore>,
        num_workers: usize,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<ProcessingJob>(100);
        let results: Arc<tokio::sync::RwLock<std::collections::HashMap<String, ProcessingResult>>> =
            Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new()));
        let results_clone = results.clone();

        let rx = Arc::new(tokio::sync::Mutex::new(rx));

        for worker_id in 0..num_workers {
            let dp = doc_processor.clone();
            let ec = embedding_client.clone();
            let vs = vector_store.clone();
            let results = results_clone.clone();
            let rx = rx.clone();

            tokio::spawn(async move {
                tracing::info!("Document worker {} started", worker_id);

                loop {
                    let job = {
                        let mut guard = rx.lock().await;
                        guard.recv().await
                    };
                    
                    let Some(job) = job else { break; };

                    tracing::info!(
                        "Worker {} processing job {} for '{}'",
                        worker_id,
                        job.job_id,
                        job.source
                    );

                    let result = process_document(&job, &dp, &ec, &vs).await;

                    let mut results_lock = results.write().await;
                    results_lock.insert(job.job_id.clone(), result);
                }

                tracing::info!("Document worker {} stopped", worker_id);
            });
        }

        Self { tx, results }
    }

    pub async fn submit(&self, job: ProcessingJob) -> anyhow::Result<()> {
        self.tx.send(job).await.map_err(|e| anyhow::anyhow!(e))
    }

    pub async fn get_result(&self, job_id: &str) -> Option<ProcessingResult> {
        self.results.read().await.get(job_id).cloned()
    }

    pub async fn check_result(&self, job_id: &str) -> ProcessingStatus {
        match self.results.read().await.get(job_id) {
            Some(r) if r.success => ProcessingStatus::Done(r.clone()),
            Some(r) => ProcessingStatus::Failed(r.clone()),
            None => ProcessingStatus::Pending,
        }
    }
}

#[derive(Debug)]
pub enum ProcessingStatus {
    Pending,
    Done(ProcessingResult),
    Failed(ProcessingResult),
}

async fn process_document(
    job: &ProcessingJob,
    doc_processor: &DocumentProcessor,
    embedding_client: &EmbeddingClient,
    vector_store: &VectorStore,
) -> ProcessingResult {
    let chunks = doc_processor.chunk_text(&job.text, &job.source);

    let chunk_count = chunks.len();
    let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let vectors = match embedding_client.embed_batch(&chunk_texts).await {
        Ok(v) => v,
        Err(e) => {
            return ProcessingResult {
                job_id: job.job_id.clone(),
                success: false,
                source: job.source.clone(),
                chunk_count: 0,
                error: Some(e.to_string()),
            };
        }
    };

    if let Err(e) = vector_store.add(chunks, vectors).await {
        return ProcessingResult {
            job_id: job.job_id.clone(),
            success: false,
            source: job.source.clone(),
            chunk_count: 0,
            error: Some(e.to_string()),
        };
    }

    ProcessingResult {
        job_id: job.job_id.clone(),
        success: true,
        source: job.source.clone(),
        chunk_count,
        error: None,
    }
}