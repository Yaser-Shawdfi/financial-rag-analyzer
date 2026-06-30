use std::env;

#[derive(Clone, Debug)]
pub struct Config {
    pub bind_addr: String,
    pub ollama_url: String,
    pub embedding_model: String,
    pub llm_model: String,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub top_k: usize,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            bind_addr: env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:3000".into()),
            ollama_url: env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".into()),
            embedding_model: env::var("EMBEDDING_MODEL").unwrap_or_else(|_| "nomic-embed-text".into()),
            llm_model: env::var("LLM_MODEL").unwrap_or_else(|_| "llama3.1:8b".into()),
            chunk_size: env::var("CHUNK_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(800),
            chunk_overlap: env::var("CHUNK_OVERLAP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(150),
            top_k: env::var("TOP_K")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5),
        }
    }
}