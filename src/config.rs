use figment::{Figment, providers::{Toml, Env, Format}};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub ollama: OllamaConfig,
    pub rag: RagConfig,
    pub auth: AuthConfig,
    pub storage: StorageConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    pub bind_addr: String,
    pub max_connections: usize,
    pub request_timeout_secs: u64,
    pub max_body_size_mb: usize,
    pub cors_origins: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub url: String,
    pub embedding_model: String,
    pub llm_model: String,
    pub embedding_batch_size: usize,
    pub request_timeout_secs: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RagConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub top_k: usize,
    pub min_score: f32,
    pub max_context_tokens: usize,
    pub system_prompt: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthConfig {
    pub enabled: bool,
    pub jwt_secret: String,
    pub jwt_expiry_hours: i64,
    pub rate_limit_per_minute: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StorageConfig {
    pub sled_path: String,
    pub vector_cache_size_mb: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                bind_addr: "127.0.0.1:3000".into(),
                max_connections: 1000,
                request_timeout_secs: 120,
                max_body_size_mb: 100,
                cors_origins: vec!["*".into()],
            },
            ollama: OllamaConfig {
                url: "http://localhost:11434".into(),
                embedding_model: "nomic-embed-text".into(),
                llm_model: "llama3.1:8b".into(),
                embedding_batch_size: 10,
                request_timeout_secs: 120,
            },
            rag: RagConfig {
                chunk_size: 800,
                chunk_overlap: 150,
                top_k: 5,
                min_score: 0.15,
                max_context_tokens: 4096,
                system_prompt: "You are a financial document analyst. Answer the user's question based ONLY on the provided context from financial filings.\n\nRules:\n1. If the answer is not in the context, say \"This information is not available in the provided document.\"\n2. Cite the source section for each claim.\n3. For numerical data, present it clearly.\n4. Do not make assumptions or use external knowledge.".into(),
            },
            auth: AuthConfig {
                enabled: false,
                jwt_secret: "change-me-in-production".into(),
                jwt_expiry_hours: 24,
                rate_limit_per_minute: 60,
            },
            storage: StorageConfig {
                sled_path: "./data/vectors.sled".into(),
                vector_cache_size_mb: 256,
            },
        }
    }
}

impl AppConfig {
    pub fn load() -> Self {
        let config_path = std::env::var("RAG_CONFIG_PATH")
            .unwrap_or_else(|_| "config.toml".into());

        let figment = Figment::from(SerializedAppConfig(AppConfig::default()))
            .merge(Toml::file(&config_path))
            .merge(Env::prefixed("RAG_"));

        figment.extract().unwrap_or_else(|e| {
            tracing::warn!("Config load error (using defaults): {}", e);
            AppConfig::default()
        })
    }
}

struct SerializedAppConfig(AppConfig);
impl figment::Provider for SerializedAppConfig {
    fn metadata(&self) -> figment::Metadata {
        figment::Metadata::named("AppConfig")
    }
    fn data(&self) -> Result<figment::value::Map<figment::Profile, figment::value::Dict>, figment::Error> {
        let dict = figment::providers::Serialized::defaults(&self.0).data()?;
        Ok(dict)
    }
}