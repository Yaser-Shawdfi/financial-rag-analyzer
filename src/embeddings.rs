use serde::{Deserialize, Serialize};

use crate::config::OllamaConfig;

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

pub struct EmbeddingClient {
    config: OllamaConfig,
    client: reqwest::Client,
}

impl EmbeddingClient {
    pub fn new(config: OllamaConfig) -> Self {
        let timeout = std::time::Duration::from_secs(config.request_timeout_secs);
        Self {
            config,
            client: reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .unwrap(),
        }
    }

    pub async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let req = EmbedRequest {
            model: self.config.embedding_model.clone(),
            prompt: text.to_string(),
        };

        let resp = self
            .client
            .post(format!("{}/api/embeddings", self.config.url))
            .json(&req)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Embedding request failed: {} - {}", status, body);
        }

        let embed_resp: EmbedResponse = resp.json().await?;
        Ok(embed_resp.embedding)
    }

    pub async fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        let batch_size = self.config.embedding_batch_size.max(1);

        for (i, text) in texts.iter().enumerate() {
            let emb = self.embed(text).await?;
            results.push(emb);
            if (i + 1) % batch_size == 0 {
                tracing::info!("Embedded {}/{} chunks", i + 1, texts.len());
            }
        }

        Ok(results)
    }
}