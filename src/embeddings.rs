use serde::{Deserialize, Serialize};

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
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl EmbeddingClient {
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            base_url,
            model,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .unwrap(),
        }
    }

    /// Generate an embedding for a single text string via Ollama /api/embeddings.
    pub async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let req = EmbedRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
        };

        let resp = self
            .client
            .post(format!("{}/api/embeddings", self.base_url))
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

    /// Batch embed multiple texts sequentially (Ollama processes one at a time).
    pub async fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for (i, text) in texts.iter().enumerate() {
            let emb = self.embed(text).await?;
            results.push(emb);
            if (i + 1) % 10 == 0 {
                tracing::info!("Embedded {}/{} chunks", i + 1, texts.len());
            }
        }
        Ok(results)
    }
}