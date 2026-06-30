use serde::{Deserialize, Serialize};

use crate::embeddings::EmbeddingClient;
use crate::vector_store::ScoredChunk;

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

pub struct RagPipeline {
    embedding_client: std::sync::Arc<EmbeddingClient>,
    ollama_url: String,
    llm_model: String,
}

#[derive(Serialize, Clone)]
pub struct RagAnswer {
    pub answer: String,
    pub sources: Vec<ScoredChunk>,
}

impl RagPipeline {
    pub fn new(
        embedding_client: std::sync::Arc<EmbeddingClient>,
        ollama_url: String,
        llm_model: String,
    ) -> Self {
        Self {
            embedding_client,
            ollama_url,
            llm_model,
        }
    }

    /// Generate an answer by retrieving relevant chunks and passing them to the LLM.
    pub async fn answer(
        &self,
        question: &str,
        retrieved: Vec<ScoredChunk>,
    ) -> anyhow::Result<RagAnswer> {
        if retrieved.is_empty() {
            return Ok(RagAnswer {
                answer: "No relevant documents found. Please upload a financial report first.".into(),
                sources: vec![],
            });
        }

        // Build context from retrieved chunks
        let context_parts: Vec<String> = retrieved
            .iter()
            .enumerate()
            .map(|(i, sc)| {
                format!(
                    "[Source {} - Section: {} | Score: {:.3}]\n{}",
                    i + 1,
                    sc.chunk.section,
                    sc.score,
                    sc.chunk.text
                )
            })
            .collect();
        let context = context_parts.join("\n\n---\n\n");

        let prompt = format!(
            r#"You are a financial document analyst. Answer the user's question based ONLY on the provided context from financial filings.

Rules:
1. If the answer is not in the context, say "This information is not available in the provided document."
2. Cite the source section for each claim.
3. For numerical data, present it clearly.
4. Do not make assumptions or use external knowledge.

Context:
{}

Question: {}

Answer:"#,
            context, question
        );

        let req = OllamaRequest {
            model: self.llm_model.clone(),
            prompt,
            stream: false,
        };

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?;

        let resp = client
            .post(format!("{}/api/generate", self.ollama_url))
            .json(&req)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("LLM request failed: {} - {}", status, body);
        }

        let ollama_resp: OllamaResponse = resp.json().await?;

        Ok(RagAnswer {
            answer: ollama_resp.response,
            sources: retrieved,
        })
    }

    pub fn embedding_client(&self) -> &EmbeddingClient {
        &self.embedding_client
    }
}