use serde::{Deserialize, Serialize};

use crate::config::RagConfig;
use crate::vector_store::ScoredChunk;

#[derive(Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
    done: bool,
}

#[derive(Serialize, Clone)]
pub struct RagAnswer {
    pub answer: String,
    pub sources: Vec<ScoredChunk>,
    pub elapsed_ms: u64,
}

pub struct RagPipeline {
    config: RagConfig,
    ollama_url: String,
    llm_model: String,
    client: reqwest::Client,
}

impl RagPipeline {
    pub fn new(config: RagConfig, ollama_url: String, llm_model: String) -> Self {
        let timeout = std::time::Duration::from_secs(config.max_context_tokens as u64);
        Self {
            config,
            ollama_url,
            llm_model,
            client: reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .unwrap(),
        }
    }

    pub fn build_prompt(&self, question: &str, retrieved: &[ScoredChunk]) -> String {
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

        format!(
            "{}\n\nContext:\n{}\n\nQuestion: {}\n\nAnswer:",
            self.config.system_prompt, context, question
        )
    }

    pub async fn answer(
        &self,
        question: &str,
        retrieved: Vec<ScoredChunk>,
    ) -> anyhow::Result<RagAnswer> {
        let start = std::time::Instant::now();

        if retrieved.is_empty() {
            return Ok(RagAnswer {
                answer: "No relevant documents found. Please upload a financial report first.".into(),
                sources: vec![],
                elapsed_ms: 0,
            });
        }

        let prompt = self.build_prompt(question, &retrieved);

        let req = OllamaGenerateRequest {
            model: self.llm_model.clone(),
            prompt,
            stream: false,
        };

        let resp = self
            .client
            .post(format!("{}/api/generate", self.ollama_url))
            .json(&req)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("LLM request failed: {} - {}", status, body);
        }

        let ollama_resp: OllamaGenerateResponse = resp.json().await?;

        Ok(RagAnswer {
            answer: ollama_resp.response,
            sources: retrieved,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    pub async fn answer_stream(
        &self,
        question: &str,
        retrieved: Vec<ScoredChunk>,
    ) -> anyhow::Result<tokio_stream::wrappers::ReceiverStream<Result<String, std::io::Error>>> {
        if retrieved.is_empty() {
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            let _ = tx.send(Ok("No relevant documents found.".to_string())).await;
            return Ok(tokio_stream::wrappers::ReceiverStream::new(rx));
        }

        let prompt = self.build_prompt(question, &retrieved);

        let req_body = serde_json::json!({
            "model": self.llm_model,
            "prompt": prompt,
            "stream": true,
        });

        let resp = self
            .client
            .post(format!("{}/api/generate", self.ollama_url))
            .json(&req_body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("LLM stream request failed: {} - {}", status, body);
        }

        let byte_stream = resp.bytes_stream();
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        tokio::spawn(async move {
            use tokio_stream::StreamExt;
            let mut byte_stream = byte_stream;
            let mut buffer = String::new();

            while let Some(chunk_result) = byte_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        buffer.push_str(&String::from_utf8_lossy(&chunk));
                        while let Some(pos) = buffer.find('\n') {
                            let line = buffer[..pos].to_string();
                            buffer = buffer[pos + 1..].to_string();

                            if line.is_empty() {
                                continue;
                            }

                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                                if let Some(token) = json.get("response").and_then(|v| v.as_str()) {
                                    if !token.is_empty() {
                                        if tx.send(Ok(token.to_string())).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            e.to_string(),
                        ))).await;
                        break;
                    }
                }
            }
        });

        Ok(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}