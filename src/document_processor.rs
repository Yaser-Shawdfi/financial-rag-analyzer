use std::path::Path;

use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    pub source: String,
    pub section: String,
    pub index: usize,
}

pub struct DocumentProcessor {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl DocumentProcessor {
    pub fn new() -> Self {
        Self {
            chunk_size: 800,
            chunk_overlap: 150,
        }
    }

    pub fn with_params(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
        }
    }

    /// Extract text from a file based on its extension.
    pub fn extract_text(&self, file_path: &Path) -> anyhow::Result<String> {
        let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");

        match ext {
            "pdf" => self.extract_pdf(file_path),
            "html" | "htm" => self.extract_html(file_path),
            "txt" => Ok(std::fs::read_to_string(file_path)?),
            _ => Ok(std::fs::read_to_string(file_path)?),
        }
    }

    fn extract_pdf(&self, path: &Path) -> anyhow::Result<String> {
        let text = pdf_extract::extract_text(path)?;
        Ok(self.clean_text(&text))
    }

    fn extract_html(&self, path: &Path) -> anyhow::Result<String> {
        let raw = std::fs::read_to_string(path)?;
        Ok(self.strip_html(&raw))
    }

    /// Remove HTML tags, scripts, styles; keep text content.
    fn strip_html(&self, html: &str) -> String {
        // Remove script/style blocks
        let script_re = Regex::new(r"(?is)<(script|style)[^>]*>.*?</\1>").unwrap();
        let tag_re = Regex::new(r"(?s)<[^>]+>").unwrap();
        let entity_re = Regex::new(r"&[a-zA-Z#0-9]+;").unwrap();

        let no_scripts = script_re.replace_all(html, " ");
        let no_tags = tag_re.replace_all(&no_scripts, " ");
        let no_entities = entity_re.replace_all(&no_tags, " ");

        self.clean_text(&no_entities)
    }

    fn clean_text(&self, text: &str) -> String {
        let whitespace_re = Regex::new(r"\s+").unwrap();
        whitespace_re.replace_all(text, " ").trim().to_string()
    }

    /// Split text into overlapping chunks, with section detection for 10-K style documents.
    pub fn chunk_text(&self, text: &str, source: &str) -> Vec<Chunk> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return vec![];
        }

        let section_re =
            Regex::new(r"(?i)(?:^|\s)(Item\s+\d+[A-Z]?\.?)|(?:^|\s)(PART\s+[IVX]+)").ok();

        let mut chunks = Vec::new();
        let mut idx = 0;
        let mut pos = 0;

        while pos < words.len() {
            let end = (pos + self.chunk_size).min(words.len());
            let chunk_words = &words[pos..end];
            let chunk_text = chunk_words.join(" ");

            // Detect section from chunk content
            let section = if let Some(re) = &section_re {
                re.captures(&chunk_text)
                    .map(|c| {
                        c.get(0)
                            .map(|m| m.as_str().trim().to_string())
                            .unwrap_or_else(|| "Unknown".into())
                    })
                    .unwrap_or_else(|| {
                        if pos == 0 {
                            "Header/Overview".into()
                        } else {
                            "Body".into()
                        }
                    })
            } else {
                "Unknown".into()
            };

            chunks.push(Chunk {
                id: format!("{}_{}", source, idx),
                text: chunk_text,
                source: source.to_string(),
                section,
                index: idx,
            });

            idx += 1;

            // Advance with overlap
            let advance = self.chunk_size.saturating_sub(self.chunk_overlap);
            if advance == 0 {
                break;
            }
            pos += advance;
        }

        tracing::info!("Chunked '{}' into {} chunks", source, chunks.len());
        chunks
    }
}