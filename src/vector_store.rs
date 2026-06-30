use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::document_processor::Chunk;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoredChunk {
    pub chunk: Chunk,
    pub score: f32,
}

pub struct VectorStore {
    /// Stores chunks and their embedding vectors
    entries: Vec<(Chunk, Vec<f32>)>,
    /// Maps source document name -> count of chunks
    doc_counts: HashMap<String, usize>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            doc_counts: HashMap::new(),
        }
    }

    /// Add chunks with their embedding vectors.
    pub fn add(&mut self, chunks: Vec<Chunk>, vectors: Vec<Vec<f32>>) {
        let source = chunks
            .first()
            .map(|c| c.source.clone())
            .unwrap_or_default();

        let count = chunks.len();
        for (chunk, vec) in chunks.into_iter().zip(vectors.into_iter()) {
            self.entries.push((chunk, vec));
        }

        *self.doc_counts.entry(source).or_insert(0) += count;
    }

    /// Remove all chunks from a given source document.
    pub fn remove_document(&mut self, source: &str) -> usize {
        let before = self.entries.len();
        self.entries.retain(|(c, _)| c.source != source);
        let removed = before - self.entries.len();
        self.doc_counts.remove(source);
        removed
    }

    /// Search for top-K most similar chunks using cosine similarity.
    pub fn search(&self, query_vec: &[f32], top_k: usize) -> Vec<ScoredChunk> {
        let mut scored: Vec<ScoredChunk> = self
            .entries
            .iter()
            .map(|(chunk, vec)| {
                let score = cosine_similarity(query_vec, vec);
                ScoredChunk {
                    chunk: chunk.clone(),
                    score,
                }
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored.into_iter().take(top_k).collect()
    }

    pub fn doc_names(&self) -> Vec<String> {
        self.doc_counts.keys().cloned().collect()
    }

    pub fn total_chunks(&self) -> usize {
        self.entries.len()
    }

    pub fn doc_info(&self) -> Vec<(String, usize)> {
        self.doc_counts
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}