use std::sync::Arc;

use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::document_processor::Chunk;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoredChunk {
    pub chunk: Chunk,
    pub score: f32,
}

#[derive(Serialize, Deserialize)]
struct StoredEntry {
    chunk: Chunk,
    vector: Vec<f32>,
}

const TABLE: TableDefinition<&str, Vec<u8>> = TableDefinition::new("vectors");

pub struct VectorStore {
    db: Arc<Database>,
    cache: Arc<RwLock<Vec<(Chunk, Vec<f32>)>>>,
}

impl VectorStore {
    pub fn new(path: &str) -> anyhow::Result<Self> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let db = Database::create(path)?;
        let db_arc = Arc::new(db);

        {
            let txn = db_arc.begin_write()?;
            let _ = txn.open_table(TABLE)?;
            txn.commit()?;
        }

        let mut entries = Vec::new();
        {
            let txn = db_arc.begin_read()?;
            if let Ok(table) = txn.open_table(TABLE) {
                for item in table.iter()? {
                    let (_key, value) = item?;
                    let bytes = value.value();
                    if let Ok(entry) = bincode::deserialize::<StoredEntry>(&bytes) {
                        entries.push((entry.chunk, entry.vector));
                    }
                }
            }
        }

        tracing::info!("Loaded {} vectors from redb", entries.len());

        Ok(Self {
            db: db_arc,
            cache: Arc::new(RwLock::new(entries)),
        })
    }

    pub async fn add(&self, chunks: Vec<Chunk>, vectors: Vec<Vec<f32>>) -> anyhow::Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(TABLE)?;
            for (chunk, vector) in chunks.iter().zip(vectors.iter()) {
                let entry = StoredEntry {
                    chunk: chunk.clone(),
                    vector: vector.clone(),
                };
                let bytes = bincode::serialize(&entry)?;
                table.insert(chunk.id.as_str(), bytes)?;
            }
        }
        txn.commit()?;

        let mut cache = self.cache.write().await;
        for (chunk, vector) in chunks.into_iter().zip(vectors.into_iter()) {
            cache.push((chunk, vector));
        }

        Ok(())
    }

    pub async fn search(
        &self,
        query_vec: &[f32],
        top_k: usize,
        min_score: f32,
        source_filter: Option<&str>,
    ) -> Vec<ScoredChunk> {
        let cache = self.cache.read().await;

        let mut scored: Vec<ScoredChunk> = cache
            .iter()
            .filter(|(chunk, _)| {
                if let Some(filter) = source_filter {
                    chunk.source == filter
                } else {
                    true
                }
            })
            .map(|(chunk, vector)| {
                let score = cosine_similarity(query_vec, vector);
                ScoredChunk {
                    chunk: chunk.clone(),
                    score,
                }
            })
            .filter(|sc| sc.score >= min_score)
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored.into_iter().take(top_k).collect()
    }

    pub fn remove_document(&self, source: &str) -> usize {
        let cache_keys: Vec<String> = {
            let cache = self.cache.try_write();
            match cache {
                Ok(c) => c
                    .iter()
                    .filter(|(chunk, _)| chunk.source == source)
                    .map(|(chunk, _)| chunk.id.clone())
                    .collect(),
                Err(_) => vec![],
            }
        };

        let removed = cache_keys.len();

        if let Ok(txn) = self.db.begin_write() {
            if let Ok(mut table) = txn.open_table(TABLE) {
                for key in &cache_keys {
                    let _ = table.remove(key.as_str());
                }
            }
            let _ = txn.commit();
        }

        {
            if let Ok(mut cache) = self.cache.try_write() {
                cache.retain(|(chunk, _)| chunk.source != source);
            }
        }

        removed
    }

    pub fn doc_info(&self) -> Vec<(String, usize)> {
        let cache = self.cache.try_read();
        match cache {
            Ok(c) => {
                let mut counts: std::collections::HashMap<String, usize> =
                    std::collections::HashMap::new();
                for (chunk, _) in c.iter() {
                    *counts.entry(chunk.source.clone()).or_insert(0) += 1;
                }
                counts.into_iter().collect()
            }
            Err(_) => vec![],
        }
    }

    pub fn total_chunks(&self) -> usize {
        self.cache.try_read().map(|c| c.len()).unwrap_or(0)
    }

    pub fn doc_names(&self) -> Vec<String> {
        self.doc_info().into_iter().map(|(n, _)| n).collect()
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