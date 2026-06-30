# Financial RAG Analyzer - Enterprise Edition v2.0

AI-Powered Financial Document Analyzer using Retrieval-Augmented Generation (RAG), built entirely in Rust.

Upload financial reports (10-K, annual reports, PDF, TXT, HTML) and ask natural-language questions about them. The system retrieves relevant passages using vector similarity search and generates grounded answers with source citations via a local LLM.

## Enterprise Features (v2.0)

| Feature | Description |
|---|---|
| **Persistent Storage** | redb embedded database - survives restarts, no data loss |
| **JWT Authentication** | Pure-Rust HMAC-SHA256 JWT tokens, configurable on/off |
| **Rate Limiting** | Per-user request limits (configurable, default 60/min) |
| **Async Document Queue** | Background worker pool processes uploads non-blocking |
| **Streaming Responses** | SSE (Server-Sent Events) for token-by-token LLM output |
| **Configurable** | TOML config file + environment variable overrides |
| **Request Limits** | Body size limits, request timeouts |
| **Structured Logging** | tracing crate with env-filter, JSON output support |
| **Docker Ready** | Multi-stage Dockerfile + docker-compose with Ollama |
| **Source Filtering** | Filter retrieval by document name |
| **Job Status API** | Poll async document processing status |
| **CORS** | Configurable per-origin or wildcard |
| **Trace Layer** | HTTP request tracing middleware |

## Architecture

```
Upload Flow (Async):
  File -> Extract Text -> Submit to Queue -> Worker Pool processes:
    Chunk (800 words, 150 overlap) -> Embed (nomic-embed-text) -> Store (redb persistent)

Query Flow:
  Question -> Embed -> Vector Search (top-K, cosine similarity) -> Prompt Construction
           -> LLM Generation (llama3.1:8b) -> Answer + Source Citations

Streaming Flow:
  Question -> Embed -> Vector Search -> SSE Stream -> Token-by-token LLM output
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Rust (edition 2021) |
| Web Framework | Axum 0.7 (async, Tokio runtime) |
| Database | redb (pure-Rust embedded KV store) |
| Auth | HMAC-SHA256 JWT (pure Rust, no C deps) |
| PDF Extraction | pdf-extract 0.7 |
| Embeddings | nomic-embed-text (via Ollama API) |
| LLM | llama3.1:8b (via Ollama API) |
| Frontend | Single-page HTML/JS (dark theme, SSE support) |
| Config | figment (TOML + env vars) |
| Container | Docker multi-stage build |

## Prerequisites

1. **Rust** (rustup): https://rustup.rs
2. **Ollama** running locally with models:
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

## How to Run

### Option A: Direct

```bash
cargo build
cargo run
# Open http://localhost:3000
```

### Option B: Docker Compose

```bash
docker-compose up -d
# Server on http://localhost:3000
# Ollama on http://localhost:11434
# Pull models: docker exec -it <ollama-container> ollama pull llama3.1:8b
```

## Configuration

Edit `config.toml` or override with environment variables prefixed `RAG_`:

```bash
RAG_SERVER_BIND_ADDR=0.0.0.0:8080
RAG_AUTH_ENABLED=true
RAG_AUTH_JWT_SECRET=your-secret-key
RAG_OLLAMA_URL=http://ollama:11434
```

### Full Config Reference

```toml
[server]
bind_addr = "127.0.0.1:3000"
max_connections = 1000
request_timeout_secs = 120
max_body_size_mb = 100
cors_origins = ["*"]

[ollama]
url = "http://localhost:11434"
embedding_model = "nomic-embed-text"
llm_model = "llama3.1:8b"
embedding_batch_size = 10
request_timeout_secs = 120

[rag]
chunk_size = 800
chunk_overlap = 150
top_k = 5
min_score = 0.15
max_context_tokens = 4096
system_prompt = "..."

[auth]
enabled = false
jwt_secret = "change-me-in-production"
jwt_expiry_hours = 24
rate_limit_per_minute = 60

[storage]
sled_path = "./data/vectors.sled"
vector_cache_size_mb = 256
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Chat UI (HTML) |
| `/api/health` | GET | Health check with version, model info, chunk count |
| `/api/login` | POST | Get JWT token (`{"username":"admin"}`) |
| `/api/docs` | GET | List uploaded documents |
| `/api/upload` | POST | Upload file (multipart) - returns job_id for async processing |
| `/api/ask` | POST | Ask a question (JSON) - blocking response |
| `/api/ask/stream` | POST | Ask a question - SSE streaming response |
| `/api/job/:job_id` | GET | Check async document processing status |
| `/api/delete/:name` | DELETE | Delete a document from the index |

## Project Structure

```
financial-rag/
├── Cargo.toml                 # Dependencies and package metadata
├── config.toml                # Default configuration
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # Docker Compose with Ollama
├── .cargo/config.toml         # Linker configuration for Windows
├── src/
│   ├── main.rs                # Entry point, server startup
│   ├── config.rs              # Structured config (figment: TOML + env)
│   ├── auth.rs                # JWT auth + rate limiting middleware
│   ├── document_processor.rs  # Text extraction + section-aware chunking
│   ├── document_queue.rs      # Async processing queue with worker pool
│   ├── embeddings.rs          # Ollama embedding API client
│   ├── vector_store.rs        # Persistent vector store (redb + cosine sim)
│   ├── rag_pipeline.rs        # RAG: retrieve + generate + streaming
│   └── server.rs              # Axum web server, all API routes
├── static/
│   └── index.html             # Chat UI (dark theme, SSE streaming)
├── sample_data/
│   └── apple_10k_sample.txt   # Synthetic Apple 10-K for testing
└── README.md
```

## Usage Examples

### Upload and process a document
```bash
curl -F "file=@apple_10k.pdf" http://localhost:3000/api/upload
# -> {"success":true, "job_id":"abc-123", "document":"apple_10k.pdf", "message":"..."}

# Check processing status
curl http://localhost:3000/api/job/abc-123
# -> {"job_id":"abc-123", "status":"done", "result":{"chunk_count":15}}
```

### Ask a question (blocking)
```bash
curl -X POST http://localhost:3000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the main risk factors?"}'
```

### Ask a question (streaming via SSE)
```bash
curl -X POST http://localhost:3000/api/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"What was total revenue?"}'
```

### With authentication enabled
```bash
# Get token
TOKEN=$(curl -s -X POST http://localhost:3000/api/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin"}' | jq -r .token)

# Use token
curl -H "Authorization: Bearer $TOKEN" http://localhost:3000/api/docs
```

## License

MIT