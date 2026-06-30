# Financial RAG Analyzer

AI-Powered Financial Document Analyzer using Retrieval-Augmented Generation (RAG), built entirely in Rust.

Upload financial reports (10-K, annual reports, PDF, TXT, HTML) and ask natural-language questions about them. The system retrieves relevant passages using vector similarity search and generates grounded answers with source citations via a local LLM.

## Architecture

```
Upload Flow:
  File -> Text Extraction (PDF/HTML/TXT) -> Section-Aware Chunking (800 words, 150 overlap)
       -> Embedding (nomic-embed-text via Ollama) -> Vector Store (in-memory cosine similarity)

Query Flow:
  Question -> Embed -> Cosine Similarity Search (top-5) -> Prompt Construction
           -> LLM Generation (llama3.1:8b via Ollama) -> Answer + Source Citations
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Rust 1.96 (edition 2021) |
| Web Framework | Axum 0.7 (async, Tokio runtime) |
| HTTP Client | Reqwest 0.12 |
| PDF Extraction | pdf-extract 0.7 |
| HTML Parsing | Regex-based tag stripping |
| Embeddings | nomic-embed-text (via Ollama API) |
| LLM | llama3.1:8b (via Ollama API) |
| Vector Store | Custom in-memory with cosine similarity |
| Frontend | Single-page HTML/JS (dark theme, no dependencies) |

## Prerequisites

1. **Rust** (rustup): https://rustup.rs
2. **Ollama** running locally with models:
   ```bash
   ollama pull llama3.1:8b       # LLM for answer generation
   ollama pull nomic-embed-text  # Embedding model
   ```

## How to Run

```bash
# Build
cargo build

# Start the server (Ollama must be running on localhost:11434)
cargo run

# Open the UI in your browser
# http://localhost:3000
```

### Environment Variables (optional)

| Variable | Default | Description |
|---|---|---|
| `BIND_ADDR` | `127.0.0.1:3000` | Server bind address |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `LLM_MODEL` | `llama3.1:8b` | Ollama LLM model |
| `CHUNK_SIZE` | `800` | Words per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap words between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Chat UI (HTML) |
| `/api/health` | GET | Health check with model info and chunk count |
| `/api/docs` | GET | List uploaded documents |
| `/api/upload` | POST | Upload file (multipart, PDF/TXT/HTML) |
| `/api/ask` | POST | Ask a question (JSON body: `{"question": "..."}`) |
| `/api/delete/:name` | DELETE | Delete a document from the index |

## Usage Examples

### Upload a document
```bash
curl -F "file=@apple_10k.txt" http://localhost:3000/api/upload
```

Response:
```json
{"success": true, "document": "apple_10k.txt", "chunks": 8, "elapsed_ms": 44000}
```

### Ask a question
```bash
curl -X POST http://localhost:3000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the main risk factors?"}'
```

Response:
```json
{
  "answer": "According to [Source 1], the main risk factors are...",
  "sources": [
    {"chunk": {"section": "PART I", "text": "..."}, "score": 0.66},
    {"chunk": {"section": "Body", "text": "..."}, "score": 0.42}
  ],
  "elapsed_ms": 62000
}
```

## Project Structure

```
financial-rag/
├── Cargo.toml                 # Dependencies and package metadata
├── .cargo/config.toml         # Linker configuration for Windows GNU target
├── src/
│   ├── main.rs                # Entry point, server startup, shared state
│   ├── config.rs              # Configuration from environment variables
│   ├── document_processor.rs  # Text extraction (PDF/HTML/TXT) + chunking
│   ├── embeddings.rs          # Ollama embedding API client
│   ├── vector_store.rs        # In-memory vector store with cosine similarity
│   ├── rag_pipeline.rs        # RAG: retrieve chunks + LLM prompt + generation
│   └── server.rs              # Axum web server, API routes, handlers
├── static/
│   └── index.html             # Chat UI (dark theme, file upload, source citations)
├── sample_data/
│   └── apple_10k_sample.txt   # Synthetic Apple 10-K for testing
└── README.md
```

## RAG Prompt Design

The system prompt enforces anti-hallucination rules:

```
You are a financial document analyst. Answer the user's question based ONLY
on the provided context from financial filings.

Rules:
1. If the answer is not in the context, say "This information is not available
   in the provided document."
2. Cite the source section for each claim.
3. For numerical data, present it clearly.
4. Do not make assumptions or use external knowledge.
```

## Verified Test Results

Tested with a synthetic Apple 10-K filing (5,200 words):

| Question | Answer Quality | Retrieval Score | Latency |
|---|---|---|---|
| "What are the main risk factors?" | Listed all 6 risks with citations | 0.660 | 62s |
| "What was total revenue and net income?" | Correct: $383,285M revenue, $96,930M net income | 0.569 | 6.8s |

## Limitations

- **In-memory vector store**: Index is lost on restart (no persistence)
- **Embedding latency**: ~22s per chunk with nomic-embed-text on CPU
- **No PDF table extraction**: Financial tables in PDFs are extracted as flat text
- **Single-user**: Not designed for concurrent multi-user access
- **No streaming**: LLM response is received in full before returning
- **LLM accuracy**: llama3.1:8b is an 8B parameter model; larger models would improve answer quality

## Extending the Project

- **Persistent vector store**: Replace in-memory store withsled/redb or ChromaDB
- **SEC EDGAR integration**: Add automatic 10-K download by ticker symbol
- **Multi-document comparison**: Index multiple filings, filter by company metadata
- **Table extraction**: Use Camelot/tabula for structured financial table parsing
- **Streaming responses**: Use Server-Sent Events for token-by-token LLM output
- **Hybrid search**: Combine vector similarity with BM25 keyword search
- **Reranking**: Add a cross-encoder reranker for better retrieval precision

## License

MIT