FROM rust:1.96-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo/
COPY src/ src/
COPY static/ static/

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/financial-rag /app/financial-rag
COPY --from=builder /app/config.toml /app/config.toml
COPY --from=builder /app/static/ /app/static/
COPY --from=builder /app/sample_data/ /app/sample_data/

RUN mkdir -p /app/data

EXPOSE 3000

ENV RUST_LOG=info
ENV RAG_CONFIG_PATH=/app/config.toml

CMD ["/app/financial-rag"]