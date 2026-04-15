# === Build stage ===
FROM rust:1.94.1-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY AGENTS.md README.md ./
COPY src/ src/
COPY config/ config/
COPY tests/ tests/

RUN cargo build --release --locked

# === Runtime stage (must match builder's glibc) ===
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/rein /usr/local/bin/rein

# Data directory for memories.db + Tantivy/HNSW indexes
RUN mkdir -p /data
VOLUME /data

ENV REIN_DB=/data/memories.db
ENV REIN_LOG=info
ENV REIN_SSE_BIND=0.0.0.0
ENV REIN_SSE_PORT=8680

EXPOSE 8680

# Default: SSE server bound to 0.0.0.0 (container-accessible)
CMD ["sh", "-lc", "if [ -z \"${REIN_HTTP_TOKEN:-}\" ]; then echo 'REIN_HTTP_TOKEN must be set when exposing rein HTTP/SSE from the container.' >&2; exit 64; fi; exec rein serve --sse"]
