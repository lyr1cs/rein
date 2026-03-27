# === Build stage ===
FROM rust:latest AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY config/ config/
COPY tests/ tests/

RUN cargo build --release

# === Runtime stage (must match builder's glibc) ===
FROM debian:trixie-slim

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
CMD ["rein", "serve", "--sse"]
