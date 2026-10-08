# syntax=docker/dockerfile:1

# ---- Build stage ----
FROM rust:1.98-bookworm AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Build dependencies first so they're cached separately from app code.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src \
    && echo "fn main() {}" > src/main.rs \
    && echo "" > src/lib.rs \
    && cargo build --release --bin rusty_rich \
    && rm -rf src

COPY src ./src
RUN touch src/main.rs src/lib.rs \
    && cargo build --release --bin rusty_rich

# ---- Runtime stage ----
FROM debian:bookworm-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/rusty_rich ./rusty_rich
COPY frontend ./frontend

ENV RUST_LOG=info
EXPOSE 8080

CMD ["./rusty_rich"]
