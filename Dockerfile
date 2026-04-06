# ══════════════════════════════════════════════════════════════════════════════
# Aegis-ML — Multi-stage Docker build using UV
# Target image: ~600 MB (base deps only, no HF/torch)
#               ~2.5 GB with HF extra
#
# Build:
#   docker build -t aegis-ml .
#   docker build --build-arg EXTRAS=hf -t aegis-ml:hf .
#
# Run:
#   docker run -p 8000:8000 -e CLASSIFIER_TYPE=sklearn aegis-ml
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: UV + dependency resolver ─────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:0.5 AS uv-base

# ── Stage 2: Python builder (compile all deps) ────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

# Install system build deps (needed by some Python packages with C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential libffi-dev libssl-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Copy UV binary from the official UV image
COPY --from=uv-base /uv /usr/local/bin/uv
COPY --from=uv-base /uvx /usr/local/bin/uvx

WORKDIR /build

# Copy only dependency files first (layer cache optimisation)
COPY pyproject.toml ./

# Which optional extras to install (default: base only; override with --build-arg EXTRAS=hf)
ARG EXTRAS=""

# Install dependencies into a venv using UV
# --mount=type=cache speeds up repeated builds by caching the UV download cache
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -z "$EXTRAS" ]; then \
        uv venv /app/.venv && \
        uv pip install --python /app/.venv/bin/python .; \
    else \
        uv venv /app/.venv && \
        uv pip install --python /app/.venv/bin/python ".[$EXTRAS]"; \
    fi

# ── Stage 3: Final slim runtime image ─────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash aegis

# Minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built venv from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY --chown=aegis:aegis app/ ./app/
COPY --chown=aegis:aegis demo/ ./demo/
COPY --chown=aegis:aegis training/ ./training/
COPY --chown=aegis:aegis pyproject.toml ./

# Create required directories
RUN mkdir -p /app/models /app/logs /app/data && \
    chown -R aegis:aegis /app/models /app/logs /app/data

# Use the venv's Python exclusively
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Default environment (override via docker-compose or -e flags)
ENV HOST=0.0.0.0
ENV PORT=8000
ENV CLASSIFIER_TYPE=sklearn
ENV CONFIDENCE_THRESHOLD=0.70
ENV LOG_LEVEL=INFO

USER aegis

EXPOSE 8000
EXPOSE 7860

# Health check — verify the /health endpoint is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command: run the FastAPI service
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
