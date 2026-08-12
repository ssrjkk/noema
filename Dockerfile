# =============================================================================
# Stage 1: Builder — install dependencies and compile
# =============================================================================
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml README.md ./
COPY noema/ noema/

RUN pip install --upgrade pip && \
    pip install build && \
    pip install -e ".[dev,db,full,sentry]"

# =============================================================================
# Stage 2: Runtime — minimal image
# =============================================================================
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NOEMA_API__HOST=0.0.0.0 \
    NOEMA_API__PORT=8000 \
    NOEMA_LLM__PROVIDER=fallback \
    NOEMA_NS__ENABLED=false

# Create non-root user
RUN groupadd -r noema && useradd -r -g noema -d /app -s /sbin/nologin noema && \
    mkdir -p /app /app/data /app/.noema && \
    chown -R noema:noema /app

WORKDIR /app

# Copy only runtime deps from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /build/noema/ noema/
COPY --from=builder /build/pyproject.toml .

# Runtime data directories
VOLUME ["/app/data", "/app/.noema"]

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

USER noema
EXPOSE 8000

ENTRYPOINT ["python", "-m", "uvicorn", "noema.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
