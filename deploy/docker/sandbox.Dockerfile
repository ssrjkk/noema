# =============================================================================
# noema sandbox image — Docker environment for safe code execution.
#
# This image ships the trusted tooling required by the Docker sandbox stages:
#   * ruff   (used by the `lint` stage)
#   * pytest (used by the `run_tests` stage)
#
# The sandbox engine keeps `--network=none` at runtime for all stages, so the
# tools must be baked into the image at build time — they cannot be installed
# at runtime (no network). Plain `python:3.12-slim` does NOT contain ruff or
# pytest, which is why the stock image broke lint/tests.
#
# Build:
#   docker build -f deploy/docker/sandbox.Dockerfile -t noema-sandbox:3.12 .
# Push (or otherwise make available on the host) before pointing
#   NOEMA_SANDBOX_DOCKER_IMAGE (or the code default) at it.
# =============================================================================

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --upgrade pip && \
    pip install ruff pytest pytest-timeout

# Default execution stage: the sandbox engine overrides the command
# (python/ruff/pytest) per stage, so no CMD is required.
