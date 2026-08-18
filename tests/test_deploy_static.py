"""Static checks for deployment artifacts (Dockerfile, docker-compose, Helm).

These run without docker/helm installed: they catch the class of mistakes
that previously broke the image build (python version mismatch between
builder/runtime, editable-install ``.pth`` pointing outside the image,
plaintext default credentials).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_python_versions_match():
    d = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    from_stages = [line for line in d.splitlines() if line.startswith("FROM python:")]
    assert len(from_stages) == 2
    versions = {line.split()[1] for line in from_stages}
    assert versions == {"python:3.12-slim"}, f"stage versions must match, got {versions}"
    assert "/usr/local/lib/python3.12/site-packages/" in d


def test_dockerfile_install_is_not_editable():
    d = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "-e " not in d, "editable installs leave .pth files pointing at the builder dir"
    assert 'pip install ".[dev,db,full,sentry]"' in d


def test_compose_redis_requires_password():
    c = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "--requirepass" in c
    assert "NOEMA_REDIS__URL" in c


def test_compose_no_hardcoded_plaintext_db_password():
    c = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "POSTGRES_PASSWORD: noema" not in c
    assert "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-noema}" in c


def test_compose_env_template_committed():
    assert (ROOT / "compose.env.example").is_file()
    example = (ROOT / "compose.env.example").read_text(encoding="utf-8")
    assert "POSTGRES_PASSWORD" in example and "REDIS_PASSWORD" in example


def test_compose_env_is_gitignored():
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "compose.env" in gi


def test_helm_no_plaintext_default_passwords():
    values = (ROOT / "deploy/helm/noema/values.yaml").read_text(encoding="utf-8")
    assert "password: noema" not in values
    assert 'password: ""' in values, "sub-chart passwords must default empty (auto-generate)"


def test_helm_secrets_via_secret_not_configmap():
    dep = (ROOT / "deploy/helm/noema/templates/deployment.yaml").read_text(encoding="utf-8")
    assert "secretKeyRef" in dep
