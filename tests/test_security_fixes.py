"""Regression tests for security fixes: scaffolder traversal, token
signing, merkle import verification, healer timeouts, tracer stack."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from noema.audit.merkle import MerkleChainAudit
from noema.healer.engine import HealingStrategy, SelfHealer
from noema.modules.auth.kernel import TokenManager
from noema.scaffolder.generator import ProjectScaffolder
from noema.tracing.tracer import Tracer

# ── Scaffolder: path traversal ───────────────────────────────────────────


def _solution_with_filename(tmp_path, filename):
    from noema.core.types import CodeBlock, Solution, Task, TechStack

    task = Task(title="Safe Scaffold", description="desc")
    solution = Solution(
        task_id=task.id,
        title="Safe Scaffold",
        summary="s",
        stack=TechStack(languages=["python"]),
        code_blocks=[
            CodeBlock(
                filename=filename,
                language="python",
                content="print('ok')",
            )
        ],
    )
    return solution, task


def test_scaffolder_sanitizes_traversal(tmp_path):
    scaffolder = ProjectScaffolder(output_dir=str(tmp_path))
    solution, task = _solution_with_filename(tmp_path, "../../evil.py")
    result = asyncio.run(scaffolder.scaffold(solution, task))
    project_root = tmp_path / "safe_scaffold"
    files = result["files"]
    assert files, "expected files to be created"
    # Nothing may land outside the project dir; the payload file must exist
    # under its sanitized name.
    assert any(Path(f).name == "evil.py" for f in files), f"unexpected files: {files}"
    assert all((project_root / f).resolve().is_relative_to(project_root.resolve()) for f in files)
    assert (project_root / "src" / "safe_scaffold" / "evil.py").is_file()


def test_scaffolder_rejects_absolute_paths(tmp_path):
    scaffolder = ProjectScaffolder(output_dir=str(tmp_path))
    solution, task = _solution_with_filename(tmp_path, "/etc/cron.d/evil")
    asyncio.run(scaffolder.scaffold(solution, task))
    assert not (tmp_path / "etc").exists(), "absolute path must not escape"
    assert (tmp_path / "safe_scaffold").exists()


# ── Auth: token signing ──────────────────────────────────────────────────


def test_token_manager_default_secret_is_random():
    a = TokenManager()
    b = TokenManager()
    assert a.secret != b.secret
    assert len(a.secret) >= 32


def test_token_manager_signature_not_truncated():
    tm = TokenManager(secret="test-secret")
    pair = tm.create_tokens("user-1", ["admin"])
    body, sig = pair.access_token.rsplit(".", 1)
    import hashlib
    import hmac

    expected = hmac.new(b"test-secret", body.encode(), hashlib.sha256).hexdigest()
    assert sig == expected, "MAC must not be truncated"
    assert len(sig) == 64


def test_token_manager_verify_roundtrip():
    tm = TokenManager(secret="test-secret")
    pair = tm.create_tokens("user-1", ["admin"])
    payload = tm.verify_token(pair.access_token)
    assert payload is not None
    assert payload["sub"] == "user-1"


def test_token_manager_rejects_forged_token():
    tm = TokenManager(secret="test-secret")
    pair = tm.create_tokens("user-1")
    forged = pair.access_token[:-8] + "deadbeef"
    assert tm.verify_token(forged) is None


# ── Merkle: import verification ──────────────────────────────────────────


def test_merkle_import_roundtrip_ok():
    chain = MerkleChainAudit(chain_id="chain-1")
    chain.append(payload={"task_id": "t1", "tenant_id": "default"})
    chain.append(payload={"task_id": "t2", "tenant_id": "default"})
    blocks = chain.export_blocks()
    imported = MerkleChainAudit.import_blocks("chain-1", blocks)
    assert imported.verify_chain()
    assert imported.tip_hash == chain.tip_hash


def test_merkle_import_rejects_tampered_chain():
    chain = MerkleChainAudit(chain_id="chain-1")
    chain.append(payload={"task_id": "t1", "tenant_id": "default"})
    chain.append(payload={"task_id": "t2", "tenant_id": "default"})
    blocks = chain.export_blocks()
    # Tamper: rewrite a block hash (simulating a forged export).
    tampered = [dict(b) for b in blocks]
    tampered[-1]["block_hash"] = "00" * 32
    with pytest.raises(ValueError, match="verification"):
        MerkleChainAudit.import_blocks("chain-1", tampered)


def test_merkle_import_rejects_broken_link():
    chain = MerkleChainAudit(chain_id="chain-1")
    chain.append(payload={"task_id": "t1", "tenant_id": "default"})
    blocks = chain.export_blocks()
    tampered = [dict(b) for b in blocks]
    tampered[-1]["prev_hash"] = "11" * 32
    with pytest.raises(ValueError, match="verification"):
        MerkleChainAudit.import_blocks("chain-1", tampered)


# ── Healer: hung coroutine must time out ─────────────────────────────────


async def test_healer_timeouts_hung_coroutine():
    strategy = HealingStrategy(timeout=0.05, max_retries=1)

    async def _hung(*args, **kwargs):
        await asyncio.sleep(10)

    healer = SelfHealer(strategy=strategy)
    result = await healer.execute_with_healing(_hung, fallback="fb")
    assert result == "fb"
    assert healer.history[-1].action_taken == "fallback"


# ── Tracer: explicit spans must not leak the stack ───────────────────────


def test_tracer_stack_popped_with_explicit_span():
    tracer = Tracer()
    span = tracer.start_span("outer")
    inner = tracer.start_span("inner")
    tracer.end_span(inner)
    tracer.end_span(span)
    assert tracer._stack == []
    assert len(tracer._spans) == 2


# ── Semantic cache: no cross-tenant leaks ────────────────────────────────


class _DeterministicEmbedder:
    """Maps text to fixed orthogonal vectors so similarity is predictable."""

    is_semantic = True

    def embed_one(self, text: str):
        import numpy as np

        if text.strip().startswith("hit seed") or text.strip() == "want a hit":
            vec = [1.0, 0.0, 0.0, 0.0]
        elif text.strip() == "secret b prompt":
            vec = [0.0, 1.0, 0.0, 0.0]
        else:
            vec = [0.0, 0.0, 1.0, 0.0]
        return np.asarray([vec], dtype=np.float32)


def test_semantic_cache_scoped_per_tenant():
    from noema.cache import SemanticCache

    cache = SemanticCache(similarity_threshold=0.5)
    cache._embedder = _DeterministicEmbedder()
    # >10 entries so the semantic scan path is exercised.
    for i in range(11):
        cache.set(
            [{"role": "user", "content": f"hit seed {i}"}],
            f"resp-a-{i}",
            "m1",
            tenant_id="a",
        )
    cache.set(
        [{"role": "user", "content": "secret b prompt"}],
        "secret-b-response",
        "m1",
        tenant_id="b",
    )

    # Same-tenant semantic hit still works.
    hit = cache.get([{"role": "user", "content": "want a hit"}], "m1", tenant_id="a")
    assert hit is not None and hit.startswith("resp-a-")

    # Tenant "a" must never receive tenant "b"'s cached response.
    miss = cache.get([{"role": "user", "content": "secret b prompt"}], "m1", tenant_id="a")
    assert miss is None

    # Tenant "b" still gets its own exact hit.
    own = cache.get([{"role": "user", "content": "secret b prompt"}], "m1", tenant_id="b")
    assert own == "secret-b-response"
