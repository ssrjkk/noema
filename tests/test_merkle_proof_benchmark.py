"""Benchmarks for Merkle inclusion proofs — proving O(log N) proof size and
fast verification at enterprise scale."""

import json

from noema.audit.merkle_proof import (
    _safe_hash,
    _serialize,
    compute_root,
    generate_inclusion_proof,
    verify_inclusion_proof,
)

N = 100_000
_DATA, _HASHES = None, None


def _dataset():
    global _DATA, _HASHES
    if _DATA is None:
        leaf_data = [{"task_id": f"task-{i}", "result": f"r-{i}"} for i in range(N)]
        _DATA = leaf_data
        _HASHES = [_safe_hash(_serialize(d)) for d in leaf_data]
    return _DATA, _HASHES


def test_proof_path_is_log_n():
    data, hashes = _dataset()
    proof = generate_inclusion_proof(data[N // 2], N // 2, hashes)
    expected_depth = (N - 1).bit_length()
    assert len(proof.path) == expected_depth


def test_proof_size_in_bytes():
    data, hashes = _dataset()
    proof = generate_inclusion_proof(data[0], 0, hashes)
    doc = json.dumps(proof.to_dict())
    assert len(doc.encode("utf-8")) <= 4096  # ~2-3 KB for 100K events


def test_verify_fast(benchmark):
    data, hashes = _dataset()
    proof = generate_inclusion_proof(data[0], 0, hashes)

    def _verify():
        assert verify_inclusion_proof(proof, data[0]) is True

    benchmark(_verify)


def test_generate_fast(benchmark):
    data, hashes = _dataset()

    def _generate():
        return generate_inclusion_proof(data[12345], 12345, hashes)

    benchmark(_generate)


def test_root_stable_across_generation():
    data, hashes = _dataset()
    root = compute_root(hashes)
    for i in (0, 50_000, N - 1):
        proof = generate_inclusion_proof(data[i], i, hashes)
        assert proof.root_hash == root
