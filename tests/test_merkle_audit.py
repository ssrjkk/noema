from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from noema.audit.merkle import (
    AuditBlock,
    MerkleChainAudit,
    _compute_commitment,
    _hash,
)

# ── Property: Hash functions are deterministic ──────────────────────────────────


@given(st.binary(min_size=1, max_size=128))
@settings(max_examples=100)
def test_hash_deterministic(data):
    h1 = _hash(data)
    h2 = _hash(data)
    assert h1 == h2
    assert len(h1) == 32


@given(st.binary(min_size=1, max_size=64), st.binary(min_size=1, max_size=64))
@settings(max_examples=100)
def hash_different_inputs_different_hashes(a, b):
    assume(a != b)
    assert _hash(a) != _hash(b)


# ── Property: Commitment hides payload ──────────────────────────────────────────


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.one_of(
            st.integers(), st.text(max_size=20), st.floats(allow_nan=False, allow_infinity=False)
        ),
        min_size=1,
        max_size=5,
    )
)
@settings(max_examples=50)
def test_commitment_deterministic(payload):
    c1 = _compute_commitment(payload)
    c2 = _compute_commitment(payload)
    assert c1 == c2
    assert len(c1) == 32


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.one_of(st.integers(), st.text(max_size=20)),
        min_size=1,
        max_size=5,
    )
)
@settings(max_examples=50)
def test_commitment_length(payload):
    commitment = _compute_commitment(payload)
    assert len(commitment) == 32


# ── Property: Merkle Chain Initialization ───────────────────────────────────────


@given(st.text(min_size=1, max_size=16, alphabet="abcdef0123456789"))
@settings(max_examples=20)
def test_chain_initialization(chain_id):
    chain = MerkleChainAudit(chain_id=chain_id)
    assert chain.chain_id == chain_id
    assert chain.height == 1
    assert chain.tip.index == 0
    assert chain.verify_chain()


@given(st.text(min_size=0, max_size=8, alphabet="abcdef0123456789"))
@settings(max_examples=20)
def test_chain_id_default(chain_id):
    chain = MerkleChainAudit(chain_id=chain_id or None)
    if chain_id:
        assert chain.chain_id == chain_id
    else:
        assert len(chain.chain_id) == 8


# ── Property: Append preserves chain integrity ──────────────────────────────────


@given(st.integers(min_value=1, max_value=20), st.text(min_size=1, max_size=20))
@settings(max_examples=50)
def test_chain_append_preserves_integrity(count, tenant_id):
    chain = MerkleChainAudit()
    for i in range(count):
        chain.append({"task_id": f"task_{i}", "tenant_id": tenant_id, "event": "test"})
    assert chain.height == count + 1
    assert chain.verify_chain()


@given(
    st.lists(
        st.fixed_dictionaries(
            {
                "task_id": st.text(min_size=1, max_size=16),
                "tenant_id": st.text(min_size=1, max_size=8),
                "event": st.just("task_completed"),
            }
        ),
        min_size=1,
        max_size=30,
    )
)
@settings(max_examples=50)
def test_chain_append_multiple_events(events):
    chain = MerkleChainAudit()
    for ev in events:
        chain.append(ev)
    assert chain.height == len(events) + 1
    assert chain.verify_chain()


# ── Property: Merkle Proof ──────────────────────────────────────────────────────


@given(st.integers(min_value=2, max_value=16))
@settings(max_examples=30)
def test_merkle_proof_roundtrip(count):
    chain = MerkleChainAudit()
    for i in range(count):
        chain.append({"event": f"e_{i}"})
    for idx in range(count + 1):
        proof = chain.prove_inclusion(idx)
        assert proof is not None
        assert proof.block_index == idx
        assert chain.verify_proof(proof)


@given(st.integers(min_value=0, max_value=5))
@settings(max_examples=20)
def test_merkle_proof_nonexistent_index(invalid_idx):
    chain = MerkleChainAudit()
    proof = chain.prove_inclusion(10 + invalid_idx)
    assert proof is None
    proof = chain.prove_inclusion(-1 - invalid_idx)
    assert proof is None


# ── Property: Chain export/import roundtrip ─────────────────────────────────────


@given(st.integers(min_value=1, max_value=10), st.text(min_size=1, max_size=16))
@settings(max_examples=30)
def test_chain_export_import_roundtrip(count, chain_id):
    original = MerkleChainAudit(chain_id=chain_id)
    for i in range(count):
        original.append({"event": f"e_{i}"})
    exported = original.export_blocks()
    imported = MerkleChainAudit.import_blocks(chain_id, exported)
    assert imported.chain_id == original.chain_id
    assert imported.height == original.height
    assert imported.root == original.root
    assert imported.verify_chain()


# ── Property: Audit proof creation ──────────────────────────────────────────────


@given(
    st.text(min_size=1, max_size=16),
    st.text(min_size=1, max_size=8),
)
@settings(max_examples=30)
def test_create_audit_proof_structure(task_id, tenant_id):
    chain = MerkleChainAudit()
    proof_data = chain.create_audit_proof(task_id, tenant_id)
    assert proof_data["chain_id"] == chain.chain_id
    assert proof_data["block_index"] > 0
    assert len(proof_data["block_hash"]) == 64
    assert proof_data["verified"] is True
    assert proof_data["proof"] is not None
    assert "siblings" in proof_data["proof"]
    assert "directions" in proof_data["proof"]


# ── Property: Tampering is detected ─────────────────────────────────────────────


@given(st.integers(min_value=2, max_value=8))
@settings(max_examples=20)
def test_tampered_chain_detected(count):
    chain = MerkleChainAudit()
    for i in range(count):
        chain.append({"event": f"e_{i}"})
    blocks = chain._blocks
    if len(blocks) > 1:
        tampered_block = blocks[len(blocks) // 2]
        tampered = AuditBlock(
            index=tampered_block.index,
            prev_hash=tampered_block.prev_hash,
            timestamp=tampered_block.timestamp,
            commitment=b"\xff" * 32,
            metadata=tampered_block.metadata,
            block_hash=tampered_block.block_hash,
        )
        chain._blocks[len(blocks) // 2] = tampered
        assert not chain.verify_chain()


# ── Property: to_dict provides valid summary ────────────────────────────────────


@given(st.integers(min_value=1, max_value=8))
@settings(max_examples=20)
def test_to_dict_structure(count):
    chain = MerkleChainAudit()
    for i in range(count):
        chain.append({"event": f"e_{i}"})
    info = chain.to_dict()
    assert info["chain_id"] == chain.chain_id
    assert info["height"] == count + 1
    assert info["verified"] is True
    assert len(info["tip_hash"]) == 64
    assert len(info["root"]) == 64
