import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from noema.audit.merkle_proof import (
    InclusionProof,
    IncrementalMerkleTree,
    _safe_hash,
    _serialize,
    compute_root,
    generate_inclusion_proof,
    verify_inclusion_proof,
)


@given(
    st.lists(
        st.dictionaries(st.text(min_size=1, max_size=5), st.text(min_size=1, max_size=5)),
        min_size=1,
        max_size=32,
    ),
    st.integers(min_value=0, max_value=31),
)
def test_inclusion_proof_validity(leaf_data_list, target_index):
    target_index = target_index % len(leaf_data_list)
    leaf_hashes = [_safe_hash(_serialize(d)) for d in leaf_data_list]
    target_data = leaf_data_list[target_index]
    proof = generate_inclusion_proof(target_data, target_index, leaf_hashes)
    assert verify_inclusion_proof(proof, target_data)


@given(
    st.lists(
        st.dictionaries(st.text(min_size=1, max_size=5), st.text(min_size=1, max_size=5)),
        min_size=1,
        max_size=32,
    ),
    st.integers(min_value=0, max_value=31),
    st.integers(min_value=0, max_value=1000),
)
def test_inclusion_proof_rejects_wrong_data(leaf_data_list, target_index, wrong_seed):
    target_index = target_index % len(leaf_data_list)
    assume(len(leaf_data_list) > 0)
    leaf_hashes = [_safe_hash(_serialize(d)) for d in leaf_data_list]
    target_data = leaf_data_list[target_index]
    wrong_data = {"_tamper": str(wrong_seed)}
    assume(wrong_data != target_data)
    proof = generate_inclusion_proof(target_data, target_index, leaf_hashes)
    assert not verify_inclusion_proof(proof, wrong_data)


@given(
    st.lists(
        st.dictionaries(st.text(min_size=1, max_size=5), st.text(min_size=1, max_size=5)),
        min_size=1,
        max_size=32,
    ),
)
def test_all_proofs_verify(leaf_data_list):
    leaf_hashes = [_safe_hash(_serialize(d)) for d in leaf_data_list]
    for i, data in enumerate(leaf_data_list):
        proof = generate_inclusion_proof(data, i, leaf_hashes)
        assert verify_inclusion_proof(proof, data)


@given(
    st.lists(
        st.dictionaries(st.text(min_size=1, max_size=5), st.text(min_size=1, max_size=5)),
        min_size=1,
        max_size=32,
    ),
)
def test_all_proofs_have_same_root(leaf_data_list):
    leaf_hashes = [_safe_hash(_serialize(d)) for d in leaf_data_list]
    roots = set()
    for i, data in enumerate(leaf_data_list):
        proof = generate_inclusion_proof(data, i, leaf_hashes)
        roots.add(proof.root_hash)
    assert len(roots) == 1


@given(
    st.lists(
        st.dictionaries(st.text(min_size=1, max_size=5), st.text(min_size=1, max_size=5)),
        min_size=1,
        max_size=32,
    ),
)
def test_root_matches_compute_root(leaf_data_list):
    leaf_hashes = [_safe_hash(_serialize(d)) for d in leaf_data_list]
    expected = compute_root(leaf_hashes)
    if leaf_data_list:
        proof = generate_inclusion_proof(leaf_data_list[0], 0, leaf_hashes)
        assert proof.root_hash == expected


@given(
    st.lists(
        st.dictionaries(st.text(min_size=1, max_size=5), st.text(min_size=1, max_size=5)),
        min_size=1,
        max_size=32,
    ),
    st.integers(min_value=0, max_value=31),
)
def test_proof_to_dict_roundtrip(leaf_data_list, target_index):
    target_index = target_index % len(leaf_data_list)
    leaf_hashes = [_safe_hash(_serialize(d)) for d in leaf_data_list]
    target_data = leaf_data_list[target_index]
    proof = generate_inclusion_proof(target_data, target_index, leaf_hashes)
    d = proof.to_dict()
    restored = InclusionProof.from_dict(d)
    assert restored == proof
    assert verify_inclusion_proof(restored, target_data)


@given(
    st.lists(
        st.dictionaries(st.text(min_size=1, max_size=5), st.text(min_size=1, max_size=5)),
        min_size=2,
        max_size=32,
    ),
    st.integers(min_value=0, max_value=31),
)
def test_proof_path_is_log_n(leaf_data_list, target_index):
    target_index = target_index % len(leaf_data_list)
    leaf_hashes = [_safe_hash(_serialize(d)) for d in leaf_data_list]
    target_data = leaf_data_list[target_index]
    proof = generate_inclusion_proof(target_data, target_index, leaf_hashes)
    n_padded = 1 << (len(leaf_data_list) - 1).bit_length() if len(leaf_data_list) > 0 else 1
    import math

    expected_depth = math.ceil(math.log2(n_padded))
    assert len(proof.path) == expected_depth


def test_empty_leaf_list_raises():
    with pytest.raises(ValueError, match="empty"):
        generate_inclusion_proof({"x": 1}, 0, [])


def test_index_out_of_range_raises():
    hashes = [_safe_hash(b"test")]
    with pytest.raises(IndexError):
        generate_inclusion_proof({"x": 1}, 5, hashes)
    with pytest.raises(IndexError):
        generate_inclusion_proof({"x": 1}, -1, hashes)


@given(
    st.lists(
        st.dictionaries(st.text(min_size=1, max_size=5), st.text(min_size=1, max_size=5)),
        min_size=2,
        max_size=32,
    ),
    st.integers(min_value=0, max_value=31),
    st.integers(min_value=0, max_value=31),
)
def test_different_leaves_have_different_proofs(leaf_data_list, i, j):
    i = i % len(leaf_data_list)
    j = j % len(leaf_data_list)
    assume(i != j)
    leaf_hashes = [_safe_hash(_serialize(d)) for d in leaf_data_list]
    p1 = generate_inclusion_proof(leaf_data_list[i], i, leaf_hashes)
    p2 = generate_inclusion_proof(leaf_data_list[j], j, leaf_hashes)
    assert p1.path != p2.path or p1.block_index != p2.block_index


# ── IncrementalMerkleTree: must match static builders ────────────────────────


@given(st.lists(st.binary(min_size=32, max_size=32), min_size=1, max_size=64))
def test_incremental_root_matches_static(leaf_hashes):
    tree = IncrementalMerkleTree()
    for h in leaf_hashes:
        tree.append(h)
    assert tree.root == compute_root(leaf_hashes)
    assert tree.count == len(leaf_hashes)


@given(st.lists(st.binary(min_size=32, max_size=32), min_size=1, max_size=64))
def test_incremental_proof_matches_static(leaf_hashes):
    tree = IncrementalMerkleTree()
    for h in leaf_hashes:
        tree.append(h)
    for i in range(len(leaf_hashes)):
        inc_path = tree.proof(i)
        static_proof = generate_inclusion_proof({"x": i}, i, leaf_hashes)
        assert [h for h, _ in inc_path] == [h for h, _ in static_proof.path]
        assert [d for _, d in inc_path] == [d for _, d in static_proof.path]


@given(st.lists(st.binary(min_size=32, max_size=32), min_size=1, max_size=64))
def test_incremental_bulk_init_matches_static(leaf_hashes):
    tree = IncrementalMerkleTree(leaf_hashes)
    assert tree.root == compute_root(leaf_hashes)
    for i in range(len(leaf_hashes)):
        static_proof = generate_inclusion_proof({"x": i}, i, leaf_hashes)
        inc_path = tree.proof(i)
        assert [h for h, _ in inc_path] == [h for h, _ in static_proof.path]
        assert [d for _, d in inc_path] == [d for _, d in static_proof.path]


@given(st.lists(st.binary(min_size=32, max_size=32), min_size=1, max_size=64))
def test_incremental_append_step_by_step(leaf_hashes):
    tree = IncrementalMerkleTree()
    for i in range(len(leaf_hashes)):
        tree.append(leaf_hashes[i])
        assert tree.root == compute_root(leaf_hashes[: i + 1])
        assert tree.count == i + 1


@given(
    st.lists(
        st.dictionaries(st.text(min_size=1, max_size=5), st.text(min_size=1, max_size=5)),
        min_size=1,
        max_size=64,
    )
)
def test_incremental_inclusion_proof_verifies(leaf_data_list):
    tree = IncrementalMerkleTree()
    for d in leaf_data_list:
        tree.append(_safe_hash(_serialize(d)))
    for i, d in enumerate(leaf_data_list):
        proof = tree.inclusion_proof(d, i)
        assert verify_inclusion_proof(proof, d)
        assert proof.root_hash == tree.root
        assert proof.leaf_hash == _safe_hash(_serialize(d))


def test_incremental_empty():
    tree = IncrementalMerkleTree()
    assert tree.root == b"\x00" * 32
    assert tree.count == 0
    with pytest.raises(IndexError):
        tree.proof(0)
    tree.append(_safe_hash(b"first"))
    assert tree.count == 1
    assert tree.root == _safe_hash(b"first")


@given(st.lists(st.binary(min_size=32, max_size=32), min_size=1, max_size=63))
def test_incremental_bulk_init_then_append_matches_static(leaf_hashes):
    extra = _safe_hash(b"appended-later")
    leaves = leaf_hashes + [extra]
    tree = IncrementalMerkleTree(leaf_hashes)
    tree.append(extra)
    assert tree.root == compute_root(leaves)
    for i in range(len(leaves)):
        inc_path = tree.proof(i)
        static_proof = generate_inclusion_proof({"x": i}, i, leaves)
        assert [h for h, _ in inc_path] == [h for h, _ in static_proof.path]
        assert [d for _, d in inc_path] == [d for _, d in static_proof.path]
