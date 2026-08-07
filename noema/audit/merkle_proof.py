from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from noema.utils.json_utils import serialize_to_bytes

if TYPE_CHECKING:
    from collections.abc import Sequence

DOMAIN_SEPARATOR = b"NOEMA_FRAMEWORK_AUDIT_V1"


def _safe_hash(data: bytes) -> bytes:
    return hashlib.sha256(DOMAIN_SEPARATOR + data).digest()


def _hash_pair(left: bytes, right: bytes) -> bytes:
    if left < right:
        return _safe_hash(left + right)
    return _safe_hash(right + left)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


def _pad_sequence(seq: Sequence[bytes]) -> list[bytes]:
    if not seq:
        raise ValueError("Cannot pad empty sequence")
    n = len(seq)
    target = _next_pow2(n)
    out = list(seq)
    out.extend([seq[-1]] * (target - n))
    return out


def _serialize(data: Any) -> bytes:
    return serialize_to_bytes(data)


@dataclass(frozen=True)
class InclusionProof:
    leaf_hash: bytes
    path: list[tuple[bytes, bool]]
    root_hash: bytes
    block_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "leaf_hash": self.leaf_hash.hex(),
            "path": [(h.hex(), d) for h, d in self.path],
            "root_hash": self.root_hash.hex(),
            "block_index": self.block_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InclusionProof:
        return cls(
            leaf_hash=bytes.fromhex(data["leaf_hash"]),
            path=[(bytes.fromhex(h), bool(d)) for h, d in data["path"]],
            root_hash=bytes.fromhex(data["root_hash"]),
            block_index=data["block_index"],
        )


def _build_levels(leaves: Sequence[bytes]) -> list[list[bytes]]:
    levels = [_pad_sequence(leaves)]
    while len(levels[-1]) > 1:
        level = levels[-1]
        parent: list[bytes] = []
        for i in range(0, len(level), 2):
            parent.append(_hash_pair(level[i], level[i + 1]))
        levels.append(parent)
    return levels


def compute_root(leaf_hashes: Sequence[bytes]) -> bytes:
    if not leaf_hashes:
        return b"\x00" * 32
    levels = _build_levels(leaf_hashes)
    return levels[-1][0]


def generate_inclusion_proof(
    leaf_data: Any,
    leaf_index: int,
    all_leaf_hashes: Sequence[bytes],
) -> InclusionProof:
    if not all_leaf_hashes:
        raise ValueError("Cannot generate proof from empty leaf list")
    if leaf_index < 0 or leaf_index >= len(all_leaf_hashes):
        raise IndexError(f"leaf_index {leaf_index} out of range [0, {len(all_leaf_hashes)})")
    leaf_hash = _safe_hash(_serialize(leaf_data))
    levels = _build_levels(all_leaf_hashes)
    path: list[tuple[bytes, bool]] = []
    idx = leaf_index
    for level in levels[:-1]:
        if idx % 2 == 0:
            path.append((level[idx + 1], False))
        else:
            path.append((level[idx - 1], True))
        idx //= 2
    return InclusionProof(
        leaf_hash=leaf_hash,
        path=path,
        root_hash=levels[-1][0],
        block_index=leaf_index,
    )


def verify_inclusion_proof(proof: InclusionProof, leaf_data: Any) -> bool:
    current = _safe_hash(_serialize(leaf_data))
    if current != proof.leaf_hash:
        return False
    for sibling, is_right in proof.path:
        current = _hash_pair(sibling, current) if is_right else _hash_pair(current, sibling)
    return current == proof.root_hash


class IncrementalMerkleTree:
    """Append-only Merkle tree with O(log N) append and O(log N) proofs.

    Level nodes that fall entirely inside the current padding region (the
    duplicate of the last leaf) are stored as ``None`` and resolved lazily
    against the current last leaf, so appending only recomputes the O(log N)
    nodes on the path to the root. Produces byte-identical roots and proofs to
    the static `compute_root` / `generate_inclusion_proof` builders.
    """

    __slots__ = ("_levels", "_real", "_pad_hashes")

    def __init__(self, leaves: Sequence[bytes] | None = None) -> None:
        self._real: list[bytes] = list(leaves) if leaves else []
        self._levels: list[list[bytes | None]]
        if self._real:
            self._levels = cast("list[list[bytes | None]]", _build_levels(self._real))
            self._nullify_pads(len(self._real))
        else:
            self._levels = [[]]
        self._pad_hashes = self._compute_pad_hashes()

    def _compute_pad_hashes(self) -> list[bytes]:
        leaf = self._real[-1] if self._real else b"\x00" * 32
        out = [leaf]
        h = leaf
        for _ in range(len(self._levels) - 1):
            h = _hash_pair(h, h)
            out.append(h)
        return out

    def _nullify_pads(self, n: int) -> None:
        for k in range(len(self._levels)):
            start = (n + (1 << k) - 1) >> k
            level = self._levels[k]
            for j in range(start, len(level)):
                level[j] = None

    def _node(self, k: int, j: int) -> bytes:
        level = self._levels[k]
        if j < len(level):
            value = level[j]
            if value is not None:
                return value
        return self._pad_hashes[k]

    @property
    def count(self) -> int:
        return len(self._real)

    @property
    def root(self) -> bytes:
        if not self._real:
            return b"\x00" * 32
        return self._node(len(self._levels) - 1, 0)

    @property
    def depth(self) -> int:
        return len(self._levels) - 1

    def append(self, leaf: bytes) -> None:
        if self._levels == [[]]:
            self._levels = [[leaf]]
            self._real.append(leaf)
            self._pad_hashes = self._compute_pad_hashes()
            return
        self._real.append(leaf)
        padded = _next_pow2(len(self._real))
        if len(self._levels) < padded.bit_length():
            self._levels.append([])
        level0 = self._levels[0]
        while len(level0) <= len(self._real) - 1:
            level0.append(None)
        level0[len(self._real) - 1] = leaf
        self._pad_hashes = self._compute_pad_hashes()
        self._refresh_path(len(self._real) - 1)

    def _refresh_path(self, idx: int) -> None:
        for k in range(1, len(self._levels)):
            idx >>= 1
            level = self._levels[k]
            while len(level) <= idx:
                level.append(None)
            level[idx] = _hash_pair(
                self._node(k - 1, 2 * idx),
                self._node(k - 1, 2 * idx + 1),
            )

    def proof(self, leaf_index: int) -> list[tuple[bytes, bool]]:
        if not (0 <= leaf_index < len(self._real)):
            raise IndexError(f"leaf_index {leaf_index} out of range [0, {len(self._real)})")
        path: list[tuple[bytes, bool]] = []
        idx = leaf_index
        for k in range(len(self._levels) - 1):
            sibling = self._node(k, idx + 1) if idx % 2 == 0 else self._node(k, idx - 1)
            path.append((sibling, idx % 2 != 0))
            idx //= 2
        return path

    def inclusion_proof(self, leaf_data: Any, leaf_index: int) -> InclusionProof:
        if not (0 <= leaf_index < len(self._real)):
            raise IndexError(f"leaf_index {leaf_index} out of range [0, {len(self._real)})")
        return InclusionProof(
            leaf_hash=_safe_hash(_serialize(leaf_data)),
            path=self.proof(leaf_index),
            root_hash=self.root,
            block_index=leaf_index,
        )

    def to_static(self) -> list[bytes]:
        return list(self._real)
