from __future__ import annotations

import hashlib
import hmac
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from noema.logging import get_logger
from noema.utils.json_utils import serialize_to_bytes

logger = get_logger(__name__)

_HASH_ALGO = "blake2b"
_HASH_DIGEST_SIZE = 32
_BLAKE2_SALT = b"noema-audit-v1"


def _hash(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=_HASH_DIGEST_SIZE, salt=_BLAKE2_SALT).digest()


def _compute_commitment(payload: dict[str, Any], nonce: bytes | None = None) -> bytes:
    encoded = serialize_to_bytes(payload)
    if nonce is None:
        return _hash(encoded)
    return _hash(nonce + encoded)


def _compute_block_hash(
    index: int,
    prev_hash: bytes,
    commitment: bytes,
    timestamp: float,
    metadata: bytes,
) -> bytes:
    data = (
        index.to_bytes(8, "big")
        + prev_hash
        + commitment
        + int(timestamp * 1_000_000).to_bytes(8, "big")
        + metadata
    )
    return _hash(data)


@dataclass(frozen=True)
class AuditBlock:
    index: int
    prev_hash: bytes
    timestamp: float
    commitment: bytes
    metadata: bytes
    block_hash: bytes
    nonce: bytes = field(default_factory=lambda: uuid.uuid4().bytes)

    def verify(self) -> bool:
        expected = _compute_block_hash(
            self.index, self.prev_hash, self.commitment, self.timestamp, self.metadata
        )
        return hmac.compare_digest(expected, self.block_hash)


@dataclass(frozen=True)
class MerkleProof:
    block_index: int
    block_hash: bytes
    siblings: list[tuple[bytes, bool]]  # (sibling_hash, is_left)

    def verify(self, root: bytes, leaf: bytes) -> bool:
        current = leaf
        for sibling, is_left in self.siblings:
            current = _hash(sibling + current) if is_left else _hash(current + sibling)
        return hmac.compare_digest(current, root)


class MerkleChainAudit:
    def __init__(self, chain_id: str | None = None) -> None:
        self._chain_id = chain_id or str(uuid.uuid4())[:8]
        self._blocks: list[AuditBlock] = []
        self._dirty = False
        self._tree_levels: list[list[bytes]] | None = None
        genesis_commitment = _compute_commitment(
            {"event": "chain_created", "chain_id": self._chain_id, "version": "1.0"}
        )
        genesis_hash = _compute_block_hash(0, b"\x00" * 32, genesis_commitment, 0, b"genesis")
        self._genesis = AuditBlock(
            index=0,
            prev_hash=b"\x00" * 32,
            timestamp=0,
            commitment=genesis_commitment,
            metadata=b"genesis",
            block_hash=genesis_hash,
        )
        self._blocks.append(self._genesis)
        logger.info("merkle_chain_created", chain_id=self._chain_id)

    @property
    def chain_id(self) -> str:
        return self._chain_id

    @property
    def tip(self) -> AuditBlock:
        return self._blocks[-1]

    @property
    def tip_hash(self) -> bytes:
        return self.tip.block_hash

    @property
    def root(self) -> bytes:
        return self._build_levels()[-1][0]

    @property
    def height(self) -> int:
        return len(self._blocks)

    def append(
        self,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> AuditBlock:
        prev = self.tip
        index = prev.index + 1
        commitment = _compute_commitment(payload)
        meta_bytes = serialize_to_bytes(metadata or {})
        now = time.time()
        block_hash = _compute_block_hash(index, prev.block_hash, commitment, now, meta_bytes)
        block = AuditBlock(
            index=index,
            prev_hash=prev.block_hash,
            timestamp=now,
            commitment=commitment,
            metadata=meta_bytes,
            block_hash=block_hash,
        )
        if not block.verify():
            raise RuntimeError(f"Block {index} failed self-verification")
        self._blocks.append(block)
        self._dirty = True
        self._tree_levels = None
        logger.debug("block_appended", index=index, chain_id=self._chain_id)
        return block

    def verify_chain(self) -> bool:
        for i, block in enumerate(self._blocks):
            if not block.verify():
                logger.error("block_verification_failed", index=i, chain_id=self._chain_id)
                return False
            if i > 0:
                expected_prev = self._blocks[i - 1].block_hash
                if not hmac.compare_digest(block.prev_hash, expected_prev):
                    logger.error("chain_link_broken", index=i, chain_id=self._chain_id)
                    return False
        return True

    def prove_inclusion(self, block_index: int) -> MerkleProof | None:
        if block_index < 0 or block_index >= len(self._blocks):
            return None
        block = self._blocks[block_index]
        levels = self._build_levels()
        if len(levels[0]) == 1:
            return MerkleProof(block_index=block_index, block_hash=block.block_hash, siblings=[])
        proof_siblings: list[tuple[bytes, bool]] = []
        idx = block_index
        for level in levels[:-1]:
            if idx % 2 == 0:
                if idx + 1 < len(level):
                    proof_siblings.append((level[idx + 1], False))
            else:
                proof_siblings.append((level[idx - 1], True))
            idx //= 2
        return MerkleProof(
            block_index=block_index, block_hash=block.block_hash, siblings=proof_siblings
        )

    def verify_proof(self, proof: MerkleProof) -> bool:
        if proof.block_index >= len(self._blocks):
            return False
        block = self._blocks[proof.block_index]
        leaf = block.block_hash
        if not hmac.compare_digest(leaf, proof.block_hash):
            return False
        return proof.verify(self.root, leaf)

    def get_block(self, index: int) -> AuditBlock | None:
        if 0 <= index < len(self._blocks):
            return self._blocks[index]
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self._chain_id,
            "height": self.height,
            "tip_hash": self.tip_hash.hex(),
            "root": self.root.hex(),
            "verified": self.verify_chain(),
        }

    def export_blocks(self) -> list[dict[str, Any]]:
        return [
            {
                "index": b.index,
                "prev_hash": b.prev_hash.hex(),
                "timestamp": b.timestamp,
                "commitment": b.commitment.hex(),
                "block_hash": b.block_hash.hex(),
                "metadata": b.metadata.decode("utf-8", errors="replace"),
            }
            for b in self._blocks
        ]

    @classmethod
    def import_blocks(cls, chain_id: str, blocks_data: list[dict[str, Any]]) -> MerkleChainAudit:
        instance = cls(chain_id=chain_id)
        instance._blocks = [instance._genesis]
        for bd in blocks_data:
            if bd.get("index") == 0:
                continue
            raw_meta = bd.get("metadata", "")
            meta_bytes = raw_meta.encode("utf-8") if isinstance(raw_meta, str) else raw_meta
            block = AuditBlock(
                index=bd["index"],
                prev_hash=bytes.fromhex(bd["prev_hash"]),
                timestamp=bd["timestamp"],
                commitment=bytes.fromhex(bd["commitment"]),
                metadata=meta_bytes,
                block_hash=bytes.fromhex(bd["block_hash"]),
            )
            instance._blocks.append(block)
        instance._tree_levels = None
        # Fail closed: a tampered export (broken prev_hash linkage, forged
        # block hashes or an inconsistent merkle root) must not be accepted
        # as a valid chain.
        if not instance.verify_chain():
            raise ValueError("imported blocks failed chain verification")
        return instance

    @staticmethod
    def _build_merkle_tree(leaves: list[bytes]) -> list[bytes]:
        if not leaves:
            return []
        level = leaves[:]
        while len(level) > 1:
            next_level: list[bytes] = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    combined = _hash(level[i] + level[i + 1])
                    next_level.append(combined)
                else:
                    next_level.append(level[i])
            level = next_level
        return level

    def _build_levels(self) -> list[list[bytes]]:
        """Build (and memoize) all merkle levels: leaves at index 0, root at -1."""
        if self._tree_levels is not None:
            return self._tree_levels
        levels: list[list[bytes]] = [[b.block_hash for b in self._blocks]]
        while len(levels[-1]) > 1:
            level = levels[-1]
            next_level: list[bytes] = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    next_level.append(_hash(level[i] + level[i + 1]))
                else:
                    next_level.append(level[i])
            levels.append(next_level)
        self._tree_levels = levels
        return levels

    def _build_merkle_root(self) -> bytes:
        if not self._blocks:
            return b"\x00" * 32
        tree = self._build_levels()
        return tree[-1][0]

    def create_audit_proof(self, task_id: str, tenant_id: str) -> dict[str, Any]:
        block = self.append(
            payload={"task_id": task_id, "tenant_id": tenant_id, "event": "task_completed"},
            metadata={"audit_type": "solution_generation"},
        )
        proof = self.prove_inclusion(block.index)
        return {
            "chain_id": self._chain_id,
            "block_index": block.index,
            "block_hash": block.block_hash.hex(),
            "root": self.root.hex(),
            "proof": {
                "siblings": [s.hex() for s, _ in proof.siblings] if proof else [],
                "directions": ["left" if is_left else "right" for _, is_left in proof.siblings]
                if proof
                else [],
            }
            if proof
            else None,
            "verified": self.verify_chain(),
        }
