"""CLI commands for cryptographic audit trails - Merkle inclusion proofs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from noema.audit.merkle_proof import InclusionProof, verify_inclusion_proof
from noema.cli.ui import console, data_table, ok, section

audit_app = typer.Typer(help="Cryptographic audit trail commands", rich_markup_mode="rich")


@audit_app.command()
def proof(
    tenant_id: str = typer.Option(..., "--tenant", "-t", help="Tenant ID"),
    task_id: str = typer.Option(..., "--task", help="Task ID"),
    output: str = typer.Option("", "--output", "-o", help="Write proof JSON to file"),
    fallback_dir: str = typer.Option(".noema/audit", "--fallback-dir", help="File fallback dir"),
) -> None:
    """Generate a Merkle inclusion proof for a completed task."""

    async def _run() -> None:
        from noema.audit.logger import AuditLogger

        audit = AuditLogger(pg_pool=None, fallback_dir=fallback_dir)
        await audit.initialize()
        proof_dict = await audit.get_proof_for_task(tenant_id=tenant_id, task_id=task_id)
        result = {"tenant_id": tenant_id, "task_id": task_id, "proof": proof_dict}
        if output:
            Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
            ok(f"Proof saved: [path]{output}[/path]")
        else:
            console.print_json(json.dumps(result))

    asyncio.run(_run())


@audit_app.command()
def verify(
    proof_file: str = typer.Argument(..., help="Path to proof JSON file"),
    leaf_data: str = typer.Option("", "--leaf-data", help="JSON string of the original leaf data"),
) -> None:
    """Verify an inclusion proof (client-side, no server trust)."""
    if not Path(proof_file).exists():
        console.print(f"[err]File not found: {proof_file}[/err]")
        raise typer.Exit(code=1)
    doc = json.loads(Path(proof_file).read_text(encoding="utf-8"))
    proof_dict = doc.get("proof") or doc
    leaf = (
        json.loads(leaf_data) if leaf_data else doc.get("leaf_data") or proof_dict.get("leaf_data")
    )
    if leaf is None:
        console.print(
            "[err]Provide --leaf-data JSON or include 'leaf_data' in the proof file[/err]"
        )
        raise typer.Exit(code=1)
    proof = InclusionProof.from_dict(proof_dict)
    valid = verify_inclusion_proof(proof, leaf)
    valid_mark = "[ok]VALID[/ok]" if valid else "[err]INVALID[/err]"
    data_table(
        "Proof Verification",
        ["Field", "Value"],
        [
            ["Block index", str(proof.block_index)],
            ["Leaf hash", proof.leaf_hash.hex()[:24] + "..."],
            ["Root hash", proof.root_hash.hex()[:24] + "..."],
            ["Path length", str(len(proof.path))],
            ["Valid", valid_mark],
        ],
    )
    raise typer.Exit(code=0 if valid else 1)


@audit_app.command()
def chain(
    action: str = typer.Argument("stats", help="stats|append|export|verify"),
    chain_id: str = typer.Option("", "--chain-id", help="Chain ID (defaults to auto)"),
    event: str = typer.Option("", "--event", help="Event JSON for append"),
    export_path: str = typer.Option("", "--export", "-o", help="Export chain to file"),
    import_path: str = typer.Option("", "--import", "-i", help="Import chain from file"),
) -> None:
    """Inspect or manage a MerkleChainAudit hash chain."""
    from noema.audit.merkle import MerkleChainAudit

    if action == "stats":
        chain_obj = MerkleChainAudit(chain_id=chain_id or None)
        data = chain_obj.to_dict()
        rows = []
        for key, val in data.items():
            if key == "root":
                rows.append(["root", str(val)[:24] + "..."])
            else:
                rows.append([key, str(val)])
        data_table("Merkle Chain", ["Field", "Value"], rows)

    elif action == "append":
        if not event:
            console.print("[err]Specify --event JSON[/err]")
            raise typer.Exit(code=1)
        chain_obj = MerkleChainAudit(chain_id=chain_id or None)
        block = chain_obj.append(json.loads(event))
        ok(f"Block appended: index={block.index} hash={block.block_hash.hex()[:24]}...")

    elif action == "export":
        if not export_path:
            console.print("[err]Specify --export FILE[/err]")
            raise typer.Exit(code=1)
        chain_obj = MerkleChainAudit(chain_id=chain_id or None)
        doc = {"chain_id": chain_obj.chain_id, "blocks": chain_obj.export_blocks()}
        Path(export_path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
        ok(f"Chain exported: [path]{export_path}[/path]")

    elif action == "import":
        if not import_path:
            console.print("[err]Specify --import FILE[/err]")
            raise typer.Exit(code=1)
        doc = json.loads(Path(import_path).read_text(encoding="utf-8"))
        chain_obj = MerkleChainAudit.import_blocks(doc["chain_id"], doc["blocks"])
        verified = chain_obj.verify_chain()
        section("Chain Import")
        ok(f"chain_id={chain_obj.chain_id} height={chain_obj.height} verified={verified}")

    else:
        console.print(f"[err]Unknown action: {action}[/err]")
        raise typer.Exit(code=1)
