"""Tests for the federation protocol (T3.2): delegation, retries, circuits."""

from __future__ import annotations

import asyncio

import pytest

from noema.billing.ledger import ContributionLedger
from noema.federation.router import DelegationResult, FederationRouter, PeerNode


class FakePeer:
    """Transport-injectable peer client with scriptable failures."""

    def __init__(self, fail_times: int = 0, latency: float = 0.0, response: dict | None = None):
        self.fail_times = fail_times
        self.latency = latency
        self.calls = 0
        self.response = response or {"solution_id": "s1", "quality": "good"}

    async def think(self, title: str, description: str) -> dict:
        self.calls += 1
        if self.latency:
            await asyncio.sleep(self.latency)
        if self.calls <= self.fail_times:
            raise ConnectionError("peer down")
        return self.response


def make_router(peers: list[str], clients: dict[str, FakePeer], **kwargs) -> FederationRouter:
    ledger = ContributionLedger()
    router = FederationRouter(
        peers=peers,
        client_factory=lambda addr: clients[addr],
        ledger=ledger,
        base_delay=0.0,
        **kwargs,
    )
    return router


class TestPeerNode:
    def test_parse_valid(self):
        p = PeerNode.parse("10.0.0.5:50051")
        assert (p.host, p.port, p.address) == ("10.0.0.5", 50051, "10.0.0.5:50051")

    def test_parse_ipv6ish_host(self):
        p = PeerNode.parse("[::1]:1234")
        assert p.port == 1234

    @pytest.mark.parametrize("bad", ["no-port", "host:notaport", ":5000"])
    def test_parse_rejects_garbage(self, bad: str):
        with pytest.raises(ValueError, match="host:port"):
            PeerNode.parse(bad)


class TestDelegation:
    async def test_delegates_round_robin(self):
        clients = {"a:1": FakePeer(), "b:2": FakePeer()}
        router = make_router(["a:1", "b:2"], clients)
        results = [await router.delegate(f"s{i}", "desc") for i in range(4)]
        assert all(r.status == "delegated" for r in results)
        assert clients["a:1"].calls == 2
        assert clients["b:2"].calls == 2
        await router.aclose()

    async def test_retry_then_success(self):
        clients = {"a:1": FakePeer(fail_times=2)}
        router = make_router(["a:1"], clients)
        result = await router.delegate("s1", "desc")
        assert result.status == "delegated"
        assert result.attempts == 3
        await router.aclose()

    async def test_retries_exhausted_falls_back_local(self):
        clients = {"a:1": FakePeer(fail_times=99)}

        async def local_executor(subtask_id: str, desc: str) -> dict:
            return {"solution_id": "local-1"}

        router = make_router(["a:1"], clients, local_executor=local_executor, max_retries=1)
        result = await router.delegate("s1", "desc")
        assert result.status == "local"
        assert result.response["solution_id"] == "local-1"
        assert "ConnectionError" in result.error
        await router.aclose()

    async def test_no_peers_runs_local(self):
        async def local_executor(subtask_id: str, desc: str) -> dict:
            return {"solution_id": "local-2"}

        router = make_router([], {}, local_executor=local_executor)
        result = await router.delegate("s1", "desc")
        assert result.status == "local"
        assert "no healthy peers" in result.error
        await router.aclose()

    async def test_no_local_executor_reports_failed(self):
        router = make_router(["a:1"], {"a:1": FakePeer(fail_times=99)}, max_retries=0)
        result = await router.delegate("s1", "desc")
        assert result.status == "failed"
        assert result.error
        await router.aclose()

    async def test_timeout_falls_back_local(self):
        clients = {"a:1": FakePeer(latency=5.0)}

        async def local_executor(subtask_id: str, desc: str) -> dict:
            return {"solution_id": "local-3"}

        router = make_router(
            ["a:1"], clients, local_executor=local_executor, max_retries=0, request_timeout=0.05
        )
        result = await router.delegate("s1", "desc")
        assert result.status == "local"
        assert "timeout" in result.error
        await router.aclose()


class TestCircuitBreaking:
    async def test_circuit_opens_after_threshold(self):
        clients = {"a:1": FakePeer(fail_times=99), "b:2": FakePeer()}
        router = make_router(["a:1", "b:2"], clients, failure_threshold=2, max_retries=0)
        # First two delegations fail on a:1 (round-robin also uses b:2, which
        # succeeds and keeps its own circuit closed).
        for _ in range(4):
            await router.delegate("s", "desc")
        assert router.circuit_stats()["a:1"]["state"] == "open"
        assert "b:2" in router.healthy_peers()
        assert "a:1" not in router.healthy_peers()
        # All further work goes to the healthy peer only.
        calls_b = clients["b:2"].calls
        for _ in range(3):
            result = await router.delegate("s", "desc")
            assert result.peer == "b:2"
        assert clients["b:2"].calls == calls_b + 3
        await router.aclose()

    async def test_all_circuits_open_falls_back(self):
        clients = {"a:1": FakePeer(fail_times=99)}
        router = make_router(["a:1"], clients, failure_threshold=1, max_retries=0)

        async def local_executor(subtask_id: str, desc: str) -> dict:
            return {"solution_id": "l"}

        router.local_executor = local_executor
        await router.delegate("s1", "d")  # trips the circuit
        result = await router.delegate("s2", "d")
        assert result.status == "local"
        assert "circuit open" in result.error or "no healthy peers" in result.error
        await router.aclose()


class TestExecute:
    async def test_split_and_rejoin_in_order(self):
        clients = {"a:1": FakePeer(), "b:2": FakePeer()}
        router = make_router(["a:1", "b:2"], clients)
        summary = await router.execute("Root task", ["sub-1", "sub-2", "sub-3", "sub-4"])
        assert summary["total"] == 4
        assert summary["delegated"] == 4
        assert summary["failed"] == 0
        assert summary["peers_used"] == ["a:1", "b:2"]
        # Join is positional: subtask i lands at index i.
        descs = [s["description"] for s in summary["subtasks"]]
        assert descs == ["sub-1", "sub-2", "sub-3", "sub-4"]
        await router.aclose()

    async def test_empty_subtask_list_runs_root(self):
        router = make_router(
            [], {}, local_executor=lambda *a: asyncio.sleep(0, {"ok": True}) or None
        )
        # no peers and no real executor → the root task itself fails honestly
        summary = await router.execute("Root only", [])
        assert summary["total"] == 1
        assert summary["subtasks"][0]["description"] == "Root only"
        await router.aclose()

    async def test_ledger_records_every_subtask(self):
        clients = {"a:1": FakePeer(response={"solution_id": "x", "cost_usd": 0.1})}
        router = make_router(["a:1"], clients)
        await router.execute("Root", ["s1", "s2"])
        audit = router.ledger.audit()
        assert audit["count"] == 2
        assert audit["entries"][0]["peer"] == "a:1"
        assert audit["entries"][0]["kind"] == "subtask"
        await router.aclose()


class TestDelegationResult:
    def test_to_dict_shape(self):
        r = DelegationResult(subtask_id="s1", description="d", peer="p:1", status="delegated")
        d = r.to_dict()
        assert d["subtask_id"] == "s1"
        assert d["peer"] == "p:1"
        assert d["attempts"] == 0
