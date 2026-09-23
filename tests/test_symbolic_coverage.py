"""Edge-case tests for noema/neurosymbolic/symbolic.py.

All z3 dependencies are mocked so the tests run without a real Z3 install.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from noema.neurosymbolic.symbolic import (
    Constraint,
    SymbolicEngine,
    SymbolicVerificationError,
    TaskGraph,
    _coerce_value,
    _extract_bounds,
    _unique_names,
    _z3_unavailable,
)


class Z3Exception(Exception):  # noqa: N818
    """Stand-in for ``z3.z3types.Z3Exception``.

    The real class is only importable once ``libz3`` loads, so it cannot be
    imported here; the production check keys off the type name, which this
    reproduces exactly.
    """


def _z3_import_raises(exc: Exception):
    """Patch the import machinery so ``from z3 import ...`` raises ``exc``."""
    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def _fail(name, *args, **kwargs):
        if name == "z3":
            raise exc
        return real_import(name, *args, **kwargs)

    return patch("builtins.__import__", side_effect=_fail)


# ---------------------------------------------------------------------------
# Helpers – fake z3 classes used by _coerce_value and the engine methods
# ---------------------------------------------------------------------------


def _make_fake_z3():
    """Return a module-like object with the z3 symbols the source imports."""
    fake = types.ModuleType("z3")

    class BoolRef:
        pass

    class ArithRef:
        def __init__(self, is_int=False):
            self._is_int = is_int

        def is_int(self):
            return self._is_int

    class _Unsat:
        """Sentinel that compares equal to the z3.unsat singleton."""

        def __eq__(self, other):
            return other is self or getattr(other, "__class__", None) is _Unsat

        def __hash__(self):
            return hash("_unsat_sentinel")

    unsat = _Unsat()

    class _Term:
        """Stand-in for a z3 term; supports the comparisons the engine makes."""

        def __init__(self, name):
            self.name = name

        def __ge__(self, other):
            return (self.name, ">=", other)

        def __le__(self, other):
            return (self.name, "<=", other)

        def __gt__(self, other):
            return (self.name, ">", other)

        def __lt__(self, other):
            return (self.name, "<", other)

        def __eq__(self, other):
            return (self.name, "==", other)

        def __hash__(self):
            return hash(("term", self.name))

        def __bool__(self):
            return True

    fake.BoolRef = BoolRef
    fake.ArithRef = ArithRef
    fake.unsat = unsat
    fake.Solver = MagicMock
    fake.Int = MagicMock(side_effect=lambda name: _Term(name))
    fake.Real = MagicMock(side_effect=lambda name: _Term(name))
    fake.Bool = MagicMock(side_effect=lambda name: _Term(name))
    fake.And = MagicMock()
    return fake


# ---------------------------------------------------------------------------
# _unique_names (inner while-loop when candidate collides)
# ---------------------------------------------------------------------------


class TestUniqueNamesCollision:
    def test_inner_while_loop_triggered(self):
        # The third "a" collides with *seen*, and "a_1" is taken too, so the
        # collision loop must keep advancing to "a_2".
        result = _unique_names(["a", "a_1", "a"])
        assert result == ["a", "a_1", "a_2"]

    def test_multiple_collisions(self):
        # "a" appears three times and "a_1" / "a_2" are pre-occupied.
        result = _unique_names(["a", "a_1", "a_2", "a", "a"])
        assert result[0] == "a"
        assert result[1] == "a_1"
        assert result[2] == "a_2"
        # Third "a" should skip a_1 and a_2 → a_3
        assert result[3] == "a_3"
        # Fourth "a" → a_4
        assert result[4] == "a_4"


# ---------------------------------------------------------------------------
# _extract_bounds (<= pattern)
# ---------------------------------------------------------------------------


class TestExtractBoundsLePattern:
    def test_le_pattern_in_description(self):
        req = {"description": "x <= 42"}
        lower, upper = _extract_bounds(req)
        assert upper == 42.0

    def test_le_pattern_in_constraints_list(self):
        req = {"constraints": ["value <= 99.5"]}
        lower, upper = _extract_bounds(req)
        assert upper == 99.5

    def test_le_combined_with_ge(self):
        req = {"description": "x >= 5 and x <= 10"}
        lower, upper = _extract_bounds(req)
        assert lower == 5.0
        assert upper == 10.0


# ---------------------------------------------------------------------------
# _coerce_value
# ---------------------------------------------------------------------------


class TestCoerceValue:
    """Exercise every branch of _coerce_value with mocked z3 types."""

    def _bool_ref(self):
        fake = _make_fake_z3()
        return fake.BoolRef(), fake

    def _arith_ref(self, is_int=False):
        fake = _make_fake_z3()
        return fake.ArithRef(is_int=is_int), fake

    # -- BoolRef branches --

    def test_boolref_with_bool(self):
        var, fake_z3 = self._bool_ref()
        with patch.dict(sys.modules, {"z3": fake_z3}):
            assert _coerce_value(var, True) is True
            assert _coerce_value(var, False) is False

    def test_boolref_with_int(self):
        var, fake_z3 = self._bool_ref()
        with patch.dict(sys.modules, {"z3": fake_z3}):
            assert _coerce_value(var, 1) is True
            assert _coerce_value(var, 0) is False

    def test_boolref_with_float(self):
        var, fake_z3 = self._bool_ref()
        with patch.dict(sys.modules, {"z3": fake_z3}):
            assert _coerce_value(var, 3.14) is True
            assert _coerce_value(var, 0.0) is False

    def test_boolref_with_string_true_variants(self):
        var, fake_z3 = self._bool_ref()
        with patch.dict(sys.modules, {"z3": fake_z3}):
            for s in ("true", "1", "yes", "on"):
                assert _coerce_value(var, s) is True

    def test_boolref_with_string_false_variants(self):
        var, fake_z3 = self._bool_ref()
        with patch.dict(sys.modules, {"z3": fake_z3}):
            for s in ("false", "0", "no", "off"):
                assert _coerce_value(var, s) is False

    def test_boolref_with_unrecognised_string_returns_none(self):
        var, fake_z3 = self._bool_ref()
        with patch.dict(sys.modules, {"z3": fake_z3}):
            assert _coerce_value(var, "maybe") is None

    # -- ArithRef branches --

    def test_arithref_real_with_valid_number(self):
        var, fake_z3 = self._arith_ref(is_int=False)
        with patch.dict(sys.modules, {"z3": fake_z3}):
            assert _coerce_value(var, 3.14) == 3.14

    def test_arithref_int_with_valid_integer(self):
        var, fake_z3 = self._arith_ref(is_int=True)
        with patch.dict(sys.modules, {"z3": fake_z3}):
            assert _coerce_value(var, 7) == 7

    def test_arithref_int_with_non_integer_float_returns_none(self):
        """ArithRef.is_int() but value is not integer-valued."""
        var, fake_z3 = self._arith_ref(is_int=True)
        with patch.dict(sys.modules, {"z3": fake_z3}):
            # 3.5 is not an integer → None
            assert _coerce_value(var, 3.5) is None

    def test_arithref_int_with_integer_valued_float(self):
        """ArithRef.is_int() and float is integer-valued."""
        var, fake_z3 = self._arith_ref(is_int=True)
        with patch.dict(sys.modules, {"z3": fake_z3}):
            assert _coerce_value(var, 5.0) == 5

    def test_arithref_with_uncoercible_value_returns_none(self):
        """_to_num returns None."""
        var, fake_z3 = self._arith_ref(is_int=False)
        with patch.dict(sys.modules, {"z3": fake_z3}):
            assert _coerce_value(var, "not_a_number") is None
            assert _coerce_value(var, True) is None  # bool → _to_num → None

    def test_unknown_var_type_returns_none(self):
        """var is neither BoolRef nor ArithRef."""
        with patch.dict(sys.modules, {"z3": _make_fake_z3()}):
            assert _coerce_value("not_a_z3_var", 42) is None


# ---------------------------------------------------------------------------
# SymbolicEngine.initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    @pytest.mark.asyncio
    async def test_already_initialized_returns_immediately(self):
        """early return when _initialized is True."""
        engine = SymbolicEngine()
        engine._initialized = True
        # Should be a no-op; no exception.
        await engine.initialize()

    @pytest.mark.asyncio
    async def test_z3_import_error_degraded_mode(self):
        """z3 package missing -> degraded mode."""
        engine = SymbolicEngine()

        with _z3_import_raises(ImportError("z3 not available")):
            await engine.initialize()

        assert engine._degraded is True
        assert engine._initialized is True

    @pytest.mark.asyncio
    async def test_z3_native_library_unloadable_degrades(self):
        """z3 installed but its native library blocked -> degraded, not a crash.

        On Windows, Smart App Control / WDAC refuses to load ``libz3.dll``, and
        z3 raises its own ``Z3Exception`` from the import rather than
        ``ImportError``. The engine must still degrade.
        """
        engine = SymbolicEngine()

        with _z3_import_raises(Z3Exception("libz3.dll not found.")):
            await engine.initialize()

        assert engine._degraded is True
        assert engine._initialized is True
        assert engine._solver_pool.empty()

    @pytest.mark.asyncio
    async def test_unrelated_import_failure_propagates(self):
        """A failure that is not z3 availability must not be swallowed."""
        engine = SymbolicEngine()

        with _z3_import_raises(RuntimeError("disk on fire")), pytest.raises(RuntimeError):
            await engine.initialize()

        assert engine._degraded is False
        assert engine._initialized is False


class TestZ3UnavailablePredicate:
    """The predicate is the only thing standing between a broken z3 and a crash."""

    def test_import_error_is_unavailable(self):
        assert _z3_unavailable(ImportError("no module named z3")) is True

    def test_module_not_found_error_is_unavailable(self):
        assert _z3_unavailable(ModuleNotFoundError("z3")) is True

    def test_z3_exception_is_unavailable(self):
        assert _z3_unavailable(Z3Exception("libz3.so not found.")) is True

    def test_other_errors_are_not_unavailable(self):
        assert _z3_unavailable(RuntimeError("boom")) is False
        assert _z3_unavailable(ValueError("bad value")) is False


# ---------------------------------------------------------------------------
# SymbolicEngine.parse_task
# ---------------------------------------------------------------------------


class TestParseTaskMaxConstraints:
    @pytest.mark.asyncio
    async def test_max_constraints_limits_requirements(self):
        """break when max_constraints reached for requirements."""
        engine = SymbolicEngine(max_constraints=1)
        # Mock _parse_requirement to avoid real z3.
        engine._parse_requirement = AsyncMock(
            return_value=Constraint(name="r", expression=True, description="d", variable=None)
        )
        task = {
            "requirements": [
                {"name": "a", "type": "numeric", "min": 0, "max": 1},
                {"name": "b", "type": "numeric", "min": 0, "max": 2},
                {"name": "c", "type": "numeric", "min": 0, "max": 3},
            ],
        }
        graph = await engine.parse_task(task)
        # Only 1 requirement should be parsed before the break.
        assert len(graph.requirements) == 1

    @pytest.mark.asyncio
    async def test_max_constraints_limits_constraints(self):
        """break when max_constraints reached for constraints."""
        engine = SymbolicEngine(max_constraints=1)
        engine._parse_requirement = AsyncMock(return_value=None)
        engine._parse_constraint = AsyncMock(
            return_value=Constraint(name="c", expression=True, description="d", variable=None)
        )
        task = {
            "requirements": [],
            "constraints": [
                {"name": "c1", "condition": "true"},
                {"name": "c2", "condition": "false"},
            ],
        }
        graph = await engine.parse_task(task)
        assert len(graph.constraints) == 1


# ---------------------------------------------------------------------------
# SymbolicEngine.verify_solution
# ---------------------------------------------------------------------------


class TestVerifySolutionEdgeCases:
    @pytest.mark.asyncio
    async def test_degraded_mode_returns_false(self):
        """degraded engine cannot verify."""
        engine = SymbolicEngine()
        engine._degraded = True
        engine._initialized = True
        tg = TaskGraph()
        valid, violations = await engine.verify_solution({}, tg)
        assert valid is False
        assert "symbolic_engine_unavailable_z3_not_installed" in violations

    @pytest.mark.asyncio
    async def test_symbolic_verification_error_reraises(self):
        """SymbolicVerificationError is re-raised."""
        engine = SymbolicEngine()
        engine._degraded = False
        engine._initialized = True

        async def _raise_sve():
            raise SymbolicVerificationError("pool exhausted", ["resource_exhaustion"])

        # Patch _get_solver to raise SymbolicVerificationError
        async def _bad_solver_ctx():
            raise SymbolicVerificationError("pool exhausted", ["resource_exhaustion"])
            yield  # unreachable, makes this an async generator

        # We need to make the async context manager raise.
        engine._get_solver = MagicMock(side_effect=SymbolicVerificationError("boom", ["x"]))

        with pytest.raises(SymbolicVerificationError):
            await engine.verify_solution(
                {}, TaskGraph(requirements=[Constraint(name="x", expression=True, description="d")])
            )

    @pytest.mark.asyncio
    async def test_generic_exception_returns_error(self):
        """generic exception caught and returned."""
        engine = SymbolicEngine()
        engine._degraded = False
        engine._initialized = True

        tg = TaskGraph(
            requirements=[Constraint(name="x", expression=True, description="d")],
            variables={},
        )

        # Make _get_solver raise a RuntimeError
        class _BadCM:
            async def __aenter__(self):
                raise RuntimeError("unexpected failure")

            async def __aexit__(self, *a):
                return False

        engine._get_solver = lambda: _BadCM()

        valid, violations = await engine.verify_solution({"x": 1}, tg)
        assert valid is False
        assert any("verification_error" in v for v in violations)


# ---------------------------------------------------------------------------
# SymbolicEngine._check_solution
# ---------------------------------------------------------------------------


class TestCheckSolution:
    @pytest.mark.asyncio
    async def test_constraints_unsatisfiable(self):
        """solver.check() returns unsat on the requirements."""
        fake_z3 = _make_fake_z3()
        engine = SymbolicEngine()

        solver = MagicMock()
        solver.check.return_value = fake_z3.unsat

        tg = TaskGraph(
            requirements=[Constraint(name="x", expression="expr", description="d")],
            variables={},
        )

        with patch.dict(sys.modules, {"z3": fake_z3}):
            valid, violations = await engine._check_solution(solver, {}, tg)
        assert valid is False
        assert "constraints_unsatisfiable" in violations

    @pytest.mark.asyncio
    async def test_solution_variable_not_in_graph_continues(self):
        """var is None → continue (skip that solution key)."""
        fake_z3 = _make_fake_z3()
        engine = SymbolicEngine()

        solver = MagicMock()
        solver.check.return_value = "sat"  # not unsat

        tg = TaskGraph(
            requirements=[Constraint(name="x", expression="expr", description="d")],
            variables={"x": MagicMock()},  # only "x" is known
        )

        with patch.dict(sys.modules, {"z3": fake_z3}):
            # "y" is not in task_graph.variables → skipped
            valid, violations = await engine._check_solution(solver, {"y": 10}, tg)
        # "y" was skipped, checked==0, solution is non-empty → no_solution_variable_matched
        assert valid is False
        assert "no_solution_variable_matched" in violations

    @pytest.mark.asyncio
    async def test_coerce_value_returns_none_for_solution(self):
        """_coerce_value returns None → uncheckable."""
        fake_z3 = _make_fake_z3()
        engine = SymbolicEngine()

        solver = MagicMock()
        solver.check.return_value = "sat"

        # Use a BoolRef variable but supply a value that can't be coerced
        bool_var = fake_z3.BoolRef()
        tg = TaskGraph(
            requirements=[Constraint(name="flag", expression="expr", description="d")],
            variables={"flag": bool_var},
        )

        with patch.dict(sys.modules, {"z3": fake_z3}):
            # "maybe" is not a recognised boolean string → _coerce_value → None
            valid, violations = await engine._check_solution(solver, {"flag": "maybe"}, tg)
        assert valid is False
        assert any("uncheckable" in v for v in violations)

    @pytest.mark.asyncio
    async def test_no_solution_variable_matched(self):
        """solution keys don't match any known variable."""
        fake_z3 = _make_fake_z3()
        engine = SymbolicEngine()

        solver = MagicMock()
        solver.check.return_value = "sat"

        tg = TaskGraph(
            requirements=[Constraint(name="x", expression="expr", description="d")],
            variables={"x": MagicMock()},
        )

        with patch.dict(sys.modules, {"z3": fake_z3}):
            # "z" is not in variables → checked stays 0
            valid, violations = await engine._check_solution(solver, {"z": 99}, tg)
        assert valid is False
        assert "no_solution_variable_matched" in violations


# ---------------------------------------------------------------------------
# SymbolicEngine._parse_requirement
# ---------------------------------------------------------------------------


class TestParseRequirement:
    @pytest.mark.asyncio
    async def test_z3_import_error_returns_none(self):
        """z3 not installed -> requirement skipped."""
        engine = SymbolicEngine()

        with _z3_import_raises(ImportError("no z3")):
            result = await engine._parse_requirement(
                {"name": "x", "type": "numeric", "min": 0, "max": 1}
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_z3_native_library_unloadable_returns_none(self):
        """z3 present but its native library unusable -> skipped, not raised."""
        engine = SymbolicEngine()

        with _z3_import_raises(Z3Exception("libz3.dll not found.")):
            result = await engine._parse_requirement(
                {"name": "x", "type": "numeric", "min": 0, "max": 1}
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_unrelated_import_failure_propagates(self):
        """A non-z3 import failure is a real bug and must surface."""
        engine = SymbolicEngine()

        with _z3_import_raises(RuntimeError("boom")), pytest.raises(RuntimeError):
            await engine._parse_requirement({"name": "x", "type": "numeric", "min": 0, "max": 1})

    @pytest.mark.asyncio
    async def test_non_dict_with_model_dump(self):
        """non-dict req with model_dump() → converted."""
        fake_z3 = _make_fake_z3()
        engine = SymbolicEngine()

        class FakeReq:
            def model_dump(self):
                return {"name": "x", "type": "numeric", "min": 0, "max": 10}

        with patch.dict(sys.modules, {"z3": fake_z3}):
            result = await engine._parse_requirement(FakeReq())
        assert result is not None
        assert result.name == "x"

    @pytest.mark.asyncio
    async def test_non_dict_without_model_dump_raises_typeerror(self):
        """non-dict without model_dump → TypeError → caught → None."""
        fake_z3 = _make_fake_z3()
        engine = SymbolicEngine()

        with patch.dict(sys.modules, {"z3": fake_z3}):
            result = await engine._parse_requirement(42)  # int has no model_dump
        assert result is None

    @pytest.mark.asyncio
    async def test_numeric_no_bounds_returns_none(self):
        """numeric type but no extractable bounds → None."""
        fake_z3 = _make_fake_z3()
        engine = SymbolicEngine()

        req = {"name": "x", "type": "numeric"}  # no min/max/description
        with patch.dict(sys.modules, {"z3": fake_z3}):
            result = await engine._parse_requirement(req)
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_during_parsing_returns_none(self):
        """generic exception → return None."""
        fake_z3 = _make_fake_z3()
        engine = SymbolicEngine()

        # Make _extract_bounds blow up
        req = {"name": "x", "type": "numeric", "min": 0, "max": 10}

        with (
            patch.dict(sys.modules, {"z3": fake_z3}),
            patch("noema.neurosymbolic.symbolic._extract_bounds", side_effect=RuntimeError("boom")),
        ):
            result = await engine._parse_requirement(req)
        assert result is None


# ---------------------------------------------------------------------------
# SymbolicEngine._parse_constraint
# ---------------------------------------------------------------------------


class TestParseConstraint:
    @pytest.mark.asyncio
    async def test_non_dict_with_model_dump(self):
        """non-dict const with model_dump()."""
        fake_z3 = _make_fake_z3()
        engine = SymbolicEngine()

        class FakeConst:
            def model_dump(self):
                return {"name": "flag", "condition": "must hold"}

        with patch.dict(sys.modules, {"z3": fake_z3}):
            result = await engine._parse_constraint(FakeConst())
        assert result is not None
        assert result.name == "flag"

    @pytest.mark.asyncio
    async def test_non_dict_without_model_dump_raises_typeerror(self):
        """non-dict without model_dump → TypeError → caught."""
        fake_z3 = _make_fake_z3()
        engine = SymbolicEngine()

        with patch.dict(sys.modules, {"z3": fake_z3}):
            result = await engine._parse_constraint("not_a_dict")
        # "not_a_dict" is a str, which is not a dict and has no model_dump → TypeError → None
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_during_parsing_returns_none(self):
        """generic exception → return None."""
        fake_z3 = _make_fake_z3()
        engine = SymbolicEngine()

        fake_z3.Bool = MagicMock(side_effect=RuntimeError("z3 error"))
        with patch.dict(sys.modules, {"z3": fake_z3}):
            result = await engine._parse_constraint({"name": "c", "condition": "x"})
        assert result is None
