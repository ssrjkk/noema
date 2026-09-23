"""Comprehensive tests for noema.sandbox.environment.

Covers every public function/method and all internal helpers in
``noema/sandbox/environment.py`` to drive coverage to 100%.

All Docker / subprocess / resource-limit interactions are mocked so the suite
runs without Docker, network access, or a Unix host.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from noema.sandbox.environment import (
    CodeValidationResult,
    DockerEnvironment,
    Environment,
    LocalEnvironment,
    SandboxConfig,
    SandboxResult,
    ValidationLevel,
    _build_isolated_env,
    _check_bubblewrap,
    _check_docker,
    _parse_pytest_counts,
    _safe_join,
    _set_resource_limits,
)

IS_UNIX = platform.system() != "Windows"

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> SandboxConfig:
    return SandboxConfig()


@pytest.fixture()
def local_env(config: SandboxConfig) -> LocalEnvironment:
    return LocalEnvironment(config)


@pytest.fixture()
def docker_env(config: SandboxConfig) -> DockerEnvironment:
    return DockerEnvironment(config)


@pytest.fixture()
def mock_log():
    """Capture the module logger so suppression paths are observable."""
    with patch("noema.sandbox.environment.log") as log_mock:
        yield log_mock


@pytest.fixture()
def unix_only():
    """Pretend the host is POSIX so the Unix-only branches execute here."""
    with patch("noema.sandbox.environment._IS_UNIX", True):
        yield


def _make_result(files: list[dict[str, str]] | None = None) -> SandboxResult:
    """Build a SandboxResult with one CodeValidationResult per file."""
    result = SandboxResult()
    for f in files or []:
        result.files.append(
            CodeValidationResult(
                file_path=f.get("path", f.get("filename", "main.py")),
                language=f.get("language", "python"),
            )
        )
    return result


def _py_file(
    content: str = "x = 1\n", path: str = "main.py", language: str = "python"
) -> dict[str, str]:
    return {"content": content, "path": path, "language": language}


# ===================================================================
# ValidationLevel enum
# ===================================================================


class TestValidationLevel:
    def test_members(self):
        assert ValidationLevel.AST.value == "ast"
        assert ValidationLevel.SYNTAX.value == "syntax"
        assert ValidationLevel.LINT.value == "lint"
        assert ValidationLevel.TYPE_CHECK.value == "type_check"
        assert ValidationLevel.RUN.value == "run"
        assert ValidationLevel.TEST.value == "test"


# ===================================================================
# SandboxConfig / dataclass defaults
# ===================================================================


class TestSandboxConfig:
    def test_defaults(self, config: SandboxConfig):
        assert config.enabled is True
        assert config.timeout == 60.0
        assert config.max_memory_mb == 256
        assert config.max_cpu_seconds == 10.0
        assert config.max_cpus == 0.5
        assert config.network_isolation is True
        assert config.docker_image == "noema-sandbox:3.12"
        assert config.static_check_enabled is True
        assert config.max_parallel == 4
        assert config.temp_dir == ""

    def test_custom_values(self):
        cfg = SandboxConfig(timeout=120, max_memory_mb=512, docker_image="custom:1.0")
        assert cfg.timeout == 120
        assert cfg.max_memory_mb == 512
        assert cfg.docker_image == "custom:1.0"


# ===================================================================
# SandboxResult / CodeValidationResult
# ===================================================================


class TestSandboxResult:
    def test_defaults(self):
        r = SandboxResult()
        assert r.all_valid is False
        assert r.files == []
        assert r.tests_passed == 0
        assert r.tests_failed == 0
        assert r.total_duration_ms == 0.0
        assert r.summary == ""
        assert r.details == {}


class TestCodeValidationResult:
    def test_defaults(self):
        cvr = CodeValidationResult(file_path="a.py", language="python")
        assert cvr.ast_valid is True
        assert cvr.ast_errors == []
        assert cvr.lint_passed is True
        assert cvr.run_passed is True
        assert cvr.duration_ms == 0.0


# ===================================================================
# _safe_join
# ===================================================================


class TestSafeJoin:
    def test_simple_relative(self, tmp_path: Path):
        result = _safe_join(tmp_path, "sub/file.py")
        assert result == (tmp_path.resolve() / "sub" / "file.py")

    def test_empty_defaults_to_main(self, tmp_path: Path):
        result = _safe_join(tmp_path, "")
        assert result.name == "main.py"

    def test_strips_leading_slash(self, tmp_path: Path):
        result = _safe_join(tmp_path, "/etc/passwd")
        assert str(result).startswith(str(tmp_path.resolve()))

    def test_rejects_traversal(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unsafe path"):
            _safe_join(tmp_path, "../../etc/passwd")

    def test_rejects_dotdot(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unsafe path"):
            _safe_join(tmp_path, "../escape.py")


# ===================================================================
# _build_isolated_env
# ===================================================================


class TestBuildIsolatedEnv:
    def test_contains_safe_vars(self):
        env = _build_isolated_env()
        assert "NO_PROXY" in env
        assert env["NO_PROXY"] == "*"
        assert env["no_proxy"] == "*"
        assert env["HTTP_PROXY"] == ""
        assert env["HTTPS_PROXY"] == ""
        assert env["http_proxy"] == ""
        assert env["https_proxy"] == ""
        assert env["PYTHONIOENCODING"] == "utf-8"
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"

    def test_blocks_proxy_vars(self):
        env = _build_isolated_env()
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            assert env[key] == ""

    def test_whitelists_path(self):
        with patch.dict(os.environ, {"PATH": "/usr/bin", "HOME": "/home/user", "SECRET": "nope"}):
            env = _build_isolated_env()
            assert "PATH" in env
            assert "HOME" in env
            assert "SECRET" not in env


# ===================================================================
# _parse_pytest_counts
# ===================================================================


class TestParsePytestCounts:
    def test_passed_and_failed(self):
        output = "collected 5 items\n3 passed, 2 failed\n"
        p, f = _parse_pytest_counts(output, 1)
        assert p == 3
        assert f == 2

    def test_with_errors(self):
        output = "1 passed, 1 error\n"
        p, f = _parse_pytest_counts(output, 1)
        assert p == 1
        assert f == 1

    def test_no_summary_returncode_zero(self):
        p, f = _parse_pytest_counts("some output", 0)
        assert p == 1
        assert f == 0

    def test_no_summary_returncode_nonzero(self):
        p, f = _parse_pytest_counts("some output", 1)
        assert p == 0
        assert f == 1

    def test_only_failed(self):
        output = "2 failed\n"
        p, f = _parse_pytest_counts(output, 1)
        assert p == 0
        assert f == 2

    def test_only_errors(self):
        output = "3 error\n"
        p, f = _parse_pytest_counts(output, 1)
        assert p == 0
        assert f == 3

    def test_large_output_truncated(self):
        output = "x" * 5000 + "5 passed, 1 failed\n"
        p, f = _parse_pytest_counts(output, 1)
        assert p == 5
        assert f == 1


# ===================================================================
# _set_resource_limits
# ===================================================================


class TestSetResourceLimits:
    """``resource`` is injected through ``sys.modules`` and ``_IS_UNIX`` is
    forced True, so these exercise the real limit-setting path on any host.
    """

    def test_sets_limits(self, unix_only):
        mock_resource = MagicMock()
        mock_resource.RLIMIT_CPU = 0
        mock_resource.RLIMIT_AS = 1
        mock_resource.RLIMIT_NPROC = 2
        mock_resource.RLIMIT_NOFILE = 3
        mock_resource.RLIMIT_FSIZE = 4
        with patch.dict("sys.modules", {"resource": mock_resource}):
            _set_resource_limits(cpu_sec=5, mem_mb=128)
        assert mock_resource.setrlimit.call_count == 5
        assert mock_resource.setrlimit.call_args_list[0] == call(0, (5, 10))
        assert mock_resource.setrlimit.call_args_list[1] == call(1, (128 * 1024 * 1024,) * 2)
        assert mock_resource.setrlimit.call_args_list[2] == call(2, (10, 10))
        assert mock_resource.setrlimit.call_args_list[3] == call(3, (64, 64))
        assert mock_resource.setrlimit.call_args_list[4] == call(4, (10 * 1024 * 1024,) * 2)

    def test_handles_import_error(self, unix_only, mock_log):
        with patch.dict("sys.modules", {"resource": None}):
            _set_resource_limits()
        mock_log.warning.assert_called_once_with("resource_limits_unavailable")

    def test_handles_os_error(self, unix_only, mock_log):
        mock_resource = MagicMock()
        mock_resource.RLIMIT_CPU = 0
        mock_resource.setrlimit.side_effect = OSError("nope")
        with patch.dict("sys.modules", {"resource": mock_resource}):
            _set_resource_limits()
        mock_log.warning.assert_called_once_with("resource_limits_unavailable")
        assert mock_resource.setrlimit.call_count == 1

    def test_handles_value_error_for_non_int_limit(self, unix_only, mock_log):
        mock_resource = MagicMock()
        mock_resource.RLIMIT_CPU = "not_an_int"
        with patch.dict("sys.modules", {"resource": mock_resource}):
            _set_resource_limits()
        mock_log.warning.assert_called_once_with("resource_limits_unavailable")
        mock_resource.setrlimit.assert_not_called()

    def test_noop_on_non_unix(self, mock_log):
        with (
            patch("noema.sandbox.environment._IS_UNIX", False),
            patch.dict("sys.modules", {"resource": None}),
        ):
            _set_resource_limits()
        mock_log.warning.assert_not_called()


# ===================================================================
# _check_bubblewrap
# ===================================================================


class TestCheckBubblewrap:
    def test_returns_false_on_non_unix(self):
        with patch("noema.sandbox.environment._IS_UNIX", False):
            assert _check_bubblewrap() is False

    def test_returns_true_when_available(self, unix_only):
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            assert _check_bubblewrap() is True
            mock_run.assert_called_once_with(
                ["bwrap", "--version"], capture_output=True, text=True, timeout=5
            )

    def test_returns_false_on_nonzero(self, unix_only):
        mock_result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_result):
            assert _check_bubblewrap() is False

    def test_returns_false_on_file_not_found(self, unix_only):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _check_bubblewrap() is False

    def test_returns_false_on_timeout(self, unix_only):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("bwrap", 5)):
            assert _check_bubblewrap() is False


# ===================================================================
# _check_docker
# ===================================================================


class TestCheckDocker:
    def test_returns_true_when_available(self):
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result):
            assert _check_docker() is True

    def test_returns_false_on_nonzero(self):
        mock_result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_result):
            assert _check_docker() is False

    def test_returns_false_on_file_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _check_docker() is False

    def test_returns_false_on_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 5)):
            assert _check_docker() is False


# ===================================================================
# Environment ABC
# ===================================================================


class TestEnvironmentABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Environment()  # type: ignore[abstract]

    def test_subclass_must_implement_methods(self):
        class Incomplete(Environment):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


# ===================================================================
# LocalEnvironment
# ===================================================================


class TestLocalEnvironment:
    def test_id_and_description(self, local_env: LocalEnvironment):
        assert local_env.id == "local"
        assert (
            "subprocess" in local_env.description.lower()
            or "resource" in local_env.description.lower()
        )

    def test_is_available(self, local_env: LocalEnvironment):
        assert local_env.is_available() is True

    def test_bwrap_disabled_returns_cmd_unchanged(self, local_env: LocalEnvironment):
        cmd = ["python", "main.py"]
        assert local_env.bwrap_cmd(cmd, Path("/tmp")) == cmd

    def test_bwrap_enabled_wraps_command(self, local_env: LocalEnvironment):
        local_env.bwrap_enabled = True
        cmd = ["python", "main.py"]
        wrapped = local_env.bwrap_cmd(cmd, Path("/tmp/sandbox"))
        assert wrapped[0] == "bwrap"
        assert "--unshare-all" in wrapped
        assert "/sandbox" in wrapped
        assert "--" in wrapped
        assert "python" in wrapped
        assert "main.py" in wrapped

    def test_resource_limits_preexec_non_unix(self, local_env: LocalEnvironment):
        with patch("noema.sandbox.environment._IS_UNIX", False):
            assert local_env.resource_limits_preexec() is None

    def test_resource_limits_preexec_unix(self, local_env: LocalEnvironment, unix_only):
        fn = local_env.resource_limits_preexec()
        assert fn is not None
        assert callable(fn)

    def test_resource_limits_preexec_binds_configured_limits(
        self, local_env: LocalEnvironment, unix_only
    ):
        fn = local_env.resource_limits_preexec()
        assert fn is not None
        mock_resource = MagicMock()
        mock_resource.RLIMIT_CPU = 0
        mock_resource.RLIMIT_AS = 1
        with patch.dict("sys.modules", {"resource": mock_resource}):
            fn()
        cpu = local_env.config.max_cpu_seconds
        mem = local_env.config.max_memory_mb
        mock_resource.setrlimit.assert_any_call(0, (int(cpu), int(cpu) + 5))
        mock_resource.setrlimit.assert_any_call(1, (mem * 1024 * 1024,) * 2)

    def test_resource_limits_preexec_callable_invokes_setrlimit(
        self, local_env: LocalEnvironment, unix_only
    ):
        fn = local_env.resource_limits_preexec()
        assert fn is not None
        mock_resource = MagicMock()
        mock_resource.RLIMIT_CPU = 0
        mock_resource.RLIMIT_AS = 1
        mock_resource.RLIMIT_NPROC = 2
        mock_resource.RLIMIT_NOFILE = 3
        mock_resource.RLIMIT_FSIZE = 4
        with patch.dict("sys.modules", {"resource": mock_resource}):
            fn()
        assert mock_resource.setrlimit.call_count == 5


# ===================================================================
# LocalEnvironment._write_files
# ===================================================================


class TestLocalWriteFiles:
    def test_writes_files(self, local_env: LocalEnvironment, tmp_path: Path):
        files = [
            {"content": "x = 1\n", "path": "main.py", "language": "python"},
            {"content": "y = 2\n", "path": "sub/helper.py", "language": "python"},
        ]
        local_env._write_files(files, tmp_path)
        assert (tmp_path / "main.py").read_text(encoding="utf-8") == "x = 1\n"
        assert (tmp_path / "sub" / "helper.py").read_text(encoding="utf-8") == "y = 2\n"

    def test_defaults_filename(self, local_env: LocalEnvironment, tmp_path: Path):
        files = [{"content": "z = 3\n", "language": "python"}]
        local_env._write_files(files, tmp_path)
        assert (tmp_path / "main.py").exists()

    def test_uses_filename_key(self, local_env: LocalEnvironment, tmp_path: Path):
        files = [{"content": "z = 3\n", "filename": "alt.py", "language": "python"}]
        local_env._write_files(files, tmp_path)
        assert (tmp_path / "alt.py").exists()


# ===================================================================
# LocalEnvironment.lint
# ===================================================================


class TestLocalLint:
    @pytest.mark.asyncio
    async def test_lint_success(self, local_env: LocalEnvironment):
        files = [_py_file("x = 1\n")]
        result = _make_result(files)
        mock_proc = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_proc):
            result = await local_env.lint(result, files)
        assert result.files[0].lint_passed is True

    @pytest.mark.asyncio
    async def test_lint_failure(self, local_env: LocalEnvironment):
        files = [_py_file("import os\n")]
        result = _make_result(files)
        mock_proc = MagicMock(returncode=1, stdout="E501 line too long", stderr="")
        with patch("subprocess.run", return_value=mock_proc):
            result = await local_env.lint(result, files)
        assert result.files[0].lint_passed is False
        assert len(result.files[0].lint_errors) > 0

    @pytest.mark.asyncio
    async def test_lint_non_python_skipped(self, local_env: LocalEnvironment):
        files = [_py_file("console.log(1)", language="javascript", path="main.js")]
        result = _make_result(files)
        with patch("subprocess.run") as mock_run:
            result = await local_env.lint(result, files)
        mock_run.assert_not_called()
        assert result.files[0].lint_passed is True

    @pytest.mark.asyncio
    async def test_lint_timeout(self, local_env: LocalEnvironment, mock_log, tmp_path: Path):
        files = [_py_file()]
        result = _make_result(files)
        work = tmp_path / "lint-work"
        with (
            patch("tempfile.mkdtemp", return_value=str(work)),
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ruff", 10)),
        ):
            result = await local_env.lint(result, files)
        mock_log.warning.assert_called_once_with("lint_timeout")
        assert result.files[0].lint_passed is True
        assert not work.exists()

    @pytest.mark.asyncio
    async def test_lint_generic_exception(
        self, local_env: LocalEnvironment, mock_log, tmp_path: Path
    ):
        files = [_py_file()]
        result = _make_result(files)
        work = tmp_path / "lint-work"
        with (
            patch("tempfile.mkdtemp", return_value=str(work)),
            patch("subprocess.run", side_effect=RuntimeError("boom")),
        ):
            result = await local_env.lint(result, files)
        mock_log.warning.assert_called_once_with("lint_direct_error", error="boom")
        assert result.files[0].lint_passed is True
        assert not work.exists()

    @pytest.mark.asyncio
    async def test_lint_empty_output(self, local_env: LocalEnvironment):
        files = [_py_file()]
        result = _make_result(files)
        mock_proc = MagicMock(returncode=1, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_proc):
            result = await local_env.lint(result, files)
        assert result.files[0].lint_passed is False
        assert result.files[0].lint_errors == ["lint produced no output"]

    @pytest.mark.asyncio
    async def test_lint_multiple_files(self, local_env: LocalEnvironment):
        files = [_py_file("a = 1\n", path="a.py"), _py_file("b = 2\n", path="b.py")]
        result = _make_result(files)
        mock_proc = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_proc):
            result = await local_env.lint(result, files)
        assert all(f.lint_passed for f in result.files)


# ===================================================================
# LocalEnvironment.run_code
# ===================================================================


class TestLocalRunCode:
    @pytest.mark.asyncio
    async def test_run_success(self, local_env: LocalEnvironment):
        files = [_py_file("print('hello')")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"hello\n", b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await local_env.run_code(result, files)
        assert result.files[0].run_passed is True
        assert "hello" in result.files[0].run_output

    @pytest.mark.asyncio
    async def test_run_failure_nonzero(self, local_env: LocalEnvironment):
        files = [_py_file("raise ValueError('oops')")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Traceback..."))
        mock_proc.returncode = 1
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await local_env.run_code(result, files)
        assert result.files[0].run_passed is False
        assert "Traceback" in result.files[0].run_errors

    @pytest.mark.asyncio
    async def test_run_non_python_skipped(self, local_env: LocalEnvironment):
        files = [_py_file("console.log(1)", language="javascript", path="main.js")]
        result = _make_result(files)
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            result = await local_env.run_code(result, files)
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_static_failed_skipped(self, local_env: LocalEnvironment):
        files = [_py_file()]
        result = _make_result(files)
        result.files[0].static_passed = False
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            result = await local_env.run_code(result, files)
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_timeout(self, local_env: LocalEnvironment):
        files = [_py_file("while True: pass")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await local_env.run_code(result, files)
        assert result.files[0].run_passed is False
        assert "timed out" in result.files[0].run_errors.lower()

    @pytest.mark.asyncio
    async def test_run_generic_exception(self, local_env: LocalEnvironment):
        files = [_py_file()]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=RuntimeError("boom"))
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await local_env.run_code(result, files)
        assert result.files[0].run_passed is False
        assert "boom" in result.files[0].run_errors

    @pytest.mark.asyncio
    async def test_run_bwrap_enabled(self, local_env: LocalEnvironment):
        local_env.bwrap_enabled = True
        files = [_py_file("print('ok')")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"ok\n", b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            result = await local_env.run_code(result, files)
        # Verify bwrap wrapping was used
        call_args = mock_exec.call_args
        assert call_args[0][0] == "bwrap"

    @pytest.mark.asyncio
    async def test_run_kills_hung_proc(self, local_env: LocalEnvironment):
        files = [_py_file()]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await local_env.run_code(result, files)
        mock_proc.kill.assert_called_once()
        mock_proc.wait.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_outer_exception(self, local_env: LocalEnvironment, mock_log, tmp_path: Path):
        files = [_py_file()]
        result = _make_result(files)
        work = tmp_path / "run-work"
        with (
            patch("tempfile.mkdtemp", return_value=str(work)),
            patch.object(local_env, "_write_files", side_effect=RuntimeError("disk full")),
        ):
            result = await local_env.run_code(result, files)
        mock_log.warning.assert_called_once_with("run_code_direct_error", error="disk full")
        assert result.files[0].run_passed is True
        assert not work.exists()

    @pytest.mark.asyncio
    async def test_run_output_truncated(self, local_env: LocalEnvironment):
        files = [_py_file()]
        result = _make_result(files)
        long_output = b"x" * 1000
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(long_output, b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await local_env.run_code(result, files)
        assert len(result.files[0].run_output) <= 500

    @pytest.mark.asyncio
    async def test_run_errors_truncated(self, local_env: LocalEnvironment):
        files = [_py_file()]
        result = _make_result(files)
        long_error = b"e" * 2000
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", long_error))
        mock_proc.returncode = 1
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await local_env.run_code(result, files)
        assert len(result.files[0].run_errors) <= 1000


# ===================================================================
# LocalEnvironment.run_tests
# ===================================================================


class TestLocalRunTests:
    @pytest.mark.asyncio
    async def test_no_test_files_returns_immediately(self, local_env: LocalEnvironment):
        files = [_py_file(path="main.py")]
        result = _make_result(files)
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            result = await local_env.run_tests(result, files)
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_pytest(self, local_env: LocalEnvironment):
        files = [_py_file("def test_ok(): assert True", path="test_main.py")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"1 passed\n", b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await local_env.run_tests(result, files)
        assert result.tests_passed == 1
        assert result.tests_failed == 0

    @pytest.mark.asyncio
    async def test_pytest_failure(self, local_env: LocalEnvironment):
        files = [_py_file("def test_bad(): assert False", path="test_fail.py")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"1 failed\n", b""))
        mock_proc.returncode = 1
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await local_env.run_tests(result, files)
        assert result.tests_failed >= 1

    @pytest.mark.asyncio
    async def test_tests_timeout(self, local_env: LocalEnvironment, mock_log):
        files = [_py_file(path="test_slow.py")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await local_env.run_tests(result, files)
        mock_log.warning.assert_called_once_with("tests_timeout")
        assert (result.tests_passed, result.tests_failed) == (0, 0)
        mock_proc.kill.assert_called_once()
        mock_proc.wait.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tests_generic_exception(self, local_env: LocalEnvironment, mock_log):
        files = [_py_file(path="test_err.py")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=RuntimeError("boom"))
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await local_env.run_tests(result, files)
        mock_log.warning.assert_called_once_with("test_direct_error", error="boom")
        assert (result.tests_passed, result.tests_failed) == (0, 0)
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_tests_kills_hung_proc(self, local_env: LocalEnvironment):
        files = [_py_file(path="test_hang.py")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await local_env.run_tests(result, files)
        mock_proc.kill.assert_called_once()
        mock_proc.wait.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tests_with_bwrap(self, local_env: LocalEnvironment):
        local_env.bwrap_enabled = True
        files = [_py_file(path="test_bwrap.py")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"2 passed\n", b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            result = await local_env.run_tests(result, files)
        call_args = mock_exec.call_args
        assert call_args[0][0] == "bwrap"
        assert result.tests_passed == 2

    @pytest.mark.asyncio
    async def test_tests_outer_exception(self, local_env: LocalEnvironment, mock_log, tmp_path):
        files = [_py_file(path="test_outer.py")]
        result = _make_result(files)
        work = tmp_path / "tests-work"
        with (
            patch("tempfile.mkdtemp", return_value=str(work)),
            patch.object(local_env, "_write_files", side_effect=RuntimeError("disk full")),
        ):
            result = await local_env.run_tests(result, files)
        mock_log.warning.assert_called_once_with("test_direct_error", error="disk full")
        assert (result.tests_passed, result.tests_failed) == (0, 0)
        assert not work.exists()

    @pytest.mark.asyncio
    async def test_detects_test_in_filename_key(self, local_env: LocalEnvironment):
        files = [{"content": "def test_x(): pass", "filename": "test_foo.py", "language": "python"}]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"1 passed\n", b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await local_env.run_tests(result, files)
        assert result.tests_passed == 1


# ===================================================================
# DockerEnvironment
# ===================================================================


class TestDockerEnvironment:
    def test_id_and_description(self, docker_env: DockerEnvironment):
        assert docker_env.id == "docker"
        assert "docker" in docker_env.description.lower()

    def test_is_available_delegates_to_check_docker(self, docker_env: DockerEnvironment):
        with patch("noema.sandbox.environment._check_docker", return_value=True):
            assert docker_env.is_available() is True
        with patch("noema.sandbox.environment._check_docker", return_value=False):
            assert docker_env.is_available() is False

    def test_docker_run_flags_no_network(self, docker_env: DockerEnvironment):
        docker_env.config.network_isolation = True
        flags = docker_env.docker_run_flags()
        assert "--rm" in flags
        assert "--network=none" in flags
        assert "--memory" in flags
        assert "256m" in flags
        assert "--cpus" in flags
        assert "0.5" in flags

    def test_docker_run_flags_with_network(self, docker_env: DockerEnvironment):
        docker_env.config.network_isolation = False
        flags = docker_env.docker_run_flags()
        assert "--network=none" not in flags

    def test_docker_run_flags_custom_limits(self):
        cfg = SandboxConfig(max_memory_mb=512, max_cpus=2.0, network_isolation=False)
        env = DockerEnvironment(cfg)
        flags = env.docker_run_flags()
        assert "512m" in flags
        assert "2.0" in flags


# ===================================================================
# DockerEnvironment._write_files
# ===================================================================


class TestDockerWriteFiles:
    def test_writes_files(self, docker_env: DockerEnvironment, tmp_path: Path):
        files = [{"content": "x = 1\n", "path": "main.py", "language": "python"}]
        docker_env._write_files(files, tmp_path)
        assert (tmp_path / "main.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_creates_subdirs(self, docker_env: DockerEnvironment, tmp_path: Path):
        files = [{"content": "y = 2\n", "path": "pkg/mod.py", "language": "python"}]
        docker_env._write_files(files, tmp_path)
        assert (tmp_path / "pkg" / "mod.py").exists()


# ===================================================================
# DockerEnvironment.lint
# ===================================================================


class TestDockerLint:
    @pytest.mark.asyncio
    async def test_lint_success(self, docker_env: DockerEnvironment):
        files = [_py_file("x = 1\n")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.lint(result, files)
        assert result.files[0].lint_passed is True

    @pytest.mark.asyncio
    async def test_lint_failure(self, docker_env: DockerEnvironment):
        files = [_py_file("bad code")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"E501 too long\n", b""))
        mock_proc.returncode = 1
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.lint(result, files)
        assert result.files[0].lint_passed is False
        assert "E501" in result.files[0].lint_errors[0]

    @pytest.mark.asyncio
    async def test_lint_failure_empty_output(self, docker_env: DockerEnvironment):
        files = [_py_file("bad")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"some stderr\n"))
        mock_proc.returncode = 1
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.lint(result, files)
        assert result.files[0].lint_passed is False

    @pytest.mark.asyncio
    async def test_lint_timeout(self, docker_env: DockerEnvironment):
        files = [_py_file()]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.lint(result, files)
        assert result.files[0].lint_passed is False
        assert "timed out" in result.files[0].lint_errors[0].lower()

    @pytest.mark.asyncio
    async def test_lint_kills_hung_proc(self, docker_env: DockerEnvironment):
        files = [_py_file()]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.lint(result, files)
        mock_proc.kill.assert_called_once()
        mock_proc.wait.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lint_non_python_skipped(self, docker_env: DockerEnvironment):
        files = [_py_file("code", language="rust", path="main.rs")]
        result = _make_result(files)
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            result = await docker_env.lint(result, files)
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_lint_outer_exception(
        self, docker_env: DockerEnvironment, mock_log, tmp_path: Path
    ):
        files = [_py_file()]
        result = _make_result(files)
        work = tmp_path / "docker-lint-work"
        with (
            patch("tempfile.mkdtemp", return_value=str(work)),
            patch.object(docker_env, "_write_files", side_effect=RuntimeError("disk full")),
        ):
            result = await docker_env.lint(result, files)
        mock_log.warning.assert_called_once_with("lint_docker_error", error="disk full")
        assert result.files[0].lint_passed is True
        assert not work.exists()


# ===================================================================
# DockerEnvironment.run_code
# ===================================================================


class TestDockerRunCode:
    @pytest.mark.asyncio
    async def test_run_success(self, docker_env: DockerEnvironment):
        files = [_py_file("print('hello')")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"hello\n", b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.run_code(result, files)
        assert result.files[0].run_passed is True
        assert "hello" in result.files[0].run_output

    @pytest.mark.asyncio
    async def test_run_failure(self, docker_env: DockerEnvironment):
        files = [_py_file("raise ValueError")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Traceback\n"))
        mock_proc.returncode = 1
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.run_code(result, files)
        assert result.files[0].run_passed is False
        assert "Traceback" in result.files[0].run_errors

    @pytest.mark.asyncio
    async def test_run_non_python_skipped(self, docker_env: DockerEnvironment):
        files = [_py_file("code", language="go", path="main.go")]
        result = _make_result(files)
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            result = await docker_env.run_code(result, files)
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_static_failed_skipped(self, docker_env: DockerEnvironment):
        files = [_py_file()]
        result = _make_result(files)
        result.files[0].static_passed = False
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            result = await docker_env.run_code(result, files)
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_timeout(self, docker_env: DockerEnvironment):
        files = [_py_file("while True: pass")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.run_code(result, files)
        assert result.files[0].run_passed is False
        assert "timed out" in result.files[0].run_errors.lower()

    @pytest.mark.asyncio
    async def test_run_generic_exception(self, docker_env: DockerEnvironment):
        files = [_py_file()]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=RuntimeError("container error"))
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.run_code(result, files)
        assert result.files[0].run_passed is False
        assert "container error" in result.files[0].run_errors

    @pytest.mark.asyncio
    async def test_run_kills_hung_proc(self, docker_env: DockerEnvironment):
        files = [_py_file()]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.run_code(result, files)
        mock_proc.kill.assert_called_once()
        mock_proc.wait.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_outer_exception(
        self, docker_env: DockerEnvironment, mock_log, tmp_path: Path
    ):
        files = [_py_file()]
        result = _make_result(files)
        work = tmp_path / "docker-run-work"
        with (
            patch("tempfile.mkdtemp", return_value=str(work)),
            patch.object(docker_env, "_write_files", side_effect=RuntimeError("disk full")),
        ):
            result = await docker_env.run_code(result, files)
        mock_log.warning.assert_called_once_with("run_code_docker_error", error="disk full")
        assert result.files[0].run_passed is True
        assert not work.exists()

    @pytest.mark.asyncio
    async def test_run_output_truncated(self, docker_env: DockerEnvironment):
        files = [_py_file()]
        result = _make_result(files)
        long_output = b"x" * 1000
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(long_output, b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.run_code(result, files)
        assert len(result.files[0].run_output) <= 500

    @pytest.mark.asyncio
    async def test_run_errors_truncated(self, docker_env: DockerEnvironment):
        files = [_py_file()]
        result = _make_result(files)
        long_error = b"e" * 2000
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", long_error))
        mock_proc.returncode = 1
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.run_code(result, files)
        assert len(result.files[0].run_errors) <= 1000

    @pytest.mark.asyncio
    async def test_run_duration_recorded(self, docker_env: DockerEnvironment):
        files = [_py_file("print(1)")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"1\n", b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.run_code(result, files)
        assert result.files[0].duration_ms >= 0


# ===================================================================
# DockerEnvironment.run_tests
# ===================================================================


class TestDockerRunTests:
    @pytest.mark.asyncio
    async def test_no_test_files_returns_immediately(self, docker_env: DockerEnvironment):
        files = [_py_file(path="main.py")]
        result = _make_result(files)
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            result = await docker_env.run_tests(result, files)
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_pytest_in_docker(self, docker_env: DockerEnvironment):
        files = [_py_file("def test_ok(): assert True", path="test_main.py")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"1 passed\n", b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.run_tests(result, files)
        assert result.tests_passed == 1
        assert result.tests_failed == 0

    @pytest.mark.asyncio
    async def test_pytest_failure(self, docker_env: DockerEnvironment):
        files = [_py_file("def test_bad(): assert False", path="test_fail.py")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"1 failed\n", b""))
        mock_proc.returncode = 1
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.run_tests(result, files)
        assert result.tests_failed >= 1

    @pytest.mark.asyncio
    async def test_tests_timeout(self, docker_env: DockerEnvironment, mock_log):
        files = [_py_file(path="test_slow.py")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.run_tests(result, files)
        mock_log.warning.assert_called_once_with("tests_docker_timeout")
        assert (result.tests_passed, result.tests_failed) == (0, 0)

    @pytest.mark.asyncio
    async def test_tests_generic_exception(self, docker_env: DockerEnvironment, mock_log):
        files = [_py_file(path="test_err.py")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=RuntimeError("boom"))
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.run_tests(result, files)
        mock_log.warning.assert_called_once_with("test_docker_error", error="boom")
        assert (result.tests_passed, result.tests_failed) == (0, 0)

    @pytest.mark.asyncio
    async def test_tests_kills_hung_proc(self, docker_env: DockerEnvironment):
        files = [_py_file(path="test_hang.py")]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.run_tests(result, files)
        mock_proc.kill.assert_called_once()
        mock_proc.wait.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tests_outer_exception(
        self, docker_env: DockerEnvironment, mock_log, tmp_path: Path
    ):
        files = [_py_file(path="test_outer.py")]
        result = _make_result(files)
        work = tmp_path / "docker-tests-work"
        with (
            patch("tempfile.mkdtemp", return_value=str(work)),
            patch.object(docker_env, "_write_files", side_effect=RuntimeError("disk full")),
        ):
            result = await docker_env.run_tests(result, files)
        mock_log.warning.assert_called_once_with("test_docker_error", error="disk full")
        assert (result.tests_passed, result.tests_failed) == (0, 0)
        assert not work.exists()

    @pytest.mark.asyncio
    async def test_detects_test_in_filename_key(self, docker_env: DockerEnvironment):
        files = [{"content": "def test_x(): pass", "filename": "test_foo.py", "language": "python"}]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"1 passed\n", b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.run_tests(result, files)
        assert result.tests_passed == 1

    @pytest.mark.asyncio
    async def test_mixed_files_only_tests_run(self, docker_env: DockerEnvironment):
        files = [
            _py_file("x = 1\n", path="main.py"),
            _py_file("def test_x(): pass", path="test_x.py"),
        ]
        result = _make_result(files)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"1 passed\n", b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await docker_env.run_tests(result, files)
        assert result.tests_passed == 1
