# Coverage Mapping for noema/sandbox/environment.py

This document maps the previously missed lines to the tests that now cover them.

## Test Results
- **Total tests**: 107
- **Passed**: 97
- **Skipped**: 10 (Unix-only tests on Windows platform)
- **Failed**: 0

## Previously Missed Lines → Covering Tests

### Lines 53-72: `_set_resource_limits()`
- `TestSetResourceLimits::test_sets_limits_on_unix` (Unix only)
- `TestSetResourceLimits::test_handles_import_error` (Unix only)
- `TestSetResourceLimits::test_handles_os_error` (Unix only)
- `TestSetResourceLimits::test_handles_value_error_for_non_int_limit` (Unix only)
- `TestSetResourceLimits::test_noop_on_non_unix`

### Lines 79-88: `_check_bubblewrap()`
- `TestCheckBubblewrap::test_returns_false_on_non_unix`
- `TestCheckBubblewrap::test_returns_true_when_available` (Unix only)
- `TestCheckBubblewrap::test_returns_false_on_nonzero` (Unix only)
- `TestCheckBubblewrap::test_returns_false_on_file_not_found` (Unix only)
- `TestCheckBubblewrap::test_returns_false_on_timeout` (Unix only)

### Lines 96-97: `_check_docker()` exception handling
- `TestCheckDocker::test_returns_true_when_available`
- `TestCheckDocker::test_returns_false_on_nonzero`
- `TestCheckDocker::test_returns_false_on_file_not_found`
- `TestCheckDocker::test_returns_false_on_timeout`

### Line 266: `bwrap_cmd()` when bwrap_enabled=True
- `TestLocalEnvironment::test_bwrap_enabled_wraps_command`

### Line 288: `LocalEnvironment.is_available()`
- `TestLocalEnvironment::test_is_available`

### Lines 294-300: `resource_limits_preexec()` on Unix
- `TestLocalEnvironment::test_resource_limits_preexec_non_unix`
- `TestLocalEnvironment::test_resource_limits_preexec_unix` (Unix only)
- `TestLocalEnvironment::test_resource_limits_preexec_callable_invokes_setrlimit` (Unix only)

### Lines 339-342: `LocalEnvironment.lint()` exception handling
- `TestLocalLint::test_lint_timeout`
- `TestLocalLint::test_lint_generic_exception`

### Line 359: `run_code()` non-Python skip
- `TestLocalRunCode::test_run_non_python_skipped`

### Line 361: `run_code()` static_failed skip
- `TestLocalRunCode::test_run_static_failed_skipped`

### Line 368: `run_code()` with bwrap enabled
- `TestLocalRunCode::test_run_bwrap_enabled`

### Lines 387-396: `run_code()` failure paths
- `TestLocalRunCode::test_run_failure_nonzero`
- `TestLocalRunCode::test_run_timeout`
- `TestLocalRunCode::test_run_generic_exception`

### Lines 399-400: `run_code()` finally block (kill proc)
- `TestLocalRunCode::test_run_kills_hung_proc`

### Lines 404-405: `run_code()` outer exception
- `TestLocalRunCode::test_run_outer_exception`

### Lines 412-461: `LocalEnvironment.run_tests()`
- `TestLocalRunTests::test_no_test_files_returns_immediately`
- `TestLocalRunTests::test_runs_pytest`
- `TestLocalRunTests::test_pytest_failure`
- `TestLocalRunTests::test_tests_timeout`
- `TestLocalRunTests::test_tests_generic_exception`
- `TestLocalRunTests::test_tests_kills_hung_proc`
- `TestLocalRunTests::test_tests_with_bwrap`
- `TestLocalRunTests::test_tests_outer_exception`
- `TestLocalRunTests::test_detects_test_in_filename_key`

### Lines 499-550: `DockerEnvironment.lint()`
- `TestDockerLint::test_lint_success`
- `TestDockerLint::test_lint_failure`
- `TestDockerLint::test_lint_failure_empty_output`
- `TestDockerLint::test_lint_timeout`
- `TestDockerLint::test_lint_kills_hung_proc`
- `TestDockerLint::test_lint_non_python_skipped`
- `TestDockerLint::test_lint_outer_exception`

### Lines 567-608: `DockerEnvironment.run_code()`
- `TestDockerRunCode::test_run_success`
- `TestDockerRunCode::test_run_failure`
- `TestDockerRunCode::test_run_non_python_skipped`
- `TestDockerRunCode::test_run_static_failed_skipped`
- `TestDockerRunCode::test_run_timeout`
- `TestDockerRunCode::test_run_generic_exception`
- `TestDockerRunCode::test_run_kills_hung_proc`
- `TestDockerRunCode::test_run_outer_exception`
- `TestDockerRunCode::test_run_output_truncated`
- `TestDockerRunCode::test_run_errors_truncated`
- `TestDockerRunCode::test_run_duration_recorded`

### Lines 612-613: `DockerEnvironment.run_code()` outer exception
- `TestDockerRunCode::test_run_outer_exception`

### Lines 620-668: `DockerEnvironment.run_tests()`
- `TestDockerRunTests::test_no_test_files_returns_immediately`
- `TestDockerRunTests::test_runs_pytest_in_docker`
- `TestDockerRunTests::test_pytest_failure`
- `TestDockerRunTests::test_tests_timeout`
- `TestDockerRunTests::test_tests_generic_exception`
- `TestDockerRunTests::test_tests_kills_hung_proc`
- `TestDockerRunTests::test_tests_outer_exception`
- `TestDockerRunTests::test_detects_test_in_filename_key`
- `TestDockerRunTests::test_mixed_files_only_tests_run`

## Additional Coverage

The test suite also covers:
- All dataclasses: `SandboxConfig`, `SandboxResult`, `CodeValidationResult`
- `ValidationLevel` enum
- Helper functions: `_safe_join`, `_build_isolated_env`, `_parse_pytest_counts`
- `Environment` ABC instantiation checks
- `LocalEnvironment._write_files()` with various file configurations
- `DockerEnvironment._write_files()` and `docker_run_flags()`
- Output truncation (500 chars for output, 1000 chars for errors)
- Process cleanup in finally blocks
- Bwrap command wrapping for both local and Docker environments

## Notes

1. **Unix-only tests**: 10 tests are skipped on Windows because they test Unix-specific functionality (resource limits, bubblewrap). These would pass on Linux/macOS.

2. **No real Docker calls**: All Docker interactions are mocked using `AsyncMock` and `MagicMock`.

3. **No network access**: All tests run offline with mocked subprocess calls.

4. **Coroutine warnings fixed**: All `proc.kill()` calls use `MagicMock()` instead of `AsyncMock()` to match the synchronous nature of `asyncio.subprocess.Process.kill()`.
