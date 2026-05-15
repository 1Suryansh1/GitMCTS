"""Test Oracle - Run test suite as objective judge for branch scoring."""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result of running test suite in a worktree."""
    passed: int
    failed: int
    errors: int
    score: float  # passed / total
    summary: str


def run_tests(worktree_path: Path, test_cmd: str = "pytest", timeout: int = 120) -> TestResult:
    """
    Run the test suite in the worktree. This IS the value function.
    No LLM judge. No subjective evaluation. Tests pass or they don't.
    """
    report_file = ".mcts_test_report.json"

    cmd_parts = test_cmd.split()
    if "pytest" in test_cmd:
        cmd_parts.extend([
            "--json-report",
            "--json-report-file=" + report_file,
            "-x", "-q"
        ])

    try:
        result = subprocess.run(
            cmd_parts,
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        report_path = worktree_path / report_file
        if report_path.exists():
            report = json.loads(report_path.read_text())
            summary = report.get("summary", {})
            passed = summary.get("passed", 0)
            failed = summary.get("failed", 0)
            errors = summary.get("error", 0)
            total = passed + failed + errors
            score = passed / total if total > 0 else 0.0
            return TestResult(
                passed=passed,
                failed=failed,
                errors=errors,
                score=score,
                summary=f"{passed}/{total} tests passing"
            )
    except subprocess.TimeoutExpired:
        return TestResult(
            passed=0,
            failed=0,
            errors=0,
            score=0.0,
            summary="Test run timed out"
        )
    except Exception as e:
        return TestResult(
            passed=0,
            failed=0,
            errors=0,
            score=0.0,
            summary=f"Error running tests: {str(e)}"
        )

    stdout = result.stdout
    passed = 0
    failed = 0

    for line in stdout.split("\n"):
        if "passed" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part.replace(".", "").isdigit() and i + 1 < len(parts) and "passed" in parts[i + 1]:
                    try:
                        passed = int(part.replace(".", ""))
                    except ValueError:
                        pass

    return TestResult(
        passed=passed,
        failed=failed,
        errors=0,
        score=1.0 if passed > 0 else 0.0,
        summary=stdout[:200]
    )


def get_test_count(worktree_path: Path, test_cmd: str = "pytest") -> int:
    """Get total number of tests in the suite without running them."""
    try:
        result = subprocess.run(
            test_cmd.split() + ["--collect-only", "-q"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        for line in result.stdout.split("\n"):
            if "test" in line.lower() and any(c.isdigit() for c in line):
                parts = line.split()
                for part in parts:
                    if part.isdigit():
                        return int(part)
    except Exception:
        pass
    return 0