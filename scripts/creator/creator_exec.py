"""Code extraction, tool-build detection, and sandboxed execution for the CREATOR
fork (see docs/CREATOR-fork-plan.md).

The repo has no code executor; this is the missing piece. Model output is run in a
short-lived subprocess with CPU/memory/time limits, and the printed numeric answer
is compared to the dataset's ground-truth `answer`.

- extract_code(text)         -> the python from ```python blocks (or the whole text)
- built(code)                -> True iff a function is defined AND called (a reusable
                                tool, vs. inline-only arithmetic) -- ToolWorld's built()
- execute_and_check(code, g) -> (exec_answer, correct, error)
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile

# Language-tagged python blocks, and bare ``` blocks, as separate patterns: dataset
# `solution`s append a bare-fenced "Output:" block we must NOT execute, so prefer
# python-tagged blocks and only fall back to bare fences when none exist.
_PY_FENCE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_BARE_FENCE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
# A signed number, optionally $-prefixed and with thousands separators / decimals.
_NUM = re.compile(r"-?\$?\s*\d[\d,]*(?:\.\d+)?")
# An explicit final-answer line the prompt asks the model to emit.
_ANSWER_LINE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)


def extract_code(text: str) -> str:
    """Concatenate ```python blocks; fall back to bare ``` blocks, then whole text.
    Python-tagged blocks win so a dataset `solution`'s bare-fenced "Output:" block
    (and model prose) is not mistaken for executable code."""
    blocks = _PY_FENCE.findall(text or "")
    if not blocks:
        blocks = _BARE_FENCE.findall(text or "")
    if blocks:
        return "\n\n".join(b.strip() for b in blocks)
    return (text or "").strip()


def built(code: str) -> bool:
    """True iff the code defines a function and calls it elsewhere -- the CREATOR
    analogue of holding-and-fusing in ToolWorld. Inline-only code (no def, or a def
    that is never invoked) is recognition-negative. Syntax errors -> False."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return False
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if not defined:
        return False
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in defined:
            return True
    return False


def _to_float(token: str) -> float | None:
    token = token.replace("$", "").replace(",", "").strip()
    try:
        return float(token)
    except ValueError:
        return None


def _parse_answer(stdout: str) -> float | None:
    """Prefer the last `ANSWER:` line; else the last number anywhere in stdout."""
    ans_lines = _ANSWER_LINE.findall(stdout or "")
    for raw in reversed(ans_lines):
        nums = _NUM.findall(raw)
        if nums:
            v = _to_float(nums[-1])
            if v is not None:
                return v
    nums = _NUM.findall(stdout or "")
    return _to_float(nums[-1]) if nums else None


def _limit_resources():  # pragma: no cover - runs only in the child process
    """Best-effort CPU + address-space caps in the subprocess (preexec_fn)."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (5, 6))
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))  # 1 GiB
    except Exception:
        pass


def _run(code: str, timeout: float = 5.0) -> tuple[str | None, str | None]:
    """Run `code` in a sandboxed subprocess (isolated, CPU/mem-limited, temp cwd).
    Returns (stdout, None) on a clean exit, or (None, error) on timeout / spawn
    failure / non-zero exit. Shared by execute_and_check and the held-out scorer."""
    if not (code or "").strip():
        return None, "empty code"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True, text=True, timeout=timeout,
                cwd=tmp, preexec_fn=_limit_resources,
            )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as e:  # spawn failure
        return None, f"{type(e).__name__}: {e}"
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        return None, f"exit {proc.returncode}: {err[-1] if err else ''}"
    return proc.stdout, None


def correct_within_tol(ans: float | None, gold) -> bool:
    """Relative+absolute tolerance: |a - gold| <= 1e-2 * max(1, |gold|)."""
    if ans is None:
        return False
    try:
        return abs(ans - float(gold)) <= 1e-2 * max(1.0, abs(float(gold)))
    except (TypeError, ValueError):
        return False


def execute_and_check(code: str, gold: float, timeout: float = 5.0
                      ) -> tuple[float | None, bool, str | None]:
    """Run `code` in a sandboxed subprocess; compare its printed answer to `gold`.
    Returns (exec_answer, correct, error)."""
    stdout, err = _run(code, timeout)
    if err is not None:
        return None, False, err
    ans = _parse_answer(stdout)
    if ans is None:
        return None, False, "no numeric answer in stdout"
    return ans, correct_within_tol(ans, gold), None
