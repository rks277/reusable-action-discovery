"""Shared core for the MCP-based CREATOR eval — the single source of truth for the
four-tool logic, used by BOTH the real MCP server (server.py) and the in-process
backend (backends.py) so behavior and docstrings can never drift.

Protocol: the model gets ONE worked example, explores with write_script/run_script,
then calls begin_test to receive 20 held-out problems (same formula, new values) and
submits answers. Phase gating is symmetric:
  explore -> write_script OK, run_script OK, begin_test OK, submit_answers REFUSED
  test    -> write_script REFUSED, run_script OK, submit_answers OK

A "reference answer" is the dataset item's canonical solution re-executed on a row's
input values (precomputed in make_hard_batch as `golds`); we surface it as
`reference_answer(s)` for clarity. Test reference answers stay private (scoring only).

  build_episode(item, item_idx, seed, n_test=20) -> EpisodeState
  EpisodeState.op_write_script / op_run_script / op_begin_test / op_submit_answers
  EpisodeState.score()
  TOOL_SCHEMAS(arg_names) -> [openai/mcp tool schema, ...]  (4 model-facing tools)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scripts.creator.creator_eval_hard import make_hard_batch
from scripts.creator.creator_exec import _parse_answer, _run, correct_within_tol
from scripts.creator.creator_heldout import has_solve

TOOL_NAMES = ("write_script", "run_script", "begin_test", "submit_answers")


# ---------------------------------------------------------------- construction
def build_episode(item: dict, item_idx: int, seed: int, n_test: int = 20):
    """Build an EpisodeState from a CC.jsonl item: one shown example + n_test hidden
    held-out problems (same template/formula, hard-resampled values). Returns None if
    the item can't yield n_test+1 distinct finite rows."""
    batch = make_hard_batch(item, N=n_test + 1, seed=seed, withhold=False)
    if batch is None:
        return None
    # Present input dicts keyed by the UPPERCASE names the template shows, so the model
    # sees one consistent naming (template var SIDES <-> inputs["SIDES"]).
    keys = [n.upper() for n in batch["arg_names"]]

    def to_inputs(row: dict) -> dict:
        return {n.upper(): row[n] for n in batch["arg_names"]}

    rows, golds = batch["rows"], batch["golds"]
    return EpisodeState(
        item_idx=item_idx,
        keys=keys,
        template=batch["template"],
        example_inputs=to_inputs(rows[0]),
        example_reference=float(golds[0]),
        test_inputs=[to_inputs(r) for r in rows[1:n_test + 1]],
        _test_references=[float(g) for g in golds[1:n_test + 1]],
    )


@dataclass
class EpisodeState:
    item_idx: int
    keys: list[str]                       # input dict keys (UPPERCASE), == template var names
    template: str
    example_inputs: dict
    example_reference: float
    test_inputs: list[dict]
    _test_references: list[float]         # PRIVATE — scoring only, never returned to model

    scripts: dict[str, str] = field(default_factory=dict)
    test_phase: bool = False
    submitted: dict[int, float] = field(default_factory=dict)   # problem_no(1-based) -> answer
    # protocol counters/flags
    n_write_attempts: int = 0             # write_script calls INCLUDING ones refused in test phase
    n_write_calls: int = 0                # successful writes (explore phase)
    n_run_calls: int = 0
    n_run_calls_in_test: int = 0
    called_begin_test: bool = False
    n_refused: int = 0

    @property
    def n_test(self) -> int:
        return len(self.test_inputs)

    # ------------------------------------------------------------------ tools
    def op_write_script(self, name: str, code: str) -> dict:
        self.n_write_attempts += 1
        if self.test_phase:
            self.n_refused += 1
            return {"ok": False, "error": "write_script is DISABLED during the test phase. "
                    "Use run_script to run a script you already wrote, then submit_answers."}
        if not isinstance(name, str) or not name:
            return {"ok": False, "error": "name must be a non-empty string"}
        if not isinstance(code, str) or not code.strip():
            return {"ok": False, "error": "code must be a non-empty string"}
        self.n_write_calls += 1
        self.scripts[name] = code
        warn = "" if has_solve(code) else (
            " WARNING: this code does not define `def solve(inputs):` — run_script will fail "
            "until it does.")
        return {"ok": True, "message": f"Saved script '{name}' ({len(code)} chars).{warn}",
                "scripts": sorted(self.scripts)}

    def op_run_script(self, name: str, inputs: dict | None = None) -> dict:
        """Execute scripts[name] by calling solve(inputs) in the sandbox."""
        if name not in self.scripts:
            return {"ok": False, "error": f"no script named '{name}'. Available: "
                    f"{sorted(self.scripts) or '(none)'}"}
        if inputs is None:
            inputs = self.example_inputs
        if not isinstance(inputs, dict):
            return {"ok": False, "error": "inputs must be a JSON object mapping "
                    f"{self.keys} to numbers"}
        code = self.scripts[name]
        self.n_run_calls += 1
        if self.test_phase:
            self.n_run_calls_in_test += 1
        if not has_solve(code):
            return {"ok": False, "error": "script does not define `def solve(inputs):`. "
                    "Define exactly that function; read each value as inputs['NAME']."}
        harness = f"{code}\n\nprint('ANSWER:', solve({inputs!r}))\n"
        stdout, err = _run(harness)
        if err:
            return {"ok": False, "error": _friendly_error(err), "stdout": (stdout or "")[:1000]}
        val = _parse_answer(stdout or "")
        return {"ok": True, "return_value": val, "stdout": (stdout or "").strip()[:1000]}

    def op_begin_test(self) -> dict:
        if self.test_phase:
            return {"ok": False, "error": "test phase already started"}
        self.test_phase = True
        self.called_begin_test = True
        problems = [{"problem": i + 1, "inputs": inp}
                    for i, inp in enumerate(self.test_inputs)]
        return {
            "ok": True,
            "message": (f"Test phase started. write_script is now DISABLED. Solve all "
                        f"{self.n_test} problems below (same word problem, new inputs) and "
                        f"call submit_answers with a list of {self.n_test} numbers in order. "
                        f"You may use run_script on a script you already wrote, or compute "
                        f"answers yourself."),
            "template": self.template,
            "problems": problems,
        }

    def op_submit_answers(self, answers) -> dict:
        if not self.test_phase:
            self.n_refused += 1
            return {"ok": False, "error": "submit_answers is only available after begin_test."}
        parsed = _coerce_answers(answers, self.n_test)
        if parsed is None:
            return {"ok": False, "error": f"answers must be a list of {self.n_test} numbers "
                    f"(in problem order), or an object mapping problem number -> number."}
        self.submitted = parsed
        return {"ok": True, "message": f"Recorded {len(parsed)} answers. You may resubmit to "
                "overwrite. The episode ends when you stop or the token budget is exhausted."}

    # ----------------------------------------------------------------- scoring
    def score(self) -> dict:
        n_correct = sum(
            1 for i in range(1, self.n_test + 1)
            if correct_within_tol(self.submitted.get(i), self._test_references[i - 1]))
        per = [bool(correct_within_tol(self.submitted.get(i), self._test_references[i - 1]))
               for i in range(1, self.n_test + 1)]
        return {
            "item_idx": self.item_idx,
            "N": self.n_test,
            "n_correct": n_correct,
            "frac_correct": round(n_correct / self.n_test, 3),
            "all_correct": n_correct == self.n_test,
            "per_problem_correct": per,
            "wrote_script": self.n_write_calls > 0,
            "attempted_build": self.n_write_attempts > 0,   # incl. writes refused post-begin_test
            "n_write_attempts": self.n_write_attempts,
            "locked_out": self.n_write_calls == 0 and self.n_write_attempts > 0,  # tried to build too late
            "n_write_calls": self.n_write_calls,
            "n_run_calls": self.n_run_calls,
            "called_begin_test": self.called_begin_test,
            "submitted": bool(self.submitted),
            "never_submitted": not self.submitted,
            "used_run_in_test": self.n_run_calls_in_test > 0,
            "n_run_calls_in_test": self.n_run_calls_in_test,
            "n_refused_tool_calls": self.n_refused,
        }

    # -------------------------------------------------------------- dispatch
    def call(self, name: str, args: dict) -> dict:
        """Route a model tool call to the matching op_*; structured error on bad name/args."""
        args = args or {}
        if name == "write_script":
            return self.op_write_script(args.get("name"), args.get("code"))
        if name == "run_script":
            return self.op_run_script(args.get("name"), args.get("inputs"))
        if name == "begin_test":
            return self.op_begin_test()
        if name == "submit_answers":
            return self.op_submit_answers(args.get("answers"))
        return {"ok": False, "error": f"unknown tool '{name}'. Tools: {list(TOOL_NAMES)}"}


# ------------------------------------------------------------------ helpers
def _friendly_error(err: str) -> str:
    e = err.lower()
    if "eoferror" in e:
        return (f"{err} -- the sandbox has NO stdin; never call input(). Read every value "
                "from the inputs dict, e.g. inputs['X'].")
    if "nameerror" in e:
        return (f"{err} -- all inputs live in the inputs dict (e.g. inputs['X']); there are "
                "no bare top-level variables.")
    if "timeout" in e:
        return f"{err} -- your code ran too long (5s CPU limit); avoid loops over input()."
    return err


def _coerce_answers(answers, n: int) -> dict[int, float] | None:
    """Accept a length-n list (problem order) or a {problem_no: number} object."""
    out: dict[int, float] = {}
    if isinstance(answers, list):
        if len(answers) != n:
            return None
        for i, v in enumerate(answers, 1):
            f = _as_float(v)
            if f is not None:
                out[i] = f
        return out
    if isinstance(answers, dict):
        for k, v in answers.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                return None
            f = _as_float(v)
            if f is not None and 1 <= idx <= n:
                out[idx] = f
        return out
    return None


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------ tool schemas
def TOOL_SCHEMAS(keys: list[str]) -> list[dict]:
    """OpenAI/MCP tool schemas with the I/O contract + phase availability documented.
    `keys` are the exact input-dict keys for this episode (also the template var names)."""
    keylist = ", ".join(keys)
    contract = (
        "Scripts must define exactly this function:\n"
        "    def solve(inputs: dict) -> float:\n"
        f"`inputs` maps each input name to its numeric value; the keys are exactly: [{keylist}]. "
        "Read every value as inputs['NAME']. RETURN a single number (the answer). "
        "The sandbox has NO stdin (never call input()), no files, no network, and no bare "
        "top-level variables — only inputs[...] and your own locals exist. 5s CPU, 1GiB RAM.\n"
        "Example:\n"
        "    def solve(inputs: dict) -> float:\n"
        f"        return inputs['{keys[0]}'] * 2"
    )
    return [
        {"type": "function", "function": {
            "name": "write_script",
            "description": ("Save a named Python program. AVAILABLE ONLY BEFORE begin_test — "
                            "once you call begin_test this tool is permanently disabled for the "
                            "rest of the episode, so write (and test) any script first. "
                            + contract),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "a short name for the script"},
                "code": {"type": "string", "description": "Python source defining solve(inputs)"},
            }, "required": ["name", "code"]}}},
        {"type": "function", "function": {
            "name": "run_script",
            "description": ("Execute a saved script by calling solve(inputs) in the sandbox and "
                            "return its return value, stdout, and any error. Available in both "
                            "phases."),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "name of a script you wrote"},
                "inputs": {"type": "object", "description": (
                    f"object mapping {keylist} to numbers; omit to use the worked example's "
                    "inputs")},
            }, "required": ["name"]}}},
        {"type": "function", "function": {
            "name": "begin_test",
            "description": ("Start the test and receive the held-out test problems (same word "
                            "problem, new inputs). IMPORTANT: this PERMANENTLY disables "
                            "write_script for the rest of the episode (run_script stays "
                            "available) — if you intend to write any script, do so BEFORE calling "
                            "this."),
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "submit_answers",
            "description": ("Submit your final answers (TEST PHASE ONLY — refused before "
                            "begin_test). Provide a list of numbers in problem order (problem 1 "
                            "first), one per test problem. You may resubmit to overwrite."),
            "parameters": {"type": "object", "properties": {
                "answers": {"type": "array", "items": {"type": "number"},
                            "description": "20 numeric answers in problem order"},
            }, "required": ["answers"]}}},
    ]
