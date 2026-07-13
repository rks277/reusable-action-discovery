"""Shared core for the tool-disposition benchmark — ONE persistent session over N distinct
problems, answered one at a time. Scripts persist across the session and there is a GLOBAL
write budget of 0.1N. Stream-backed R3 sessions bind each script to its hidden problem class;
generic sessions without hidden labels retain the legacy global-script behavior. This is the
single source of truth for the tool logic.

Tools exposed to the model:
  write_script(name, code)  -- save a reusable solve(inputs)->float; refused once 0.1N writes used
  run_script(name, inputs)  -- execute a saved script when its hidden-class binding permits
  list_scripts()            -- names of all scripts saved so far (persist across problems)
  read_script(name)         -- source of a saved script (recall prior work)
  submit_answer(value)      -- final answer for the CURRENT problem; advances to the next

Grading is exact-match at each problem's stated significant figures (grading.correct_to_sigfigs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scripts.creator.creator_exec import _parse_answer, _run
from scripts.creator.creator_heldout import has_solve
from scripts.creator.tool_disposition_benchmark.grading import (
    correct_to_sigfigs, correct_exact_int, as_exact_int)

TOOL_NAMES = ("write_script", "run_script", "list_scripts", "read_script", "submit_answer")
CLASS_BOUND_VERSION = "class-bound-r3-v1"
GLOBAL_SCRIPT_VERSION = "global-scripts-v1"


def write_budget(n: int) -> int:
    """Global cap on scripts written for the whole session = 0.1N (at least 1)."""
    return max(1, round(0.1 * n))


@dataclass
class SessionState:
    problems: list[dict]                       # each: idx, keys, question, inputs, gold, sig_figs
    budget: int                                # global write budget (0.1N) — ENFORCED (refusal cap)
    announce_budget: int | None = None         # budget reported to the model via writes_remaining;
    #   defaults to `budget`. Set below `budget` to announce a tighter cap than is enforced (the
    #   awareness-vs-enforcement decoupling): the counter hits 0 at `announce_budget` but writes are
    #   only refused at `budget`. None → identical to the old behavior.
    announce_recurrence: bool = False          # awareness arm: disclose the recurring-type structure
    announce_n_types: int | None = None        # A2 arm: also disclose the exact number of distinct
    #   types -- pi*'s Dirichlet-multinomial predictive is CONSTRUCTED WITH exact N, so without this
    #   the "same-information" regret comparison isn't actually same-information (2026-07-03 audit).
    class_bound: bool | None = None             # infer from server-only _class_id labels by default

    scripts: dict[str, str] = field(default_factory=dict)
    script_class: dict[str, object] = field(default_factory=dict)
    script_artifact: dict[str, str] = field(default_factory=dict)
    script_writes: list[dict] = field(default_factory=list)
    cur: int = 0                               # index of the problem currently being solved
    n_write_calls: int = 0
    n_write_attempts: int = 0
    n_run_calls: int = 0
    n_list_calls: int = 0
    n_read_calls: int = 0
    n_refused: int = 0
    n_cross_class_run_refused: int = 0
    # per-problem record (lazily created), and global script->{problem idx} attribution
    records: list[dict] = field(default_factory=list)
    attribution: dict[str, set] = field(default_factory=dict)

    def __post_init__(self):
        has_labels = bool(self.problems) and all("_class_id" in p for p in self.problems)
        if self.class_bound is None:
            self.class_bound = has_labels
        if self.class_bound and not has_labels:
            raise ValueError("class-bound sessions require a server-only _class_id on every problem")
        if not self.records:
            self.records = [{"idx": p["idx"], "item_idx": p.get("item_idx"),
                             "used_script": False, "scripts_run": [], "scripts_authored": [],
                             "artifacts_run": [], "artifacts_authored": [],
                             "n_run_calls": 0, "runs": [], "submitted": False, "answer": None,
                             "correct": False}
                            for p in self.problems]

    @property
    def n(self) -> int:
        return len(self.problems)

    @property
    def announced(self) -> int:
        """Budget reported to the model (writes_remaining). Defaults to the enforced budget."""
        return self.budget if self.announce_budget is None else self.announce_budget

    @property
    def writes_remaining(self) -> int:
        return max(0, self.announced - self.n_write_calls)

    @property
    def done(self) -> bool:
        return self.cur >= self.n

    def current(self) -> dict | None:
        return self.problems[self.cur] if not self.done else None

    # ------------------------------------------------------------------ tools
    def op_write_script(self, name: str, code: str) -> dict:
        self.n_write_attempts += 1
        if self.n_write_calls >= self.budget:
            self.n_refused += 1
            return {"ok": False, "error": f"write budget exhausted ({self.budget} scripts for the "
                    "whole session). write_script is disabled; reuse a saved script with run_script "
                    "(see list_scripts) or compute by hand."}
        if not isinstance(name, str) or not name:
            return {"ok": False, "error": "name must be a non-empty string"}
        if not isinstance(code, str) or not code.strip():
            return {"ok": False, "error": "code must be a non-empty string"}
        existed = name in self.scripts
        self.scripts[name] = code
        # Every successful authoring event consumes budget, including replacing an existing name.
        # Otherwise a model can rewrite one generic filename indefinitely and bypass the scarce-build
        # treatment entirely (observed in the GPT-5.6 Sol R3 calibration, 2026-07-11).
        self.n_write_calls += 1
        artifact = f"write_{self.n_write_calls}:{name}" if self.class_bound else name
        self.script_artifact[name] = artifact
        bound_class = self.current().get("_class_id") if self.class_bound and not self.done else None
        if self.class_bound:
            self.script_class[name] = bound_class
        self.script_writes.append({
            "artifact": artifact,
            "name": name,
            "class_id": bound_class,
            "problem_idx": self.current().get("idx") if not self.done else None,
            "replaced": existed,
        })
        if not self.done:
            self.records[self.cur].setdefault("scripts_authored", []).append(name)
            self.records[self.cur].setdefault("artifacts_authored", []).append(artifact)
        warn = "" if has_solve(code) else (
            " WARNING: this code does not define `def solve(inputs):` — run_script will fail "
            "until it does.")
        return {"ok": True,
                "message": f"{'Replaced' if existed else 'Saved'} script '{name}' "
                           f"({len(code)} chars); this used one write.{warn}",
                "scripts": sorted(self.scripts),
                "writes_remaining": self.writes_remaining}

    def op_run_script(self, name: str, inputs: dict | None = None) -> dict:
        if name not in self.scripts:
            return {"ok": False, "error": f"no script named '{name}'. Available: "
                    f"{sorted(self.scripts) or '(none)'}"}
        if inputs is None:
            inputs = {}
        if not isinstance(inputs, dict):
            return {"ok": False, "error": "inputs must be a JSON object mapping your script's "
                    "input names to numbers"}
        self.n_run_calls += 1
        if self.class_bound:
            current_class = self.current().get("_class_id") if not self.done else None
            if self.script_class.get(name) != current_class:
                self.n_cross_class_run_refused += 1
                return {
                    "ok": False,
                    "error": (f"script '{name}' is bound to a different hidden problem type and "
                              "cannot be run on the current problem."),
                }
        code = self.scripts[name]
        if not has_solve(code):
            return {"ok": False, "error": "script does not define `def solve(inputs):`."}
        harness = f"{code}\n\nprint('ANSWER:', solve({inputs!r}))\n"
        stdout, err = _run(harness)
        if err:
            return {"ok": False, "error": _friendly_error(err), "stdout": (stdout or "")[:1000]}
        val = _parse_answer(stdout or "")
        # success: attribute this run to the current problem (drives persistence + used_script,
        # and the run's return value is kept so score() can attribute *benefit* to the script)
        if not self.done:
            rec = self.records[self.cur]
            rec["used_script"] = True
            rec["n_run_calls"] += 1
            if name not in rec["scripts_run"]:
                rec["scripts_run"].append(name)
            artifact = self.script_artifact.get(name, name)
            if artifact not in rec["artifacts_run"]:
                rec["artifacts_run"].append(artifact)
            rec["runs"].append({"script": name, "artifact": artifact, "ret": val})
            self.attribution.setdefault(artifact, set()).add(self.cur)
        return {"ok": True, "return_value": val, "stdout": (stdout or "").strip()[:1000]}

    def op_list_scripts(self) -> dict:
        self.n_list_calls += 1
        return {"ok": True, "scripts": sorted(self.scripts),
                "writes_remaining": self.writes_remaining}

    def op_read_script(self, name: str) -> dict:
        self.n_read_calls += 1
        if name not in self.scripts:
            return {"ok": False, "error": f"no script named '{name}'. Available: "
                    f"{sorted(self.scripts) or '(none)'}"}
        return {"ok": True, "name": name, "code": self.scripts[name]}

    def op_submit_answer(self, value) -> dict:
        if self.done:
            return {"ok": False, "error": "all problems already submitted; the session is over."}
        prob = self.problems[self.cur]
        rec = self.records[self.cur]
        rec["submitted"] = True
        if prob.get("exact_int"):                       # arbitrary-precision integer match
            ans = as_exact_int(value)
            rec["correct"] = correct_exact_int(ans, prob["gold"])
        else:
            ans = _as_float(value)
            rec["correct"] = correct_to_sigfigs(ans, prob["gold"], prob["sig_figs"])
        rec["answer"] = ans
        self.cur += 1
        remaining = self.n - self.cur
        msg = (f"Recorded answer for problem {prob['idx']}. "
               + (f"{remaining} problem(s) remaining." if remaining else "That was the last problem."))
        return {"ok": True, "message": msg, "problems_remaining": remaining}

    # ----------------------------------------------------------------- scoring
    def score(self) -> dict:
        recs = self.records
        n_correct = sum(1 for r in recs if r["correct"])
        used = [r for r in recs if r["used_script"]]
        not_used = [r for r in recs if not r["used_script"]]
        # persistence: distinct problems each written script was RUN on (breadth of application)
        persistence_counts = {name: len(idxs) for name, idxs in self.attribution.items()}
        # reusability: distinct problems each script was BENEFICIAL toward — i.e. the script's
        # output was correct AND was the submitted (correct) answer. Averaged over the same scripts
        # as persistence, so scripts that were run but never paid off count as 0.
        by_idx = {p["idx"]: p for p in self.problems}   # records carry the problem's own idx, which
        #   need not equal its position (e.g. a 1-problem isolation session keeps its original idx)
        benef_attr: dict[str, set] = {}
        for r in recs:
            if not r["correct"]:
                continue
            prob = by_idx[r["idx"]]
            for run in r.get("runs", []):
                ret = run["ret"]
                if ret is None:
                    continue
                if prob.get("exact_int"):
                    ok = correct_exact_int(r["answer"], ret) and correct_exact_int(ret, prob["gold"])
                else:
                    ok = (correct_to_sigfigs(r["answer"], ret, prob["sig_figs"])
                          and correct_to_sigfigs(ret, prob["gold"], prob["sig_figs"]))
                if ok:
                    benef_attr.setdefault(run.get("artifact", run["script"]), set()).add(r["idx"])
        reusability_counts = {name: len(benef_attr.get(name, set())) for name in self.attribution}
        return {
            "N": self.n,
            "budget": self.budget,
            "n_correct": n_correct,
            "solve_rate": round(n_correct / self.n, 4) if self.n else 0.0,
            "n_submitted": sum(1 for r in recs if r["submitted"]),
            "n_scripts_written": self.n_write_calls,
            "n_run_calls": self.n_run_calls,
            "n_problems_used_script": len(used),
            # efficiency: P(correct | used a script on that problem) vs P(correct | by hand)
            "eff_solve_given_script": round(sum(r["correct"] for r in used) / len(used), 4)
                if used else None,
            "eff_solve_by_hand": round(sum(r["correct"] for r in not_used) / len(not_used), 4)
                if not_used else None,
            # persistence = breadth of application (run on); reusability = breadth of *beneficial*
            # application (correct output submitted as the answer)
            "persistence_counts": persistence_counts,
            "persistence": round(sum(persistence_counts.values()) / len(persistence_counts), 3)
                if persistence_counts else None,
            "reusability_counts": reusability_counts,
            "reusability": round(sum(reusability_counts.values()) / len(reusability_counts), 3)
                if reusability_counts else None,
            "n_refused_tool_calls": self.n_refused,
            "n_cross_class_run_refused": self.n_cross_class_run_refused,
            "benchmark_version": CLASS_BOUND_VERSION if self.class_bound else GLOBAL_SCRIPT_VERSION,
            "class_bound": self.class_bound,
            "script_bindings": dict(self.script_class) if self.class_bound else {},
            "script_writes": list(self.script_writes),
            "records": recs,
        }

    # -------------------------------------------------------------- dispatch
    def call(self, name: str, args: dict) -> dict:
        args = args or {}
        if name == "write_script":
            return self.op_write_script(args.get("name"), args.get("code"))
        if name == "run_script":
            return self.op_run_script(args.get("name"), args.get("inputs"))
        if name == "list_scripts":
            return self.op_list_scripts()
        if name == "read_script":
            return self.op_read_script(args.get("name"))
        if name == "submit_answer":
            return self.op_submit_answer(args.get("value"))
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
        return f"{err} -- your code ran too long (5s CPU limit)."
    return err


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------ tool schemas
def TOOL_SCHEMAS(class_bound: bool = False) -> list[dict]:
    """OpenAI/Anthropic tool schemas, optionally enforcing the R3 hidden-type contract."""
    scope = (
        " Each script is bound to the hidden problem type on which it is written and may be run "
        "on the current problem or later problems only when they have that same hidden type. "
        "Problem types remain hidden and unlabeled."
    ) if class_bound else " Scripts persist across ALL problems and can be reused."
    contract = (
        "A script must define exactly this function:\n"
        "    def solve(inputs: dict) -> float:\n"
        "`inputs` maps names YOU choose to numbers; read them as inputs['NAME'] and RETURN a "
        f"single number.{scope} The sandbox has "
        "no stdin (never call input()), no files, no network; 5s CPU, 1GiB RAM.\n"
        "Example:\n"
        "    def solve(inputs: dict) -> float:\n"
        "        return inputs['A'] * inputs['B']"
    )
    run_scope = (
        "Execution is allowed only when the current problem has the same hidden type as the "
        "problem on which the script was written. "
    ) if class_bound else "Always available. "
    return [
        {"type": "function", "function": {
            "name": "write_script",
            "description": ("Save a named, reusable Python program. There is a GLOBAL budget on "
                            "how many scripts you may write for the whole session; once it is used "
                            "up this tool is disabled (run_script still works on saved scripts). "
                            "Every successful call consumes one write, including replacing an "
                            "existing name. " + contract),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "a short name for the script"},
                "code": {"type": "string", "description": "Python source defining solve(inputs)"},
            }, "required": ["name", "code"]}}},
        {"type": "function", "function": {
            "name": "run_script",
            "description": ("Execute a saved script by calling solve(inputs) in the sandbox; "
                            "returns its return value, stdout, and any error. "
                            + run_scope
                            + "You may call it several times per problem (e.g. chain primitives)."),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "name of a script you saved"},
                "inputs": {"type": "object", "description": "object mapping your script's input "
                           "names to numbers (omit for an empty dict)"},
            }, "required": ["name"]}}},
        {"type": "function", "function": {
            "name": "list_scripts",
            "description": "List the names of all scripts you have saved so far (they persist "
                           "across every problem in the session).",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "read_script",
            "description": "Return the source code of a saved script (to recall or adapt prior work).",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "name of a saved script"},
            }, "required": ["name"]}}},
        {"type": "function", "function": {
            "name": "submit_answer",
            "description": ("Submit your final numeric answer for the CURRENT problem, rounded to "
                            "the requested number of significant figures. This advances to the next "
                            "problem (you cannot return to a submitted problem)."),
            "parameters": {"type": "object", "properties": {
                "value": {"type": "number", "description": "the numeric answer for this problem"},
            }, "required": ["value"]}}},
    ]
