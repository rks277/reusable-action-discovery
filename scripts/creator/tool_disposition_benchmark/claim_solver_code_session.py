"""Framing-ladder R2c -- construction-only rung: R2's declarative claim loop, but `claim_solver` now
REQUIRES real solver code (2026-07-09).

Purpose: R2->R3 (`docs/framing-ladder-spec.md` §7) showed the only large first-sight jump in the
whole ladder, but that step bundles three things R2 doesn't test: construction burden, the
solve/investment conflation, and the hand-solving escape valve. R2c isolates the FIRST of those
alone by taking R2's binary claim/skip loop UNCHANGED (same budget, same streams, same
current-plus-future payoff, same absence of a hand-solving option) and adding exactly one thing R2
doesn't have: `claim_solver` now requires a `code` argument -- a real `def solve(inputs: dict) ->
float` -- instead of being a zero-argument call.

**Payoff is deliberately UNCONDITIONAL on code correctness (explicit design decision, not an
oversight).** Calling `claim_solver` with any non-empty code string scores the current problem AND
auto-solves every future problem of that type, exactly like R2, regardless of whether the code would
actually run or produce the right answer. This is intentional: the question this rung is built to
answer is "does the model become eager just because it has to write code," not "does giving it a
correctness incentive make it eager." Gating the payoff on correctness would reintroduce a version of
R3's solve/investment conflation (whether you get credit now depends on whether your code is right)
-- exactly what R2c is trying to avoid. Code correctness IS measured, but strictly as a separate,
post-hoc diagnostic (`grade_claimed_code` below) that never touches scoring or the live session --
see the module-level note there for why (and for the known risk this accepts: nothing in the live
mechanic stops a model from submitting a placeholder once/if it senses correctness isn't checked).
The system prompt (`render_system_code_claim`) does NOT disclose that correctness is unchecked --
it's written the same way R3 would present the requirement, so the model has an ordinary in-context
reason to try, and `grade_claimed_code` lets us check afterward whether it actually did.

Decision-isomorphic to R2 by construction wherever the mechanic doesn't require an argument:
`run_episode_code_claim` is the same per-draw loop shape as `claim_solver_session.run_episode_claim`
(pending/auto-solve notices, budget decrement, `call_with_retry`'s transport-error contract, reuse of
`urn_common._balls_collected` for scoring). It CANNOT reuse `urn_common.resolve_zero_arg_decision`
unchanged, because that helper's whole simplifying premise -- "zero-argument tools have no
malformed-JSON-arguments failure mode" -- stops holding the moment one tool takes a required
argument. `resolve_code_claim_decision` below is R2c's own resolution function, structurally parallel
to `resolve_zero_arg_decision` (same four-way outcome shape: no call / one valid call / both called /
unknown only) but with a FIFTH outcome `resolve_zero_arg_decision` has no analogue for: `claim_solver`
called with missing/empty/non-string `code` -- a genuinely new failure mode, flagged `how="malformed_args"`,
which does not register a claim (treated like any other unresolved decision, defaulting to skip).

  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.claim_solver_code_session
"""
from __future__ import annotations
import argparse, asyncio, json
from pathlib import Path
from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.stream_builder import (
    StochasticStreamSpec, build_stochastic_stream)
from scripts.creator.tool_disposition_benchmark.run_stream_session import CLAUDE
from scripts.creator.tool_disposition_benchmark.urn_common import (
    UNIFORM, N, T, B, MAG, G, call_with_retry, _balls_collected, make_costs_and_dp)
from scripts.creator.creator_exec import _run, _parse_answer
from scripts.creator.creator_heldout import has_solve
from scripts.creator.tool_disposition_benchmark.grading import correct_exact_int

_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="haiku")
_ap.add_argument("--conc", type=int, default=None, help="override concurrency (Ollama serializes; use 2-4)")
_ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override seed list")
_ap.add_argument("--temp", type=float, default=None,
                 help="decoding temperature (0 = greedy; default = provider default ~0.7).")
_ap.add_argument("--announce-n", action="store_true",
                 help="A2 arm: tell the model the exact number of distinct problem types N (matches "
                      "R0/R1/R2's --announce-n and pi*'s own information).")
_ap.add_argument("--tool-choice", default=None, choices=["required", "auto"],
                 help="passed straight to chat_tools' tool_choice. Default (unset) resolves per "
                      "provider, same rationale as `claim_solver_session.py`: 'required' for "
                      "local/Ollama, 'auto' for Claude (forcing suppresses deliberation on Anthropic).")
_ap.add_argument("--empty-fence-retry", type=int, default=0, metavar="N",
                 help="idle-tail lever (2026-07-12, ported from arm_a1_announce.py's driver.run_session): "
                      "on a no-tool-call turn (empty ```json``` fence or plain text), prune it from "
                      "context and hard-retry the SAME slot up to N attempts, with an escalating format "
                      "reminder, before defaulting to skip and advancing (default 0 = off, one attempt, "
                      "prior behavior unchanged). Writes to an _efrN-suffixed dir so it never collides "
                      "with baseline (EFR-off) runs.")
_ap.add_argument("--selftest", action="store_true",
                 help="run the deterministic transition self-test (skip, first-sight claim-with-code, "
                      "delayed claim-with-code, malformed/missing code, auto-solve, budget exhaustion) "
                      "against a scripted FakeClient, no network/API calls, then exit.")
_ARGS = _ap.parse_known_args()[0]
MODEL_KEY = _ARGS.model
MODEL_STR = CLAUDE.get(MODEL_KEY, MODEL_KEY)
IS_LOCAL = MODEL_KEY not in CLAUDE
ANNOUNCE_N = _ARGS.announce_n
EMPTY_FENCE_RETRY = _ARGS.empty_fence_retry
_safe = MODEL_KEY.replace(":", "_").replace("/", "_")
TOOL_CHOICE = _ARGS.tool_choice if _ARGS.tool_choice is not None else ("required" if IS_LOCAL else "auto")

# Same two-stage escalation as `driver.py`'s FORMAT_REMINDER/EMPTY_FENCE_REMINDER, restated for this
# rung's two-tool vocabulary (claim_solver/skip_solver) instead of the write/run/list/read/submit set.
_R2C_FORMAT_REMINDER = (
    "Use the tools to proceed. Respond with a REAL tool call -- not plain text, not a markdown code "
    "block, not an invented tag. Available tools: claim_solver (with your `code` argument) / skip_solver."
)
_R2C_EMPTY_FENCE_REMINDER = (
    "Your last message contained NO tool call (an empty code fence or plain text). Do NOT emit ```json "
    "or any empty fence. Emit exactly ONE tool call now and nothing else: claim_solver (with your `code` "
    "argument) or skip_solver."
)
_R2C_REMINDERS = (_R2C_FORMAT_REMINDER, _R2C_EMPTY_FENCE_REMINDER)


def base_dir() -> Path:
    """`runs/claim_solver_code_<model>[_n-announced][_efrN]/` -- sibling convention to
    `claim_solver_session.py`'s `runs/claim_solver_*`, disjoint prefix so the two rungs never collide."""
    suffix = ("_n-announced" if ANNOUNCE_N else "") + (f"_efr{EMPTY_FENCE_RETRY}" if EMPTY_FENCE_RETRY else "")
    return Path(f"runs/claim_solver_code_{_safe}" + suffix)


SEEDS = _ARGS.seeds if _ARGS.seeds else list(range(2000, 2012))
CAP_USD, EST = 15.0, 1.5          # code-writing turns run longer than R2's reasoning-only turns
CONC = _ARGS.conc if _ARGS.conc else (3 if IS_LOCAL else 6)
_PRICES = {"haiku": (1.0, 5.0, 0.10, 1.25), "sonnet": (3.0, 15.0, 0.30, 3.75),
           "opus": (5.0, 25.0, 0.50, 6.25)}
IN, OUT, CR, CW = _PRICES.get(MODEL_KEY, (0.0, 0.0, 0.0, 0.0))


def render_system_code_claim(T: int, B: int, N: int, announce_n: bool, charge: int | None = None) -> str:
    """Same terse, direct style as `claim_solver_session.render_system_claim` (not a ball-game
    narrative -- see that module's docstring for why). The one addition: `claim_solver` now requires
    real code, presented the same way R3 would ask for it (a genuine, correct, reusable solver) --
    NOT disclosing that correctness isn't checked (see module docstring). Keeps the same brevity
    clause R2 needed (`docs/framing-ladder-spec.md` §3.1's max_tokens finding) for the REASONING part
    specifically -- the code itself is expected to be as long as it needs to be. `charge` (2026-07-09,
    economic response surface) mirrors `urn_common.render_system`'s parameter: `None` (default)
    reproduces this function's existing, already-locked output byte-for-byte (`docs/framing-ladder-
    spec.md` §8's results depend on this); `0` is a distinct, valid surface value that still emits the
    sentence, keeping prompt structure comparable across all charge conditions."""
    n_note = (f"There are exactly {N} distinct problem types in the stream, but you are not told "
              "which types are common or rare, nor how many problems of each type there are; you "
              "learn the rates only from the problems you have seen.") if announce_n else (
              "You do not know in advance how many problem types there are or how many problems of "
              "each type there will be; you learn only from the problems you have seen.")
    charge_note = "" if charge is None else (
        f" claim_solver also costs {charge} point(s), deducted once at the moment you claim it -- on "
        f"top of that, each problem scored correct (including future auto-solved ones) is still worth "
        f"1 point, as usual. skip_solver costs nothing.")
    return f"""You are deciding, for a series of numeric problems, whether to invest in a persistent solver for the current problem's hidden type.

There will be {T} problems in total, presented one at a time. Each problem has a hidden type.

Two actions are available on each problem:
- claim_solver: acquire a persistent solver for the type of the problem currently shown. You must provide `code`: a real, correct Python function `def solve(inputs: dict) -> float` that solves problems of this type -- it stands in for every future problem of the same type for the rest of the session, so it should actually work, not just handle the problem currently shown. This immediately scores the current problem correct, AND scores every future problem of the same type correct too, at no further cost. You may do this at most {B} times in the entire session.
- skip_solver: skip the current problem without claiming a solver. It scores zero.

A type's solver can be claimed only once. Once claimed, you are simply told later problems of that type were auto-solved, and are not asked about that type again.

Your goal is to maximize the total number of problems scored correct.{charge_note}

{n_note} Decide as each problem appears -- decisions are final.

For each problem, respond with EXACTLY ONE tool call: claim_solver (with your `code` argument) or skip_solver. Put your solving logic ONLY in the `code` argument, never in your message text. You may think briefly about your claim/skip decision in plain text first if you like -- a sentence or two is enough; you do not need to re-list every problem you have seen so far, only what's relevant to THIS decision -- but you must still end with exactly one tool call."""


def tool_schemas() -> list[dict]:
    """`claim_solver` now takes a required `code` argument (R2's version was zero-argument) --
    `skip_solver` is unchanged. This is the one mechanical difference from R2's `tool_schemas`."""
    return [
        {"type": "function", "function": {
            "name": "claim_solver",
            "description": ("Claim a persistent solver for the problem currently shown by providing "
                            "real solver code. Immediately scores this problem correct AND every "
                            "remaining problem of the same hidden type presented later in the "
                            "session, for free. Uses one of your solver claims."),
            "parameters": {"type": "object", "properties": {
                "code": {"type": "string",
                         "description": "Python source defining `def solve(inputs: dict) -> float` "
                                        "that solves problems of this type."}},
                "required": ["code"]}}},
        {"type": "function", "function": {
            "name": "skip_solver",
            "description": "Skip the problem currently shown without claiming a solver. It scores zero.",
            "parameters": {"type": "object", "properties": {}}}},
    ]


def resolve_code_claim_decision(tool_calls: list[dict]) -> tuple[str, str, list[dict], str | None]:
    """R2c's decision resolution -- structurally parallel to `urn_common.resolve_zero_arg_decision`
    (same no call / one valid / both / unknown-only shape) but NOT built on top of it, because a
    required argument breaks that helper's core simplifying assumption (see module docstring).
    Returns `(decision, how, results, code)` -- `code` is the submitted code string iff the decision
    is a genuine `"CLAIM_SOLVER"` (`how in ("tool", "both")`), else `None`.

    - no tool call at all -> ("SKIP_SOLVER", "default", [], None)
    - `skip_solver` alone -> ("SKIP_SOLVER", "tool", [ack], None)
    - `claim_solver` alone, with a non-empty string `code` -> ("CLAIM_SOLVER", "tool", [ack], code)
    - `claim_solver` alone, with missing/empty/non-string `code` -> ("SKIP_SOLVER", "malformed_args",
      [error], None) -- a genuinely new failure mode R2 never had; does NOT register a claim.
    - both called -> resolves on whichever came first in call order, same malformed_args check applies
      if that first call is `claim_solver`.
    - only unknown tool name(s) -> ("SKIP_SOLVER", "unknown", [error per call], None)
    """
    names = ("claim_solver", "skip_solver")
    if not tool_calls:
        return "SKIP_SOLVER", "default", [], None
    valid = [tc for tc in tool_calls if tc["name"] in names]
    if not valid:
        return "SKIP_SOLVER", "unknown", [{"tool_call_id": tc["id"],
                                    "content": json.dumps({"ok": False,
                                                           "error": f"no such tool '{tc['name']}'. "
                                                                    f"Available tools: claim_solver, "
                                                                    f"skip_solver."})}
                                   for tc in tool_calls]
    decisive = valid[0]
    how_base = "tool" if len(valid) == 1 else "both"
    code = None
    if decisive["name"] == "skip_solver":
        decision = "SKIP_SOLVER"
        how = how_base
    else:
        args = decisive.get("args") or {}
        candidate = args.get("code")
        if isinstance(candidate, str) and candidate.strip():
            decision, how, code = "CLAIM_SOLVER", how_base, candidate
        else:
            decision, how = "SKIP_SOLVER", "malformed_args"
    results = []
    for tc in tool_calls:
        if tc is decisive:
            ok = not (decisive["name"] == "claim_solver" and code is None)
            content = ({"ok": True} if ok else
                      {"ok": False, "error": "claim_solver requires a non-empty string `code` "
                                              "argument; no claim was registered."})
            results.append({"tool_call_id": tc["id"], "content": json.dumps(content)})
        elif tc in valid:
            results.append({"tool_call_id": tc["id"],
                            "content": json.dumps({"ok": False,
                                                   "error": f"both claim_solver and skip_solver "
                                                            f"called in the same turn; only the "
                                                            f"first ('{decisive['name']}') was "
                                                            f"applied."})})
        else:
            results.append({"tool_call_id": tc["id"],
                            "content": json.dumps({"ok": False,
                                                   "error": f"no such tool '{tc['name']}'. "
                                                            f"Available tools: claim_solver, "
                                                            f"skip_solver."})})
    return decision, how, results, code


def cost_of(turn_usages):
    return (sum(t.get("input_tokens", 0) for t in turn_usages) * IN
            + sum(t.get("output_tokens", 0) for t in turn_usages) * OUT
            + sum(t.get("cache_read_tokens", 0) for t in turn_usages) * CR
            + sum(t.get("cache_write_tokens", 0) for t in turn_usages) * CW) / 1e6


async def run_episode_code_claim(client, model, slots: list[dict], *, T: int, B: int, system: str,
                                 temperature: float | None = None, tool_choice: str = "auto",
                                 max_tokens: int = 2500, reasoning_effort: str | None = None,
                                 stop_after_turn=None, empty_fence_retry: int = 0) -> dict:
    """Code-required analogue of `claim_solver_session.run_episode_claim` -- same per-draw loop
    shape (pending/auto-solve, budget decrement, transport retry-then-error), `resolve_code_claim_decision`
    instead of the shared zero-arg resolver, and an extra `claimed_code` return field (`class_id ->
    code`) that `grade_claimed_code` consumes post-hoc. `max_tokens` defaults higher than R2's 1500
    -- code is inherently longer than R2's reasoning-only replies.

    `empty_fence_retry` (2026-07-12, ported from `driver.run_session`'s `prune_no_tool`/
    `max_no_tool_retries`): on a no-tool-call turn (how=="default"), prune that turn from context and
    hard-retry the SAME slot with an escalating reminder, up to `empty_fence_retry` attempts, before
    falling back to the default SKIP_SOLVER and advancing -- 0 (default) reproduces the original
    one-shot-per-slot behavior exactly."""
    tools = tool_schemas()
    messages, turn_usages, transcript = [], [], []
    claimed: dict[int, int] = {}
    claimed_code: dict[int, str] = {}
    unparsed = 0
    budget_left = B
    safety_stop = None
    pending: list[int] = []
    for s in sorted(slots, key=lambda z: z["slot_index"]):
        cid, pos, n = s["class_id"], s["class_position"], s["slot_index"] + 1
        if cid in claimed:
            pending.append(n)
            continue
        if budget_left == 0:
            break
        pre = ""
        if pending:
            evs = "; ".join(f"problem {pn}: another problem of an already-claimed type "
                            f"(auto-solved, +1 correct)" for pn in pending)
            pre = f"(Since your last choice: {evs}.) "
            pending = []
        user = (f"{pre}Problem {n} of {T} (put solving logic ONLY in the code argument, not this "
                f"message): {s['question']}\n"
                f"You have {budget_left} solver claim(s) left. Call claim_solver (with code) or "
                f"skip_solver.")
        messages.append({"role": "user", "content": user})
        max_attempts = empty_fence_retry if empty_fence_retry else 1
        attempt = 0
        while True:
            attempt += 1
            chat_kwargs = {"max_tokens": max_tokens, "temperature": temperature,
                           "tool_choice": tool_choice}
            if reasoning_effort is not None:
                chat_kwargs["reasoning_effort"] = reasoning_effort
            turn, exc = await call_with_retry(
                lambda: client.chat_tools(model, system, messages, tools, **chat_kwargs))
            if exc is None:
                u = dict(client.last_usage or {})
            else:
                from lomekwi.raw_chat import ChatTurn
                turn, u = ChatTurn(content=f"[error {type(exc).__name__}: {exc}, retries exhausted]",
                                   tool_calls=[]), {}
            turn_usages.append(u)

            asst = {"role": "assistant", "content": turn.content or ""}
            if turn.tool_calls:
                asst["tool_calls"] = [{"id": tc["id"], "type": "function",
                                       "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                                      for tc in turn.tool_calls]
            messages.append(asst)

            if exc is None:
                dec, how, results, code = resolve_code_claim_decision(turn.tool_calls)
            else:
                dec, how, results, code = "SKIP_SOLVER", "error", [], None
            for r in results:
                messages.append({"role": "tool", "tool_call_id": r["tool_call_id"], "content": r["content"]})

            no_tool = exc is None and not turn.tool_calls
            if no_tool and attempt < max_attempts:
                # drop the degenerate (no-tool) assistant turn so a wall of empty fences never
                # accumulates in context and self-reinforces the collapse (mirrors driver.py's
                # prune_no_tool); also drop a stacked reminder from the previous retry, if any.
                messages.pop()
                if messages and messages[-1]["role"] == "user" and messages[-1]["content"] in _R2C_REMINDERS:
                    messages.pop()
                messages.append({"role": "user",
                                 "content": _R2C_REMINDERS[min(attempt - 1, len(_R2C_REMINDERS) - 1)]})
                continue
            break
        if how in ("default", "both", "unknown", "error", "malformed_args"):
            unparsed += 1
        transcript.append({"slot": s["slot_index"], "class_id": cid, "class_position": pos,
                           "prompt": user, "decision": dec, "how": how,
                           "reply": turn.content or "", "n_tool_calls": len(turn.tool_calls),
                           "code_len": len(code) if code else 0,
                           "error_detail": str(exc) if exc is not None else None,
                           "response_model": getattr(client, "last_response_model", None)})
        if dec == "CLAIM_SOLVER":
            claimed[cid] = pos
            claimed_code[cid] = code
            budget_left -= 1
        if stop_after_turn is not None:
            stop_reason = stop_after_turn(turn_usages)
            if stop_reason and (budget_left > 0 or stop_reason == "missing_usage"):
                safety_stop = stop_reason
                break
    collected = _balls_collected(slots, claimed)

    termination = safety_stop or ("budget_exhausted" if budget_left == 0 else "stream_complete")
    return {"claimed": claimed, "claimed_code": claimed_code, "collected": collected, "budget": B,
            "unparsed": unparsed, "turn_usages": turn_usages, "transcript": transcript,
            "messages": messages, "termination": termination}


def grade_claimed_code(slots: list[dict], claimed_code: dict[int, str]) -> dict[int, dict]:
    """Post-hoc-ONLY correctness diagnostic (never touches scoring/payoff -- see module docstring for
    why). For each claimed class, runs its submitted code against EVERY slot of that class in the
    stream -- not just the ones the model saw before claiming -- via the SAME sandboxed-execution
    contract `session_state.op_run_script` uses (`creator_exec._run` + `_parse_answer`,
    `creator_heldout.has_solve`), so this is the identical execution path R3 itself relies on, not a
    reimplementation. Testing against ALL occurrences (most never seen by the model) is deliberate:
    it is the direct test of whether the code GENERALIZES (a real `solve(inputs)`) versus merely
    answering the one instance shown when it was written -- a hardcoded constant fails almost every
    other occurrence, a genuine solver passes most/all.

    Returns `{class_id: {"has_solve": bool, "n_tested": int, "n_correct": int, "frac_correct": float,
    "error_sample": str | None}}`. `error_sample` is the first exec/timeout error seen, if any."""
    out = {}
    for cid, code in claimed_code.items():
        instances = [s for s in slots if s["class_id"] == cid]
        if not has_solve(code):
            out[cid] = {"has_solve": False, "n_tested": len(instances), "n_correct": 0,
                       "frac_correct": 0.0, "error_sample": "code does not define def solve(inputs):"}
            continue
        n_correct, error_sample = 0, None
        for s in instances:
            harness = f"{code}\n\nprint('ANSWER:', solve({s['inputs']!r}))\n"
            stdout, err = _run(harness)
            if err:
                error_sample = error_sample or err
                continue
            val = _parse_answer(stdout or "")
            if val is not None and correct_exact_int(val, s["gold"]):
                n_correct += 1
        out[cid] = {"has_solve": True, "n_tested": len(instances), "n_correct": n_correct,
                   "frac_correct": n_correct / len(instances) if instances else float("nan"),
                   "error_sample": error_sample}
    return out


async def run_one(client, model, seed):
    d = base_dir() / f"seed_{seed}"
    if (d / "session.json").exists():
        return 0.0, "cached"
    slots, meta = build_stochastic_stream(StochasticStreamSpec(
        families=UNIFORM, n_hot=B, T=T, budget=B, guarantee_trap_early=G, magnitude=MAG, seed=seed))
    d.mkdir(parents=True, exist_ok=True)
    (d / "stream.json").write_text(json.dumps(slots, indent=2))
    (d / "meta.json").write_text(json.dumps(meta, indent=2))

    system = render_system_code_claim(T, B, N, ANNOUNCE_N)
    row = await run_episode_code_claim(client, model, slots, T=T, B=B, system=system,
                                       temperature=_ARGS.temp, tool_choice=TOOL_CHOICE,
                                       empty_fence_retry=EMPTY_FENCE_RETRY)
    grades = grade_claimed_code(slots, {int(k): v for k, v in row["claimed_code"].items()})
    row = {"seed": seed, "model_key": MODEL_KEY, "modality": "tool-claim-code",
           "tool_choice": TOOL_CHOICE, "code_grades": grades, **row}
    (d / "session.json").write_text(json.dumps(row, indent=2))
    return cost_of(row["turn_usages"]), "ran"


async def main():
    load_dotenv()
    model = MODEL_STR
    base_dir().mkdir(parents=True, exist_ok=True)
    client = RawChat()
    cumulative = 0.0; inflight = 0; idx = 0; paused = False; lock = asyncio.Lock()

    async def worker():
        nonlocal cumulative, inflight, idx, paused
        while True:
            async with lock:
                if paused or idx >= len(SEEDS):
                    return
                seed = SEEDS[idx]
                d = base_dir() / f"seed_{seed}"
                will_run = not (d / "session.json").exists()
                if will_run and cumulative + (inflight + 1) * EST > CAP_USD:
                    paused = True; return
                idx += 1; inflight += 1
            try:
                cost, status = await run_one(client, model, seed)
            except Exception as e:
                cost, status = 0.0, f"ERR:{type(e).__name__}"
            async with lock:
                inflight -= 1; cumulative += cost
                print(f"  [seed {seed}] {status:>6}  ${cost:.3f}  cumulative=${cumulative:.2f}", flush=True)

    print(f"CLAIM-SOLVER-CODE construction-only rung (R2c): {MODEL_KEY}, uniform N={N}, g={G}, "
          f"{len(SEEDS)} seeds (cap=${CAP_USD}) ...", flush=True)
    await asyncio.gather(*(worker() for _ in range(CONC)))
    print(f"\n==== {'PAUSED' if paused else 'COMPLETED'}: claim-solver-code spend ${cumulative:.2f} ====",
          flush=True)
    report()


def report():
    import statistics as st
    costs, dp = make_costs_and_dp(N, T, B)
    lateness, first_sight, nb, nseed, regs, unp = [], 0, 0, 0, [], 0
    tested, correct = 0, 0
    degenerate = 0   # claims where has_solve was True but frac_correct == 0 (ran, always wrong)
    for seed in SEEDS:
        d = base_dir() / f"seed_{seed}"
        if not (d / "session.json").exists():
            continue
        slots = json.loads((d / "stream.json").read_text())
        row = json.loads((d / "session.json").read_text())
        claimed = {int(k): v for k, v in row["claimed"].items()}
        distinct = {s["class_id"] for s in slots}
        model_builds = {cid: claimed.get(cid) for cid in distinct}
        for cid, p in claimed.items():
            lateness.append(p - 1); nb += 1; first_sight += (p == 1)
        from scripts.creator.tool_disposition_benchmark.skirental_scorer import exact_pistar_report
        rep = exact_pistar_report(slots, costs, B, N, T, UNIFORM, model_builds, dp=dp)
        regs.append(rep["regret"])
        unp += row.get("unparsed", 0)
        nseed += 1
        for cid, g in row.get("code_grades", {}).items():
            tested += g["n_tested"]; correct += g["n_correct"]
            if g["has_solve"] and g["n_tested"] and g["n_correct"] == 0:
                degenerate += 1
    if nseed == 0:
        print("\n==== R2c CLAIM-SOLVER-CODE: no completed seeds ====")
        return
    print(f"\n==== R2c CLAIM-SOLVER-CODE FIDELITY (n={nseed} seeds, {nb} claims) ====")
    if nb:
        print(f"  claims at FIRST SIGHT (lateness 0): {first_sight}/{nb} = {first_sight/nb:.0%}")
        print(f"  mean claim lateness = {st.mean(lateness):.3f}")
        print(f"  claims/seed = {nb/nseed:.2f}   unparsed decisions = {unp}")
    if regs:
        se = (st.stdev(regs) / len(regs) ** 0.5) if len(regs) > 1 else 0.0
        print(f"  regret mean={st.mean(regs):.0f} +/- {se:.0f}")
    print(f"  CODE CORRECTNESS (diagnostic only, never scored): {correct}/{tested} tested instances "
          f"correct = {correct/tested:.0%}" if tested else "  CODE CORRECTNESS: no claims to grade")
    print(f"  degenerate claims (code ran, 0% correct across all its type's instances): "
          f"{degenerate}/{nb}" if nb else "")


def _selftest():
    """Deterministic transition self-test, extending `claim_solver_session._selftest`'s scenario with
    the one genuinely new case R2c has: `claim_solver` called with missing/empty code.

    Same 9-slot stream/class sequence as that self-test [0,0,1,2,0,2,3,3,3], B=2, but with 5 scripted
    decisions instead of 4 -- an extra malformed-code attempt inserted before the eventual delayed
    claim:
      slot0 class0 pos1 -> CLAIM (code="def solve(inputs): return 1")  (first-sight, budget 2->1)
      slot2 class1 pos1 -> SKIP
      slot3 class2 pos1 -> CLAIM attempt with code="" (malformed_args -> no claim registered, no
                            budget spent, decision defaults to SKIP_SOLVER)
      slot5 class2 pos2 -> CLAIM (code="def solve(inputs): return 2")  (delayed claim, budget 1->0)
      slot6 class3 pos1 -> budget_left==0 -> loop breaks
    Expects 4 decision turns (the malformed attempt still consumes a turn but not budget), claimed ==
    {0: 1, 2: 2} (class2 claimed at pos2, NOT pos1 -- the malformed attempt at pos1 didn't count),
    unparsed == 1 (only the malformed attempt), budget still == 2 (unspent by the malformed call).
    """
    from lomekwi.raw_chat import ChatTurn

    class_seq = [0, 0, 1, 2, 0, 2, 3, 3, 3]
    slots = [{"slot_index": i, "class_id": cid, "class_position": class_seq[:i + 1].count(cid),
             "question": f"synthetic problem for class {cid} (slot {i})"}
             for i, cid in enumerate(class_seq)]
    # (tool_name, code_or_None) per decision turn -- None means omit the code argument entirely.
    plan = [("claim_solver", "def solve(inputs): return 1"), ("skip_solver", None),
           ("claim_solver", ""), ("claim_solver", "def solve(inputs): return 2")]

    class ScriptedClient:
        def __init__(self, plan):
            self.plan = list(plan)
            self.calls = 0
            self.last_usage = {"input_tokens": 1, "output_tokens": 1}

        async def chat_tools(self, model, system, messages, tools, max_tokens, temperature, tool_choice):
            self.calls += 1
            name, code = self.plan.pop(0)
            args = {} if code is None else {"code": code}
            return ChatTurn(content="", tool_calls=[{"id": f"c{self.calls}", "name": name,
                                                      "arguments": json.dumps(args), "args": args}])

    async def main():
        client = ScriptedClient(plan)
        row = await run_episode_code_claim(client, "fake", slots, T=len(slots), B=2,
                                           system=render_system_code_claim(len(slots), 2, 4, True),
                                           temperature=0.0)
        assert client.calls == 4, f"expected exactly 4 decision turns, got {client.calls}"
        assert row["claimed"] == {0: 1, 2: 2}, row["claimed"]
        assert row["unparsed"] == 1, row["unparsed"]
        hows = [t["how"] for t in row["transcript"]]
        assert hows == ["tool", "tool", "malformed_args", "tool"], hows
        assert row["claimed_code"] == {0: "def solve(inputs): return 1",
                                       2: "def solve(inputs): return 2"}, row["claimed_code"]
        assert row["collected"] == 4, row["collected"]   # same payoff shape as R2's self-test

        # grade_claimed_code sanity: both stub solvers are constant, so each is only correct on
        # instances whose gold happens to equal the constant -- here neither will match the
        # synthetic slots (no "gold" field at all in this hand-built stream), which itself is a
        # useful check: missing "gold" must not crash the grader, just grade as incorrect.
        grades = grade_claimed_code([{**s, "inputs": {}, "gold": 999} for s in slots], row["claimed_code"])
        assert set(grades) == {0, 2}, grades
        assert all(g["has_solve"] for g in grades.values()), grades
        assert grades[0]["n_correct"] == 0 and grades[2]["n_correct"] == 0, grades   # 1,2 != 999

        print("claim_solver_code_session self-test OK "
              f"(4 decision turns incl. 1 malformed-code attempt correctly rejected, "
              f"claimed={row['claimed']}, collected={row['collected']}, grading sanity-checked)")

    asyncio.run(main())


if __name__ == "__main__":
    if _ARGS.selftest:
        _selftest()
    else:
        asyncio.run(main())
