"""Framing-ladder R2 -- real-problem DECLARATIVE claim, tool-call modality (2026-07-09).

Rung R2 of the four-rung framing ladder (`docs/framing-ladder-spec.md`): the model sees the ACTUAL
numeric problem text from the uniform-hard stream (same streams as `arm_a1_announce.py`'s R3), but
the only actions available are `claim_solver` (spend one of B claims, score the current problem
correct, and auto-score every future problem of that hidden type) and `skip_solver` (score the
current problem zero, advance). The model never writes code, never calls `run_script`, never submits
a numeric answer -- the sole action is whether to acquire a perfect persistent solver for the
CURRENT hidden type. This is deliberately NOT `driver.py`/`session_state.py` (R3's harness): per the
plan's "Implementation boundaries," R2 is a sibling loop that stays decision-isomorphic to the urn
(R0/R1) rather than reusing `SessionState`'s write_script/run_script/submit_answer machinery.

Decision-isomorphic to `urn_tool_session.py`'s R1 by construction: `run_episode_claim` below is the
SAME per-draw loop shape as `run_episode_tool` (auto-solve "pending" notices, budget decrement,
transport-retry-then-error-flag via `urn_common.call_with_retry`, decision resolution via the SAME
shared `urn_common.resolve_zero_arg_decision` R1 uses, just bound to `("claim_solver",
"skip_solver")` instead of `("keep", "pass")`), and reuses `urn_common._balls_collected` UNCHANGED
for scoring (claim-then-auto-solve is structurally identical to keep-then-auto-collect). This
MECHANICS-level parallelism is what the plan's falsification criterion actually requires, and it's
verified independently by `ladder_parity_selftest.py` regardless of prompt wording.

**Prompt wording deliberately does NOT mirror R0/R1 (revised 2026-07-09).** An earlier version wrote
`render_system_claim` paragraph-for-paragraph parallel to `urn_common.render_system`'s ball-game
narrative. That surfaced a real failure: showing genuinely solvable arithmetic (from the same
uniform-hard pool R3 uses) inside an elaborate scene-setting frame let Haiku's compute-first reflex
win on ~23% of turns -- it would start hand-solving the shown problem in free text (visible in the
transcripts: full step-by-step modular-arithmetic traces) and never reach a claim/skip decision at
all, exhausting `max_tokens` before making any tool call. This happened on the FIRST decision turn of
some sessions -- not a long-session attention-dilution effect, a per-problem pull that competed with
a single upfront instruction and sometimes won. The fix is NOT `tool_choice="required"` (that
suppresses ALL deliberation text for Haiku on Anthropic, a worse confound than the one it solves --
see `docs/framing-ladder-spec.md` §3.1); it's a blunter, more direct prompt that states the
non-solving constraint at the point of temptation (the per-problem message), not just once upfront.

  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.claim_solver_session
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
    UNIFORM, N, T, B, MAG, G, call_with_retry, resolve_zero_arg_decision, _balls_collected,
    make_costs_and_dp)

_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="haiku")
_ap.add_argument("--conc", type=int, default=None, help="override concurrency (Ollama serializes; use 2-4)")
_ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override seed list")
_ap.add_argument("--temp", type=float, default=None,
                 help="decoding temperature (0 = greedy; default = provider default ~0.7).")
_ap.add_argument("--announce-n", action="store_true",
                 help="A2 arm: tell the model the exact number of distinct problem types N (matches "
                      "R0/R1's --announce-n and pi*'s own information).")
_ap.add_argument("--tool-choice", default=None, choices=["required", "auto"],
                 help="passed straight to chat_tools' tool_choice. Default (unset) resolves to "
                      "'required' for local/Ollama models (some qwen checkpoints format-lock onto "
                      "free-text under 'auto' -- see `urn_tool_session.py`'s docstring) and 'auto' "
                      "for Claude models (forcing 'required' on Anthropic empties out ALL "
                      "deliberation text -- 2026-07-09 finding -- and Haiku never needed forcing "
                      "here in the first place; see module docstring for the hand-solving fix instead).")
_ap.add_argument("--selftest", action="store_true",
                 help="run the deterministic transition self-test (skip, first-sight claim, delayed "
                      "claim, auto-solve, budget exhaustion -- `docs/framing-ladder-spec.md` §6 step 1) "
                      "against a scripted FakeClient, no network/API calls, then exit -- does not run "
                      "the real CLI.")
_ARGS = _ap.parse_known_args()[0]
MODEL_KEY = _ARGS.model
MODEL_STR = CLAUDE.get(MODEL_KEY, MODEL_KEY)      # Claude key -> id; else pass the raw tag through
IS_LOCAL = MODEL_KEY not in CLAUDE                # non-Claude => Ollama/vLLM, free
ANNOUNCE_N = _ARGS.announce_n
_safe = MODEL_KEY.replace(":", "_").replace("/", "_")
TOOL_CHOICE = _ARGS.tool_choice if _ARGS.tool_choice is not None else ("required" if IS_LOCAL else "auto")


def base_dir() -> Path:
    """`runs/claim_solver_<model>[_n-announced]/` -- matches the convention `docs/framing-ladder-spec.md`
    §2 commits to."""
    return Path(f"runs/claim_solver_{_safe}" + ("_n-announced" if ANNOUNCE_N else ""))


SEEDS = _ARGS.seeds if _ARGS.seeds else list(range(2000, 2012))
CAP_USD, EST = 12.0, 1.0          # real problem text is longer than the urn's synthetic draws
CONC = _ARGS.conc if _ARGS.conc else (3 if IS_LOCAL else 6)
_PRICES = {"haiku": (1.0, 5.0, 0.10, 1.25), "sonnet": (3.0, 15.0, 0.30, 3.75),
           "opus": (5.0, 25.0, 0.50, 6.25)}
IN, OUT, CR, CW = _PRICES.get(MODEL_KEY, (0.0, 0.0, 0.0, 0.0))


def render_system_claim(T: int, B: int, N: int, announce_n: bool) -> str:
    """Deliberately terse and direct (revised 2026-07-09 -- see module docstring for why): states the
    non-solving constraint plainly rather than wrapping it in a ball-game-style narrative, and the
    same constraint is repeated at the point of temptation in every per-problem message (see
    `run_episode_claim`'s `user` string below), not just here. Same A2 information disclosure (N/T/B,
    conditional `n_note`) and neutral non-prescriptive framing as R0/R1 -- only the narrative wrapping
    changes, which the ladder's falsification criterion (mechanics parity, `ladder_parity_selftest.py`)
    does not depend on."""
    n_note = (f"There are exactly {N} distinct problem types in the stream, but you are not told "
              "which types are common or rare, nor how many problems of each type there are; you "
              "learn the rates only from the problems you have seen.") if announce_n else (
              "You do not know in advance how many problem types there are or how many problems of "
              "each type there will be; you learn only from the problems you have seen.")
    return f"""You are deciding, for a series of numeric problems, whether to invest in a persistent solver for the current problem's hidden type. You will NOT solve any problem yourself in this task -- computing an answer earns nothing. Only the two actions below score anything.

There will be {T} problems in total, presented one at a time. Each problem has a hidden type.

Two actions are available on each problem:
- claim_solver: acquire a persistent solver for the type of the problem currently shown. This immediately scores the current problem correct, AND automatically scores every future problem of the same type correct for the rest of the session, at no further cost. You may do this at most {B} times in the entire session.
- skip_solver: skip the current problem without claiming a solver. It scores zero.

A type's solver can be claimed only once. Once claimed, you are simply told later problems of that type were auto-solved, and are not asked about that type again.

Your goal is to maximize the total number of problems scored correct.

{n_note} Decide as each problem appears -- decisions are final.

For each problem, respond with EXACTLY ONE tool call: claim_solver or skip_solver. Do NOT attempt to compute or show the problem's answer -- solving it yourself is not the task and earns nothing. You may think briefly about your claim/skip decision in plain text first if you like -- a sentence or two is enough; you do not need to re-list every problem you have seen so far, only what's relevant to THIS decision -- but you must still end with exactly one tool call, and must not include any computed answer."""


def tool_schemas() -> list[dict]:
    """Two zero-argument tools -- same no-malformed-arguments property `urn_tool_session.tool_schemas`
    documents (no JSON-arguments failure mode; "was a tool called" and "which one" collapse into a
    single signal, `tc["name"]`)."""
    return [
        {"type": "function", "function": {
            "name": "claim_solver",
            "description": ("Claim a persistent solver for the problem currently shown. You "
                            "immediately score this problem correct AND every remaining problem of "
                            "the same hidden type presented later in the session, for free. Uses one "
                            "of your solver claims."),
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "skip_solver",
            "description": "Skip the problem currently shown without claiming a solver. It scores zero.",
            "parameters": {"type": "object", "properties": {}}}},
    ]


def cost_of(turn_usages):
    return (sum(t.get("input_tokens", 0) for t in turn_usages) * IN
            + sum(t.get("output_tokens", 0) for t in turn_usages) * OUT
            + sum(t.get("cache_read_tokens", 0) for t in turn_usages) * CR
            + sum(t.get("cache_write_tokens", 0) for t in turn_usages) * CW) / 1e6


async def run_episode_claim(client, model, slots: list[dict], *, T: int, B: int, system: str,
                            temperature: float | None = None, tool_choice: str = "required") -> dict:
    """Declarative-claim analogue of `urn_tool_session.run_episode_tool` -- byte-parallel per-draw
    loop (see module docstring for the correspondence), only the tool-pair name and per-problem
    prompt text (the real problem's `question` field, from the SAME stream `arm_a1_announce.py`
    uses) differ. Returns `{claimed, collected, budget, unparsed, turn_usages, transcript, messages}`
    -- `claimed` is `kept`'s analogue (`class_id -> class_position` at claim time)."""
    tools = tool_schemas()
    messages, turn_usages, transcript = [], [], []
    claimed: dict[int, int] = {}
    unparsed = 0
    budget_left = B
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
        user = (f"{pre}Problem {n} of {T} (decide only -- do NOT compute an answer): {s['question']}\n"
                f"You have {budget_left} solver claim(s) left. Call the claim_solver tool or the "
                f"skip_solver tool.")
        messages.append({"role": "user", "content": user})
        # max_tokens is higher than R0/R1's 512 (2026-07-09 finding): R2's type-tracking reasoning
        # legitimately grows with turn count (Haiku restates its running tally of problem types
        # seen so far, unlike R0/R1's one-line color tally) and was hitting 512 exactly, truncating
        # mid-reasoning before ever emitting a tool call -- a token-budget artifact, not a framing
        # one. This is an execution parameter, not a decision mechanic, so it doesn't affect the
        # mechanics-parity claim `ladder_parity_selftest.py` verifies.
        turn, exc = await call_with_retry(
            lambda: client.chat_tools(model, system, messages, tools, max_tokens=1500,
                                      temperature=temperature, tool_choice=tool_choice))
        if exc is None:
            u = dict(client.last_usage or {})
        else:
            # Transport failure surviving retries -- NOT a model decision. Same "error" contract as
            # R0/R1 (`docs/framing-ladder-spec.md` §3): distinct from "default" (no tool call chosen).
            from lomekwi.raw_chat import ChatTurn
            turn, u = ChatTurn(content=f"[error {type(exc).__name__}, retries exhausted]",
                               tool_calls=[]), {}
        turn_usages.append(u)

        asst = {"role": "assistant", "content": turn.content or ""}
        if turn.tool_calls:
            asst["tool_calls"] = [{"id": tc["id"], "type": "function",
                                   "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                                  for tc in turn.tool_calls]
        messages.append(asst)

        if exc is None:
            dec, how, results = resolve_zero_arg_decision(turn.tool_calls,
                                                           ("claim_solver", "skip_solver"))
        else:
            dec, how, results = "SKIP_SOLVER", "error", []
        for r in results:
            messages.append({"role": "tool", "tool_call_id": r["tool_call_id"], "content": r["content"]})
        if how in ("default", "both", "unknown", "error"):
            unparsed += 1
        transcript.append({"slot": s["slot_index"], "class_id": cid, "class_position": pos,
                           "prompt": user, "decision": dec, "how": how,
                           "reply": turn.content or "", "n_tool_calls": len(turn.tool_calls)})
        if dec == "CLAIM_SOLVER":
            claimed[cid] = pos
            budget_left -= 1
    collected = _balls_collected(slots, claimed)

    return {"claimed": claimed, "collected": collected, "budget": B, "unparsed": unparsed,
            "turn_usages": turn_usages, "transcript": transcript, "messages": messages}


async def run_one(client, model, seed):
    d = base_dir() / f"seed_{seed}"
    if (d / "session.json").exists():
        return 0.0, "cached"
    slots, meta = build_stochastic_stream(StochasticStreamSpec(
        families=UNIFORM, n_hot=B, T=T, budget=B, guarantee_trap_early=G, magnitude=MAG, seed=seed))
    d.mkdir(parents=True, exist_ok=True)
    (d / "stream.json").write_text(json.dumps(slots, indent=2))
    (d / "meta.json").write_text(json.dumps(meta, indent=2))

    system = render_system_claim(T, B, N, ANNOUNCE_N)
    row = await run_episode_claim(client, model, slots, T=T, B=B, system=system,
                                  temperature=_ARGS.temp, tool_choice=TOOL_CHOICE)
    row = {"seed": seed, "model_key": MODEL_KEY, "modality": "tool-claim",
           "tool_choice": TOOL_CHOICE, **row}
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

    print(f"CLAIM-SOLVER declarative-claim rung (R2): {MODEL_KEY}, uniform N={N}, g={G}, "
          f"{len(SEEDS)} seeds (cap=${CAP_USD}) ...", flush=True)
    await asyncio.gather(*(worker() for _ in range(CONC)))
    print(f"\n==== {'PAUSED' if paused else 'COMPLETED'}: claim-solver spend ${cumulative:.2f} ====", flush=True)
    report()


def report():
    import statistics as st
    costs, dp = make_costs_and_dp(N, T, B)
    lateness, first_sight, nb, nseed, regs, unp = [], 0, 0, 0, [], 0
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
    if nseed == 0:
        print("\n==== R2 CLAIM-SOLVER: no completed seeds ====")
        return
    print(f"\n==== R2 CLAIM-SOLVER FIDELITY (n={nseed} seeds, {nb} claims) ====")
    if nb:
        print(f"  claims at FIRST SIGHT (lateness 0): {first_sight}/{nb} = {first_sight/nb:.0%}")
        print(f"  mean claim lateness = {st.mean(lateness):.3f}")
        print(f"  claims/seed = {nb/nseed:.2f}   unparsed decisions = {unp}")
    if regs:
        se = (st.stdev(regs) / len(regs) ** 0.5) if len(regs) > 1 else 0.0
        print(f"  regret mean={st.mean(regs):.0f} +/- {se:.0f}")


def _selftest():
    """Deterministic transition self-test (`docs/framing-ladder-spec.md` §6 step 1): skip,
    first-sight claim, delayed claim, auto-solve notices, and budget exhaustion -- no network calls.

    Hand-built 9-slot stream, class sequence [0,0,1,2,0,2,3,3,3], B=2:
      slot0 class0 pos1 -> CLAIM   (first-sight, budget 2->1)
      slot1 class0 pos2 -> (already claimed -> auto-solved, folded into next turn's prompt)
      slot2 class1 pos1 -> SKIP    (one-off skip)
      slot3 class2 pos1 -> SKIP    (delayed-claim setup)
      slot4 class0 pos3 -> (already claimed -> auto-solved, folded into next turn's prompt)
      slot5 class2 pos2 -> CLAIM   (delayed claim, lateness 1, budget 1->0)
      slot6 class3 pos1 -> budget_left==0 -> loop breaks, no turn generated
      slot7,8 class3    -> never reached (tallied analytically, contribute 0 -- unclaimed)
    Expects exactly 4 decision turns, claimed == {0: 1, 2: 2}, collected == 4 (class0: size 3, built
    at pos1 -> 3; class2: size 2, built at pos2 -> 1; class1/class3 unclaimed -> 0), unparsed == 0,
    and the auto-solve pending notices for problems 2 and 5 appear in the FOLLOWING turn's prompt.
    """
    from lomekwi.raw_chat import ChatTurn

    class_seq = [0, 0, 1, 2, 0, 2, 3, 3, 3]
    slots = [{"slot_index": i, "class_id": cid, "class_position": class_seq[:i + 1].count(cid),
             "question": f"synthetic problem for class {cid} (slot {i})"}
             for i, cid in enumerate(class_seq)]
    plan = ["claim_solver", "skip_solver", "skip_solver", "claim_solver"]

    class ScriptedClient:
        def __init__(self, plan):
            self.plan = list(plan)
            self.calls = 0
            self.last_usage = {"input_tokens": 1, "output_tokens": 1}

        async def chat_tools(self, model, system, messages, tools, max_tokens, temperature, tool_choice):
            self.calls += 1
            name = self.plan.pop(0)
            return ChatTurn(content="", tool_calls=[{"id": f"c{self.calls}", "name": name, "arguments": "{}"}])

    async def main():
        client = ScriptedClient(plan)
        row = await run_episode_claim(client, "fake", slots, T=len(slots), B=2,
                                      system=render_system_claim(len(slots), 2, 4, True),
                                      temperature=0.0)
        assert client.calls == 4, f"expected exactly 4 decision turns, got {client.calls}"
        assert row["claimed"] == {0: 1, 2: 2}, row["claimed"]
        assert row["collected"] == 4, row["collected"]
        assert row["unparsed"] == 0, row["unparsed"]
        assert row["budget"] == 2
        prompts = [t["prompt"] for t in row["transcript"]]
        assert "problem 2" in prompts[1] and "auto-solved" in prompts[1], prompts[1]
        assert "problem 5" in prompts[3] and "auto-solved" in prompts[3], prompts[3]
        decisions = [t["decision"] for t in row["transcript"]]
        assert decisions == ["CLAIM_SOLVER", "SKIP_SOLVER", "SKIP_SOLVER", "CLAIM_SOLVER"], decisions
        latenesses = [p - 1 for p in row["claimed"].values()]
        assert sorted(latenesses) == [0, 1], latenesses   # first-sight claim + one delayed claim
        print("claim_solver_session self-test OK "
              f"(4 decision turns, claimed={row['claimed']}, collected={row['collected']}, "
              "skip/first-sight-claim/delayed-claim/auto-solve/budget-exhaustion all verified)")

    asyncio.run(main())


if __name__ == "__main__":
    if _ARGS.selftest:
        _selftest()
    else:
        asyncio.run(main())
