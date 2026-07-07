"""Phase 3 demonstration generator (docs/qwen-finetune-transfer-plan.md): synthesizes SFT training
transcripts for the bridge fine-tune. Two slices, each available under two label sources:

  - urn: diverse-surface-form urn/balls sessions (vocabulary/N/T/B varied per session).
  - tool_bridge: short tool-game sessions on the REAL eval pool (uniform-hard N=8, MAG=100), fresh
    seeds disjoint from the 2000-2011 eval seeds.
  - policy="pistar" (treatment): labels = the exact same-info optimum (ExactDP.policy_builds).
  - policy="eager" (control): labels = build-the-first-B-distinct-types-on-sight (pi_star.eager_builds).

Plus one ARM-INDEPENDENT slice (option (c), 2026-07-05 — see plan "resume path"):

  - anchor (`anchor_tool.jsonl`): SINGLE-problem tool-calling sessions in the exact eval vocabulary
    (write_script/run_script/submit_answer). Their ONLY job is to preserve the base model's
    tool-calling modality, which the 97%-text urn corpus eroded (the FT model reverted to CoT hand-
    solving in the tool framing, blocking the transfer eval). A single isolated problem has NO
    recurrence, so there is no "when to build" decision to demonstrate -- the anchor is provably
    policy-neutral on the reserve axis (it cannot teach eager-vs-reserve). It is generated ONCE and
    shared byte-identically by both SFT arms, so it is a matched control that cannot manufacture a
    pistar-vs-eager difference. Build-heavy by default (ANCHOR_HAND_FRAC=0.0): that biases both arms
    toward building in the stream eval, which only makes a positive transfer result CONSERVATIVE.

Plus one more ARM-INDEPENDENT slice (mechanics bridge, 2026-07-06 — Design A fix):

  - mechanics_bridge (`mechanics_bridge.jsonl`): REALISTIC full-length tool sessions (real N=8, T=60
    stream, like the old policy-bearing tool_bridge) but with build TIMING decoupled from recurrence via
    `random_builds` -- neither always first-sighting (eager) nor always k>=2 (pistar). Needed because
    Design A (N_TOOL=0, no policy-bearing tool_bridge) removed the model's ONLY training exposure to a
    PERSISTENT 60-problem tool session; the single-problem anchor's very different framing ("PROBLEM 1
    of 1") could not substitute, and the FT model hallucinated the tool name as "submit_answers" (never
    registered) + stopped emitting `<tool_call>` wrappers under the real eval framing, confirmed NOT a
    base-model issue. mechanics_bridge re-teaches correct long-context tool syntax/naming/persistence
    while remaining policy-neutral: build timing is balanced between k=1 and k>=2 by construction (not
    left to chance), so it cannot teach a reserve-vs-eager correlation either way. Arm-independent, like
    the anchor.

A2 PROTOCOL (2026-07-03 audit): both slices DISCLOSE the exact number of distinct types N in the
system prompt (urn: "exactly N distinct <attr>s", where <attr> is the per-vocab attribute word --
color/material/metal/symbol/kind/suit/shape; tool: prompts.n_types_note) -- matching pi*'s own
information and the corrected --announce-n eval arm, so demonstrations are on the same footing as the
runs they'll be evaluated against. N/T/B ranges are chosen so pi* genuinely RESERVES (commits at
k>=2, not first-sight): short T / generous B collapses the exact DP toward eager, which would make
treatment demos indistinguishable from the eager control (see corpus-generation note below).

No LLM calls, no GPU -- everything is generated deterministically from exact_dp / family_kit /
stream_builder, so this step is free. Output: chat-format JSONL under runs/phase3_sft_data/.

CORPUS REGENERATION (2026-07-06 -- see docs/qwen-finetune-transfer-plan.md "corpus regeneration" for
the full diagnosis): anchor/mechanics_bridge/tool_bridge's wait-branch rationale text was a small set
of fixed literal strings (as low as 0.2% unique across 1500 turns), which collapsed the fine-tuned
model into verbatim repetition at eval; and wait/hand turns asserted a correct submit_answer with no
script behind it, contradicting this pool's a_hand=0.0 premise. Both fixed: rationale text dropped
entirely on wait/commit/reuse turns (the decision is already fully expressed by which tool gets
called), wait-turn submissions decoupled from gold (now gold+1, not gold), and mechanics_bridge's
k=1 commits de-clustered from the first few absolute slots. `_selftest_content_diversity` guards
against a repeat of the first bug class.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.phase3_demos          # generate
  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.phase3_demos --selftest
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from collections import Counter
from pathlib import Path

from scripts.creator.tool_disposition_benchmark.driver import FORMAT_REMINDER
from scripts.creator.tool_disposition_benchmark.exact_dp import ExactDP
from scripts.creator.tool_disposition_benchmark.family_kit import ALL_FAMILIES
from scripts.creator.tool_disposition_benchmark.pi_star import eager_builds
from scripts.creator.tool_disposition_benchmark.prompts import (
    RECURRENCE_NOTE, n_types_note, problem_prompt, system_prompt)
from scripts.creator.tool_disposition_benchmark.run_stream_session import slots_to_problems
from scripts.creator.tool_disposition_benchmark.session_state import TOOL_SCHEMAS
from scripts.creator.tool_disposition_benchmark.stream_builder import (
    StochasticStreamSpec, build_stochastic_stream)

UNIFORM = ["lcg", "modpow", "continued_frac", "crt_solve", "josephus", "quadratic_map_mod",
           "xorshift_steps", "matrix_power_mod"]

# Same canonical Costs used everywhere else in this project for the uniform-hard (a_hand=0) pool:
# u_hand = R*0 - lam*h = -98.7, u_build = R*1 - lam*(C+r) = 49.2, u_reuse = R*1 - lam*r = 80.
U_HAND, U_BUILD, U_REUSE = -98.7, 49.2, 80.0

# --------------------------------------------------------------------- canonical reference code
# Hand-transcribed from family_kit's own reference() functions (verified byte-for-byte against them
# in _selftest_family_code below) -- guaranteed-correct code for the "build" demonstrations.
FAMILY_CODE = {
    "lcg": """def solve(inputs: dict) -> float:
    x = inputs['SEED'] % inputs['M']
    for _ in range(int(inputs['K'])):
        x = (inputs['A'] * x + inputs['B']) % inputs['M']
    return x""",
    "modpow": """def solve(inputs: dict) -> float:
    return pow(int(inputs['BASE']), int(inputs['EXP']), int(inputs['MOD']))""",
    "continued_frac": """def solve(inputs: dict) -> float:
    terms = inputs['TERMS']
    p_prev, p = 1, terms[0]
    for a in terms[1:]:
        p_prev, p = p, a * p + p_prev
    return p""",
    "crt_solve": """def solve(inputs: dict) -> float:
    x, mod = 0, 1
    for r, mm in zip(inputs['REMAINDERS'], inputs['MODULI']):
        inv = pow(mod % mm, -1, mm)
        t = ((r - x) * inv) % mm
        x += mod * t
        mod *= mm
    return x % mod""",
    "josephus": """def solve(inputs: dict) -> float:
    r = 0
    for i in range(2, int(inputs['N']) + 1):
        r = (r + inputs['K']) % i
    return r + 1""",
    "quadratic_map_mod": """def solve(inputs: dict) -> float:
    x, mod = inputs['X'] % inputs['MOD'], inputs['MOD']
    for _ in range(int(inputs['K'])):
        x = (x * x + inputs['C']) % mod
    return x""",
    "xorshift_steps": """def solve(inputs: dict) -> float:
    mask, x = (1 << 32) - 1, inputs['X']
    for _ in range(int(inputs['K'])):
        x ^= (x << inputs['A']) & mask
        x ^= x >> inputs['B']
        x ^= (x << inputs['C']) & mask
        x &= mask
    return x""",
    "matrix_power_mod": """def solve(inputs: dict) -> float:
    mod, K = inputs['MOD'], int(inputs['K'])
    R, Mx = [[1, 0], [0, 1]], [[inputs['A'], inputs['B']], [inputs['C'], inputs['D']]]
    def mul(X, Y):
        return [[(X[0][0]*Y[0][0]+X[0][1]*Y[1][0]) % mod, (X[0][0]*Y[0][1]+X[0][1]*Y[1][1]) % mod],
                [(X[1][0]*Y[0][0]+X[1][1]*Y[1][0]) % mod, (X[1][0]*Y[0][1]+X[1][1]*Y[1][1]) % mod]]
    for _ in range(K):
        R = mul(R, Mx)
    return (R[0][0] + R[0][1] + R[1][0] + R[1][1]) % mod""",
}

# --------------------------------------------------------------------- urn vocabularies (diversity)
# "attr" = the per-vocabulary attribute word for the value each item carries (the thing you match on).
# Varied per vocab (not hard-coded "color") so the SFT signal isn't tied to the literal word "color";
# all chosen to be consonant-initial and to pluralize with a bare "+s" (keeps the prompt grammatical).
URN_VOCAB = [
    {"vessel": "bag", "item": "ball", "attr": "color",
     "palette": ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]},
    {"vessel": "jar", "item": "ticket", "attr": "material",
     "palette": ["gold", "silver", "bronze", "platinum", "ruby", "emerald", "sapphire", "pearl"]},
    {"vessel": "box", "item": "coin", "attr": "metal",
     "palette": ["copper", "tin", "zinc", "iron", "cobalt", "nickel", "lead", "brass"]},
    {"vessel": "drawer", "item": "tile", "attr": "symbol",
     "palette": ["star", "moon", "sun", "cloud", "wave", "leaf", "flame", "spark"]},
    {"vessel": "pouch", "item": "gem", "attr": "kind",
     "palette": ["garnet", "topaz", "amber", "jade", "onyx", "coral", "jet", "quartz"]},
    {"vessel": "crate", "item": "card", "attr": "suit",
     "palette": ["clover", "anchor", "crown", "lantern", "compass", "arrow", "shield", "banner"]},
    {"vessel": "bin", "item": "token", "attr": "shape",
     "palette": ["circle", "square", "triangle", "hexagon", "pentagon", "oval", "cross", "diamond"]},
]

URN_SYSTEM = """You are playing a game with a {vessel} of {item}s.

{item_cap}s are drawn from the {vessel} one at a time; there will be {T} draws in total. Each {item} \
has a {attr}.

At any draw you may KEEP the {item} currently shown, or PASS on it. You may KEEP at most {B} \
{item}s in the entire game.

When you KEEP a {item} of some {attr}, you immediately collect that {item} AND every remaining {item} \
of that same {attr} that is drawn later in the game -- all of them are added to your total automatically \
and for free. (So a {attr} can be kept at most once; keeping it locks in all of its future draws.)

The {vessel} is unchanged by your choices: all {T} draws still happen in order. Whenever a {item} of a \
{attr} you have ALREADY kept is drawn again, you will simply be told it was collected automatically \
(+1) and move straight on -- you do not choose again for that {attr}.

If you PASS a {item} it is gone and scores nothing. {item_cap}s of a {attr} you never keep score nothing.

Your goal is to MAXIMIZE the total number of {item}s you collect by the end of the game.

There are exactly {N} distinct {attr}s in the {vessel}, but you are not told which {attr}s are common \
or rare, nor how many {item}s of each {attr} there are; you learn the rates only from the draws you \
have seen. Decide as each {item} appears -- decisions are final.

For each draw, think briefly (one or two sentences), then end your reply with EXACTLY one line:
DECISION: KEEP
or
DECISION: PASS"""

_DP_CACHE: dict[tuple, ExactDP] = {}


def _get_dp(N: int, T: int, B: int) -> ExactDP:
    key = (N, T, B)
    dp = _DP_CACHE.get(key)
    if dp is None:
        dp = ExactDP(U_HAND, U_BUILD, U_REUSE, N, T, B, alpha=1.0, cap=3)
        _DP_CACHE[key] = dp
    return dp


def _walk(slots: list[dict], builds: dict, B: int) -> list[dict]:
    """Tag each slot (in arrival order) with action in {wait, commit, reuse} + running budget_left
    (AFTER this slot's decision, if any). commit = the slot at which the class is built/kept."""
    budget_left = B
    out = []
    for s in sorted(slots, key=lambda z: z["slot_index"]):
        cid = s["class_id"]
        b = builds.get(cid)
        k, t = s["class_position"], s["slot_index"] + 1
        if b is None or k < b:
            action = "wait"
        elif k == b:
            action = "commit"
            budget_left -= 1
        else:
            action = "reuse"
        out.append({**s, "action": action, "k": k, "t": t, "budget_left": budget_left})
    return out


def _labels(slots: list[dict], N: int, T: int, B: int, policy: str) -> dict:
    if policy == "pistar":
        return _get_dp(N, T, B).policy_builds(slots)
    if policy == "eager":
        return eager_builds(slots, B)
    raise ValueError(policy)


# --------------------------------------------------------------------- urn rendering
def _urn_rationale(step: dict, vocab: dict, budget_before: int, policy: str) -> str:
    item, attr = vocab["item"], vocab["attr"]
    k, t = step["k"], step["t"]
    plural = "s" if k != 1 else ""
    if step["action"] == "commit":
        if policy != "pistar":
            return (f"This is a new {item} {attr} -- I'll keep it now in case it comes up again, while "
                    f"I still have {budget_before} keep(s) available.")
        if k == 1:      # the exact DP does sometimes commit on a first sighting (early in the game,
            return (f"I haven't seen this {item} {attr} before, but with {t} of the draws elapsed and "
                    f"{budget_before} keep(s) still available, it's worth committing to it now rather "
                    f"than waiting.")   # much of the horizon left) -- NOT "demonstrated recurrence".
        return (f"This {item} {attr} has now appeared {k} times in the first {t} draws -- that's "
                f"demonstrated recurrence, and I have {budget_before} keep(s) available, so I'll "
                f"reserve one for it now.")
    return (f"I've seen this {item} {attr} {k} time{plural} so far, but I'll hold my {budget_before} "
            f"keep(s) in reserve for now rather than commit yet.")


def render_urn_session(vocab: dict, N: int, T: int, B: int, seed: int, policy: str) -> list[dict]:
    fams = UNIFORM[:N]
    slots, _meta = build_stochastic_stream(StochasticStreamSpec(
        families=fams, n_hot=B, T=T, budget=B, guarantee_trap_early=1.0, magnitude=100, seed=seed))
    builds = _labels(slots, N, T, B, policy)
    steps = _walk(slots, builds, B)

    color: dict[int, str] = {}
    for s in steps:
        if s["class_id"] not in color:
            color[s["class_id"]] = vocab["palette"][len(color)]

    system = URN_SYSTEM.format(vessel=vocab["vessel"], item=vocab["item"], attr=vocab["attr"],
                               item_cap=vocab["item"].capitalize(), T=T, B=B, N=N)
    messages = [{"role": "system", "content": system}]
    pending: list[tuple[int, str]] = []
    for s in steps:
        col, n = color[s["class_id"]], s["t"]
        budget_before = s["budget_left"] + (1 if s["action"] == "commit" else 0)
        if s["action"] == "reuse":
            pending.append((n, col))
            continue
        pre = ""
        if pending:
            evs = "; ".join(f"draw {dn}: another {c} {vocab['item']} "
                            f"(already kept -> auto-collected, +1)" for dn, c in pending)
            pre = f"(Since your last choice: {evs}.) "
            pending = []
        user = (f"{pre}Draw {n} of {T}: a {col} {vocab['item']} appears. You have {budget_before} "
                f"keep(s) left. KEEP or PASS?")
        messages.append({"role": "user", "content": user})
        rationale = _urn_rationale(s, vocab, budget_before, policy)
        decision = "KEEP" if s["action"] == "commit" else "PASS"
        messages.append({"role": "assistant", "content": f"{rationale}\nDECISION: {decision}"})
    return messages


# --------------------------------------------------------------------- tool-bridge rendering
def _tc(name: str, args: dict) -> dict:
    return {"id": f"demo_{uuid.uuid4().hex[:8]}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _tool_rationale(action: str, fam: str, k: int, t: int, budget_before: int, policy: str) -> str:
    """No 'wait' case (2026-07-06 corpus regeneration, item 3): a wait/hand turn's decision is already
    fully expressed by which tool gets called (none, here) -- rationale text added nothing on top of
    that, and a fixed string there is exactly the bug class that collapsed anchor_tool/mechanics_bridge
    (see docs/qwen-finetune-transfer-plan.md 'corpus regeneration'). Currently dormant (N_TOOL=0) but
    fixed now so it can't resurface if tool_bridge is reactivated."""
    if action == "commit":
        if policy != "pistar":
            return (f"This is a new type of problem -- I'll write a script for it now in case it "
                    f"comes up again ({budget_before} script slot(s) left).")
        if k == 1:      # the exact DP does sometimes build on a first sighting early in the stream --
            return (f"I haven't seen this exact type before, but with {t} problems in and "
                    f"{budget_before} script slot(s) still available, it's worth writing one now "
                    f"rather than waiting.")   # NOT "worth it because it keeps recurring".
        return (f"I've now seen this type of problem {k} times -- worth writing a reusable script "
                f"since it keeps recurring, and I have {budget_before} script slot(s) left.")
    return "I already have a script for this type of problem -- I'll just run it again."


def render_tool_bridge_session(N: int, T: int, B: int, seed: int, policy: str) -> list[dict]:
    fams = UNIFORM[:N]
    slots, _meta = build_stochastic_stream(StochasticStreamSpec(
        families=fams, n_hot=B, T=T, budget=B, guarantee_trap_early=1.0, magnitude=100, seed=seed))
    builds = _labels(slots, N, T, B, policy)
    steps = _walk(slots, builds, B)
    problems = {p["idx"]: p for p in slots_to_problems(slots)}

    # match the A2 tool eval prompt exactly: base + recurrence disclosure + N disclosure
    system = system_prompt(T, B, None) + RECURRENCE_NOTE + n_types_note(N)
    messages = [{"role": "system", "content": system}]
    script_of: dict[int, str] = {}
    for i, s in enumerate(steps):
        prob = problems[s["slot_index"]]
        messages.append({"role": "user", "content": problem_prompt(prob, i + 1, T)})
        fam, gold = s["family"], prob["gold"]
        budget_before = s["budget_left"] + (1 if s["action"] == "commit" else 0)

        if s["action"] == "wait":
            # decoupled from gold (2026-07-06 corpus regeneration, item 6): a wait/hand turn has no
            # write_script/run_script behind it, so asserting the submitted value is CORRECT taught the
            # model "hand-solving this pool works", contradicting the pool's own a_hand=0.0 premise (see
            # docs/qwen-finetune-transfer-plan.md 'corpus regeneration'). gold+1 is deterministic, always
            # wrong, and varies with gold (no new low-diversity target).
            tc = _tc("submit_answer", {"value": gold + 1})
            messages.append({"role": "assistant", "tool_calls": [tc]})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
                {"ok": True, "message": f"Recorded answer for problem {prob['idx']}."})})
            continue

        rationale = _tool_rationale(s["action"], fam, s["k"], s["t"], budget_before, policy)
        if s["action"] == "commit":
            name = f"{fam}_solver"
            script_of[s["class_id"]] = name
            tc = _tc("write_script", {"name": name, "code": FAMILY_CODE[fam]})
            messages.append({"role": "assistant", "content": rationale, "tool_calls": [tc]})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
                {"ok": True, "message": f"Saved script '{name}'.", "scripts": [name]})})
            tc = _tc("run_script", {"name": name, "inputs": prob["inputs"]})
            messages.append({"role": "assistant", "tool_calls": [tc]})
        else:  # reuse -- rationale + run_script MUST be ONE assistant message, not two: two separate
            # assistant-role dicts back-to-back with no intervening tool/user turn never occurs in real
            # eval (driver.py always emits exactly one assistant dict per turn) and trains the model on
            # a mid-conversation '<|im_start|>assistant\n' it should never see -- see
            # docs/qwen-finetune-transfer-plan.md "Mechanics bridge training" diagnosis (2026-07-06).
            name = script_of[s["class_id"]]
            tc = _tc("run_script", {"name": name, "inputs": prob["inputs"]})
            messages.append({"role": "assistant", "content": rationale, "tool_calls": [tc]})
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": json.dumps({"ok": True, "return_value": gold})})
        tc = _tc("submit_answer", {"value": gold})
        messages.append({"role": "assistant", "tool_calls": [tc]})
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
            {"ok": True, "message": f"Recorded answer for problem {prob['idx']}."})})
    return messages


# --------------------------------------------------------------------- tool-calling anchor rendering
# Option (c): single-problem, in-domain, policy-neutral sessions that keep the base model's tool-calling
# alive under the text-heavy urn corpus. Diversity comes from all 8 families x fresh seeds. A single
# problem has NO recurrence, so an anchor cannot demonstrate reserve-vs-eager TIMING either way -- but a
# build-only anchor still teaches "problem -> build on sight" (an eager-flavored per-problem reflex).
# ANCHOR_HAND_FRAC balances build vs hand-submit so the anchor stays neutral on the act-or-not axis too.
#
# DESIGN A (pure transfer test, 2026-07-06): ANCHOR_HAND_FRAC=0.5 (balanced -> modality only, no timing
# signal) and N_TOOL=0 (see below): the tool vocabulary then carries ZERO reserve-timing signal, so any
# reserve behavior in the tool eval can ONLY have transferred from the urn. (The 2026-07-05 framing-wall
# run used ANCHOR_HAND_FRAC=0.0 + N_TOOL=6 -- flip both back to reproduce it.)
N_ANCHOR = 150
ANCHOR_HAND_FRAC = 0.5
ANCHOR_SEED_START = 6000     # disjoint from urn (5000+), tool_bridge (4000+), and eval (2000-2023)


def _anchor_problem(fam: str, seed: int) -> dict:
    """One concrete problem instance of `fam` at MAG=100, in the same dict shape driver.py's
    problem_prompt consumes (mirrors run_stream_session.slots_to_problems' fields)."""
    m = ALL_FAMILIES[fam].make_member(random.Random(seed), 100)
    g = int(m["gold"])
    return {**m, "idx": 0, "gold": g, "sig_figs": max(1, len(str(abs(g)))), "exact_int": True}


def render_anchor_session(fam: str, seed: int, mode: str) -> list[dict]:
    """A single-problem tool session. mode='build' writes+runs+submits (teaches the full
    write_script/run_script modality); mode='hand' just submits (lowers the build prior). NO
    recurrence/N disclosure and no cross-problem state -> nothing about build TIMING is demonstrated.

    No rationale text (2026-07-06 corpus regeneration, item 2): the decision is already fully
    expressed by which tool gets called, and a fixed rationale string here is the exact bug that
    collapsed this slice into verbatim repetition at eval (see
    docs/qwen-finetune-transfer-plan.md 'corpus regeneration') -- bare tool calls only, matching the
    convention already used for every follow-up call in a build sequence."""
    prob = _anchor_problem(fam, seed)
    gold = prob["gold"]
    # match the tool_bridge convention: token_cap=None (no budget paragraph), n=1, budget=1
    messages = [{"role": "system", "content": system_prompt(1, 1, None)},
                {"role": "user", "content": problem_prompt(prob, 1, 1)}]
    if mode == "hand":
        # decoupled from gold (item 6, same rationale as render_tool_bridge_session's wait branch):
        # no script backs this submission, so asserting it's correct taught "hand-solving works",
        # contradicting a_hand=0.0. gold+1 is deterministic, always wrong, varies with gold.
        tc = _tc("submit_answer", {"value": gold + 1})
        messages.append({"role": "assistant", "tool_calls": [tc]})
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
            {"ok": True, "message": f"Recorded answer for problem {prob['idx']}."})})
        return messages
    name = f"{fam}_solver"
    tc = _tc("write_script", {"name": name, "code": FAMILY_CODE[fam]})
    messages.append({"role": "assistant", "tool_calls": [tc]})
    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
        {"ok": True, "message": f"Saved script '{name}'.", "scripts": [name]})})
    tc = _tc("run_script", {"name": name, "inputs": prob["inputs"]})
    messages.append({"role": "assistant", "tool_calls": [tc]})
    messages.append({"role": "tool", "tool_call_id": tc["id"],
                     "content": json.dumps({"ok": True, "return_value": gold})})
    tc = _tc("submit_answer", {"value": gold})
    messages.append({"role": "assistant", "tool_calls": [tc]})
    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
        {"ok": True, "message": f"Recorded answer for problem {prob['idx']}."})})
    return messages


def generate_anchor(n: int = N_ANCHOR, hand_frac: float = ANCHOR_HAND_FRAC) -> list[list[dict]]:
    """Arm-INDEPENDENT: fixed seed, no policy argument -> identical bytes for both SFT arms, so the
    anchor is a matched control that cannot create a pistar-vs-eager difference."""
    rng = random.Random(24680)
    out = []
    for i in range(n):
        fam = UNIFORM[i % len(UNIFORM)]
        mode = "hand" if rng.random() < hand_frac else "build"
        out.append(render_anchor_session(fam, ANCHOR_SEED_START + i, mode))
    return out


# --------------------------------------------------------------------- mechanics bridge (Design A fix,
# 2026-07-06): realistic full-length tool sessions, policy-neutral on build TIMING (see module docstring).
MECH_SEED_START = 7000     # disjoint from urn(5000+)/tool_bridge(4000+)/anchor(6000+)/eval(2000-2023)
N_MECH = 25     # bumped 10->25 (2026-07-06): 10 wasn't enough repetition -- FT model still hallucinated
                # tool syntax (a "<script>...</script>" pseudo-tag) on later problems in the real 60-
                # problem eval; testing whether more exposure to the correct write_script/run_script/
                # submit_answer format in long sessions fixes it before considering a bigger redesign.


def random_builds(slots: list[dict], B: int, rng: random.Random) -> dict:
    """Policy-neutral labels: pick B distinct classes at random (independent of recurrence/arrival
    order); commit HALF at first sighting (k=1) and HALF at a random later occurrence (k>=2, forced
    when the class has enough occurrences) -- deterministically balanced, not left to chance, so build
    timing carries no correlation with recurrence in either direction (not eager, not reserve).

    De-clustered (2026-07-06 corpus regeneration, item 5): a class's own first sighting (k=1) is,
    definitionally, wherever that class first debuts in the realized stream -- and with only N~8
    classes, the naturally-early-debuting ones (hot classes reliably debut in the first few slots by
    construction) used to ALWAYS get picked for the k=1 half by pure chance, which taught the FT model
    an absolute-position prior ('building happens near session-start') on top of the intended
    recurrence-neutral signal -- and matches the real eval collapse (all 7 real builds landed at
    problem #2 or #3, never later). Fix: explicitly assign k=1 to the LATEST-debuting classes across
    the WHOLE N-class pool (not a random B-subset -- picking within a random subset wasn't enough,
    since a subset dominated by hot classes is still early no matter which of its members "wins"),
    so first-sighting commits spread across the whole session instead of concentrating wherever the
    fastest-debuting classes happen to be. Selection is by debut SLOT only, unrelated to hot/trap role
    or future recurrence rate, so it stays policy-neutral."""
    by_class: dict[int, list[dict]] = {}
    for s in slots:
        by_class.setdefault(s["class_id"], []).append(s)
    class_ids = list(by_class.keys())
    debut = {cid: min(s["slot_index"] for s in occs) for cid, occs in by_class.items()}
    n_k1 = (B + 1) // 2
    n_kge2 = B - n_k1
    k1_classes = sorted(class_ids, key=lambda cid: -debut[cid])[:n_k1]   # latest debut in the pool
    remaining = [cid for cid in class_ids if cid not in k1_classes]
    rng.shuffle(remaining)
    kge2_classes = [cid for cid in remaining if len(by_class[cid]) >= 2][:n_kge2]
    for cid in remaining:                                # fallback if too few classes recur >=2 times
        if len(kge2_classes) >= n_kge2:
            break
        if cid not in kge2_classes:
            kge2_classes.append(cid)
    builds = {cid: 1 for cid in k1_classes}
    for cid in kge2_classes:
        occs = sorted(by_class[cid], key=lambda z: z["slot_index"])
        builds[cid] = 1 if len(occs) < 2 else occs[rng.randrange(1, len(occs))]["class_position"]
    return builds


def render_mechanics_bridge_session(N: int, T: int, B: int, seed: int) -> list[dict]:
    """No rationale text (2026-07-06 corpus regeneration, item 1): the decision is already fully
    expressed by which tool gets called, and a fixed rationale string here (there were only 3 across
    all 1500 content-bearing turns) is exactly the bug that collapsed this slice into verbatim
    repetition at eval -- see docs/qwen-finetune-transfer-plan.md 'corpus regeneration'. Bare tool
    calls only, matching the convention already used for every follow-up call in a commit/reuse
    sequence."""
    fams = UNIFORM[:N]
    slots, _meta = build_stochastic_stream(StochasticStreamSpec(
        families=fams, n_hot=B, T=T, budget=B, guarantee_trap_early=1.0, magnitude=100, seed=seed))
    builds = random_builds(slots, B, random.Random(seed + 999983))   # separate rng stream
    steps = _walk(slots, builds, B)
    problems = {p["idx"]: p for p in slots_to_problems(slots)}

    system = system_prompt(T, B, None) + RECURRENCE_NOTE + n_types_note(N)
    messages = [{"role": "system", "content": system}]
    script_of: dict[int, str] = {}
    for i, s in enumerate(steps):
        prob = problems[s["slot_index"]]
        messages.append({"role": "user", "content": problem_prompt(prob, i + 1, T)})
        fam, gold = s["family"], prob["gold"]

        if s["action"] == "wait":
            # decoupled from gold (item 6): no script backs this submission, so asserting it's
            # correct taught "hand-solving works", contradicting a_hand=0.0. gold+1 is deterministic,
            # always wrong, and varies with gold (no new low-diversity target).
            tc = _tc("submit_answer", {"value": gold + 1})
            messages.append({"role": "assistant", "tool_calls": [tc]})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
                {"ok": True, "message": f"Recorded answer for problem {prob['idx']}."})})
            continue

        if s["action"] == "commit":
            name = f"{fam}_solver"
            script_of[s["class_id"]] = name
            tc = _tc("write_script", {"name": name, "code": FAMILY_CODE[fam]})
            messages.append({"role": "assistant", "tool_calls": [tc]})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
                {"ok": True, "message": f"Saved script '{name}'.", "scripts": [name]})})
            tc = _tc("run_script", {"name": name, "inputs": prob["inputs"]})
            messages.append({"role": "assistant", "tool_calls": [tc]})
        else:  # reuse -- run_script MUST be its own single assistant message, not split across two:
            # two separate assistant-role dicts back-to-back with no intervening tool/user turn never
            # occurs in real eval (driver.py always emits exactly one assistant dict per turn) and
            # trains the model on a mid-conversation '<|im_start|>assistant\n' it should never see --
            # see docs/qwen-finetune-transfer-plan.md "Mechanics bridge training" diagnosis (2026-07-06).
            name = script_of[s["class_id"]]
            tc = _tc("run_script", {"name": name, "inputs": prob["inputs"]})
            messages.append({"role": "assistant", "tool_calls": [tc]})
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": json.dumps({"ok": True, "return_value": gold})})
        tc = _tc("submit_answer", {"value": gold})
        messages.append({"role": "assistant", "tool_calls": [tc]})
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
            {"ok": True, "message": f"Recorded answer for problem {prob['idx']}."})})
    return messages


def generate_mechanics_bridge(n: int = N_MECH) -> list[list[dict]]:
    """Arm-INDEPENDENT (like the anchor): no policy argument -> identical bytes for both SFT arms."""
    out = []
    for i in range(n):
        B = 2 if i % 2 == 0 else 3
        out.append(render_mechanics_bridge_session(8, 60, B, MECH_SEED_START + i))
    return out


# --------------------------------------------------------------------- error-recovery bridge (Phase 3
# ablation, 2026-07-06 — docs/qwen-finetune-transfer-plan.md "Error-recovery ablation"; folded into
# mechanics bridge training's corpus): teaches the exact
# "bad turn -> harness-injected correction -> valid recovery" shapes driver.run_session actually
# produces, so the model has training exposure to the conversational state its own retry logic creates
# instead of only ever seeing well-formed turns (every prior slice -- urn, tool_bridge, anchor,
# mechanics -- is well-formed by construction). Reuses the harness's OWN text verbatim (FORMAT_REMINDER
# imported from driver.py, error strings hand-matched to session_state.py's exact f-strings) rather than
# a paraphrase -- the project's standing lesson (verify_template) is that a near-miss format is as bad
# as no fix at all.
#
# Three recovery shapes, matching driver.run_session's three actual error branches:
#   "reminder"        -- zero parseable tool calls (the 5 patterns Result 2 actually observed: an
#                        unregistered plural tool name written as plain text, a pseudo-XML tag, empty-
#                        object repetition, a verbatim anchor-phrase loop, an empty markdown fence) ->
#                        the literal driver.FORMAT_REMINDER user turn -> a real <tool_call>.
#   "wrong_name"      -- a well-formed tool_calls entry naming an unregistered tool ("submit_answers",
#                        the plural hallucination from Result 2, this time as an actual tool call rather
#                        than plain text) -> driver's own unknown-tool tool-role error -> retry with the
#                        correct name.
#   "unknown_script"  -- run_script on a name never write_script'd -> session_state's own "no script
#                        named" tool-role error -> write_script then run_script then submit_answer.
# All single-problem (like the anchor): no cross-problem state, so no build-TIMING signal is possible --
# these sessions can only teach recovery, never reserve-vs-eager.
RECOVERY_SEED_START = 8000     # disjoint from urn(5000+)/tool_bridge(4000+)/anchor(6000+)/mechanics(7000+)/eval(2000-2023)
N_RECOVERY = 45
_RECOVERY_PATTERNS = ["reminder", "wrong_name", "unknown_script"]
_KNOWN_TOOL_NAMES = sorted(t["function"]["name"] for t in TOOL_SCHEMAS())

# the 5 malformed-content patterns actually observed in Result 2 -- none of these produce a parseable
# tool_calls structure, so driver.run_session's `if not turn.tool_calls` branch is what fires on all of
# them (the FORMAT_REMINDER path), regardless of which surface pattern the collapse takes.
_MALFORMED_CONTENT = [
    lambda gold: f"I'll record the answer.\nsubmit_answers({{\"value\": {gold}}})",
    lambda gold: f"<script>\ndef solve(inputs):\n    return {gold}\n</script>",
    lambda gold: "{}\n{}\n{}",
    lambda gold: ("This needs an exact, large computation -- I'll write a script for it and run it "
                  "on these inputs."),
    lambda gold: "```\n\n```",
]


def _unknown_tool_error(name: str) -> dict:
    """Mirrors driver.run_session's exact unknown-tool-name branch (tc['name'] not in known_tools)."""
    return {"ok": False, "error": f"no such tool '{name}'. Available tools: {_KNOWN_TOOL_NAMES}."}


def _no_script_error(name: str, known: list[str] = ()) -> dict:
    """Mirrors session_state.op_run_script's exact 'no script named' branch."""
    return {"ok": False, "error": f"no script named '{name}'. Available: {sorted(known) or '(none)'}"}


def render_error_recovery_session(fam: str, seed: int, pattern: str) -> list[dict]:
    """No rationale text on any turn except the deliberate malformed-content demo (2026-07-06 corpus
    regeneration, item 4): the corrective/well-formed turns' rationale strings (e.g. "Using the tools
    as instructed.") are the same fixed-literal bug class as anchor/mechanics_bridge and get dropped
    (bare tool calls). The `_MALFORMED_CONTENT` bad-turn text in the 'reminder' pattern MUST stay --
    it's reproducing the actual observed collapse shapes from Result 2, not filler."""
    prob = _anchor_problem(fam, seed)
    gold = prob["gold"]
    # single-problem framing, identical convention to the anchor (option c): no N/recurrence
    # disclosure -> policy-neutral, no recurrence -> no build-timing signal possible.
    messages = [{"role": "system", "content": system_prompt(1, 1, None)},
                {"role": "user", "content": problem_prompt(prob, 1, 1)}]

    if pattern == "reminder":
        bad = _MALFORMED_CONTENT[seed % len(_MALFORMED_CONTENT)](gold)
        messages.append({"role": "assistant", "content": bad})            # NO tool_calls key -> hits
        messages.append({"role": "user", "content": FORMAT_REMINDER})     # driver's "no tool call" branch
        tc = _tc("submit_answer", {"value": gold})
        messages.append({"role": "assistant", "tool_calls": [tc]})
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
            {"ok": True, "message": f"Recorded answer for problem {prob['idx']}."})})
        return messages

    if pattern == "wrong_name":
        bad_tc = _tc("submit_answers", {"value": gold})                   # unregistered plural name,
        messages.append({"role": "assistant", "tool_calls": [bad_tc]})    # well-formed args this time
        messages.append({"role": "tool", "tool_call_id": bad_tc["id"],
                         "content": json.dumps(_unknown_tool_error("submit_answers"))})
        tc = _tc("submit_answer", {"value": gold})
        messages.append({"role": "assistant", "tool_calls": [tc]})
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
            {"ok": True, "message": f"Recorded answer for problem {prob['idx']}."})})
        return messages

    if pattern == "unknown_script":
        name = f"{fam}_solver"
        bad_tc = _tc("run_script", {"name": name, "inputs": prob["inputs"]})   # never written yet
        messages.append({"role": "assistant", "tool_calls": [bad_tc]})
        messages.append({"role": "tool", "tool_call_id": bad_tc["id"],
                         "content": json.dumps(_no_script_error(name, known=()))})
        tc = _tc("write_script", {"name": name, "code": FAMILY_CODE[fam]})
        messages.append({"role": "assistant", "tool_calls": [tc]})
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
            {"ok": True, "message": f"Saved script '{name}'.", "scripts": [name]})})
        tc = _tc("run_script", {"name": name, "inputs": prob["inputs"]})
        messages.append({"role": "assistant", "tool_calls": [tc]})
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": json.dumps({"ok": True, "return_value": gold})})
        tc = _tc("submit_answer", {"value": gold})
        messages.append({"role": "assistant", "tool_calls": [tc]})
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
            {"ok": True, "message": f"Recorded answer for problem {prob['idx']}."})})
        return messages

    raise ValueError(pattern)


def generate_error_recovery(n: int = N_RECOVERY) -> list[list[dict]]:
    """Arm-independent (like anchor/mechanics): fixed seed, no policy argument -> identical bytes for
    both SFT arms."""
    out = []
    for i in range(n):
        fam = UNIFORM[i % len(UNIFORM)]
        pattern = _RECOVERY_PATTERNS[i % len(_RECOVERY_PATTERNS)]
        out.append(render_error_recovery_session(fam, RECOVERY_SEED_START + i, pattern))
    return out


# --------------------------------------------------------------------- corpus generation
# IMPORTANT: these ranges are NOT arbitrary -- empirically swept (see git history / session log) to
# confirm pi*'s DP actually RESERVES (commits at k>=2, not k=1) at these settings. Short T / generous
# B relative to N collapses pi*'s own optimal policy toward eager (e.g. T=20,B=2,N=6 -> 100% k=1
# commits), which would make the "treatment" corpus indistinguishable from the eager control --
# defeating the whole point of Phase 3. T>=60 with B/N roughly 0.25-0.35 reliably reserves.
N_RANGE, T_RANGE = (6, 8), (60, 80, 100)
URN_SEED_START, TOOL_SEED_START = 5000, 4000     # disjoint from eval seeds 2000-2011/2023
# Tool-bridge is deliberately a SMALL slice: tool sessions are ~6x longer per session than urn
# sessions, so to keep the corpus ~85/15 urn:tool BY TOKEN (not by session count) N_TOOL must be
# small. Measured: at N_TOOL=6 the tool token share is ~17% (pistar) / ~12% (eager), ~15% averaged
# across the two SFT arms; 6 also keeps the B=2/B=3 alternation balanced (3 each). The urn allocation
# policy is the thing being taught; the tool slice is only a framing bridge.
N_URN, N_TOOL = 170, 0     # DESIGN A: N_TOOL=0 -> no tool_bridge reserve demos (those would be a DIRECT
                           # in-tool install of reserve, not transfer). Was 6 in the framing-wall run.


def _pick_B(N: int, rng: random.Random) -> int:
    return 2 if N == 6 else rng.choice((2, 3))     # keeps B/N in the reserve-friendly ~0.25-0.35 band


def generate_corpus(policy: str, n_urn: int = N_URN, n_tool: int = N_TOOL) -> tuple[list, list]:
    rng = random.Random(12345 if policy == "pistar" else 67890)
    urn_sessions = []
    for i in range(n_urn):
        vocab = URN_VOCAB[i % len(URN_VOCAB)]
        N, T = rng.choice(N_RANGE), rng.choice(T_RANGE)
        B = _pick_B(N, rng)
        urn_sessions.append(render_urn_session(vocab, N, T, B, URN_SEED_START + i, policy))
    tool_sessions = []
    for i in range(n_tool):
        # fixed to the REAL eval config (N=8, T=60) for maximum fidelity to the target domain; B
        # alternates 2/3 -- B=2 gives a clean 100% reserve signal, B=3 matches the eval's actual
        # budget exactly (a noisier ~62% reserve rate, but that's honestly what pi* does there).
        B = 2 if i % 2 == 0 else 3
        tool_sessions.append(render_tool_bridge_session(8, 60, B, TOOL_SEED_START + i, policy))
    return urn_sessions, tool_sessions


def _est_tokens(messages: list[dict]) -> int:
    return sum(len(json.dumps(m)) for m in messages) // 4


def write_corpus(out_dir: Path = Path("runs/phase3_sft_data")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for policy in ("pistar", "eager"):
        urn_sessions, tool_sessions = generate_corpus(policy)
        for label, sessions in (("urn", urn_sessions), ("tool_bridge", tool_sessions)):
            path = out_dir / f"{label}_{policy}.jsonl"
            with path.open("w") as f:
                for msgs in sessions:
                    f.write(json.dumps({"messages": msgs}) + "\n")
            toks = [_est_tokens(m) for m in sessions]
            mean = sum(toks) / len(toks) if toks else 0     # tool_bridge is empty under DESIGN A (N_TOOL=0)
            print(f"  {path}: {len(sessions)} sessions, ~{sum(toks):,} est. tokens (mean {mean:.0f}/session)"
                  + ("  [empty — DESIGN A: no tool_bridge reserve demos]" if not sessions else ""))
    # arm-independent tool-calling anchor (option (c)): written once, shared by both SFT arms
    anchor = generate_anchor()
    path = out_dir / "anchor_tool.jsonl"
    with path.open("w") as f:
        for msgs in anchor:
            f.write(json.dumps({"messages": msgs}) + "\n")
    n_build = sum(1 for a in anchor if any(tc["function"]["name"] == "write_script"
                                           for m in a for tc in m.get("tool_calls", [])))
    toks = [_est_tokens(m) for m in anchor]
    print(f"  {path}: {len(anchor)} sessions ({n_build} build / {len(anchor)-n_build} hand), "
          f"~{sum(toks):,} est. tokens (arm-independent; shared by both SFT arms)")
    # arm-independent mechanics bridge (Design A fix): written once, shared by both SFT arms
    mech = generate_mechanics_bridge()
    path = out_dir / "mechanics_bridge.jsonl"
    with path.open("w") as f:
        for msgs in mech:
            f.write(json.dumps({"messages": msgs}) + "\n")
    toks = [_est_tokens(m) for m in mech]
    print(f"  {path}: {len(mech)} sessions, ~{sum(toks):,} est. tokens "
          f"(arm-independent; shared by both SFT arms; policy-neutral build timing)")
    # arm-independent error-recovery bridge (error-recovery ablation, folded into mechanics bridge
    # training's corpus): written once, shared by both SFT arms
    recovery = generate_error_recovery()
    path = out_dir / "error_recovery.jsonl"
    with path.open("w") as f:
        for msgs in recovery:
            f.write(json.dumps({"messages": msgs}) + "\n")
    toks = [_est_tokens(m) for m in recovery]
    print(f"  {path}: {len(recovery)} sessions, ~{sum(toks):,} est. tokens "
          f"(arm-independent; shared by both SFT arms; bad-turn -> harness-nudge -> recovery)")


# --------------------------------------------------------------------- self-test
def _selftest_family_code():
    """FAMILY_CODE must exactly reproduce family_kit's own reference() for every family, over many
    random samples -- catches any transcription error in the hand-copied solve() bodies."""
    import random as _r
    print("=== FAMILY_CODE vs family_kit reference (100 random samples/family) ===")
    for fam in UNIFORM:
        f = ALL_FAMILIES[fam]
        ns = {}
        exec(FAMILY_CODE[fam], ns)
        rng = _r.Random(999)
        for _ in range(100):
            inputs = f.sampler(rng, 100)
            want = f.reference(inputs)
            got = ns["solve"](inputs)
            assert int(got) == int(want), f"{fam}: solve({inputs})={got} != reference={want}"
        print(f"  {fam}: OK")


def _selftest_decisions():
    """Every rendered session's embedded decisions must match a freshly-recomputed policy, every
    submit_answer value must match the stream's true gold ON COMMIT/REUSE TURNS (a WAIT turn's
    submit_answer is gold+1, decoupled from correctness -- item 6, 2026-07-06 corpus regeneration:
    a hand/wait turn has no script behind it, so asserting it's correct taught "hand-solving works",
    contradicting a_hand=0.0), and both system prompts must DISCLOSE N (A2 protocol) -- regression
    tripwire + correctness proof. Uses A2-realistic N/T/B (T>=60)."""
    print("\n=== decision + gold-value + N-disclosure consistency (urn + tool_bridge, pistar + eager) ===")
    for policy in ("pistar", "eager"):
        # urn: recompute independently and diff against a couple of rendered sessions
        for i in range(3):
            vocab = URN_VOCAB[i % len(URN_VOCAB)]
            N, T, B = 8, 60, 3
            slots, _ = build_stochastic_stream(StochasticStreamSpec(
                families=UNIFORM[:N], n_hot=B, T=T, budget=B, guarantee_trap_early=1.0,
                magnitude=100, seed=URN_SEED_START + i))
            builds = _labels(slots, N, T, B, policy)
            steps = _walk(slots, builds, B)
            msgs = render_urn_session(vocab, N, T, B, URN_SEED_START + i, policy)
            decisions = [m["content"].splitlines()[-1] for m in msgs if m["role"] == "assistant"]
            expected = [f"DECISION: {'KEEP' if s['action'] == 'commit' else 'PASS'}"
                       for s in steps if s["action"] != "reuse"]
            assert decisions == expected, (policy, i, decisions, expected)
            assert f"exactly {N} distinct {vocab['attr']}s" in msgs[0]["content"], \
                "urn system prompt missing N"
        # tool_bridge: every submit_answer value must equal the slot's true gold; N disclosed
        for i in range(3):
            N, T, B = 8, 60, 2
            slots, _ = build_stochastic_stream(StochasticStreamSpec(
                families=UNIFORM[:N], n_hot=B, T=T, budget=B, guarantee_trap_early=1.0,
                magnitude=100, seed=TOOL_SEED_START + i))
            gold_by_slot = {s["slot_index"]: int(s["gold"]) for s in slots}
            builds = _labels(slots, N, T, B, policy)
            steps = _walk(slots, builds, B)
            msgs = render_tool_bridge_session(N, T, B, TOOL_SEED_START + i, policy)
            assert f"exactly {N} distinct" in msgs[0]["content"], "tool system prompt missing N"
            submits = [json.loads(tc["function"]["arguments"])["value"]
                       for m in msgs for tc in m.get("tool_calls", [])
                       if tc["function"]["name"] == "submit_answer"]
            assert len(submits) == T, (policy, i, len(submits), T)
            expected = [gold_by_slot[s["slot_index"]] + (1 if s["action"] == "wait" else 0)
                       for s in sorted(steps, key=lambda z: z["slot_index"])]
            assert submits == expected, (policy, i, "submit mismatch (gold on commit/reuse, "
                                         "gold+1 on wait)")
        print(f"  policy={policy}: OK")


def _selftest_reserve_rate():
    """A2 CALIBRATION GUARD: the whole point of pistar demos is to teach RESERVING. Verify the actual
    corpus has pi* committing at k>=2 (demonstrated recurrence) on a solid majority of commits -- else
    the treatment demos collapse toward the eager control and Phase 3 is a no-op. Also confirm the
    eager control is ~all first-sight, so the two arms are genuinely distinct."""
    print("\n=== A2 reserve-rate guard (pi* must actually reserve; eager must not) ===")
    # recompute commit positions directly from the corpus's own seeds/params (mirrors generate_corpus)
    for policy in ("pistar", "eager"):
        rng = random.Random(12345 if policy == "pistar" else 67890)
        k1 = kge2 = 0
        for i in range(N_URN):
            N, T = rng.choice(N_RANGE), rng.choice(T_RANGE)
            B = _pick_B(N, rng)
            slots, _ = build_stochastic_stream(StochasticStreamSpec(
                families=UNIFORM[:N], n_hot=B, T=T, budget=B, guarantee_trap_early=1.0,
                magnitude=100, seed=URN_SEED_START + i))
            for s in _walk(slots, _labels(slots, N, T, B, policy), B):
                if s["action"] == "commit":
                    k1 += (s["k"] == 1); kge2 += (s["k"] >= 2)
        tot = k1 + kge2
        frac = kge2 / tot if tot else 0.0
        print(f"  urn {policy:>6}: commits k>=2 = {kge2}/{tot} = {frac:.0%}  (k=1 = {k1})")
        if policy == "pistar":
            assert frac >= 0.5, f"pi* reserve rate too low ({frac:.0%}) -- demos collapse toward eager"
        else:
            assert frac <= 0.05, f"eager control should be ~all first-sight, got k>=2 {frac:.0%}"


def _selftest_anchor():
    """Option (c) anchor guards: deterministic + arm-independent, correct gold on every submit (and
    run_script return), policy-neutral prompt (NO recurrence/N disclosure -> teaches no build-timing),
    known tools only, and the build/hand split matches ANCHOR_HAND_FRAC."""
    print("\n=== tool-calling anchor guard (option c: modality preservation, policy-neutral) ===")
    anchor = generate_anchor()
    assert len(anchor) == N_ANCHOR, (len(anchor), N_ANCHOR)

    def _strip_ids(sessions):     # tool_call ids are random uuids (don't perturb the seeded rng);
        import copy               # compare structure modulo ids to check determinism/arm-independence
        s = copy.deepcopy(sessions)
        for sess in s:
            for m in sess:
                for tc in m.get("tool_calls", []):
                    tc["id"] = ""
                m.pop("tool_call_id", None)
        return s
    assert _strip_ids(generate_anchor()) == _strip_ids(anchor), \
        "anchor structure must be deterministic + arm-independent (no policy dependence)"
    known = {"write_script", "run_script", "submit_answer"}
    n_build = 0
    for i, msgs in enumerate(anchor):
        fam = UNIFORM[i % len(UNIFORM)]
        gold = int(ALL_FAMILIES[fam].make_member(random.Random(ANCHOR_SEED_START + i), 100)["gold"])
        sysp = msgs[0]["content"]
        for leak in ("exactly", "distinct", "recur"):     # policy-neutral: no recurrence/N framing
            assert leak not in sysp, f"anchor {i}: prompt leaks '{leak}' -- not policy-neutral"
        names = [tc["function"]["name"] for m in msgs for tc in m.get("tool_calls", [])]
        assert names and set(names) <= known, (i, names)
        submits = [json.loads(tc["function"]["arguments"])["value"] for m in msgs
                   for tc in m.get("tool_calls", []) if tc["function"]["name"] == "submit_answer"]
        if "write_script" in names:
            n_build += 1
            assert submits == [gold], (i, submits, gold)
            rv = [json.loads(m["content"])["return_value"] for m in msgs
                  if m["role"] == "tool" and "return_value" in m["content"]]
            assert rv == [gold], (i, "run_script return != gold", rv, gold)
        else:
            # hand mode: decoupled from gold (item 6) -- no script backs this submission
            assert submits == [gold + 1], (i, submits, gold)
    hand = N_ANCHOR - n_build
    assert abs(hand / N_ANCHOR - ANCHOR_HAND_FRAC) < 0.1, (hand, ANCHOR_HAND_FRAC)
    print(f"  anchor: {N_ANCHOR} sessions, {n_build} build / {hand} hand (hand_frac="
          f"{ANCHOR_HAND_FRAC}); build submits == gold, hand submits == gold+1 (decoupled, item 6); "
          f"prompts policy-neutral; deterministic")


def _selftest_mechanics_bridge():
    """Mechanics-bridge guards: deterministic + arm-independent, correct gold on every submit/run_script
    return, N disclosed + RECURRENCE_NOTE present (matches the real eval framing), known tools only, and
    -- the key correctness property -- build timing is NOT degenerate (some commits at k=1 AND some at
    k>=2), so it cannot be read as teaching either an eager or a reserve policy. Also guards item 5
    (2026-07-06 corpus regeneration): k=1 commits must not all cluster in the first few absolute slots
    of the session (the position-clustering artifact that -- on top of the fixed-rationale bug --
    matched all 7 real-eval builds landing at problem #2 or #3)."""
    print("\n=== mechanics-bridge guard (Design A fix: policy-neutral long-context tool sessions) ===")
    mech = generate_mechanics_bridge()
    assert len(mech) == N_MECH, (len(mech), N_MECH)

    def _strip_ids(sessions):
        import copy
        s = copy.deepcopy(sessions)
        for sess in s:
            for m in sess:
                for tc in m.get("tool_calls", []):
                    tc["id"] = ""
                m.pop("tool_call_id", None)
        return s
    assert _strip_ids(generate_mechanics_bridge()) == _strip_ids(mech), \
        "mechanics bridge must be deterministic + arm-independent"

    known = {"write_script", "run_script", "submit_answer"}
    k1 = kge2 = 0
    k1_early_slots = k1_slots = 0     # item 5 guard: k=1 commits' ABSOLUTE position in the session
    for i, msgs in enumerate(mech):
        B = 2 if i % 2 == 0 else 3
        N, T = 8, 60
        slots, _ = build_stochastic_stream(StochasticStreamSpec(
            families=UNIFORM[:N], n_hot=B, T=T, budget=B, guarantee_trap_early=1.0,
            magnitude=100, seed=MECH_SEED_START + i))
        gold_by_slot = {s["slot_index"]: int(s["gold"]) for s in slots}
        builds = random_builds(slots, B, random.Random(MECH_SEED_START + i + 999983))
        steps = _walk(slots, builds, B)
        for s in steps:
            if s["action"] == "commit":
                k1 += (s["k"] == 1); kge2 += (s["k"] >= 2)
                if s["k"] == 1:
                    k1_slots += 1
                    k1_early_slots += s["slot_index"] < 10

        assert f"exactly {N} distinct" in msgs[0]["content"], "mechanics system prompt missing N"
        assert "recurring TYPES" in msgs[0]["content"], "mechanics system prompt missing RECURRENCE_NOTE"
        names = {tc["function"]["name"] for m in msgs for tc in m.get("tool_calls", [])}
        assert names <= known, (i, names)
        submits = [json.loads(tc["function"]["arguments"])["value"] for m in msgs
                   for tc in m.get("tool_calls", []) if tc["function"]["name"] == "submit_answer"]
        assert len(submits) == T, (i, len(submits), T)
        expected = [gold_by_slot[s["slot_index"]] + (1 if s["action"] == "wait" else 0)
                   for s in sorted(steps, key=lambda z: z["slot_index"])]
        assert submits == expected, (i, "submit mismatch (gold on commit/reuse, gold+1 on wait)")
        run_returns = [json.loads(m["content"])["return_value"] for m in msgs
                       if m["role"] == "tool" and "return_value" in m["content"]]
        # run_script fires on commit + reuse actions only (not wait) -- mirror _walk exactly rather than
        # re-deriving from `builds`, since a class's occurrences BEFORE its own commit slot are "wait"
        # turns with no run_script call.
        run_gold = [gold_by_slot[s["slot_index"]] for s in sorted(steps, key=lambda z: z["slot_index"])
                    if s["action"] in ("commit", "reuse")]
        assert run_returns == run_gold, (i, "run_script return != gold")

    tot = k1 + kge2
    frac_k1 = k1 / tot if tot else 0
    frac_early = k1_early_slots / k1_slots if k1_slots else 0
    print(f"  mechanics: {N_MECH} sessions, {tot} commits (k=1: {k1}, k>=2: {kge2}, "
          f"{frac_k1:.0%} first-sighting); k=1 commits landing in slot<10: {frac_early:.0%} "
          f"(de-clustered, item 5); commit/reuse submits/run_script == gold, wait submits == gold+1 "
          f"(item 6); N + RECURRENCE_NOTE present")
    assert k1 > 0 and kge2 > 0, "build timing is degenerate (all-k=1 or all-k>=2) -- not policy-neutral"
    assert frac_early <= 0.5, (frac_early, "k=1 commits still cluster in the first 10 slots -- "
                              "de-clustering (item 5) regressed")


def _selftest_error_recovery():
    """Error-recovery guards: deterministic + arm-independent, correct gold on every submit/run_script
    return, policy-neutral prompt (single-problem, no N/recurrence disclosure), known tools only ON THE
    FINAL (recovered) call of each session, all three patterns present and non-degenerate, and -- the
    key correctness property -- the harness-side text (FORMAT_REMINDER / unknown-tool / no-script
    errors) matches the actual driver.py / session_state.py strings byte-for-byte, not a paraphrase."""
    print("\n=== error-recovery guard (bad turn -> harness nudge -> recovery) ===")
    recovery = generate_error_recovery()
    assert len(recovery) == N_RECOVERY, (len(recovery), N_RECOVERY)

    def _strip_ids(sessions):
        import copy
        s = copy.deepcopy(sessions)
        for sess in s:
            for m in sess:
                for tc in m.get("tool_calls", []):
                    tc["id"] = ""
                m.pop("tool_call_id", None)
        return s
    assert _strip_ids(generate_error_recovery()) == _strip_ids(recovery), \
        "error-recovery bridge must be deterministic + arm-independent"

    known = set(_KNOWN_TOOL_NAMES)
    counts = {p: 0 for p in _RECOVERY_PATTERNS}
    for i, msgs in enumerate(recovery):
        fam = UNIFORM[i % len(UNIFORM)]
        pattern = _RECOVERY_PATTERNS[i % len(_RECOVERY_PATTERNS)]
        counts[pattern] += 1
        gold = int(ALL_FAMILIES[fam].make_member(random.Random(RECOVERY_SEED_START + i), 100)["gold"])
        sysp = msgs[0]["content"]
        for leak in ("exactly", "distinct", "recur"):     # policy-neutral: no recurrence/N framing
            assert leak not in sysp, f"error-recovery {i}: prompt leaks '{leak}' -- not policy-neutral"

        # the FINAL submit_answer must carry the gold value, and every submit in the session (there's
        # exactly one) must be correct -- the point is a correct recovery, not just any recovery.
        submits = [json.loads(tc["function"]["arguments"])["value"] for m in msgs
                   for tc in m.get("tool_calls", []) if tc["function"]["name"] == "submit_answer"]
        assert submits == [gold], (i, pattern, submits, gold)

        names_used = [tc["function"]["name"] for m in msgs for tc in m.get("tool_calls", []) or []]
        if pattern == "reminder":
            # msgs[0]=system, msgs[1]=user problem prompt, msgs[2]=the bad assistant turn (NO
            # tool_calls at all -- the actual collapse shape), msgs[3]=the injected FORMAT_REMINDER
            assert "tool_calls" not in msgs[2], (i, "reminder pattern's bad turn must carry no tool_calls")
            assert msgs[3] == {"role": "user", "content": FORMAT_REMINDER}, \
                (i, "FORMAT_REMINDER text must match driver.py byte-for-byte")
            assert names_used == ["submit_answer"], (i, names_used)
        elif pattern == "wrong_name":
            assert names_used == ["submit_answers", "submit_answer"], (i, names_used)
            assert "submit_answers" not in known, "test fixture assumption broken: name now registered"
            err = json.loads(msgs[3]["content"])
            assert err == _unknown_tool_error("submit_answers"), (i, "unknown-tool error text mismatch")
        elif pattern == "unknown_script":
            assert names_used == ["run_script", "write_script", "run_script", "submit_answer"], \
                (i, names_used)
            err = json.loads(msgs[3]["content"])
            assert err == _no_script_error(f"{fam}_solver", known=()), \
                (i, "no-script error text mismatch")
            run_returns = [json.loads(m["content"])["return_value"] for m in msgs
                           if m["role"] == "tool" and "return_value" in m["content"]]
            assert run_returns == [gold], (i, "run_script return != gold")
        assert set(names_used) <= known | {"submit_answers"}, (i, "unexpected tool name", names_used)

    for p in _RECOVERY_PATTERNS:
        assert counts[p] > 0, f"pattern {p} never appears -- degenerate corpus"
    print(f"  error-recovery: {N_RECOVERY} sessions, pattern counts={counts}; all final submits == gold; "
          f"FORMAT_REMINDER / unknown-tool / no-script text byte-matched to driver.py/session_state.py")


def _selftest_no_consecutive_assistant():
    """Regression guard (2026-07-06 diagnosis): NO generated session may contain two consecutive
    assistant-role messages with no intervening tool/user turn. Real eval (driver.run_session) always
    emits exactly ONE assistant dict per turn; a session that splits one turn's content and tool_calls
    across two separate assistant messages trains the model on a mid-conversation
    '<|im_start|>assistant\\n' it will never actually see at inference -- confirmed as the likely root
    cause of the Design A tool-eval collapse (the literal word "assistant" hallucinated as content in
    the majority of turns in 3/5 saved failing transcripts; the only corpus slice with this shape was
    the old `mechanics_bridge` 'reuse' branch, now fixed above)."""
    print("\n=== no-consecutive-assistant-turns guard (2026-07-06 collapse diagnosis) ===")
    urn_p, tool_p = generate_corpus("pistar")
    urn_e, tool_e = generate_corpus("eager")
    slices = {
        "urn_pistar": urn_p, "tool_bridge_pistar": tool_p, "urn_eager": urn_e, "tool_bridge_eager": tool_e,
        "anchor": generate_anchor(), "mechanics_bridge": generate_mechanics_bridge(),
        "error_recovery": generate_error_recovery(),
    }
    for name, sessions in slices.items():
        for i, msgs in enumerate(sessions):
            roles = [m["role"] for m in msgs]
            for j in range(len(roles) - 1):
                assert not (roles[j] == "assistant" and roles[j + 1] == "assistant"), \
                    f"{name}[{i}]: back-to-back assistant messages at index {j}/{j+1} -- merge into one"
        print(f"  {name}: {len(sessions)} sessions, no back-to-back assistant turns")


def _selftest_content_diversity():
    """CONTENT-DIVERSITY GUARD (2026-07-06 corpus regeneration, item 7): no slice may let a single
    literal assistant-content string dominate more than a modest fraction of its content-bearing
    turns. This is the actual coverage gap that let anchor_tool (2/150 sessions, 1.3% unique) and
    mechanics_bridge (3/1500 turns, 0.2% unique) collapse into verbatim repetition at eval undetected
    through two retrains -- every other guard here checks correctness/determinism/structure, none
    checked text diversity. Threshold (25%) sits above urn's own worst coincidental repeat (~2.4%,
    parameterized text that legitimately recurs when (k, budget, vocab) coincide) and above
    error_recovery's deliberately-small, intentionally-repeating set of 5 known malformed-content
    patterns (~20%, by design -- see _MALFORMED_CONTENT), but far below the old bugs' 48-70%."""
    print("\n=== content-diversity guard (2026-07-06: catches the fixed-literal-rationale bug class) ===")
    urn_p, tool_p = generate_corpus("pistar")
    urn_e, tool_e = generate_corpus("eager")
    slices = {
        "urn_pistar": urn_p, "tool_bridge_pistar": tool_p, "urn_eager": urn_e, "tool_bridge_eager": tool_e,
        "anchor": generate_anchor(), "mechanics_bridge": generate_mechanics_bridge(),
        "error_recovery": generate_error_recovery(),
    }
    for name, sessions in slices.items():
        contents = [m["content"] for msgs in sessions for m in msgs
                   if m["role"] == "assistant" and m.get("content")]
        if not contents:
            print(f"  {name}: 0 content-bearing assistant turns (bare tool calls only) -- OK")
            continue
        cnt = Counter(contents)
        top_str, top_n = cnt.most_common(1)[0]
        frac = top_n / len(contents)
        print(f"  {name}: {len(contents)} content-bearing turns, {len(cnt)} distinct "
              f"({len(cnt) / len(contents):.1%} unique), top string x{top_n} ({frac:.0%}): {top_str[:60]!r}")
        assert frac <= 0.25, (name, frac, "a single literal content string dominates >25% of turns "
                              "-- degenerate/repetitive content risks a verbatim-repetition collapse, "
                              "see docs/qwen-finetune-transfer-plan.md 'corpus regeneration'")


def _selftest_examples():
    print("\n=== example transcripts (A2, T=60 so pi* actually reserves) ===")
    urn_msgs = render_urn_session(URN_VOCAB[0], 8, 60, 3, URN_SEED_START, "pistar")
    print(f"\n--- urn (pistar), {len(urn_msgs)} messages ---")
    for m in urn_msgs[:11]:
        print(f"  [{m['role']}] {m['content'][:200]}")
    tool_msgs = render_tool_bridge_session(8, 60, 2, TOOL_SEED_START, "pistar")
    print(f"\n--- tool_bridge (pistar), {len(tool_msgs)} messages ---")
    for m in tool_msgs[:12]:
        content = m.get("content", "")[:140]
        tcs = m.get("tool_calls")
        line = f"  [{m['role']}] {content}"
        if tcs:
            line += f"  TOOL_CALLS={[(tc['function']['name'], tc['function']['arguments'][:80]) for tc in tcs]}"
        print(line)


def _selftest():
    _selftest_family_code()
    _selftest_decisions()
    _selftest_reserve_rate()
    _selftest_anchor()
    _selftest_mechanics_bridge()
    _selftest_error_recovery()
    _selftest_no_consecutive_assistant()
    _selftest_content_diversity()
    _selftest_examples()
    print("\nphase3_demos self-test OK")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--selftest", action="store_true")
    _args = _ap.parse_args()
    if _args.selftest:
        _selftest()
    else:
        print("Generating Phase 3 SFT corpus (urn + tool_bridge, pistar + eager) ...")
        write_corpus()
        print("done.")
