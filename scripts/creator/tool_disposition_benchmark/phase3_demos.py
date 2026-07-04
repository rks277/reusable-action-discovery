"""Phase 3 demonstration generator (docs/qwen-finetune-transfer-plan.md): synthesizes SFT training
transcripts for the bridge fine-tune. Two slices, each available under two label sources:

  - urn: diverse-surface-form urn/balls sessions (vocabulary/N/T/B varied per session).
  - tool_bridge: short tool-game sessions on the REAL eval pool (uniform-hard N=8, MAG=100), fresh
    seeds disjoint from the 2000-2011 eval seeds.
  - policy="pistar" (treatment): labels = the exact same-info optimum (ExactDP.policy_builds).
  - policy="eager" (control): labels = build-the-first-B-distinct-types-on-sight (pi_star.eager_builds).

A2 PROTOCOL (2026-07-03 audit): both slices DISCLOSE the exact number of distinct types N in the
system prompt (urn: "exactly N distinct <attr>s", where <attr> is the per-vocab attribute word --
color/material/metal/symbol/kind/suit/shape; tool: prompts.n_types_note) -- matching pi*'s own
information and the corrected --announce-n eval arm, so demonstrations are on the same footing as the
runs they'll be evaluated against. N/T/B ranges are chosen so pi* genuinely RESERVES (commits at
k>=2, not first-sight): short T / generous B collapses the exact DP toward eager, which would make
treatment demos indistinguishable from the eager control (see corpus-generation note below).

No LLM calls, no GPU -- everything is generated deterministically from exact_dp / family_kit /
stream_builder, so this step is free. Output: chat-format JSONL under runs/phase3_sft_data/.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.phase3_demos          # generate
  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.phase3_demos --selftest
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

from scripts.creator.tool_disposition_benchmark.exact_dp import ExactDP
from scripts.creator.tool_disposition_benchmark.family_kit import ALL_FAMILIES
from scripts.creator.tool_disposition_benchmark.pi_star import eager_builds
from scripts.creator.tool_disposition_benchmark.prompts import (
    RECURRENCE_NOTE, n_types_note, problem_prompt, system_prompt)
from scripts.creator.tool_disposition_benchmark.run_stream_session import slots_to_problems
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
    if action == "wait":
        return "This looks like a new problem type -- I'll just work it out directly for now."
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
        rationale = _tool_rationale(s["action"], fam, s["k"], s["t"], budget_before, policy)

        if s["action"] == "wait":
            tc = _tc("submit_answer", {"value": gold})
            messages.append({"role": "assistant", "content": rationale, "tool_calls": [tc]})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
                {"ok": True, "message": f"Recorded answer for problem {prob['idx']}."})})
            continue

        if s["action"] == "commit":
            name = f"{fam}_solver"
            script_of[s["class_id"]] = name
            tc = _tc("write_script", {"name": name, "code": FAMILY_CODE[fam]})
            messages.append({"role": "assistant", "content": rationale, "tool_calls": [tc]})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
                {"ok": True, "message": f"Saved script '{name}'.", "scripts": [name]})})
        else:  # reuse
            name = script_of[s["class_id"]]
            messages.append({"role": "assistant", "content": rationale})

        tc = _tc("run_script", {"name": name, "inputs": prob["inputs"]})
        messages.append({"role": "assistant", "tool_calls": [tc]})
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": json.dumps({"ok": True, "return_value": gold})})
        tc = _tc("submit_answer", {"value": gold})
        messages.append({"role": "assistant", "tool_calls": [tc]})
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
            {"ok": True, "message": f"Recorded answer for problem {prob['idx']}."})})
    return messages


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
N_URN, N_TOOL = 170, 6


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
            print(f"  {path}: {len(sessions)} sessions, ~{sum(toks):,} est. tokens "
                  f"(mean {sum(toks)/len(toks):.0f}/session)")


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
    submit_answer value must match the stream's true gold, and both system prompts must DISCLOSE N
    (A2 protocol) -- regression tripwire + correctness proof. Uses A2-realistic N/T/B (T>=60)."""
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
            msgs = render_tool_bridge_session(N, T, B, TOOL_SEED_START + i, policy)
            assert f"exactly {N} distinct" in msgs[0]["content"], "tool system prompt missing N"
            submits = [json.loads(tc["function"]["arguments"])["value"]
                       for m in msgs for tc in m.get("tool_calls", [])
                       if tc["function"]["name"] == "submit_answer"]
            assert len(submits) == T, (policy, i, len(submits), T)
            golds_in_order = [gold_by_slot[s["slot_index"]]
                              for s in sorted(slots, key=lambda z: z["slot_index"])]
            assert submits == golds_in_order, (policy, i, "submit != gold")
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
