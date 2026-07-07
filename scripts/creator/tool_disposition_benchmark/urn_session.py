"""Abstraction-gap control (the "urn / balls" isomorph of the tool task).

Byte-identical DECISION to the tool benchmark, stripped of the tool cover story: balls of N colors are
drawn one at a time (the SAME stochastic stream as the tool runs, paired by seed); at <=B draws the
model may KEEP the current ball, which collects it plus every remaining ball of that color for free
(= build-a-tool-then-reuse). Passed balls, and colors never kept, score nothing (= hand-solve at
a_hand=0). Reference pi* is literally the same exact-DP; keeps map onto `model_builds`.

Headline metric = KEEP LATENESS (does it keep a color on FIRST sighting = eager, or reserve for a
color that has recurred?) compared to the tool task's build lateness. If the model RESERVES here but
builds eagerly in the tool task -> the tool/coding framing suppresses recognition (recognition failure,
not competence). If it is eager here too -> domain-general greedy myopia.

Clear color labels (approved). A0 information: told T and B, NOT which colors are frequent nor N.

  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.urn_session
"""
from __future__ import annotations
import argparse, asyncio, json, re, statistics as st
from pathlib import Path
from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.stream_builder import (
    StochasticStreamSpec, build_stochastic_stream)
from scripts.creator.tool_disposition_benchmark.run_stream_session import CLAUDE
from scripts.creator.tool_disposition_benchmark.skirental_scorer import exact_pistar_report
from scripts.creator.tool_disposition_benchmark.pi_star import Costs, clairvoyant_builds
from scripts.creator.tool_disposition_benchmark.exact_dp import ExactDP

# same uniform-hard pool as the tool A1/capability runs -> only used to fix N and to draw streams.
UNIFORM = ["lcg", "modpow", "continued_frac", "crt_solve", "josephus", "quadratic_map_mod",
           "xorshift_steps", "matrix_power_mod"]
N, T, B, MAG, G = len(UNIFORM), 60, 3, 100, 1.0

_ap = argparse.ArgumentParser()
# --model accepts a Claude key (haiku/sonnet/opus) OR any raw model tag. A tag containing ":"
# (e.g. "qwen2.5-coder:7b") routes to Ollama via RawChat (set OLLAMA_BASE_URL); local models are
# free, so pricing/spend-guard are disabled for them.
_ap.add_argument("--model", default="haiku")
_ap.add_argument("--conc", type=int, default=None, help="override concurrency (Ollama serializes; use 2-4)")
_ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override seed list")
_ap.add_argument("--temp", type=float, default=None,
                 help="decoding temperature (0 = greedy; default = provider default ~0.7). Use 0 to "
                      "read a learned/fine-tuned policy cleanly without sampling noise.")
_ap.add_argument("--announce-n", action="store_true",
                 help="A2 arm: tell the model the exact number of distinct colors N (matches pi*'s "
                      "own information -- pi*'s Dirichlet-multinomial predictive (alpha+k)/(N*alpha+t) "
                      "is CONSTRUCTED WITH exact N, so without this flag the 'same-information' "
                      "regret comparison is not actually same-information; see 2026-07-03 audit)")
_ARGS = _ap.parse_known_args()[0]
MODEL_KEY = _ARGS.model
MODEL_STR = CLAUDE.get(MODEL_KEY, MODEL_KEY)      # Claude key -> id; else pass the raw tag through
IS_LOCAL = MODEL_KEY not in CLAUDE                # non-Claude => Ollama/vLLM, free
ANNOUNCE_N = _ARGS.announce_n
_safe = MODEL_KEY.replace(":", "_").replace("/", "_")
BASE = Path(f"runs/urn_{_safe}" + ("_n-announced" if ANNOUNCE_N else ""))
SEEDS = _ARGS.seeds if _ARGS.seeds else list(range(2000, 2012))  # 12 -- same streams as Haiku/Opus urn
CAP_USD, EST = 12.0, 0.6
CONC = _ARGS.conc if _ARGS.conc else (3 if IS_LOCAL else 6)      # Ollama loads one model; keep low
_PRICES = {"haiku": (1.0, 5.0, 0.10, 1.25), "sonnet": (3.0, 15.0, 0.30, 3.75),
           "opus": (5.0, 25.0, 0.50, 6.25)}       # $/1e6 (in, out, cache_read, cache_write)
IN, OUT, CR, CW = _PRICES.get(MODEL_KEY, (0.0, 0.0, 0.0, 0.0))   # local models: no cost
PALETTE = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]
PAIR_TOOL = MODEL_KEY == "haiku"          # the A1 tool baseline (runs/arm_a1_announce) is Haiku-only

def render_system(T: int, B: int, N: int, announce_n: bool) -> str:
    """Parameterized system prompt (extracted 2026-07-07 so `rl_rollout.py` can reuse it byte-for-byte
    instead of re-deriving/paraphrasing it -- this project's standing lesson is that a near-miss format
    is as bad as no fix at all)."""
    n_note = (f"There are exactly {N} distinct colors in the bag, but you are not told which colors are "
              "common or rare, nor how many balls of each color there are; you learn the rates only from "
              "the draws you have seen.") if announce_n else (
              "You do not know in advance how many colors there are or how many balls of each color are "
              "in the bag; you learn only from the draws you have seen.")
    return f"""You are playing a game with a bag of colored balls.

Balls are drawn from the bag one at a time; there will be {T} draws in total. Each ball has a color.

At any draw you may KEEP the ball currently shown, or PASS on it. You may KEEP at most {B} balls in \
the entire game.

When you KEEP a ball of some color, you immediately collect that ball AND every remaining ball of that \
same color that is drawn later in the game -- all of them are added to your total automatically and for \
free. (So a color can be kept at most once; keeping it locks in all of its future draws.)

The bag is unchanged by your choices: all {T} draws still happen in order. Whenever a ball of a color \
you have ALREADY kept is drawn again, I will simply tell you it was collected automatically (+1) and we \
move straight on -- you do not choose again for that color. You only make a KEEP/PASS choice on a ball \
whose color you have not yet kept.

If you PASS a ball it is gone and scores nothing. Balls of a color you never keep score nothing.

Your goal is to MAXIMIZE the total number of balls you collect by the end of the game.

{n_note} Decide as each ball appears -- decisions are final.

For each draw, think briefly (one or two sentences), then end your reply with EXACTLY one line:
DECISION: KEEP
or
DECISION: PASS"""


_N_NOTE = (f"There are exactly {N} distinct colors in the bag, but you are not told which colors are "
          "common or rare, nor how many balls of each color there are; you learn the rates only from "
          "the draws you have seen.") if ANNOUNCE_N else (
          "You do not know in advance how many colors there are or how many balls of each color are "
          "in the bag; you learn only from the draws you have seen.")

SYSTEM = render_system(T, B, N, ANNOUNCE_N)


def cost_of(turn_usages):
    return (sum(t.get("input_tokens", 0) for t in turn_usages) * IN
            + sum(t.get("output_tokens", 0) for t in turn_usages) * OUT
            + sum(t.get("cache_read_tokens", 0) for t in turn_usages) * CR
            + sum(t.get("cache_write_tokens", 0) for t in turn_usages) * CW) / 1e6


def parse_decision(text: str):
    """Return (decision, how). Robust to small models that emit a bare 'KEEP'/'PASS' + rambling
    (which often contains the other word, e.g. '...nothing to pass') instead of the DECISION: line.
    Priority: (1) explicit DECISION: line (last wins); (2) a LEADING KEEP/PASS token; (3) first
    standalone token anywhere; (4) default PASS. `how` == 'default' flags a true parse failure."""
    t = (text or "").strip()
    hits = re.findall(r"DECISION:\s*(KEEP|PASS)", t, re.I)
    if hits:
        return hits[-1].upper(), "tag"
    m = re.match(r"[\W]*(KEEP|PASS)\b", t, re.I)
    if m:
        return m.group(1).upper(), "lead"
    m2 = re.search(r"\b(KEEP|PASS)\b", t, re.I)
    if m2:
        return m2.group(1).upper(), "scan"
    return "PASS", "default"


async def run_episode(client, model, slots: list[dict], *, T: int, B: int, system: str,
                      temperature: float | None = None, palette: list[str] = PALETTE) -> dict:
    """Run ONE urn episode against a pre-built `slots` stream -- no file I/O, no caching, so callers
    (the CLI eval below, and `rl_rollout.py`'s RL rollout collector) own persistence. Extracted from
    `run_one` (2026-07-07, RL Phase 1) so both reuse the EXACT same per-draw loop byte-for-byte, rather
    than a paraphrased reimplementation (this project's standing lesson: a near-miss reimplementation is
    as bad as no fix at all). Returns `{kept, collected, unparsed, turn_usages, color_of_class,
    transcript, messages}` -- `messages` (OpenAI-format user/assistant turns) is new relative to the
    original inline version, needed for RL's `build_example`-based log-prob masking."""
    # clear color labels assigned by FIRST-APPEARANCE order (leaks nothing about rate/role)
    color = {}
    for s in sorted(slots, key=lambda z: z["slot_index"]):
        if s["class_id"] not in color:
            color[s["class_id"]] = palette[len(color)]

    messages, turn_usages, transcript = [], [], []
    kept: dict[int, int] = {}                  # class_id -> class_position at which kept
    unparsed = 0
    budget_left = B
    pending: list[tuple[int, str]] = []        # kept-color draws seen since the last decision prompt
    for s in sorted(slots, key=lambda z: z["slot_index"]):
        cid, pos, n = s["class_id"], s["class_position"], s["slot_index"] + 1
        if cid in kept:                        # already kept -> auto-collected IN PLACE. Still a real
            pending.append((n, color[cid]))    #   draw: the model is TOLD (below), so it observes the
            continue                           #   full T-length stream -- OBSERVATION PARITY with the
                                                #   tool player (no removal, no gappy < T view).
        if budget_left == 0:                   # no keeps left -> no decisions remain; tally the rest
            break
        pre = ""
        if pending:                            # report the kept-color draws that occurred since the
            evs = "; ".join(f"draw {dn}: another {col} ball (already kept -> auto-collected, +1)"
                            for dn, col in pending)
            pre = f"(Since your last choice: {evs}.) "
            pending = []
        user = (f"{pre}Draw {n} of {T}: a {color[cid]} ball appears. "
                f"You have {budget_left} keep(s) left. KEEP or PASS?")
        messages.append({"role": "user", "content": user})
        try:
            reply = await client.chat(model, system, messages, max_tokens=512, temperature=temperature)
            u = dict(client.last_usage or {})
        except Exception as e:
            reply, u = f"DECISION: PASS   [error {type(e).__name__}]", {}
        turn_usages.append(u)
        messages.append({"role": "assistant", "content": reply})
        dec, how = parse_decision(reply)
        if how == "default":                       # no KEEP/PASS token at all -> true parse failure
            unparsed += 1
        transcript.append({"slot": s["slot_index"], "color": color[cid], "class_id": cid,
                           "class_position": pos, "prompt": user, "decision": dec, "how": how,
                           "reply": reply})
        if dec == "KEEP":
            kept[cid] = pos
            budget_left -= 1
    # tally any remaining draws of colors kept before budget ran out (loop `continue`d past them
    # only while budget>0; count draws that occur AFTER budget exhaustion too)
    collected = sum(1 for s in slots if s["class_id"] in kept and s["class_position"] >= kept[s["class_id"]])

    return {"kept": kept, "collected": collected, "budget": B, "unparsed": unparsed,
            "turn_usages": turn_usages, "color_of_class": color, "transcript": transcript,
            "messages": messages}


async def run_one(client, model, seed):
    d = BASE / f"seed_{seed}"
    if (d / "session.json").exists():
        return 0.0, "cached"
    slots, meta = build_stochastic_stream(StochasticStreamSpec(
        families=UNIFORM, n_hot=B, T=T, budget=B, guarantee_trap_early=G, magnitude=MAG, seed=seed))
    d.mkdir(parents=True, exist_ok=True)
    (d / "stream.json").write_text(json.dumps(slots, indent=2))
    (d / "meta.json").write_text(json.dumps(meta, indent=2))

    row = await run_episode(client, model, slots, T=T, B=B, system=SYSTEM, temperature=_ARGS.temp)
    row = {"seed": seed, "model_key": MODEL_KEY, **row}
    (d / "session.json").write_text(json.dumps(row, indent=2))
    return cost_of(row["turn_usages"]), "ran"


async def main():
    load_dotenv()
    model = MODEL_STR; BASE.mkdir(parents=True, exist_ok=True)
    client = RawChat()
    cumulative = 0.0; inflight = 0; idx = 0; paused = False; lock = asyncio.Lock()

    async def worker():
        nonlocal cumulative, inflight, idx, paused
        while True:
            async with lock:
                if paused or idx >= len(SEEDS):
                    return
                d = BASE / f"seed_{SEEDS[idx]}"
                will_run = not (d / "session.json").exists()
                if will_run and cumulative + (inflight + 1) * EST > CAP_USD:
                    paused = True; return
                seed = SEEDS[idx]; idx += 1; inflight += 1
            try:
                cost, status = await run_one(client, model, seed)
            except Exception as e:
                cost, status = 0.0, f"ERR:{type(e).__name__}"
            async with lock:
                inflight -= 1; cumulative += cost
                print(f"  [seed {seed}] {status:>6}  ${cost:.3f}  cumulative=${cumulative:.2f}", flush=True)

    print(f"URN abstraction-gap control: {MODEL_KEY}, uniform N={N}, g={G}, {len(SEEDS)} seeds "
          f"(clear labels, cap=${CAP_USD}) ...", flush=True)
    await asyncio.gather(*(worker() for _ in range(CONC)))
    print(f"\n==== {'PAUSED' if paused else 'COMPLETED'}: urn spend ${cumulative:.2f} ====", flush=True)
    report()


def _balls_collected(slots, builds):
    """Total balls a policy collects -- the urn's LITERAL objective (what the model is told to
    maximize): for each kept color, its occurrences from the keep position onward (current + all
    future same-color); un-kept colors collect nothing. Balls-regret (pi* - model) equals the
    reuse-deficit ΔM, i.e. the utility regret divided by (100*a_script + 78.7) = 178.7 here -- same
    signal, more interpretable units (see docs/online-tool-investment discussion)."""
    sizes = {}
    for s in slots:
        sizes[s["class_id"]] = sizes.get(s["class_id"], 0) + 1
    return sum(sizes[cid] - b + 1 for cid, b in builds.items() if b is not None)


def make_costs_and_dp(N: int, T: int, B: int) -> tuple[Costs, ExactDP]:
    """The urn's cost model + exact-DP reference, as a design constant of {N,T,B} (extracted 2026-07-07,
    RL Phase 1, so `rl_reward.py` builds the identical reference `report()` uses, not a re-derived one)."""
    # costs: uniform-hand a_hand=0 (passing collects nothing); Haiku A0 token constants (same as pi_star)
    costs = Costs(a_hand={f: 0.0 for f in UNIFORM}, h=987, C=308, r=200, R=100.0, lam=0.1,
                  default_a_hand=0.0)
    a_repr = st.mean(costs.ah(f) for f in UNIFORM)
    dp = ExactDP(costs.R * a_repr - costs.lam * costs.h, costs.u_build(), costs.u_reuse(),
                 N, T, B, alpha=1.0, cap=3)
    return costs, dp


def report():
    costs, dp = make_costs_and_dp(N, T, B)

    lateness, first_sight, nb, match, nseed, regs, mtr, ptr, pos_reg, unp = [], 0, 0, 0, 0, [], [], [], 0, 0
    mballs, pballs, cballs = [], [], []          # balls collected: model / pi* / clairvoyant, per seed
    for seed in SEEDS:
        d = BASE / f"seed_{seed}"
        if not (d / "session.json").exists():
            continue
        slots = json.loads((d / "stream.json").read_text())
        row = json.loads((d / "session.json").read_text())
        kept = {int(k): v for k, v in row["kept"].items()}
        distinct = {s["class_id"] for s in slots}
        model_builds = {cid: kept.get(cid) for cid in distinct}
        # first-appearance order of colors
        firstseen, seen = [], set()
        for s in sorted(slots, key=lambda z: z["slot_index"]):
            if s["class_id"] not in seen:
                seen.add(s["class_id"]); firstseen.append(s["class_id"])
        for cid, p in kept.items():
            lateness.append(p - 1); nb += 1; first_sight += (p == 1)
        built = set(kept)
        if built and built == set(firstseen[:len(built)]):
            match += 1
        rep = exact_pistar_report(slots, costs, B, N, T, UNIFORM, model_builds, dp=dp)
        regs.append(rep["regret"]); mtr.append(rep["model_traps_built"]); ptr.append(rep["pistar_traps_built"])
        pos_reg += rep["regret"] > 0
        # balls collected (urn's own objective) vs the SAME pi*; clairvoyant = hindsight best (for context)
        mballs.append(_balls_collected(slots, model_builds))
        pballs.append(_balls_collected(slots, dp.policy_builds(slots)))
        cballs.append(_balls_collected(slots, clairvoyant_builds(slots, B)))
        unp += row.get("unparsed", 0)
        nseed += 1

    print(f"\n==== URN FIDELITY (n={nseed} seeds, {nb} keeps) ====")
    if nb:
        print(f"  keeps at FIRST SIGHT (lateness 0): {first_sight}/{nb} = {first_sight/nb:.0%}")
        print(f"  mean keep lateness = {st.mean(lateness):.3f}  max = {max(lateness)}")
        print(f"  kept-set == first-B-distinct: {match}/{nseed} = {match/nseed:.0%}")
        print(f"  keeps/seed = {nb/nseed:.2f}   unparsed decisions = {unp}")
    if regs:
        se = (st.stdev(regs) / len(regs) ** 0.5) if len(regs) > 1 else 0.0
        print(f"\n==== URN vs exact pi* (same DP as tool task) ====")
        print(f"  regret mean={st.mean(regs):.0f} +/- {se:.0f}  pos={pos_reg}/{len(regs)}  "
              f"model_traps/seed={st.mean(mtr):.2f}  pi*_traps/seed={st.mean(ptr):.2f}")
    if pballs:
        breg = [p - m for p, m in zip(pballs, mballs)]
        bse = (st.stdev(breg) / len(breg) ** 0.5) if len(breg) > 1 else 0.0
        mb, pb = st.mean(mballs), st.mean(pballs)
        pct = 100 * mb / pb if pb else float("nan")
        print(f"\n==== URN BALLS COLLECTED (urn's own objective; balls-regret = utility regret / 178.7) ====")
        print(f"  balls/seed: model={mb:.1f}  pi*={pb:.1f}  clairvoyant(hindsight)={st.mean(cballs):.1f}")
        print(f"  balls-regret vs pi* = {st.mean(breg):.1f} +/- {bse:.1f} /seed  "
              f"(model collects {pct:.0f}% of pi*'s)")
    # ---- PAIRED A1 tool baseline: same seeds/streams/disclosure, only the coding framing differs.
    #      The A1 announce run is Haiku-only, so this pairing is skipped for other models. ----
    if not PAIR_TOOL:
        print(f"\n  (no paired tool baseline for {MODEL_KEY}: A1 announce was Haiku-only. "
              f"Compare urn behavior across models on the same seeds instead.)")
    else:
        from scripts.creator.tool_disposition_benchmark.skirental_scorer import (
            actions_from_session, model_builds_from_actions)
        TOOL = Path("runs/arm_a1_announce")
        tregs, tlate, tfs, tnb, tn = [], [], 0, 0, 0
        for seed in SEEDS:
            td = TOOL / f"seed_{seed}"
            if not (td / "sessions.jsonl").exists():
                continue
            tslots = json.loads((td / "stream.json").read_text())
            tsess = json.loads((td / "sessions.jsonl").read_text().splitlines()[0])
            tmb = model_builds_from_actions(actions_from_session(tsess, tslots))
            trep = exact_pistar_report(tslots, costs, B, N, T, UNIFORM, tmb, dp=dp)
            tregs.append(trep["regret"])
            for v in tmb.values():
                if v is not None:
                    tlate.append(v - 1); tnb += 1; tfs += (v == 1)
            tn += 1
        if tregs:
            tse = (st.stdev(tregs) / len(tregs) ** 0.5) if len(tregs) > 1 else 0.0
            print(f"\n==== PAIRED A1 TOOL baseline (same seeds {SEEDS[0]}-{SEEDS[-1]}, n={tn}) ====")
            print(f"  builds at FIRST SIGHT: {tfs}/{tnb} = {tfs/tnb:.0%}   mean build lateness = {st.mean(tlate):.3f}")
            print(f"  regret vs pi* mean={st.mean(tregs):.0f} +/- {tse:.0f}")
            print(f"\n  >>> PAIRED GAP (urn vs A1-tool, identical streams+disclosure):")
            print(f"      lateness  {st.mean(lateness):.2f} (urn)  vs  {st.mean(tlate):.2f} (tool)")
            print(f"      regret    {st.mean(regs):.0f} (urn)  vs  {st.mean(tregs):.0f} (tool)")

    # ---- eyeball one full seed: verify the auto-collect notices read correctly ----
    seed0 = next((s for s in SEEDS if (BASE / f"seed_{s}" / "session.json").exists()), None)
    if seed0 is not None:
        row = json.loads((BASE / f"seed_{seed0}" / "session.json").read_text())
        print(f"\n==== TRANSCRIPT echo (seed {seed0}, kept={row['kept']}) ====")
        for t in row["transcript"]:
            print(f"  [slot {t['slot']:>2}] PROMPT: {t['prompt']}")
            print(f"            -> {t['decision']:<4} | {t['reply'].replace(chr(10),' ')[:160]}")


if __name__ == "__main__":
    asyncio.run(main())
