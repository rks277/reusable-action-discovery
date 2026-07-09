"""Shared, argv-independent core for the urn abstraction-gap family (`urn_session.py`, and
`urn_tool_session.py`'s tool-calling variant). Extracted 2026-07-09 so the two harnesses reuse the
EXACT same vocab/template/scoring/reporting logic byte-for-byte instead of a paraphrased
reimplementation (this project's standing lesson: a near-miss reimplementation is as bad as no fix
at all). No argparse, no CLI, no module-level side effects tied to a particular script's argv --
safe to import from any caller regardless of that caller's own command-line flags.
"""
from __future__ import annotations
import asyncio, json, statistics as st
from pathlib import Path

from scripts.creator.tool_disposition_benchmark.skirental_scorer import exact_pistar_report
from scripts.creator.tool_disposition_benchmark.pi_star import Costs, clairvoyant_builds
from scripts.creator.tool_disposition_benchmark.exact_dp import ExactDP

TRANSPORT_RETRIES = 2       # extra attempts beyond the first, on a genuine API/transport exception
TRANSPORT_BACKOFF_S = 1.5   # seconds, multiplied by attempt number (linear backoff)


async def call_with_retry(make_call, *, retries: int = TRANSPORT_RETRIES,
                          backoff: float = TRANSPORT_BACKOFF_S):
    """Run one API call, retrying on a genuine transport/exception failure -- NOT on a model
    decision failure. `make_call` is a zero-arg callable returning a FRESH awaitable each time it's
    invoked (a coroutine object can't be re-awaited on retry). Returns `(result, None)` on success,
    or `(None, exception)` once `retries` extra attempts are exhausted.

    Added 2026-07-09 (framing-ladder strengthening review) to separate two failure modes the ladder's
    confirmatory analysis must not conflate: a TRANSPORT/API failure (worth retrying -- it says
    nothing about the model's disposition) versus a genuine MODEL decision failure (no retry -- see
    `urn_session.parse_decision` / `urn_tool_session._resolve_decision`, whose no-retry contract is
    what keeps the text-vs-tool-call modality comparison fair). Callers must flag an exhausted-retry
    result as `how="error"`, distinct from `how="default"` (a real parse/no-call failure), so both
    still count toward the same unresolved-decision budget without one silently masquerading as an
    ordinary decision (the bug this fixes: `urn_session.py` used to inject a literal
    `"DECISION: PASS   [error ...]"` string on a transport failure, which `parse_decision` then read
    as a normal tag-parsed PASS -- invisible in the `unparsed` count)."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return (await make_call()), None
        except Exception as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(backoff * (attempt + 1))
    return None, last_exc

# same uniform-hard pool as the tool A1/capability runs -> only used to fix N and to draw streams.
UNIFORM = ["lcg", "modpow", "continued_frac", "crt_solve", "josephus", "quadratic_map_mod",
           "xorshift_steps", "matrix_power_mod"]
N, T, B, MAG, G = len(UNIFORM), 60, 3, 100, 1.0

# vessel/item/attr words + a palette of N attr-values, in FIRST-APPEARANCE order (leaks nothing about
# rate/role -- same guarantee as the original hard-coded PALETTE). attr words are consonant-initial so
# "a {attr}" stays grammatical (matches phase3_demos.py's URN_VOCAB convention). treasure_chest/quiver/
# cauldron are each held out on ALL THREE axes (vessel, item, attr) from both RL's "ball"-only training
# and SFT's 7-vocab rotation (bag/ball/color, jar/ticket/material, box/coin/metal, drawer/tile/symbol,
# pouch/gem/kind, crate/card/suit, bin/token/shape) -- treasure_chest reuses the item word "coin" from
# SFT's box/coin/metal (a strictly weaker held-out claim for that one word); quiver/cauldron share no
# word with any SFT vocab on any axis.
VOCAB = {
    "ball": {"vessel": "bag", "item": "ball", "attr": "color", "adj": "colored ",
             "palette": ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]},
    "treasure_chest": {"vessel": "treasure chest", "item": "coin", "attr": "crest", "adj": "",
                        "palette": ["skull", "anchor", "dragon", "kraken", "phoenix", "serpent",
                                    "raven", "crown"]},
    "quiver": {"vessel": "quiver", "item": "arrow", "attr": "feather", "adj": "",
               "palette": ["hawk", "raven", "owl", "falcon", "eagle", "kestrel", "heron", "crane"]},
    "cauldron": {"vessel": "cauldron", "item": "potion", "attr": "label", "adj": "",
                 "palette": ["frost", "ember", "venom", "mist", "thorn", "spark", "gale", "shade"]},
}


def _art(word: str) -> str:
    """Indefinite article for a templated noun -- vocab words aren't all guaranteed consonant-initial
    (e.g. quiver's item "arrow"), so the article must be computed, not hard-coded 'a' (2026-07-09 fix:
    the hard-coded version silently produced 'a arrow' throughout)."""
    return "an" if word[:1].lower() in "aeiou" else "a"


_DECISION_LINE_INSTRUCTION = """For each draw, think briefly (one or two sentences), then end your reply with EXACTLY one line:
DECISION: KEEP
or
DECISION: PASS"""


def render_system(T: int, B: int, N: int, announce_n: bool, vocab: dict = VOCAB["ball"],
                  response_instruction: str | None = None) -> str:
    """Parameterized system prompt (extracted 2026-07-07 so `rl_rollout.py` can reuse it byte-for-byte
    instead of re-deriving/paraphrasing it -- this project's standing lesson is that a near-miss format
    is as bad as no fix at all). `vocab` generalizes the surface form (2026-07-09): with the default
    "ball" vocab this is byte-identical to the original hard-coded text; RL/SFT call sites never pass
    `vocab`, so their behavior is untouched. `response_instruction` (2026-07-09, tool-call modality
    fix) swaps the closing paragraph -- default (None) reproduces the original DECISION-line text
    byte-for-byte; `urn_tool_session.py` passes a tool-calling instruction instead. This exists
    because a smoke test found the closing "end your reply with EXACTLY one line: DECISION: KEEP/PASS"
    text is a direct, explicit competing instruction that overrides tool use even under
    `tool_choice="required"` (verified: Ollama's shim does not enforce it as a hard grammar
    constraint -- the model's own conditional distribution, driven by whatever the system prompt
    literally says, decides whether a tool gets called at all)."""
    vessel, item, attr, adj = vocab["vessel"], vocab["item"], vocab["attr"], vocab.get("adj", "")
    item_cap = item[:1].upper() + item[1:]
    vessel_a, item_a, attr_a = _art(vessel), _art(item), _art(attr)
    n_note = (f"There are exactly {N} distinct {attr}s in the {vessel}, but you are not told which "
              f"{attr}s are common or rare, nor how many {item}s of each {attr} there are; you learn "
              "the rates only from the draws you have seen.") if announce_n else (
              f"You do not know in advance how many {attr}s there are or how many {item}s of each "
              f"{attr} are in the {vessel}; you learn only from the draws you have seen.")
    return f"""You are playing a game with {vessel_a} {vessel} of {adj}{item}s.

{item_cap}s are drawn from the {vessel} one at a time; there will be {T} draws in total. Each {item} \
has {attr_a} {attr}.

At any draw you may KEEP the {item} currently shown, or PASS on it. You may KEEP at most {B} {item}s in \
the entire game.

When you KEEP {item_a} {item} of some {attr}, you immediately collect that {item} AND every remaining \
{item} of that same {attr} that is drawn later in the game -- all of them are added to your total \
automatically and for free. (So {attr_a} {attr} can be kept at most once; keeping it locks in all of \
its future draws.)

The {vessel} is unchanged by your choices: all {T} draws still happen in order. Whenever {item_a} {item} \
of {attr_a} {attr} you have ALREADY kept is drawn again, I will simply tell you it was collected \
automatically (+1) and we move straight on -- you do not choose again for that {attr}. You only make a \
KEEP/PASS choice on {item_a} {item} whose {attr} you have not yet kept.

If you PASS {item_a} {item} it is gone and scores nothing. {item_cap}s of {attr_a} {attr} you never keep \
score nothing.

Your goal is to MAXIMIZE the total number of {item}s you collect by the end of the game.

{n_note} Decide as each {item} appears -- decisions are final.

{response_instruction if response_instruction is not None else _DECISION_LINE_INSTRUCTION}"""


def resolve_zero_arg_decision(tool_calls: list[dict], valid_names: tuple[str, str]) -> tuple[str, str, list[dict]]:
    """Single-pass, no-retry decision resolution from one turn's tool calls, generalized over the
    active zero-argument tool-pair name -- `("keep", "pass")` for the urn tool-call rung (R1),
    `("claim_solver", "skip_solver")` for the real-problem declarative rung (R2) -- so both rungs
    share this logic BYTE-FOR-BYTE rather than risk a near-miss reimplementation drifting apart
    (this project's standing lesson: extracted 2026-07-09 from `urn_tool_session.py`'s original
    `_resolve_decision`, generalized only enough to swap the name pair; no other behavior changed).
    `valid_names` is `(commit_name, decline_name)` -- the second is the no-op/default side.

    Returns (decision, how, results) where `decision` is the resolved tool name, UPPERCASED, and
    `results` is a list of {"tool_call_id", "content"} to append as role="tool" messages -- one per
    call in `tool_calls`, so every id gets a matching result regardless of which one is decisive
    (required by the tool-calling protocol).

    - no tool call at all -> (decline_name.upper(), "default", [])
    - exactly one of the pair (ignoring unknown names) -> (that decision, "tool", [ack])
    - both of the pair called -> (first one in call order, "both", [ack, ack-ignored])
    - only unknown tool name(s) -> (decline_name.upper(), "unknown", [error per call])
    """
    commit_name, decline_name = valid_names
    if not tool_calls:
        return decline_name.upper(), "default", []
    valid = [tc for tc in tool_calls if tc["name"] in valid_names]
    if not valid:
        return decline_name.upper(), "unknown", [{"tool_call_id": tc["id"],
                                    "content": json.dumps({"ok": False,
                                                           "error": f"no such tool '{tc['name']}'. "
                                                                    f"Available tools: {commit_name}, "
                                                                    f"{decline_name}."})}
                                   for tc in tool_calls]
    decisive = valid[0]
    decision = decisive["name"].upper()
    how = "tool" if len(valid) == 1 else "both"
    results = []
    for tc in tool_calls:
        if tc is decisive:
            results.append({"tool_call_id": tc["id"], "content": json.dumps({"ok": True})})
        elif tc in valid:
            results.append({"tool_call_id": tc["id"],
                            "content": json.dumps({"ok": False,
                                                   "error": f"both {commit_name} and {decline_name} "
                                                            f"called in the same turn; only the first "
                                                            f"('{decision}') was applied."})})
        else:
            results.append({"tool_call_id": tc["id"],
                            "content": json.dumps({"ok": False,
                                                   "error": f"no such tool '{tc['name']}'. "
                                                            f"Available tools: {commit_name}, "
                                                            f"{decline_name}."})})
    return decision, how, results


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
    RL Phase 1, so `rl_reward.py` builds the identical reference `report_summary()` uses, not a
    re-derived one)."""
    # costs: uniform-hand a_hand=0 (passing collects nothing); Haiku A0 token constants (same as pi_star)
    costs = Costs(a_hand={f: 0.0 for f in UNIFORM}, h=987, C=308, r=200, R=100.0, lam=0.1,
                  default_a_hand=0.0)
    a_repr = st.mean(costs.ah(f) for f in UNIFORM)
    dp = ExactDP(costs.R * a_repr - costs.lam * costs.h, costs.u_build(), costs.u_reuse(),
                 N, T, B, alpha=1.0, cap=3)
    return costs, dp


def _report_one(vocab_key: str, costs, dp, base_dir_fn, seeds, modality_tag: str) -> dict | None:
    """Per-vocab fidelity/regret block. Returns a summary dict for cross-vocab pooling, or None if
    this vocab has no completed seeds. `base_dir_fn`/`seeds` are parameters (not module globals) so
    this is reusable across scripts with different argv/CLI surfaces (2026-07-09 extraction)."""
    bd = base_dir_fn(vocab_key)
    lateness, first_sight, nb, match, nseed, regs, mtr, ptr, pos_reg, unp = [], 0, 0, 0, 0, [], [], [], 0, 0
    mballs, pballs, cballs = [], [], []          # balls collected: model / pi* / clairvoyant, per seed
    for seed in seeds:
        d = bd / f"seed_{seed}"
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

    tag = f" [{modality_tag}]" if modality_tag else ""
    if nseed == 0:
        print(f"\n==== [{vocab_key}]{tag} no completed seeds ====")
        return None

    print(f"\n==== [{vocab_key}]{tag} URN FIDELITY (n={nseed} seeds, {nb} keeps) ====")
    if nb:
        print(f"  keeps at FIRST SIGHT (lateness 0): {first_sight}/{nb} = {first_sight/nb:.0%}")
        print(f"  mean keep lateness = {st.mean(lateness):.3f}  max = {max(lateness)}")
        print(f"  kept-set == first-B-distinct: {match}/{nseed} = {match/nseed:.0%}")
        print(f"  keeps/seed = {nb/nseed:.2f}   unparsed decisions = {unp}")
    if regs:
        se = (st.stdev(regs) / len(regs) ** 0.5) if len(regs) > 1 else 0.0
        print(f"  regret mean={st.mean(regs):.0f} +/- {se:.0f}  pos={pos_reg}/{len(regs)}  "
              f"model_traps/seed={st.mean(mtr):.2f}  pi*_traps/seed={st.mean(ptr):.2f}")
    breg, mb, pb, pct = None, None, None, None
    if pballs:
        breg = [p - m for p, m in zip(pballs, mballs)]
        bse = (st.stdev(breg) / len(breg) ** 0.5) if len(breg) > 1 else 0.0
        mb, pb = st.mean(mballs), st.mean(pballs)
        pct = 100 * mb / pb if pb else float("nan")
        print(f"  balls/seed: model={mb:.1f}  pi*={pb:.1f}  clairvoyant(hindsight)={st.mean(cballs):.1f}  "
              f"balls-regret={st.mean(breg):.1f} +/- {bse:.1f} ({pct:.0f}% of pi*'s)")

    return {"vocab": vocab_key, "nseed": nseed, "nb": nb,
            "first_sight_pct": (first_sight / nb) if nb else float("nan"),
            "mean_lateness": st.mean(lateness) if lateness else float("nan"),
            "regret_mean": st.mean(regs) if regs else float("nan"),
            "balls_regret_mean": st.mean(breg) if breg else float("nan"),
            "pct_of_pistar": pct}


def report_summary(vocab_keys: list[str], base_dir_fn, seeds: list[int], *, model_key: str,
                   pair_tool: bool = False, modality_label: str = "") -> list[dict]:
    """Full report: per-vocab fidelity/regret blocks, the pooled cross-vocab average (when more than
    one vocab is requested), the paired A1-tool-baseline comparison (Haiku only), and a per-vocab
    transcript echo. `modality_label` (2026-07-09, e.g. "TOOL") is printed alongside each vocab
    header so a later combined report can tell text-decision rows apart from tool-call-decision rows;
    the default "" reproduces the original single-modality output byte-for-byte. Returns the list of
    per-vocab summary dicts (mainly for programmatic reuse/testing)."""
    costs, dp = make_costs_and_dp(N, T, B)

    summaries = [s for s in (_report_one(vk, costs, dp, base_dir_fn, seeds, modality_label)
                             for vk in vocab_keys) if s is not None]

    # ---- pooled average across vocabs (2026-07-09): mean-of-vocab-means, PLUS the spread across
    #      vocabs so a generalization claim can't hide behind one cherry-picked framing. ----
    if len(vocab_keys) > 1 and summaries:
        fs = [s["first_sight_pct"] for s in summaries]
        lat = [s["mean_lateness"] for s in summaries]
        reg = [s["regret_mean"] for s in summaries]
        breg = [s["balls_regret_mean"] for s in summaries]
        print(f"\n==== POOLED ACROSS {len(summaries)} VOCABS ({', '.join(s['vocab'] for s in summaries)}) ====")
        print(f"  first-sight %: mean={st.mean(fs):.0%}  range=[{min(fs):.0%}, {max(fs):.0%}]")
        print(f"  mean lateness: mean={st.mean(lat):.3f}  range=[{min(lat):.3f}, {max(lat):.3f}]")
        print(f"  regret vs pi*: mean={st.mean(reg):.0f}  range=[{min(reg):.0f}, {max(reg):.0f}]")
        print(f"  balls-regret vs pi*: mean={st.mean(breg):.1f}  range=[{min(breg):.1f}, {max(breg):.1f}]")
        if len(summaries) > 1:
            print(f"  (stdev across vocabs: first-sight={st.stdev(fs):.1%}  lateness={st.stdev(lat):.3f}  "
                  f"regret={st.stdev(reg):.0f} -- this is BETWEEN-VOCAB spread, not within-vocab noise)")

    # ---- PAIRED A1 tool baseline: same seeds/streams/disclosure, only the coding framing differs.
    #      The A1 announce run is Haiku-only, so this pairing is skipped for other models. Compared
    #      against the POOLED urn mean across the requested vocabs (or the single vocab's mean). ----
    if not pair_tool:
        print(f"\n  (no paired tool baseline for {model_key}: A1 announce was Haiku-only. "
              f"Compare urn behavior across models on the same seeds instead.)")
    elif summaries:
        from scripts.creator.tool_disposition_benchmark.skirental_scorer import (
            actions_from_session, model_builds_from_actions)
        TOOL = Path("runs/arm_a1_announce")
        tregs, tlate, tfs, tnb, tn = [], [], 0, 0, 0
        for seed in seeds:
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
            urn_lat = st.mean([s["mean_lateness"] for s in summaries])
            urn_reg = st.mean([s["regret_mean"] for s in summaries])
            print(f"\n==== PAIRED A1 TOOL baseline (same seeds {seeds[0]}-{seeds[-1]}, n={tn}) ====")
            print(f"  builds at FIRST SIGHT: {tfs}/{tnb} = {tfs/tnb:.0%}   mean build lateness = {st.mean(tlate):.3f}")
            print(f"  regret vs pi* mean={st.mean(tregs):.0f} +/- {tse:.0f}")
            print(f"\n  >>> PAIRED GAP (urn [pooled over {len(summaries)} vocab(s)] vs A1-tool, "
                  f"identical streams+disclosure):")
            print(f"      lateness  {urn_lat:.2f} (urn)  vs  {st.mean(tlate):.2f} (tool)")
            print(f"      regret    {urn_reg:.0f} (urn)  vs  {st.mean(tregs):.0f} (tool)")

    # ---- eyeball one full seed PER VOCAB: verify the auto-collect notices read correctly ----
    for vk in vocab_keys:
        bd = base_dir_fn(vk)
        seed0 = next((s for s in seeds if (bd / f"seed_{s}" / "session.json").exists()), None)
        if seed0 is None:
            continue
        row = json.loads((bd / f"seed_{seed0}" / "session.json").read_text())
        print(f"\n==== [{vk}] TRANSCRIPT echo (seed {seed0}, kept={row['kept']}) ====")
        for t in row["transcript"]:
            print(f"  [slot {t['slot']:>2}] PROMPT: {t['prompt']}")
            print(f"            -> {t['decision']:<4} | {t['reply'].replace(chr(10),' ')[:160]}")

    return summaries


def _selftest():
    """Exercises `call_with_retry` in isolation (fake failures, no network) -- the transport-error
    contract both `urn_session.run_episode` and `urn_tool_session.run_episode_tool` depend on."""
    async def main():
        # (a) transient failure then success: retried, succeeds, no exception surfaces.
        calls = {"n": 0}
        async def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("transient")
            return "ok"
        result, exc = await call_with_retry(flaky, backoff=0.0)
        assert result == "ok" and exc is None and calls["n"] == 2, (result, exc, calls)

        # (b) permanent failure: retries exhausted, exception surfaces, call count = 1 + retries.
        calls2 = {"n": 0}
        async def always_fails():
            calls2["n"] += 1
            raise ValueError("permanent")
        result, exc = await call_with_retry(always_fails, retries=2, backoff=0.0)
        assert result is None and isinstance(exc, ValueError) and calls2["n"] == 3, (result, exc, calls2)

        # (c) immediate success: exactly one call, no retries burned.
        calls3 = {"n": 0}
        async def always_ok():
            calls3["n"] += 1
            return "fine"
        result, exc = await call_with_retry(always_ok, backoff=0.0)
        assert result == "fine" and exc is None and calls3["n"] == 1, (result, exc, calls3)

        print("urn_common self-test OK (call_with_retry: retry-then-succeed, "
              "exhausted-retries surfaces exception, no-retry-needed path all pass)")

    asyncio.run(main())


if __name__ == "__main__":
    _selftest()
