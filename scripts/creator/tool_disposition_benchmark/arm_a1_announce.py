"""A1 (announce) arm on the stochastic design: system prompt DISCLOSES the recurrence structure
(RECURRENCE_NOTE, non-prescriptive) — does the model STILL build on first sight? Uniform-hard pool (all
a_hand=0, no under-building confound), g=1. Primary metric = build lateness (fidelity).

--model accepts a Claude key (haiku/sonnet/opus) OR any raw model tag ("qwen2.5-coder:14b" -> Ollama via
RawChat; local models are free, pricing/spend-guard disabled). For local models the session uses
stop_on_budget_exhausted=True: once the write budget (B scripts) is spent, no further BUILD decisions
are possible, so the session is truncated there and the rest of the T=60 stream is valued ANALYTICALLY
(skirental_scorer.value_of_builds: reuse the rest if built, hand-solve the rest if not) -- this is the
same truncation-killing machinery already used for solve-rate scoring, just applied to save wall-clock
on a slow local model instead of API cost.

  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.arm_a1_announce [--model haiku|sonnet|opus|qwen2.5-coder:14b]
"""
from __future__ import annotations
import argparse, asyncio, json, statistics as st
from pathlib import Path
from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat, provider_for
from scripts.creator.tool_disposition_benchmark.driver import run_session
from scripts.creator.tool_disposition_benchmark.session_state import (
    CLASS_BOUND_VERSION, SessionState)
from scripts.creator.tool_disposition_benchmark.stream_builder import StochasticStreamSpec, build_stochastic_stream
from scripts.creator.tool_disposition_benchmark.run_stream_session import slots_to_problems, CLAUDE
from scripts.creator.tool_disposition_benchmark.skirental_scorer import actions_from_session, model_builds_from_actions
from scripts.creator.tool_disposition_benchmark.family_kit import set_profile

UNIFORM = ["lcg", "modpow", "continued_frac", "crt_solve", "josephus", "quadratic_map_mod",
           "xorshift_steps", "matrix_power_mod"]
N, T, B, G = len(UNIFORM), 60, 3, 1.0

_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="haiku")
_ap.add_argument("--conc", type=int, default=None, help="override concurrency (Ollama: keep <=4-6)")
_ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override seed list")
_ap.add_argument("--announce-n", action="store_true",
                 help="A2 arm: tell the model the exact number of distinct types N (matches pi*'s own "
                      "information -- see the 2026-07-03 same-information audit)")
_ap.add_argument("--empty-fence-retry", type=int, default=0, metavar="N",
                 help="idle-tail lever: on a no-tool (empty-```json```-fence) turn, prune it from context "
                      "and hard-retry the SAME problem up to N attempts before force-advancing "
                      "(default 0 = off, prior force-advance-after-2 behavior). Writes to a "
                      "_efrN-suffixed dir so it doesn't collide with baseline runs.")
_ap.add_argument("--cap", type=float, default=None,
                 help="override the USD spend guard (default: haiku/opus $12). Needed for multi-seed "
                      "Opus batches where 12*EST exceeds the default ceiling.")
_ap.add_argument("--unit-cap", type=float, default=None,
                 help="stop and save a seed after its exact cumulative API usage reaches this USD "
                      "amount, then pause the batch (may overshoot by the final in-flight API call)")
_ap.add_argument("--expected-unit", type=float, default=None,
                 help="calibrated expected seed cost used for serial global-cap reservation; pause "
                      "after a completed seed exceeds 1.5x this amount")
_ap.add_argument("--reasoning-effort", default="none",
                 help="OpenAI reasoning effort (GPT R3 calibration is locked to 'none')")
_ap.add_argument("--tool-choice", default="auto", choices=["auto", "required"],
                 help="tool selection mode (GPT R3 calibration is locked to 'auto')")
_ap.add_argument("--dry-run", action="store_true",
                 help="validate routing, pricing, caps, and canonical structure without API calls")
_ap.add_argument("--publication", action="store_true",
                 help="explicitly authorize collection on canonical publication seeds 2000--2023")
_ap.add_argument("--smoke", action="store_true",
                 help="write to a separate _smoke directory; canonical publication seeds are refused")
_ap.add_argument("--canonical-structure", action="store_true",
                 help="Opus repair arm: use MAG=1000 without pinning Josephus, assert the latent "
                      "family/role/arrival stream matches the canonical Opus urn A2 artifact, and "
                      "write to a separate _canonical-structure run directory")
_ap.add_argument("--full-stream", action="store_true",
                 help="run the ENTIRE T-problem stream even after the write budget is exhausted (the "
                      "old API-model behavior). Default now truncates the session once budget is spent "
                      "-- no build decisions remain past that point, so first-sight/lateness are "
                      "unchanged and the tail is valued analytically (value_of_builds). Truncating "
                      "avoids paying for ~57 post-budget problems of debugging on an eager model.")
_ARGS = _ap.parse_known_args()[0]
MODEL_KEY = _ARGS.model
MODEL_STR = CLAUDE.get(MODEL_KEY, MODEL_KEY)      # Claude key -> id; else pass the raw tag through
PROVIDER = provider_for(MODEL_STR)
IS_LOCAL = PROVIDER in ("ollama", "vllm")
ANNOUNCE_N = _ARGS.announce_n
EMPTY_FENCE_RETRY = _ARGS.empty_fence_retry
UNIT_CAP_USD = _ARGS.unit_cap
EXPECTED_UNIT_USD = _ARGS.expected_unit
CANONICAL_STRUCTURE = _ARGS.canonical_structure
if CANONICAL_STRUCTURE and (MODEL_KEY != "opus" or not ANNOUNCE_N):
    _ap.error("--canonical-structure requires --model opus --announce-n")
_safe = MODEL_KEY.replace(":", "_").replace("/", "_")
BASE = Path("runs/arm_a1_announce" if MODEL_KEY == "haiku" else f"runs/arm_a1_announce_{_safe}")
if ANNOUNCE_N:
    BASE = Path(str(BASE) + "_n-announced")
if CANONICAL_STRUCTURE:
    BASE = Path(str(BASE) + "_canonical-structure")
if PROVIDER == "openai":
    BASE = Path(str(BASE) + "_write-budget-v2")
BASE = Path(str(BASE) + "_class-bound-v1")
if _ARGS.smoke:
    BASE = Path(str(BASE) + "_smoke")
if EMPTY_FENCE_RETRY:
    BASE = Path(str(BASE) + f"_efr{EMPTY_FENCE_RETRY}")
SEEDS = _ARGS.seeds if _ARGS.seeds else list(range(2000, 2012))  # 12 -- paired with the urn, same seeds
_PUBLICATION_SEEDS = set(range(2000, 2024))
if _ARGS.publication and _ARGS.smoke:
    _ap.error("--publication and --smoke are mutually exclusive")
if _ARGS.smoke and _PUBLICATION_SEEDS.intersection(SEEDS):
    _ap.error("--smoke refuses canonical publication seeds 2000--2023")
if not _ARGS.dry_run and _PUBLICATION_SEEDS.intersection(SEEDS) and not _ARGS.publication:
    _ap.error("canonical seeds 2000--2023 require explicit --publication authorization")
COLLECTION_KIND = "smoke" if _ARGS.smoke else ("publication" if _ARGS.publication else "calibration")
CAP_USD, EST, CONC = (12.0, 1.0, 12) if MODEL_KEY == "haiku" else (12.0, 2.5, 12)
if IS_LOCAL:
    CAP_USD, EST = 1e9, 0.0                        # free -> spend-guard never binds
elif EXPECTED_UNIT_USD is not None:
    EST = EXPECTED_UNIT_USD                        # exact-config calibration; per-unit cap stays higher
elif UNIT_CAP_USD is not None:
    EST = UNIT_CAP_USD                             # the per-turn breaker bounds each serial unit
CONC = _ARGS.conc if _ARGS.conc else (CONC if not IS_LOCAL else 4)
if _ARGS.cap is not None:
    CAP_USD = _ARGS.cap
if PROVIDER == "openai":
    if not ANNOUNCE_N:
        _ap.error("GPT R3 calibration requires --announce-n")
    if _ARGS.cap is None or UNIT_CAP_USD is None:
        _ap.error("paid GPT R3 runs require explicit --cap and --unit-cap")
    if CONC != 1:
        _ap.error("paid GPT R3 runs require --conc 1")
    if _ARGS.reasoning_effort != "none":
        _ap.error("GPT R3 calibration locks --reasoning-effort none")
    if _ARGS.tool_choice != "auto":
        _ap.error("GPT R3 calibration locks --tool-choice auto")
    if _ARGS.full_stream:
        _ap.error("GPT R3 calibration must truncate at budget exhaustion")
if (CAP_USD <= 0 or (UNIT_CAP_USD is not None and UNIT_CAP_USD <= 0)
        or (EXPECTED_UNIT_USD is not None and EXPECTED_UNIT_USD <= 0)):
    _ap.error("spend caps must be positive")
if UNIT_CAP_USD is not None and UNIT_CAP_USD > CAP_USD:
    _ap.error("--unit-cap cannot exceed --cap")
# Truncate at budget exhaustion by default (local models always did). --full-stream restores the old
# API behavior. No build decisions occur past exhaustion, so the disposition metric is identical.
STOP_ON_BUDGET = not _ARGS.full_stream
_PRICES = {"haiku": (1.0, 5.0, 0.10, 1.25), "sonnet": (3.0, 15.0, 0.30, 3.75),
           "opus": (5.0, 25.0, 0.50, 6.25)}       # $/1e6 (in, out, cache_read, cache_write)
IN, OUT, CR, CW = _PRICES.get(MODEL_KEY, (0.0, 0.0, 0.0, 0.0))   # local models: no cost
_OPENAI_PRICES = {
    "gpt-5.4-mini-2026-03-17": {"input": 0.75, "cached": 0.075, "cache_write": 0.75,
                                "output": 4.50},
    "gpt-5.6-sol": {"input": 5.0, "cached": 0.50, "cache_write": 6.25, "output": 30.0,
                    "threshold": 272_000, "high_input": 10.0, "high_cached": 1.0,
                    "high_cache_write": 12.5, "high_output": 45.0},
}
if PROVIDER == "openai" and MODEL_STR not in _OPENAI_PRICES:
    _ap.error(f"no verified OpenAI pricing for {MODEL_STR!r}")
# MAG=100 is only hand-hard for Haiku/Qwen; at MAG=1000 crt_solve/modpow collapse to a_hand=0.00 for
# Opus too (calibrated 2026-07-03), but josephus stays hand-solvable (~0.5-0.6) at ANY magnitude for
# Opus (no closed form for general K; Opus tracks the O(N) recurrence reliably regardless of length) --
# so for Opus it is PINNED as a single forced trap at the final slot instead: by T-1 there are 0
# remaining draws, so building never pays off and it cannot influence any earlier build decision.
MAG = 1000 if MODEL_KEY == "opus" else 100
PINNED_TRAP = "josephus" if MODEL_KEY == "opus" and not CANONICAL_STRUCTURE else None
# measured pooled a_script from the 2026-07-03 Qwen-Coder calibration (a0_oracle_gap); Claude models
# assumed ~1 (never separately measured -- their scripts are ~always correct in these transcripts).
_A_SCRIPT = {"qwen2.5-coder:0.5b": 0.21, "qwen2.5-coder:1.5b": 0.35, "qwen2.5-coder:3b": 0.50,
             "qwen2.5-coder:7b": 0.75, "qwen2.5-coder:14b": 0.83, "qwen2.5-coder:32b": 0.96,
             # publication-grade tool-transfer rerun (2026-07-10, q8_0, MAG=100, a0_oracle_gap k=8
             # over the 8 uniform-hard families; a_hand=0 everywhere). q8_0 reads a bit higher than
             # the historical Q4 0.83 (closer to bf16); per-tag because base vs RL-final differ.
             "qwen-rl-base-q8:latest": 0.89, "qwen-rl-urn-final:latest": 0.94}
A_SCRIPT = _A_SCRIPT.get(MODEL_KEY, 1.0)


def cost_of(row):
    tu = row.get("turn_usages") or []
    if PROVIDER == "openai":
        price = _OPENAI_PRICES[MODEL_STR]
        total = 0.0
        for usage in tu:
            input_tokens = int(usage.get("input_tokens", 0))
            cached_tokens = min(input_tokens, int(usage.get("cache_read_tokens", 0)))
            cache_write_tokens = min(input_tokens - cached_tokens,
                                     int(usage.get("cache_write_tokens", 0)))
            uncached_tokens = input_tokens - cached_tokens - cache_write_tokens
            output_tokens = int(usage.get("output_tokens", 0))
            high = input_tokens > price.get("threshold", 10**30)
            total += (
                uncached_tokens * (price.get("high_input", price["input"]) if high else price["input"])
                + cached_tokens * (price.get("high_cached", price["cached"]) if high else price["cached"])
                + cache_write_tokens
                * (price.get("high_cache_write", price["cache_write"]) if high else price["cache_write"])
                + output_tokens * (price.get("high_output", price["output"]) if high else price["output"])
            ) / 1e6
        return total
    return (sum(t.get("input_tokens", 0) for t in tu) * IN + sum(t.get("output_tokens", 0) for t in tu) * OUT
            + sum(t.get("cache_read_tokens", 0) for t in tu) * CR + sum(t.get("cache_write_tokens", 0) for t in tu) * CW) / 1e6


_STRUCTURAL_FIELDS = ("slot_index", "family", "class_id", "class_size", "class_position",
                      "members_remaining_after", "is_recurring", "role", "rate")


def structural_projection(slots):
    return [{k: s[k] for k in _STRUCTURAL_FIELDS} for s in slots]


def make_and_validate_stream(seed):
    slots, meta = build_stochastic_stream(StochasticStreamSpec(
        families=UNIFORM, n_hot=B, T=T, budget=B, guarantee_trap_early=G, magnitude=MAG, seed=seed,
        pinned_last_trap=PINNED_TRAP))
    if CANONICAL_STRUCTURE or PROVIDER == "openai":
        reference_root = ("runs/urn_opus_n-announced" if CANONICAL_STRUCTURE
                          else "runs/urn_haiku_n-announced")
        reference_path = Path(reference_root) / f"seed_{seed}" / "stream.json"
        reference = json.loads(reference_path.read_text())
        assert structural_projection(slots) == structural_projection(reference), (
            f"seed {seed}: generated structure does not match canonical urn A2 stream")
    return slots, meta


async def run_one(client, model, seed):
    d = BASE / f"seed_{seed}"
    if (d / "sessions.jsonl").exists():
        return 0.0, "cached"
    slots, meta = make_and_validate_stream(seed)
    d.mkdir(parents=True, exist_ok=True)
    (d / "stream.json").write_text(json.dumps(slots, indent=2))
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    (d / "config.json").write_text(json.dumps({
        "budget": B, "seed": seed, "magnitude": MAG, "arm": "announce",
        "canonical_structure": CANONICAL_STRUCTURE, "pinned_last_trap": PINNED_TRAP,
        "requested_model": MODEL_KEY, "resolved_model": MODEL_STR, "provider": PROVIDER,
        "announce_n": ANNOUNCE_N, "reasoning_effort": _ARGS.reasoning_effort,
        "tool_choice": _ARGS.tool_choice, "global_cap_usd": CAP_USD,
        "unit_cap_usd": UNIT_CAP_USD, "tool_schema_version": "write-budget-v2",
        "benchmark_version": CLASS_BOUND_VERSION, "class_bound": True,
        "collection_kind": COLLECTION_KIND,
    }))
    state = SessionState(problems=slots_to_problems(slots), budget=B)
    state.announce_recurrence = True          # A1: disclose recurrence structure (non-prescriptive)
    if ANNOUNCE_N:
        state.announce_n_types = N            # A2: also disclose exact N (matches pi*'s own info)
    def _progress(n_turns, problem, n, spent, elapsed, tools, writes_remaining):
        actions = ",".join(tools) if tools else "NO_TOOL"
        print(f"    [seed {seed}] turn {n_turns:>3}  problem {problem:>2}/{n}  "
              f"actions={actions}  writes_left={writes_remaining}  "
              f"{spent/1000:.1f}k tok  {elapsed:.0f}s", flush=True)

    def _stop_after_turn(turn_usages):
        latest = turn_usages[-1] if turn_usages else {}
        if not latest or sum(int(latest.get(k, 0)) for k in
                             ("input_tokens", "output_tokens", "cache_read_tokens",
                              "cache_write_tokens", "reasoning_tokens")) <= 0:
            return "missing_usage"
        if UNIT_CAP_USD is not None and cost_of({"turn_usages": turn_usages}) >= UNIT_CAP_USD:
            return "unit_cost_cap"
        return None

    row = await run_session(client, model, state, token_cap=300_000, max_tokens=4096,
                            announce_cap=True, stop_on_budget_exhausted=STOP_ON_BUDGET, progress_cb=_progress,
                            prune_no_tool=bool(EMPTY_FENCE_RETRY),
                            max_no_tool_retries=EMPTY_FENCE_RETRY or 2,
                            stop_after_turn=_stop_after_turn,
                            reasoning_effort=_ARGS.reasoning_effort,
                            tool_choice=_ARGS.tool_choice)
    row["model_key"] = MODEL_KEY
    row["resolved_model"] = MODEL_STR
    row["provider"] = PROVIDER
    row["actual_cost_usd"] = cost_of(row)
    (d / "sessions.jsonl").write_text(json.dumps(row) + "\n")
    return cost_of(row), ("cost_cap" if row.get("stop_reason") else "ran")


async def main():
    load_dotenv(); set_profile(MODEL_KEY)
    if _ARGS.dry_run:
        for seed in SEEDS:
            make_and_validate_stream(seed)
        print(json.dumps({
            "requested_model": MODEL_KEY, "resolved_model": MODEL_STR, "provider": PROVIDER,
            "seeds": SEEDS, "concurrency": CONC, "cap_usd": CAP_USD,
            "unit_cap_usd": UNIT_CAP_USD, "expected_unit_usd": EXPECTED_UNIT_USD,
            "reasoning_effort": _ARGS.reasoning_effort,
            "tool_choice": _ARGS.tool_choice, "magnitude": MAG,
            "benchmark_version": CLASS_BOUND_VERSION, "class_bound": True,
            "collection_kind": COLLECTION_KIND,
            "canonical_structure_asserted": CANONICAL_STRUCTURE or PROVIDER == "openai",
            "network_calls": 0,
        }, indent=2))
        return
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
                will_run = not (d / "sessions.jsonl").exists()
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
                if UNIT_CAP_USD is not None and (status == "cost_cap" or cost >= UNIT_CAP_USD):
                    paused = True
                    print(f"  CIRCUIT BREAKER: seed {seed} reached ${UNIT_CAP_USD:.2f}; "
                          "no further seeds will start", flush=True)
                if (EXPECTED_UNIT_USD is not None and status == "ran"
                        and cost > 1.5 * EXPECTED_UNIT_USD):
                    paused = True
                    print(f"  CIRCUIT BREAKER: ${cost:.3f} exceeds 1.5x calibrated "
                          f"${EXPECTED_UNIT_USD:.3f}; no further seeds will start", flush=True)

    print(f"A1 announce arm: {MODEL_KEY}, uniform-hard N={N}, MAG={MAG}, pinned_trap={PINNED_TRAP}, "
          f"g={G}, {len(SEEDS)} seeds (cap=${CAP_USD}) ...", flush=True)
    await asyncio.gather(*(worker() for _ in range(CONC)))
    print(f"\n==== {'PAUSED' if paused else 'COMPLETED'}: A1 spend ${cumulative:.2f} ====", flush=True)

    # ---- fidelity report (the primary metric) ----
    lateness, first_sight, nb, match, nseed = [], 0, 0, 0, 0
    for seed in SEEDS:
        d = BASE / f"seed_{seed}"
        if not (d / "sessions.jsonl").exists():
            continue
        slots = json.loads((d / "stream.json").read_text())
        sess = json.loads((d / "sessions.jsonl").read_text().splitlines()[0])
        acts = actions_from_session(sess, slots)
        mb = {c: b for c, b in model_builds_from_actions(acts).items() if b is not None}
        firstseen, seen = [], set()
        for s in sorted(slots, key=lambda z: z["slot_index"]):
            if s["class_id"] not in seen:
                seen.add(s["class_id"]); firstseen.append(s["class_id"])
        built = set(mb)
        for c, b in mb.items():
            lateness.append(b - 1); nb += 1; first_sight += (b == 1)
        if built and built == set(firstseen[:len(built)]):
            match += 1
        nseed += 1
    print(f"\n==== A1 FIDELITY (n={nseed} seeds, {nb} builds) ====")
    if nb:
        print(f"  builds at FIRST SIGHT (lateness 0): {first_sight}/{nb} = {first_sight/nb:.0%}")
        print(f"  mean lateness = {st.mean(lateness):.3f}  max = {max(lateness)}")
        print(f"  built-set == first-B-distinct-arrivals: {match}/{nseed} = {match/nseed:.0%}")
        print(f"  builds/seed = {nb/nseed:.2f}")
        print(f"\n  => lateness ~0 CONFIRMS open-loop under disclosure; lateness>0 would mean it "
              f"closes the loop when told.")

    # ---- regret vs exact pi* (same DP as the urn) -- uses the MEASURED a_script for this model, not
    #      the ~1 default, so a weak model's lower script-correctness is priced into u_build/u_reuse
    #      for BOTH the model's realized value and pi*'s (same-info reference shares the utility fn).
    from scripts.creator.tool_disposition_benchmark.skirental_scorer import exact_pistar_report, Costs
    from scripts.creator.tool_disposition_benchmark.exact_dp import ExactDP
    costs = Costs(a_hand={f: 0.0 for f in UNIFORM}, h=987, C=308, r=200, R=100.0, lam=0.1,
                  a_script=A_SCRIPT, default_a_hand=0.0)
    dp = ExactDP(costs.R * 0.0 - costs.lam * costs.h, costs.u_build(), costs.u_reuse(),
                N, T, B, alpha=1.0, cap=3)
    regs, mtr, ptr, pos_reg, nreg = [], [], [], 0, 0
    for seed in SEEDS:
        d = BASE / f"seed_{seed}"
        if not (d / "sessions.jsonl").exists():
            continue
        slots = json.loads((d / "stream.json").read_text())
        sess = json.loads((d / "sessions.jsonl").read_text().splitlines()[0])
        mb = model_builds_from_actions(actions_from_session(sess, slots))
        rep = exact_pistar_report(slots, costs, B, N, T, UNIFORM, mb, dp=dp)
        regs.append(rep["regret"]); mtr.append(rep["model_traps_built"]); ptr.append(rep["pistar_traps_built"])
        pos_reg += rep["regret"] > 0; nreg += 1
    if regs:
        se = (st.stdev(regs) / len(regs) ** 0.5) if len(regs) > 1 else 0.0
        print(f"\n==== A1 TOOL vs exact pi* (a_script={A_SCRIPT}, n={nreg} seeds) ====")
        print(f"  regret mean={st.mean(regs):.0f} +/- {se:.0f}  pos={pos_reg}/{nreg}  "
              f"model_traps/seed={st.mean(mtr):.2f}  pi*_traps/seed={st.mean(ptr):.2f}")
        print(f"  (extrapolated analytically past any early-stop truncation -- value_of_builds prices "
              f"the untouched tail of the T={T} stream from the model's actual build set, no re-run needed)")


if __name__ == "__main__":
    asyncio.run(main())
