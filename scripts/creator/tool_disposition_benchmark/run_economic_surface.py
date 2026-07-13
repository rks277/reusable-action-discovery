"""Economic response surface driver (`docs/economic-response-surface-spec.md`).

Executes framing x B x K x seed: R0 (`urn_session.run_episode`) and R2c
(`claim_solver_code_session.run_episode_code_claim`) crossed with B in {1,3,5} and K in {0,20,24},
over the 12 canonical seeds -- 18 frame-cells x 12 seeds = 216 sessions at the full target.

Streams are held FIXED across all three B values (`economic_surface.CANONICAL_STREAM_DIR`, the same
12 streams R0's own cached A2 runs already use) -- B only changes the in-episode BUDGET, never the
stream-generation parameters, per the design in `docs/economic-response-surface-spec.md` §1. Every
cell asserts its stream is byte-identical to the canonical source before running.

`tool_choice="auto"` is passed EXPLICITLY for R2c (not inherited implicitly from any CLI default) --
forcing `"required"` on Anthropic was already found to suppress all of Haiku's deliberation text
(`docs/framing-ladder-spec.md` §3.1); R0 doesn't take a `tool_choice` at all (free-text modality).

  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.run_economic_surface
  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.run_economic_surface --selftest
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json
from pathlib import Path
from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat, provider_for
from scripts.creator.tool_disposition_benchmark.urn_common import N, T, VOCAB, render_system
from scripts.creator.tool_disposition_benchmark.urn_session import run_episode
from scripts.creator.tool_disposition_benchmark.claim_solver_code_session import (
    run_episode_code_claim, render_system_code_claim)
from scripts.creator.tool_disposition_benchmark.run_stream_session import CLAUDE
from scripts.creator.tool_disposition_benchmark.economic_surface import (
    FRAMINGS, BUDGETS, CHARGES, CELLS, CANONICAL_SEEDS, CANONICAL_STREAM_DIR, net_score)

ANNOUNCE_N = True   # fixed design choice (matches the canonical streams' own "_n-announced" dir);
                    # not a CLI flag -- the surface holds this fixed, per the spec doc's §1 "Hold ...
                    # fixed" list.

_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="haiku")
_ap.add_argument("--conc", type=int, default=None, help="override concurrency")
_ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override seed list")
_ap.add_argument("--temp", type=float, default=None)
_ap.add_argument("--cap-usd", type=float, default=None,
                 help="hard global spend cap; required explicitly for paid OpenAI runs")
_ap.add_argument("--unit-cap-usd", type=float, default=None,
                 help="per-session circuit breaker; required explicitly for paid OpenAI runs")
_ap.add_argument("--expected-unit-usd", type=float, default=None,
                 help="calibrated expected session cost; stop after a completed unit exceeds 1.5x")
_ap.add_argument("--reasoning-effort", default="none",
                 choices=["none", "low", "medium", "high", "xhigh"],
                 help="fixed OpenAI reasoning effort, applied identically to R0 and R2c")
_ap.add_argument("--tool-choice-r2c", default="auto", choices=["auto", "required"],
                 help="R2c tool policy; GPT replication locks this to auto for Claude parity")
_ap.add_argument("--cells", nargs="+", default=None,
                 help="restrict to specific cells, e.g. --cells R0:1:0 R2c:5:24 (framing:B:K)")
_ap.add_argument("--dry-run", action="store_true",
                 help="validate and print the resolved run plan without loading API clients or writing")
_ap.add_argument("--selftest", action="store_true",
                 help="run deterministic mechanics/resume/stream-hash self-tests, no network calls.")
_ARGS = _ap.parse_known_args()[0]
MODEL_KEY = _ARGS.model
MODEL_STR = CLAUDE.get(MODEL_KEY, MODEL_KEY)
PROVIDER = provider_for(MODEL_STR)
IS_LOCAL = PROVIDER in ("ollama", "vllm")
TOOL_CHOICE_R2C = _ARGS.tool_choice_r2c
_safe = MODEL_KEY.replace(":", "_").replace("/", "_")
RUN_ROOT = Path(f"runs/economic_surface_{_safe}")

SEEDS = tuple(_ARGS.seeds) if _ARGS.seeds else CANONICAL_SEEDS
CAP_USD = _ARGS.cap_usd if _ARGS.cap_usd is not None else 15.0
UNIT_CAP_USD = _ARGS.unit_cap_usd
EXPECTED_UNIT_USD = _ARGS.expected_unit_usd
EST = 0.10           # per-session spend-guard estimate; wait/never cells run closer to the full
                    # T=60 turns (budget never exhausts early), so this is deliberately higher than
                    # any single rung's own EST (`docs/framing-ladder-spec.md`'s cost-note history).
CONC = _ARGS.conc if _ARGS.conc else (3 if IS_LOCAL else (1 if PROVIDER == "openai" else 6))
_PRICES = {"haiku": (1.0, 5.0, 0.10, 1.25), "sonnet": (3.0, 15.0, 0.30, 3.75),
           "opus": (5.0, 25.0, 0.50, 6.25)}
_OPENAI_PRICES = {
    # Official OpenAI prices on 2026-07-10, USD / 1M tokens.
    "gpt-5.4-mini": {"input": 0.75, "cached": 0.075, "cache_write": 0.75,
                     "output": 4.50},
    "gpt-5.4-mini-2026-03-17": {"input": 0.75, "cached": 0.075,
                                "cache_write": 0.75, "output": 4.50},
    # >272K input tokens: 2x input and 1.5x output for the full request.
    "gpt-5.6": {"input": 5.0, "cached": 0.50, "cache_write": 6.25, "output": 30.0,
                "threshold": 272_000, "high_input": 10.0, "high_cached": 1.0,
                "high_cache_write": 12.5, "high_output": 45.0},
    "gpt-5.6-sol": {"input": 5.0, "cached": 0.50, "cache_write": 6.25,
                    "output": 30.0,
                    "threshold": 272_000, "high_input": 10.0, "high_cached": 1.0,
                    "high_cache_write": 12.5, "high_output": 45.0},
}


def _parse_cells(specs: list[str] | None) -> tuple:
    if not specs:
        return CELLS
    out = []
    for spec in specs:
        f, b, k = spec.split(":")
        out.append((f, int(b), int(k)))
    return tuple(out)


CELLS_TO_RUN = _parse_cells(_ARGS.cells)


def base_dir(framing: str, B: int, K: int) -> Path:
    """`runs/economic_surface_<model>/<frame>/B_<B>/K_<K>/` -- matches
    `docs/economic-response-surface-spec.md` §3's artifact convention."""
    return RUN_ROOT / framing / f"B_{B}" / f"K_{K}"


def _openai_cost(model: str, turn_usages) -> float:
    price = _OPENAI_PRICES.get(model)
    if price is None:
        raise ValueError(f"no verified OpenAI pricing for {model!r}; refusing to report $0")
    total = 0.0
    for usage in turn_usages:
        input_tokens = int(usage.get("input_tokens", 0))
        cached_tokens = min(input_tokens, int(usage.get("cache_read_tokens", 0)))
        cache_write_tokens = min(input_tokens - cached_tokens,
                                 int(usage.get("cache_write_tokens", 0)))
        uncached_tokens = input_tokens - cached_tokens - cache_write_tokens
        output_tokens = int(usage.get("output_tokens", 0))
        high = input_tokens > price.get("threshold", 10**30)
        input_rate = price.get("high_input", price["input"]) if high else price["input"]
        cached_rate = price.get("high_cached", price["cached"]) if high else price["cached"]
        write_rate = (price.get("high_cache_write", price["cache_write"])
                      if high else price["cache_write"])
        output_rate = price.get("high_output", price["output"]) if high else price["output"]
        # OpenAI input_tokens includes its cached subset; do not double-count it.
        total += (uncached_tokens * input_rate + cached_tokens * cached_rate
                  + cache_write_tokens * write_rate + output_tokens * output_rate) / 1e6
    return total


def cost_of(turn_usages):
    if PROVIDER == "openai":
        return _openai_cost(MODEL_STR, turn_usages)
    if MODEL_KEY in _PRICES:
        in_rate, out_rate, cache_read_rate, cache_write_rate = _PRICES[MODEL_KEY]
        return (sum(t.get("input_tokens", 0) for t in turn_usages) * in_rate
                + sum(t.get("output_tokens", 0) for t in turn_usages) * out_rate
                + sum(t.get("cache_read_tokens", 0) for t in turn_usages) * cache_read_rate
                + sum(t.get("cache_write_tokens", 0) for t in turn_usages) * cache_write_rate) / 1e6
    if IS_LOCAL:
        return 0.0
    raise ValueError(f"no verified pricing for paid provider model {MODEL_STR!r}")


def _validate_run_config() -> None:
    if PROVIDER == "openai":
        if MODEL_STR not in _OPENAI_PRICES:
            raise ValueError(f"unknown OpenAI model/pricing {MODEL_STR!r}")
        if _ARGS.cap_usd is None or UNIT_CAP_USD is None:
            raise ValueError("paid OpenAI runs require explicit --cap-usd and --unit-cap-usd")
        if CONC != 1:
            raise ValueError("paid OpenAI runs are locked to --conc 1")
        if TOOL_CHOICE_R2C != "auto":
            raise ValueError("GPT R0/R2c preregistration locks --tool-choice-r2c auto")
        if MODEL_STR in _OPENAI_PRICES and _ARGS.reasoning_effort != "none":
            raise ValueError("GPT R0/R2c preregistration locks --reasoning-effort none so function "
                             "tools work in Chat Completions and both arms share one setting")
    if CAP_USD <= 0 or (UNIT_CAP_USD is not None and UNIT_CAP_USD <= 0):
        raise ValueError("spend caps must be positive")
    if UNIT_CAP_USD is not None and UNIT_CAP_USD > CAP_USD:
        raise ValueError("--unit-cap-usd cannot exceed --cap-usd")


def _sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _stop_after_turn(turn_usages):
    latest = turn_usages[-1] if turn_usages else {}
    if not latest or sum(int(latest.get(k, 0)) for k in
                         ("input_tokens", "output_tokens", "cache_read_tokens",
                          "cache_write_tokens", "reasoning_tokens")) <= 0:
        return "missing_usage"
    if UNIT_CAP_USD is not None and cost_of(turn_usages) >= UNIT_CAP_USD:
        return "unit_cap_usd"
    return None


def load_canonical_stream(seed: int) -> list[dict]:
    return json.loads((CANONICAL_STREAM_DIR / f"seed_{seed}" / "stream.json").read_text())


def assert_canonical(seed: int, slots: list[dict]) -> None:
    ref = load_canonical_stream(seed)
    ref_seq = [s["class_id"] for s in sorted(ref, key=lambda z: z["slot_index"])]
    got_seq = [s["class_id"] for s in sorted(slots, key=lambda z: z["slot_index"])]
    if ref_seq != got_seq:
        raise AssertionError(f"seed {seed}: stream diverged from canonical (class-id sequence "
                             f"mismatch) -- economic surface REQUIRES the fixed 12 canonical "
                             f"streams (spec doc §1); refusing to run on a drifted stream.")


async def run_one(client, model, framing: str, B: int, K: int, seed: int):
    d = base_dir(framing, B, K) / f"seed_{seed}"
    if (d / "session.json").exists():
        return 0.0, "cached"
    slots = load_canonical_stream(seed)
    assert_canonical(seed, slots)
    d.mkdir(parents=True, exist_ok=True)
    (d / "stream.json").write_text(json.dumps(slots, indent=2))

    if framing == "R0":
        vocab = VOCAB["ball"]
        system = render_system(T, B, N, ANNOUNCE_N, vocab=vocab, charge=K)
        row = await run_episode(client, model, slots, T=T, B=B, system=system,
                                temperature=_ARGS.temp, palette=vocab["palette"], item=vocab["item"],
                                reasoning_effort=_ARGS.reasoning_effort,
                                stop_after_turn=_stop_after_turn)
        row["tool_choice"] = None
    elif framing == "R2c":
        system = render_system_code_claim(T, B, N, ANNOUNCE_N, charge=K)
        row = await run_episode_code_claim(client, model, slots, T=T, B=B, system=system,
                                           temperature=_ARGS.temp, tool_choice=TOOL_CHOICE_R2C,
                                           max_tokens=4000,
                                           reasoning_effort=_ARGS.reasoning_effort,
                                           stop_after_turn=_stop_after_turn)
        row["tool_choice"] = TOOL_CHOICE_R2C
    else:
        raise ValueError(f"unknown framing {framing!r}")

    config = {
        "requested_model": MODEL_KEY, "resolved_model": MODEL_STR, "provider": PROVIDER,
        "framing": framing, "B": B, "K": K, "N": N, "T": T, "announce_n": ANNOUNCE_N,
        "temperature": _ARGS.temp, "reasoning_effort": _ARGS.reasoning_effort,
        "tool_choice": row["tool_choice"], "max_completion_tokens": 4000,
        "stream_sha256": _sha256_json(slots), "system_prompt_sha256": _sha256_json(system),
        "global_cap_usd": CAP_USD, "unit_cap_usd": UNIT_CAP_USD,
    }
    row = {"seed": seed, "framing": framing, "B": B, "K": K, "model_key": MODEL_KEY,
           "resolved_model": MODEL_STR, "provider": PROVIDER, "config": config,
           "net_score": net_score(row, K), **row}
    row["provider_response_models"] = sorted({
        t["response_model"] for t in row["transcript"] if t.get("response_model")
    })
    cost = cost_of(row["turn_usages"])
    row["actual_cost_usd"] = cost
    (d / "config.json").write_text(json.dumps(config, indent=2))
    if row["termination"] in ("budget_exhausted", "stream_complete"):
        (d / "session.json").write_text(json.dumps(row, indent=2))
        return cost, "ran"
    (d / "partial_session.json").write_text(json.dumps(row, indent=2))
    return cost, f"INVALID:{row['termination']}"


async def main():
    _validate_run_config()
    if _ARGS.dry_run:
        print(json.dumps({
            "model": MODEL_KEY, "resolved_model": MODEL_STR, "provider": PROVIDER,
            "cells": CELLS_TO_RUN, "seeds": SEEDS, "concurrency": CONC,
            "cap_usd": CAP_USD, "unit_cap_usd": UNIT_CAP_USD,
            "expected_unit_usd": EXPECTED_UNIT_USD,
            "reasoning_effort": _ARGS.reasoning_effort,
            "tool_choice_r2c": TOOL_CHOICE_R2C, "network_calls": 0,
        }, indent=2))
        return
    load_dotenv()
    model = MODEL_STR
    client = RawChat()
    for framing, B, K in CELLS_TO_RUN:
        base_dir(framing, B, K).mkdir(parents=True, exist_ok=True)
    cumulative = 0.0; inflight = 0; idx = 0; paused = False; lock = asyncio.Lock()
    UNITS = [(f, b, k, seed) for (f, b, k) in CELLS_TO_RUN for seed in SEEDS]

    async def worker():
        nonlocal cumulative, inflight, idx, paused
        while True:
            async with lock:
                if paused or idx >= len(UNITS):
                    return
                framing, B, K, seed = UNITS[idx]
                d = base_dir(framing, B, K) / f"seed_{seed}"
                will_run = not (d / "session.json").exists()
                reserve = (EXPECTED_UNIT_USD if EXPECTED_UNIT_USD is not None
                           else UNIT_CAP_USD if UNIT_CAP_USD is not None else EST)
                if will_run and cumulative + (inflight + 1) * reserve > CAP_USD:
                    paused = True; return
                idx += 1; inflight += 1
            try:
                cost, status = await run_one(client, model, framing, B, K, seed)
            except Exception as e:
                cost, status = 0.0, f"ERR:{type(e).__name__}"
            async with lock:
                inflight -= 1; cumulative += cost
                print(f"  [{framing} B={B} K={K} seed={seed}] {status:>6}  ${cost:.3f}  "
                      f"cumulative=${cumulative:.2f}", flush=True)
                if status.startswith("INVALID:"):
                    paused = True
                    print(f"  CIRCUIT BREAKER: seed {seed} ended {status}; no further units start",
                          flush=True)
                if status.startswith("ERR:"):
                    paused = True
                    print(f"  CIRCUIT BREAKER: seed {seed} raised {status}; charged usage may be "
                          "incomplete, so no further units start", flush=True)
                if (EXPECTED_UNIT_USD is not None and status == "ran"
                        and cost > 1.5 * EXPECTED_UNIT_USD):
                    paused = True
                    print(f"  CIRCUIT BREAKER: ${cost:.3f} exceeds 1.5x calibrated "
                          f"${EXPECTED_UNIT_USD:.3f}; no further units start", flush=True)

    print(f"ECONOMIC RESPONSE SURFACE: {MODEL_KEY}, {len(CELLS_TO_RUN)} cells x {len(SEEDS)} seeds "
          f"(cap=${CAP_USD}) ...", flush=True)
    await asyncio.gather(*(worker() for _ in range(CONC)))
    print(f"\n==== {'PAUSED' if paused else 'COMPLETED'}: economic-surface spend ${cumulative:.2f} ====",
          flush=True)


def _selftest():
    """Mechanical/deterministic checks only -- no network calls. Verifies: canonical-stream loading
    and the drift-detection assertion, resume-skips-completed-sessions, and cell-spec parsing --
    NOT model behavior (that needs the live smoke test, `docs/economic-response-surface-spec.md`
    §5 step 2, which this repo has not run yet)."""
    # 1. canonical stream loads and self-matches.
    slots = load_canonical_stream(2000)
    assert_canonical(2000, slots)   # must not raise

    # 2. drift detection: corrupt one slot's class_id, must raise.
    corrupted = [dict(s) for s in slots]
    corrupted[0] = {**corrupted[0], "class_id": 999}
    try:
        assert_canonical(2000, corrupted)
        raise AssertionError("expected drift detection to raise on a corrupted stream")
    except AssertionError as e:
        assert "diverged from canonical" in str(e), e

    # 3. cell spec parsing.
    assert _parse_cells(None) == CELLS
    assert _parse_cells(["R0:1:0", "R2c:5:24"]) == (("R0", 1, 0), ("R2c", 5, 24))

    # 4. OpenAI cached tokens are a subset of input, not an additive category.
    usage = [{"input_tokens": 1_000_000, "cache_read_tokens": 500_000,
              "output_tokens": 100_000}]
    assert abs(_openai_cost("gpt-5.4-mini", usage) - 0.8625) < 1e-12
    # GPT-5.6's >272K request breakpoint applies to the full request.
    assert abs(_openai_cost("gpt-5.6-sol", usage) - 10.0) < 1e-12
    write_usage = [{"input_tokens": 1_000_000, "cache_write_tokens": 500_000}]
    assert abs(_openai_cost("gpt-5.6-sol", write_usage) - 11.25) < 1e-12
    assert _stop_after_turn([{}]) == "missing_usage"

    # 5. resume: isolate the fake artifact in a temporary directory. This must never write into or
    #    delete the real runs/economic_surface_<model> directory.
    async def _check_resume():
        import tempfile
        global RUN_ROOT
        original_root = RUN_ROOT
        try:
            with tempfile.TemporaryDirectory() as tmp:
                RUN_ROOT = Path(tmp) / "economic_surface_selftest"
                d = base_dir("R0", 1, 0) / f"seed_{CANONICAL_SEEDS[0]}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "session.json").write_text("{}")
                cost, status = await run_one(None, None, "R0", 1, 0, CANONICAL_SEEDS[0])
                assert (cost, status) == (0.0, "cached"), (cost, status)
        finally:
            RUN_ROOT = original_root

    asyncio.run(_check_resume())

    print("run_economic_surface self-test OK (canonical stream load + drift detection, cell-spec "
          "parsing, resume-skip verified) -- NOTE: this does not exercise live model behavior; the "
          "two-seed smoke (spec doc §5 step 2) is still required before trusting any cell.")


if __name__ == "__main__":
    if _ARGS.selftest:
        _selftest()
    else:
        asyncio.run(main())
