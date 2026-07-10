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
import argparse, asyncio, json
from pathlib import Path
from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
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
_ap.add_argument("--cap-usd", type=float, default=15.0, help="hard global spend cap (spec doc §5)")
_ap.add_argument("--cells", nargs="+", default=None,
                 help="restrict to specific cells, e.g. --cells R0:1:0 R2c:5:24 (framing:B:K)")
_ap.add_argument("--selftest", action="store_true",
                 help="run deterministic mechanics/resume/stream-hash self-tests, no network calls.")
_ARGS = _ap.parse_known_args()[0]
MODEL_KEY = _ARGS.model
MODEL_STR = CLAUDE.get(MODEL_KEY, MODEL_KEY)
IS_LOCAL = MODEL_KEY not in CLAUDE
TOOL_CHOICE_R2C = "required" if IS_LOCAL else "auto"     # same provider-conditional rule as every
                                                          # other rung -- see module docstring.
_safe = MODEL_KEY.replace(":", "_").replace("/", "_")

SEEDS = tuple(_ARGS.seeds) if _ARGS.seeds else CANONICAL_SEEDS
CAP_USD = _ARGS.cap_usd
EST = 0.10           # per-session spend-guard estimate; wait/never cells run closer to the full
                    # T=60 turns (budget never exhausts early), so this is deliberately higher than
                    # any single rung's own EST (`docs/framing-ladder-spec.md`'s cost-note history).
CONC = _ARGS.conc if _ARGS.conc else (3 if IS_LOCAL else 6)
_PRICES = {"haiku": (1.0, 5.0, 0.10, 1.25), "sonnet": (3.0, 15.0, 0.30, 3.75),
           "opus": (5.0, 25.0, 0.50, 6.25)}
IN, OUT, CR, CW = _PRICES.get(MODEL_KEY, (0.0, 0.0, 0.0, 0.0))


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
    return Path(f"runs/economic_surface_{_safe}") / framing / f"B_{B}" / f"K_{K}"


def cost_of(turn_usages):
    return (sum(t.get("input_tokens", 0) for t in turn_usages) * IN
            + sum(t.get("output_tokens", 0) for t in turn_usages) * OUT
            + sum(t.get("cache_read_tokens", 0) for t in turn_usages) * CR
            + sum(t.get("cache_write_tokens", 0) for t in turn_usages) * CW) / 1e6


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
                                temperature=_ARGS.temp, palette=vocab["palette"], item=vocab["item"])
        row["tool_choice"] = None
    elif framing == "R2c":
        system = render_system_code_claim(T, B, N, ANNOUNCE_N, charge=K)
        row = await run_episode_code_claim(client, model, slots, T=T, B=B, system=system,
                                           temperature=_ARGS.temp, tool_choice=TOOL_CHOICE_R2C)
        row["tool_choice"] = TOOL_CHOICE_R2C
    else:
        raise ValueError(f"unknown framing {framing!r}")

    row = {"seed": seed, "framing": framing, "B": B, "K": K, "model_key": MODEL_KEY,
          "net_score": net_score(row, K), **row}
    (d / "session.json").write_text(json.dumps(row, indent=2))
    return cost_of(row["turn_usages"]), "ran"


async def main():
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
                if will_run and cumulative + (inflight + 1) * EST > CAP_USD:
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

    # 4. resume: a fake completed session.json makes run_one report "cached" without touching the
    #    client (pass None as client/model -- would raise AttributeError if it tried to call it).
    async def _check_resume():
        d = base_dir("R0", 1, 0) / f"seed_{CANONICAL_SEEDS[0]}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "session.json").write_text("{}")
        cost, status = await run_one(None, None, "R0", 1, 0, CANONICAL_SEEDS[0])
        assert (cost, status) == (0.0, "cached"), (cost, status)
        import shutil
        shutil.rmtree(Path(f"runs/economic_surface_{_safe}"), ignore_errors=True)

    asyncio.run(_check_resume())

    print("run_economic_surface self-test OK (canonical stream load + drift detection, cell-spec "
          "parsing, resume-skip verified) -- NOTE: this does not exercise live model behavior; the "
          "two-seed smoke (spec doc §5 step 2) is still required before trusting any cell.")


if __name__ == "__main__":
    if _ARGS.selftest:
        _selftest()
    else:
        asyncio.run(main())
