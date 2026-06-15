"""Experiment (J): commitment probe -- is Opus's recognition gap closable?

(A)/(E) showed Opus's build-rate deficit is a RECOGNITION failure: holding both
winning ingredients, it often never issues the winning combine (82% of losses),
with no confabulation involved. (J) tests whether a one-time, mid-episode nudge
toward the combine mechanic -- WITHOUT naming the recipe pair -- recovers it.

We re-run the b<=640 PAIRED-LOSS cells (where Opus failed but a smaller model
built the identical world) under three conditions:
  baseline : no nudge (controls for regression-to-mean + run-to-run variance)
  commit   : the commitment nudge (the intervention)
  neutral  : a content-free nudge at the same trigger (rules out "any
             interruption helps")
The nudge fires once, the first turn the agent holds both recipe types and has
not built (see toolworld_v2.run's commit_nudge). Because the env runs with
hint=True (byproducts are described as combinable), the commit nudge reveals no
new information -- it is a pure commitment/priority prompt.

Metric (see scripts/analyze_commitment_probe.py): recognition rate
P(built | held both) per condition. commit >> baseline ~ neutral => closable
commitment failure; commit ~ baseline => deeper deficit.

Usage: PYTHONPATH=. python -m scripts.run_commitment_probe
       COMMIT_PROBE_RESUME=runs/<dir> PYTHONPATH=. python -m scripts.run_commitment_probe
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from scripts.toolworld_v2 import run
from scripts import sweep_config as cfg

N, T = 12, 3
MAX_BUDGET = 640                       # b<=640: pure recognition regime
MODEL = "claude-opus-4-8"

COMMIT = ("You are holding several distinct components and still have actions "
          "left. Combining the right pair of components can build a tool that "
          "opens doors far more cheaply than searching one by one. Have you "
          "tried using combine on the components you hold?")
NEUTRAL = "You still have plenty of actions left. Keep going."
CONDITIONS = [("baseline", None), ("commit", COMMIT), ("neutral", NEUTRAL)]


def max_turns_for(budget: int) -> int:
    return max(cfg.MAX_TURNS, round(budget * 1.5))


def latest(model_glob: str) -> str:
    hits = sorted(glob.glob(f"runs/{model_glob}_build_sweep_T3_n12_*"))
    if not hits:
        raise SystemExit(f"no build sweep dir for {model_glob}")
    return hits[-1]


def loss_cells() -> list[tuple[int, int]]:
    """(budget, seed) cells, budget<=MAX_BUDGET, where Opus did NOT build but a
    smaller model (Haiku or Sonnet) DID, on the identical world."""
    def load(d):
        p = Path(d) / "episodes.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines()
                if l.strip() and not json.loads(l).get("error")]
    opus = {(r["budget"], r["relabel_seed"]): r for r in load(latest("opus"))}
    small = {}
    for m in ("haiku", "sonnet"):
        for r in load(latest(m)):
            small.setdefault((r["budget"], r["relabel_seed"]), []).append(r)
    cells = []
    for (b, s), lr in opus.items():
        if b > MAX_BUDGET or lr.get("built_machine"):
            continue
        if any(x.get("built_machine") for x in small.get((b, s), [])):
            cells.append((b, s))
    return sorted(cells)


async def main():
    load_dotenv()
    resume = os.environ.get("COMMIT_PROBE_RESUME")
    if resume:
        out_dir = Path(resume)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / f"commitment_probe_T{T}_n{N}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.touch()

    done = set()
    rows: list[dict] = []
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(r)
        if not r.get("error"):
            done.add((r["budget"], r["relabel_seed"], r["condition"]))

    losses = loss_cells()
    cells = [(b, s, cn, ct)
             for (b, s) in losses
             for cn, ct in CONDITIONS
             if (b, s, cn) not in done]
    print(f"Writing to {out_path}\n"
          f"Probe: {len(losses)} b<=({MAX_BUDGET}) paired-loss cells x "
          f"{len(CONDITIONS)} conditions = {len(losses) * len(CONDITIONS)} episodes "
          f"(n={N}, T={T}); resume: {len(done)} done, {len(cells)} remaining",
          flush=True)

    sem = asyncio.Semaphore(cfg.CONCURRENCY["anthropic"])
    lock = asyncio.Lock()

    async def one(budget, seed, cond_name, cond_text):
        async with sem:
            t0 = time.time()
            try:
                result, trace = await run(
                    MODEL, n=N, n_types=T, relabel_seed=seed, drop_seed=seed,
                    hint=cfg.HINT, max_turns=max_turns_for(budget), budget=budget,
                    stop_on_build=True, no_progress_window=64,
                    commit_nudge=cond_text)
                row = {"model": MODEL, "n": N, "n_types": T, "hint": cfg.HINT,
                       "condition": cond_name,
                       "budget": budget, "relabel_seed": seed, "drop_seed": seed,
                       "labels": result["labels"],
                       "actions": [x["action"] for x in trace],
                       "agent_texts": [x["agent_text"] for x in trace],
                       "obs": [x["obs"] for x in trace],
                       "solved": result["solved"],
                       "built_machine": result["built_machine"],
                       "build_turn": result["build_turn"],
                       "total_actions": result["total_actions"],
                       "noop_total": result["noop_total"],
                       "refusals": result["refusals"],
                       "unparsed": result["unparsed"],
                       "usage": result["usage"],
                       "stopped_reason": result["stopped_reason"],
                       "nudged": result["nudged"], "nudge_turn": result["nudge_turn"],
                       "elapsed_s": round(time.time() - t0, 2)}
            except Exception as e:
                row = {"model": MODEL, "n": N, "n_types": T, "condition": cond_name,
                       "budget": budget, "relabel_seed": seed, "drop_seed": seed,
                       "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                rows.append(row)
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  {cond_name:<8} b={budget:<4} seed={seed:<2} "
                      f"built={row.get('built_machine')} "
                      f"nudged={row.get('nudged')} nudge_turn={row.get('nudge_turn')} "
                      f"actions={row.get('total_actions')} "
                      f"stop={row.get('stopped_reason')} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(*c) for c in cells))
    print(f"\nDone. Episodes at: {out_path}")

    # quick inline recognition-rate readout per condition
    ok = [r for r in rows if not r.get("error")]
    print("\ncondition | eps | held_both | built | recognition P(built|held_both)")
    for cn, _ in CONDITIONS:
        g = [r for r in ok if r.get("condition") == cn]
        hb = [r for r in g if (r.get("nudged")
                               or _held_both_from_obs(r))]
        bu = sum(1 for r in g if r.get("built_machine"))
        # recognition denominator = episodes that held both at any point
        held = [r for r in g if _held_both_from_obs(r)]
        rec = bu / len(held) if held else float("nan")
        print(f"  {cn:<8} | {len(g):>3} | {len(held):>9} | {bu:>5} | {rec:.2f}")
    print("\nFull analysis: PYTHONPATH=. python -m scripts.analyze_commitment_probe "
          f"{out_path}")


# minimal held-both check for the inline readout (full version in the analyzer)
import re as _re
_HOLD = _re.compile(r"hold (\d+) (\S+?)\(s\)")


def _held_both_from_obs(r: dict) -> bool:
    recipe = set(r.get("labels", {}).get("recipe", []))
    if len(recipe) != 2:
        return False
    seen = set()
    for o in r.get("obs") or []:
        for _, ty in _HOLD.findall(o or ""):
            seen.add(ty)
    return recipe <= seen


if __name__ == "__main__":
    asyncio.run(main())
