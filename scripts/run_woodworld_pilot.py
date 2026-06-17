"""Small woodworld pilot: one model x {hint on, hint off} x REPS, at fixed N.

Runs the episodes, writes episodes.jsonl, and emits a grouped-bar figure of
build rate and solve rate (hint on vs off) with SE(proportion) error bars.
Paired worlds across the two hint conditions (rep r => relabel_seed=gather_seed=r),
so the only difference within a rep is the hint.

Usage: PYTHONPATH=. python -m scripts.run_woodworld_pilot
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from math import sqrt
from pathlib import Path

from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.woodworld_config as cfg            # sets DEFAULT_SCHEME='letter'
from scripts.validate_woodworld import budget_for
from scripts.woodworld import run

MODEL = "claude-sonnet-4-6"
N = 10
REPS = 10
HINTS = [True, False]
CONC = cfg.CONCURRENCY["anthropic"]


async def main():
    load_dotenv()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"woodworld_pilot_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    out_path.write_text("")
    budget = budget_for(N)
    print(f"Pilot: {MODEL} N={N} budget={budget} scheme={cfg.OBFUSCATION_SCHEME} "
          f"hints={HINTS} x {REPS} reps -> {out_path}", flush=True)

    sem = asyncio.Semaphore(CONC)
    lock = asyncio.Lock()

    async def one(hint, rep):
        async with sem:
            t0 = time.time()
            try:
                result, trace = await run(MODEL, n=N, relabel_seed=rep, gather_seed=rep,
                                          hint=hint, max_turns=cfg.MAX_TURNS, budget=budget)
                row = {"model": MODEL, "n": N, "hint": hint, "budget": budget,
                       "relabel_seed": rep, "gather_seed": rep,
                       "labels": result["labels"],
                       "actions": [x["action"] for x in trace],
                       "agent_texts": [x["agent_text"] for x in trace],
                       "obs": [x["obs"] for x in trace],
                       "solved": result["solved"],
                       "built_axe": result["built_axe"],
                       "total_actions": result["total_actions"],
                       "use_axe_count": result["use_axe_count"],
                       "stopped_reason": result["stopped_reason"],
                       "usage": result["usage"], "elapsed_s": round(time.time() - t0, 2)}
            except Exception as e:
                row = {"model": MODEL, "n": N, "hint": hint, "budget": budget,
                       "relabel_seed": rep, "gather_seed": rep,
                       "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  hint={hint!s:5s} rep={rep} solved={row.get('solved')} "
                      f"built={row.get('built_axe')} actions={row.get('total_actions')}/{budget} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(h, r) for h in HINTS for r in range(REPS)))
    plot(out_path)


def plot(out_path: Path):
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]

    def rate(hint, key):
        vals = [bool(r[key]) for r in rows if r["hint"] == hint]
        n = len(vals)
        p = sum(vals) / n if n else 0.0
        se = sqrt(p * (1 - p) / n) if n else 0.0
        return p, se, n

    metrics = [("built_axe", "build rate"), ("solved", "solve rate")]
    on = [rate(True, k) for k, _ in metrics]
    off = [rate(False, k) for k, _ in metrics]

    print(f"\n{'metric':12s} {'hint=on':>16s} {'hint=off':>16s}")
    for (k, lbl), o, f in zip(metrics, on, off):
        print(f"{lbl:12s} {o[0]:.2f}±{o[1]:.2f}(n{o[2]})".ljust(29) +
              f"{f[0]:.2f}±{f[1]:.2f}(n{f[2]})")

    import numpy as np
    x = np.arange(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar(x - w / 2, [o[0] for o in on], w, yerr=[o[1] for o in on], capsize=4,
           label="hint on", color="#2ca02c")
    ax.bar(x + w / 2, [o[0] for o in off], w, yerr=[o[1] for o in off], capsize=4,
           label="hint off", color="0.6")
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylim(0, 1.05); ax.set_ylabel("rate")
    n_on, n_off = on[0][2], off[0][2]
    ax.set_title(f"woodworld pilot: {MODEL}  (N={N}, p=0.8, "
                 f"n={n_on}/{n_off} per condition)", fontsize=10)
    ax.legend(); fig.tight_layout()
    out = out_path.parent / "fig_pilot_build_solve.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
