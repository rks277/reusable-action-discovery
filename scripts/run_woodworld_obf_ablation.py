"""Woodworld obfuscation x hint ablation (Haiku, fixed cell p=0.5, N=10).

Question: is Haiku's near-zero build rate driven by the OBFUSCATION (nonsense item
names hide the recipe) or by the hidden recipe/payoff structure itself? We run the
full 2x2 -- {obfuscate on/off} x {hint on/off} -- on the SAME 40 paired seeds
(rep r => relabel_seed=gather_seed=r), so within a rep the gather draws are
identical and only the named condition changes. With obfuscate=False the items are
the real English words wood/stick/axe; the recipes and the axe's +2 payoff are
still never stated, so the agent must still discover them.

The three conditions the figure highlights -- obf+hint, obf+no-hint, no-obf+hint --
plus no-obf+no-hint (the clean obfuscation main effect under no hint) all fall out
of the 2x2. Emits episodes.jsonl + a grouped-bar figure of tried / built / solved.

Usage: PYTHONPATH=. python -m scripts.run_woodworld_obf_ablation
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
import numpy as np

import scripts.woodworld_config as cfg            # sets DEFAULT_SCHEME='letter'
from scripts.validate_woodworld import budget_for
from scripts.woodworld import run

MODEL = "claude-haiku-4-5-20251001"
N = 10
P = 0.5
REPS = 40
# (obfuscate, hint) cells of the 2x2. Set WW_ABLATION_CONDS=noobf to run only the
# no-obfuscation cells (the obfuscated cells are reused from a prior run via
# WW_ABLATION_DIR, appending to its episodes.jsonl); default runs the full 2x2.
ALL_CONDITIONS = [(True, True), (True, False), (False, True), (False, False)]
import os
CONDITIONS = ([(False, True), (False, False)]
              if os.environ.get("WW_ABLATION_CONDS") == "noobf" else ALL_CONDITIONS)
CONC = cfg.CONCURRENCY["anthropic"]


def stick_metrics(result: dict, trace: list) -> tuple[bool, int]:
    """(got_sticks, stick_crafts) from the trace: count successful combines whose
    OUTPUT side contains the stick token (the intermediate that is itself an
    ingredient). Robust to obfuscation -- read the recorded label."""
    stick_tok = result["labels"]["stick"]
    n_stick = 0
    for step in trace:
        obs = step["obs"]
        if obs.startswith("You combine") and "(new)" in obs:
            out_part = obs.split(" into ", 1)[1].split(" (new)")[0]
            if stick_tok in out_part.split():
                n_stick += 1
    return n_stick > 0, n_stick


async def main():
    load_dotenv()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # reuse an existing run dir (append the new cells) when given, else fresh
    existing = os.environ.get("WW_ABLATION_DIR")
    out_dir = Path(existing) if existing else Path("runs") / f"woodworld_obf_ablation_p05_N10_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    if not existing:
        out_path.write_text("")
    budget = budget_for(N, P)
    print(f"Obf ablation: {MODEL} N={N} p={P} budget={budget} "
          f"conds={CONDITIONS} x {REPS} reps -> {out_path}", flush=True)

    sem = asyncio.Semaphore(CONC)
    lock = asyncio.Lock()

    async def one(obfuscate, hint, rep):
        async with sem:
            t0 = time.time()
            try:
                result, trace = await run(
                    MODEL, n=N, relabel_seed=rep, gather_seed=rep, hint=hint,
                    obfuscate=obfuscate, gather_prob=P, budget=budget,
                    max_turns=cfg.MAX_TURNS, no_progress_window=round(12 / P))
                got_sticks, stick_crafts = stick_metrics(result, trace)
                row = {"model": MODEL, "n": N, "gather_prob": P, "obfuscate": obfuscate,
                       "hint": hint, "budget": budget, "rep": rep,
                       "relabel_seed": rep, "gather_seed": rep,
                       "labels": result["labels"],
                       "actions": [x["action"] for x in trace],
                       "obs": [x["obs"] for x in trace],
                       "tried": result["craft_attempts"] > 0,
                       "got_sticks": got_sticks, "stick_crafts": stick_crafts,
                       "built": bool(result["built_axe"]),
                       "solved": bool(result["solved"]),
                       "total_actions": result["total_actions"],
                       "stopped_reason": result["stopped_reason"],
                       "usage": result["usage"], "elapsed_s": round(time.time() - t0, 2)}
            except Exception as e:
                row = {"model": MODEL, "n": N, "gather_prob": P, "obfuscate": obfuscate,
                       "hint": hint, "rep": rep, "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  obf={obfuscate!s:5s} hint={hint!s:5s} rep={rep:2d} "
                      f"tried={row.get('tried')} built={row.get('built')} "
                      f"solved={row.get('solved')} act={row.get('total_actions')}/{budget} "
                      f"{'(err)' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(o, h, r) for (o, h) in CONDITIONS for r in range(REPS)))
    plot(out_path)


def _rate(rows, key):
    vals = [bool(r[key]) for r in rows if key in r]
    n = len(vals)
    p = sum(vals) / n if n else 0.0
    se = sqrt(p * (1 - p) / n) if n else 0.0
    return p, se, n


COND_LABEL = {(True, True): "obf + hint", (True, False): "obf + no-hint",
              (False, True): "no-obf + hint", (False, False): "no-obf + no-hint"}
COND_COLOR = {(True, True): "#2ca02c", (True, False): "#7fbf7f",
              (False, True): "#1f77b4", (False, False): "#9ecae1"}


def plot(out_path: Path):
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    # dedup by (obfuscate, hint, rep) keeping the LAST write (a re-run of a cell
    # supersedes earlier partial rows appended to the same file)
    by_key = {(r["obfuscate"], r["hint"], r["rep"]): r for r in rows}
    rows = list(by_key.values())
    metrics = [("tried", "tried combining"), ("built", "built axe"), ("solved", "solved")]

    print(f"\n{'condition':18s}" + "".join(f"{lbl:>20s}" for _, lbl in metrics))
    cell = {}
    for cond in ALL_CONDITIONS:
        sub = [r for r in rows if (r["obfuscate"], r["hint"]) == cond]
        cell[cond] = {k: _rate(sub, k) for k, _ in metrics}
        line = f"{COND_LABEL[cond]:18s}"
        for k, _ in metrics:
            p, se, nn = cell[cond][k]
            line += f"{p:.2f}±{se:.2f}(n{nn})".rjust(20)
        print(line)

    x = np.arange(len(metrics))
    w = 0.2
    n_per = cell[ALL_CONDITIONS[0]]["built"][2]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, cond in enumerate(ALL_CONDITIONS):
        vals = [cell[cond][k][0] for k, _ in metrics]
        errs = [cell[cond][k][1] for k, _ in metrics]
        ax.bar(x + (i - 1.5) * w, vals, w, yerr=errs, capsize=3,
               label=COND_LABEL[cond], color=COND_COLOR[cond])
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title(f"woodworld obfuscation x hint ablation: {MODEL.split('-')[1].capitalize()} "
                 f"(N={N}, p={P}, n={n_per}/cond)", fontsize=11)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    out = out_path.parent / "fig_obf_ablation.png"
    figs_out = Path("figs/woodworld/obf_ablation") / "fig_woodworld_obf_ablation_haiku_p05_N10.png"
    figs_out.parent.mkdir(parents=True, exist_ok=True)
    for dest in (out, figs_out):
        fig.savefig(dest, dpi=150)
        fig.savefig(dest.with_suffix(".pdf"))
    plt.close(fig)
    print(f"\nwrote {out} and {figs_out} (+ .pdf)")


if __name__ == "__main__":
    asyncio.run(main())
