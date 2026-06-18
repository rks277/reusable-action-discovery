"""Bar-chart figure for the playground-injection experiment (run_playground_experiment.py).

Shows, per arm (none / saw-exploration / saw-build), the build rate and solve rate
(proportions over the arm's episodes) with binomial standard-error bars, plus the
held-both and recognition P(built|held) numbers annotated. One figure.

Usage: PYTHONPATH=. python -m scripts.plot_playground_experiment <run_dir>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["none", "explore", "build"]
LABEL = {"none": "no playground", "explore": "saw-exploration\n(s9, no build)",
         "build": "saw-build\n(s4, built+used)"}


def collected(r):
    s = set()
    for o in r.get("obs", []):
        s.update(re.findall(r"and a (\w+)\.", o))
        s.update(re.findall(r"hold \d+ (\w+)\(s\)", o))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rows = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    model = rows[0]["model"].split("-")[1] if rows[0]["model"].startswith("claude-") else rows[0]["model"]
    n, t = rows[0]["n"], rows[0]["n_types"]

    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)

    def se(k, m):
        p = k / m
        return sqrt(p * (1 - p) / m) if m else 0.0

    build_p, build_se, solve_p, solve_se, rec_p, rec_se = [], [], [], [], [], []
    build_lab, solve_lab, rec_lab, ns = [], [], [], []
    for arm in ARMS:
        rs = by.get(arm, [])
        m = len(rs)
        b = sum(bool(r["built_machine"]) for r in rs)
        s = sum(bool(r["solved"]) for r in rs)
        held = [r for r in rs if set(r["labels"]["recipe"]) <= collected(r)]
        hb = sum(bool(r["built_machine"]) for r in held)
        rec = hb / len(held) if held else float("nan")
        build_p.append(b / m if m else 0); build_se.append(se(b, m)); build_lab.append(f"{b}/{m}")
        solve_p.append(s / m if m else 0); solve_se.append(se(s, m)); solve_lab.append(f"{s}/{m}")
        rec_p.append(rec if held else 0.0); rec_se.append(se(hb, len(held)) if held else 0.0)
        rec_lab.append(f"{hb}/{len(held)}" if held else "n/a")
        ns.append(m)
        print(f"{arm:<8} n={m} built={b} solved={s} held={len(held)} recognition={rec:.2f}")

    x = np.arange(len(ARMS)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5.4))
    series = [(-w, build_p, build_se, build_lab, "#2c7fb8", "P(built machine)"),
              (0.0, rec_p, rec_se, rec_lab, "#253494", "recognition P(built | held both)"),
              (w, solve_p, solve_se, solve_lab, "#7fcdbb", "P(solved)")]
    for off, ps, ses, labs, col, lab in series:
        bars = ax.bar(x + off, ps, w, yerr=ses, capsize=3, color=col, label=lab)
        for rect, p, txt in zip(bars, ps, labs):
            ax.text(rect.get_x()+rect.get_width()/2, p+0.02, txt, ha="center", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in ARMS])
    ax.set_ylabel("proportion")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"{model.capitalize()} playground-injection experiment "
                 f"(N={n}, T={t}, budget-proper; {ns[0]} reps/arm, paired worlds)\n"
                 f"recognition denominator = episodes that held both ingredients")
    ax.legend(loc="upper left", fontsize=9)

    out = Path(f"figs/toolworld/fig_playground_experiment_{model}_n{n}_T{t}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
