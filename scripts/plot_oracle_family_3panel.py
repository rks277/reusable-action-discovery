"""Regenerate the per-family oracle win-rate 3-panel figures.

Matches the style of the original per-family figures (single blue line, rollout-bootstrap
±1 SE, value labels) but with no suptitle and bold axis labels.

Usage:  python -m scripts.plot_oracle_family_3panel
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.oracle_counterfactual import score_episode

ORACLE_DIR = Path("runs/oracle")
RNG = np.random.default_rng(42)
N_BOOT = 2000
COLOR = "#4c78a8"

FAMILIES = [
    ("Qwen3.5", ORACLE_DIR / "qwen35_v2", [
        (2.0,  "2B"),
        (4.0,  "4B"),
        (9.0,  "9B"),
        (27.0, "27B"),
    ], "fig_qwen35_v2_win_rates_3panel"),
    ("Qwen3", ORACLE_DIR / "qwen3_dense_v1", [
        (0.6,  "0.6B"),
        (1.7,  "1.7B"),
        (4.0,  "4B"),
        (8.0,  "8B"),
        (14.0, "14B"),
        (32.0, "32B"),
    ], "fig_qwen3dense_win_rates_3panel"),
    ("Qwen2.5", ORACLE_DIR / "qwen25_v1", [
        (0.5,  "0.5B"),
        (1.5,  "1.5B"),
        (3.0,  "3B"),
        (7.0,  "7B"),
        (14.0, "14B"),
        (32.0, "32B"),
        (72.0, "72B"),
    ], "fig_qwen25_win_rates_3panel"),
]


def load_rows(path: Path) -> list[dict]:
    ep = path / "episodes.jsonl"
    if not ep.exists():
        return []
    rows = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
    return [r for r in rows if not r.get("error")]


def rollout_bootstrap(rows: list[dict], n_boot: int = N_BOOT):
    """Resample at rollout level -> (mean, se) for [overall, via_tool, via_not_tool]."""
    groups = []
    for r in rows:
        flags = []
        for s in score_episode(r):
            flags.append([
                int(s["win_i"]),
                int(s["win_i"] and s["win_via"] == "tool"),
                int(s["win_i"] and s["win_via"] != "tool"),
            ])
        if flags:
            groups.append(np.array(flags, dtype=float))
    if not groups:
        return np.full(3, np.nan), np.full(3, np.nan)
    n = len(groups)
    boot = np.zeros((n_boot, 3))
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        sample = np.concatenate([groups[i] for i in idx], axis=0)
        boot[b] = sample.mean(axis=0)
    all_flags = np.concatenate(groups, axis=0)
    return all_flags.mean(axis=0), boot.std(axis=0)


def make_figure(fam_name, base_dir, sizes, out_stem):
    xs, means, ses, xlabels = [], [], [], []
    for size_b, subdir in sizes:
        rows = load_rows(base_dir / subdir)
        if not rows:
            print(f"  (skip {base_dir / subdir})")
            continue
        mean, se = rollout_bootstrap(rows)
        xs.append(size_b)
        means.append(mean)
        ses.append(se)
        xlabels.append(subdir)
        print(f"  {fam_name} {subdir:6s}  overall={mean[0]:.2f}±{se[0]:.2f}  "
              f"tool={mean[1]:.2f}  not_tool={mean[2]:.2f}")

    if not xs:
        print(f"  No data for {fam_name}, skipping.")
        return

    xs = np.array(xs)
    means = np.array(means)
    ses = np.array(ses)

    # panels: not_tool (idx 2), tool (idx 1), overall (idx 0)
    panel_cfg = [
        ("P(win via manual | not solved by tool)", 2),
        ("P(win via oracle tool)", 1),
        ("P(win overall)", 0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, (title, pidx) in zip(axes, panel_cfg):
        ys = means[:, pidx]
        errs = ses[:, pidx]
        ax.errorbar(xs, ys, yerr=errs, fmt="o-", color=COLOR,
                    linewidth=1.8, markersize=5, capsize=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Model size (B params)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Probability", fontsize=10, fontweight="bold")
        ax.set_ylim(-0.05, 1.08)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.grid(alpha=0.25)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    out = base_dir / f"{out_stem}.png"
    out_pdf = base_dir / f"{out_stem}.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  wrote {out_pdf}")


def main():
    for fam_name, base_dir, sizes, out_stem in FAMILIES:
        print(f"\n=== {fam_name} ===")
        make_figure(fam_name, base_dir, sizes, out_stem)


if __name__ == "__main__":
    main()
