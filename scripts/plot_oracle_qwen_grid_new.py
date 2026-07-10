"""3x3 grid of oracle win-rate panels — for the 2026-07-10 vLLM size-sweep.

Identical layout/style to scripts/plot_oracle_qwen_grid.py, but sources rows from the
combined episodes file of the new sweep (runs/oracle/combined_episodes.jsonl), filtering
by the `model` HF id rather than the old per-size directory layout.

Rows (top to bottom): Qwen2.5, Qwen3, Qwen3.5
Columns: P(win via manual | not solved by tool) / P(win via oracle tool) / P(win overall)

Usage:  python -m scripts.plot_oracle_qwen_grid_new
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
EPISODES = ORACLE_DIR / "combined_episodes.jsonl"
RNG = np.random.default_rng(42)
N_BOOT = 2000

# (family label, color, [(size_b, model HF id)])
ROWS = [
    ("Qwen2.5", "#d94801", [
        (0.5, "Qwen/Qwen2.5-0.5B-Instruct"), (1.5, "Qwen/Qwen2.5-1.5B-Instruct"),
        (3.0, "Qwen/Qwen2.5-3B-Instruct"),   (7.0, "Qwen/Qwen2.5-7B-Instruct"),
        (14.0, "Qwen/Qwen2.5-14B-Instruct"), (32.0, "Qwen/Qwen2.5-32B-Instruct"),
    ]),
    ("Qwen3", "#238b45", [
        (0.6, "Qwen/Qwen3-0.6B"), (1.7, "Qwen/Qwen3-1.7B"), (4.0, "Qwen/Qwen3-4B"),
        (8.0, "Qwen/Qwen3-8B"),   (14.0, "Qwen/Qwen3-14B"), (32.0, "Qwen/Qwen3-32B"),
    ]),
    ("Qwen3.5", "#2171b5", [
        (2.0, "Qwen/Qwen3.5-2B"), (4.0, "Qwen/Qwen3.5-4B"),
        (9.0, "Qwen/Qwen3.5-9B"), (27.0, "Qwen/Qwen3.5-27B"),
    ]),
]

PANELS = [
    ("P(win manually | not via tool)", 2),
    ("P(win via oracle tool)", 1),
    ("P(win overall)", 0),
]


def load_all() -> dict[str, list[dict]]:
    """model HF id -> list of (non-error) rollout rows from the combined file."""
    by_model: dict[str, list[dict]] = {}
    for line in EPISODES.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        by_model.setdefault(r.get("model", "?"), []).append(r)
    return by_model


def rollout_bootstrap(rows: list[dict], n_boot: int = N_BOOT):
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


def main():
    by_model = load_all()

    all_sizes = sorted({s for _, _, sizes in ROWS for s, _ in sizes})
    all_labels = {}
    for _, _, sizes in ROWS:
        for s, mid in sizes:
            all_labels[s] = mid.split("-")[-1].replace("Instruct", "").strip() or f"{s}B"
    # nicer size labels
    all_labels = {0.5: "0.5B", 0.6: "0.6B", 1.5: "1.5B", 1.7: "1.7B", 2.0: "2B",
                  3.0: "3B", 4.0: "4B", 7.0: "7B", 8.0: "8B", 9.0: "9B",
                  14.0: "14B", 27.0: "27B", 32.0: "32B"}
    x_pad = all_sizes[-1] * 0.05
    xlim = (-x_pad, all_sizes[-1] + x_pad)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)

    for row_i, (fam_name, color, sizes) in enumerate(ROWS):
        xs, means, ses = [], [], []
        for size_b, model_id in sizes:
            rows = by_model.get(model_id, [])
            if not rows:
                print(f"  WARN: no rows for {model_id}")
                continue
            mean, se = rollout_bootstrap(rows)
            xs.append(size_b)
            means.append(mean)
            ses.append(se)

        xs = np.array(xs)
        means = np.array(means)
        ses = np.array(ses)

        for col_i, (panel_title, pidx) in enumerate(PANELS):
            ax = axes[row_i][col_i]
            ys = means[:, pidx]
            errs = ses[:, pidx]

            ax.errorbar(xs, ys, yerr=errs, fmt="o-", color=color,
                        linewidth=1.8, markersize=5, capsize=3)

            if row_i == 0:
                ax.set_title(panel_title, fontsize=11, fontweight="bold")

            if col_i == 0:
                ax.set_ylabel(f"{fam_name}\n\nProbability",
                              fontsize=10, fontweight="bold")
            else:
                ax.set_ylabel("Probability", fontsize=10, fontweight="bold")

            if row_i == 2:
                ax.set_xlabel("Model size (B params)", fontsize=10, fontweight="bold")

            ax.set_ylim(-0.05, 1.08)
            ax.set_xlim(xlim)
            label_sizes = {0.5, 2.0, 4.0, 9.0, 14.0, 27.0, 32.0}
            ax.set_xticks([s for s in all_sizes if s in label_sizes])
            ax.set_xticklabels([all_labels[s] for s in all_sizes if s in label_sizes],
                               fontsize=9, rotation=45, ha="right")
            ax.grid(alpha=0.25)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

        print(f"  {fam_name}: {len(xs)} sizes plotted")

    out = ORACLE_DIR / "fig_oracle_qwen_3x3_new.png"
    out_pdf = out.with_suffix(".pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out}")
    print(f"  wrote {out_pdf}")


if __name__ == "__main__":
    main()
