"""All Qwen families on one 3-panel oracle win-rate figure.

Same style as the per-family 3-panel figures: x = model size (B params, linear),
y = win rate averaged across all N values, rollout-bootstrap ±1 SE error bars,
value labels. Three panels: manual|not-tool / via-tool / overall.
One line per Qwen family (Qwen3.5, Qwen3, Qwen2.5).

Usage:  python -m scripts.plot_oracle_qwen_combined
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

# family -> list of (size_b, episodes_dir)
FAMILIES = [
    ("Qwen3.5", "#2171b5", [
        (2.0,  ORACLE_DIR / "qwen35_v2" / "2B"),
        (4.0,  ORACLE_DIR / "qwen35_v2" / "4B"),
        (9.0,  ORACLE_DIR / "qwen35_v2" / "9B"),
        (27.0, ORACLE_DIR / "qwen35_v2" / "27B"),
    ]),
    ("Qwen2.5", "#d94801", [
        (0.5,  ORACLE_DIR / "qwen25_v1" / "0.5B"),
        (1.5,  ORACLE_DIR / "qwen25_v1" / "1.5B"),
        (3.0,  ORACLE_DIR / "qwen25_v1" / "3B"),
        (7.0,  ORACLE_DIR / "qwen25_v1" / "7B"),
        (14.0, ORACLE_DIR / "qwen25_v1" / "14B"),
        (32.0, ORACLE_DIR / "qwen25_v1" / "32B"),
        (72.0, ORACLE_DIR / "qwen25_v1" / "72B"),
    ]),
    ("Qwen3", "#238b45", [
        (0.6,  ORACLE_DIR / "qwen3_dense_v1" / "0.6B"),
        (1.7,  ORACLE_DIR / "qwen3_dense_v1" / "1.7B"),
        (4.0,  ORACLE_DIR / "qwen3_dense_v1" / "4B"),
        (8.0,  ORACLE_DIR / "qwen3_dense_v1" / "8B"),
        (14.0, ORACLE_DIR / "qwen3_dense_v1" / "14B"),
        (32.0, ORACLE_DIR / "qwen3_dense_v1" / "32B"),
    ]),
]


def load_rows(path: Path) -> list[dict]:
    ep = path / "episodes.jsonl"
    if not ep.exists():
        return []
    rows = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
    return [r for r in rows if not r.get("error")]


def scored_flags(rows: list[dict]) -> np.ndarray:
    """Returns (M, 3) array: [win_overall, win_via_tool, win_via_not_tool] per scored row.
    One scored row per (rollout, oracle_i) pair."""
    out = []
    for r in rows:
        for s in score_episode(r):
            w = int(s["win_i"])
            via_tool = int(s["win_i"] and s["win_via"] == "tool")
            via_not = int(s["win_i"] and s["win_via"] != "tool")
            out.append([w, via_tool, via_not])
    return np.array(out, dtype=float) if out else np.zeros((0, 3))


def rollout_bootstrap(rows: list[dict], n_boot: int = N_BOOT):
    """Resample at rollout level, return (mean, se) for each of 3 metrics."""
    # group scored flags by rollout index
    groups = []
    for r in rows:
        flags = []
        for s in score_episode(r):
            w = int(s["win_i"])
            flags.append([w,
                          int(s["win_i"] and s["win_via"] == "tool"),
                          int(s["win_i"] and s["win_via"] != "tool")])
        if flags:
            groups.append(np.array(flags, dtype=float))

    if not groups:
        return np.full(3, np.nan), np.full(3, np.nan)

    n = len(groups)
    boot_means = np.zeros((n_boot, 3))
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        sample = np.concatenate([groups[i] for i in idx], axis=0)
        boot_means[b] = sample.mean(axis=0)

    all_flags = np.concatenate(groups, axis=0)
    mean = all_flags.mean(axis=0)
    se = boot_means.std(axis=0)
    return mean, se


def size_label(b: float) -> str:
    return f"{int(b)}B" if b >= 1.0 else f"{b:.1f}B"


def main():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    panel_titles = [
        "P(win via manual | not solved by tool)",
        "P(win via oracle tool)",
        "P(win overall)",
    ]
    panel_idx = [2, 1, 0]  # indices into mean/se arrays: not_tool, tool, overall

    for ax, title in zip(axes, panel_titles):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Model size (B params)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Probability", fontsize=10, fontweight="bold")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(alpha=0.25)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    handles = []
    label_offsets = {"Qwen3.5": 8, "Qwen3": -14, "Qwen2.5": 8}
    for fam_name, color, sizes in FAMILIES:
        xs, means, ses = [], [], []
        for size_b, ep_dir in sizes:
            rows = load_rows(ep_dir)
            if not rows:
                print(f"  (skip {ep_dir})")
                continue
            mean, se = rollout_bootstrap(rows)
            xs.append(size_b)
            means.append(mean)
            ses.append(se)
            print(f"  {fam_name} {size_label(size_b):5s}  "
                  f"overall={mean[0]:.2f}±{se[0]:.2f}  "
                  f"tool={mean[1]:.2f}  not_tool={mean[2]:.2f}")

        if not xs:
            continue
        xs = np.array(xs)
        means = np.array(means)   # (n_sizes, 3)
        ses = np.array(ses)       # (n_sizes, 3)

        for ax, pidx in zip(axes, panel_idx):
            ys = means[:, pidx]
            errs = ses[:, pidx]
            h = ax.errorbar(xs, ys, yerr=errs, fmt="o-", color=color,
                            linewidth=1.8, markersize=5, capsize=3,
                            label=fam_name, alpha=0.6)
            yoff = label_offsets.get(fam_name, 7)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.2f}", (x, y),
                            textcoords="offset points", xytext=(0, yoff),
                            ha="center", fontsize=7.5, color=color)
        handles.append(h)

    # x-ticks: label only "landmark" sizes to avoid crowding; minor ticks at all sizes
    all_sizes = sorted({s for _, _, szs in FAMILIES for s, _ in szs})
    label_at = {0.5, 1.5, 2.0, 4.0, 9.0, 14.0, 27.0, 32.0, 72.0}
    for ax in axes:
        ax.set_xticks(all_sizes, minor=True)
        ax.set_xticks([s for s in all_sizes if s in label_at])
        ax.set_xticklabels([size_label(s) for s in all_sizes if s in label_at],
                           fontsize=9, rotation=45, ha="right")
        ax.tick_params(axis="x", which="minor", length=3)

    fig.legend(handles=handles, loc="lower center",
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.08),
               framealpha=0.9)

    out = ORACLE_DIR / "fig_oracle_qwen_combined_3panel.png"
    out_pdf = out.with_suffix(".pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out}")
    print(f"  wrote {out_pdf}")


if __name__ == "__main__":
    main()
