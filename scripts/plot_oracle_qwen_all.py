"""Plot all Qwen oracle runs on one 3-panel figure.

Panels (x-axis = N, one line per model size):
  1. Overall win rate
  2. Win via tool   (tool path was cheaper & within budget)
  3. Win via manual | not via tool  (manual path was cheaper OR no tool win)

Families + sizes:
  Qwen3.5  : 2B 4B 9B 27B   (blue shades)
  Qwen3    : 0.6B 1.7B 4B 8B 14B 32B  (green shades)
  Qwen2.5  : 0.5B 1.5B 3B 7B 14B 32B 72B  (orange shades)

Usage:  python -m scripts.plot_oracle_qwen_all
"""

from __future__ import annotations

import json
from pathlib import Path

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.oracle_counterfactual import score_episode

# ── data layout ──────────────────────────────────────────────────────────────
ORACLE_DIR = Path("runs/oracle")

FAMILIES = {
    "Qwen3.5": {
        "dirs": {
            2.0:  ORACLE_DIR / "qwen35_v2" / "2B",
            4.0:  ORACLE_DIR / "qwen35_v2" / "4B",
            9.0:  ORACLE_DIR / "qwen35_v2" / "9B",
            27.0: ORACLE_DIR / "qwen35_v2" / "27B",
        },
        "color_start": "#cce5ff",
        "color_end":   "#084594",
    },
    "Qwen3": {
        "dirs": {
            0.6:  ORACLE_DIR / "qwen3_dense_v1" / "0.6B",
            1.7:  ORACLE_DIR / "qwen3_dense_v1" / "1.7B",
            4.0:  ORACLE_DIR / "qwen3_dense_v1" / "4B",
            8.0:  ORACLE_DIR / "qwen3_dense_v1" / "8B",
            14.0: ORACLE_DIR / "qwen3_dense_v1" / "14B",
            32.0: ORACLE_DIR / "qwen3_dense_v1" / "32B",
        },
        "color_start": "#c7e9c0",
        "color_end":   "#00441b",
    },
    "Qwen2.5": {
        "dirs": {
            0.5:  ORACLE_DIR / "qwen25_v1" / "0.5B",
            1.5:  ORACLE_DIR / "qwen25_v1" / "1.5B",
            3.0:  ORACLE_DIR / "qwen25_v1" / "3B",
            7.0:  ORACLE_DIR / "qwen25_v1" / "7B",
            14.0: ORACLE_DIR / "qwen25_v1" / "14B",
            32.0: ORACLE_DIR / "qwen25_v1" / "32B",
            72.0: ORACLE_DIR / "qwen25_v1" / "72B",
        },
        "color_start": "#fdd0a2",
        "color_end":   "#7f2704",
    },
}

NS = [4, 8, 12, 16, 20]


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def interp_color(start_hex, end_hex, t):
    s, e = hex_to_rgb(start_hex), hex_to_rgb(end_hex)
    return tuple(s[i] + (e[i] - s[i]) * t for i in range(3))


def load_rows(path: Path):
    ep = path / "episodes.jsonl"
    if not ep.exists():
        return []
    rows = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
    return [r for r in rows if not r.get("error")]


def compute_rates(rows):
    """For each N return (overall, via_tool, via_manual_or_not_tool)."""
    overall, via_tool, via_not_tool = {}, {}, {}
    for n in NS:
        sub = [r for r in rows if r["n"] == n]
        if not sub:
            overall[n] = via_tool[n] = via_not_tool[n] = float("nan")
            continue
        scored = [s for r in sub for s in score_episode(r)]
        tot = len(scored)
        wins = [s for s in scored if s["win_i"]]
        w_overall = len(wins) / tot
        w_tool = sum(1 for s in wins if s["win_via"] == "tool") / tot
        # "manual | not-tool": won, and the manual path was used (not tool)
        w_not_tool = sum(1 for s in wins if s["win_via"] != "tool") / tot
        overall[n] = w_overall
        via_tool[n] = w_tool
        via_not_tool[n] = w_not_tool
    return overall, via_tool, via_not_tool


def label_size(size_b):
    if size_b >= 1.0:
        return f"{int(size_b)}B"
    return f"{size_b:.1f}B"


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    titles = ["Overall win rate", "Win via tool", "Win via manual | not tool"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("N (number of tools)", fontsize=10)
        ax.set_ylabel("Counterfactual win rate", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(NS)
        ax.grid(alpha=0.25)

    legend_handles = []

    for fam_name, fam in FAMILIES.items():
        sizes = sorted(fam["dirs"])
        n_sizes = len(sizes)
        for k, size in enumerate(sizes):
            t = k / max(n_sizes - 1, 1)
            color = interp_color(fam["color_start"], fam["color_end"], t)
            rows = load_rows(fam["dirs"][size])
            if not rows:
                print(f"  (no data: {fam['dirs'][size]})")
                continue
            overall, via_tool, via_not_tool = compute_rates(rows)
            lbl = f"{fam_name} {label_size(size)}"
            lw = 1.4 + 0.4 * t  # thicker = larger model
            for ax, rate_dict in zip(axes, [overall, via_tool, via_not_tool]):
                ys = [rate_dict[n] for n in NS]
                h, = ax.plot(NS, ys, "o-", color=color, linewidth=lw,
                             markersize=4, label=lbl, alpha=0.85)
            legend_handles.append(
                plt.Line2D([0], [0], color=color, linewidth=lw, marker="o",
                           markersize=4, label=lbl)
            )
            print(f"  {lbl:22s}  overall={[f'{overall[n]:.2f}' for n in NS]}")

    # single legend below the figure
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.18),
               framealpha=0.9)

    out = ORACLE_DIR / "fig_oracle_qwen_all_3panel.png"
    out_pdf = out.with_suffix(".pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out}")
    print(f"  wrote {out_pdf}")


if __name__ == "__main__":
    main()
