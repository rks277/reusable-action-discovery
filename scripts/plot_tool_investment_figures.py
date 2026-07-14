"""Generate the main figures for the online tool-investment paper.

Outputs PNG and PDF files under ``figs/tool-investment``.
The Haiku panels read the preserved per-seed sessions; economic and GPT panels
read their locked analyses. The RL transfer panel uses the publication values
reported in ``docs/rl-phase1-results.md``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "figs" / "tool-investment"
SEEDS = list(range(2000, 2024))

NAVY = "#28536B"
BLUE = "#3C78A8"
ORANGE = "#E07A3F"
GREEN = "#3E8E7E"
PURPLE = "#7566A8"
GRAY = "#747B83"
LIGHT = "#D9DEE3"
INK = "#22252A"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#6B7075",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png.relative_to(ROOT)} and {pdf.relative_to(ROOT)}")


def first_sight_from_mapping(path: Path, field: str) -> float:
    session = load_json(path)
    positions = list(session[field].values())
    if not positions:
        return np.nan
    return float(np.mean(np.asarray(positions) == 1))


def first_sight_r3(seed: int) -> float:
    path = RUNS / "arm_a1_announce_n-announced" / f"seed_{seed}" / "sessions.jsonl"
    line = next(line for line in path.read_text().splitlines() if line.strip())
    session = json.loads(line)
    stream = load_json(
        RUNS / "arm_a1_announce_n-announced" / f"seed_{seed}" / "stream.json"
    )
    positions: list[int] = []
    for record in session["records"]:
        if record.get("scripts_authored"):
            positions.extend([stream[record["item_idx"]]["class_position"]] * len(record["scripts_authored"]))
    if not positions:
        return np.nan
    return float(np.mean(np.asarray(positions) == 1))


def ladder_seed_values() -> dict[str, np.ndarray]:
    specs = {
        "R0": ("urn_haiku_n-announced", "kept"),
        "R1": ("urn_tool_haiku_n-announced", "kept"),
        "R2": ("claim_solver_haiku_n-announced", "claimed"),
        "R2c": ("claim_solver_code_haiku_n-announced", "claimed"),
    }
    values: dict[str, np.ndarray] = {}
    for rung, (directory, field) in specs.items():
        values[rung] = np.asarray(
            [
                first_sight_from_mapping(
                    RUNS / directory / f"seed_{seed}" / "session.json", field
                )
                for seed in SEEDS
            ]
        )
    values["R3"] = np.asarray([first_sight_r3(seed) for seed in SEEDS])
    return values


def figure_task_and_core() -> None:
    """Figure 1: task isomorphism and Haiku A2 paired dissociation."""
    stream = load_json(RUNS / "urn_haiku_n-announced" / "seed_2000" / "stream.json")
    values = ladder_seed_values()
    urn = values["R0"] * 100
    tool = values["R3"] * 100

    fig = plt.figure(figsize=(11.2, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1])
    ax_task = fig.add_subplot(grid[:, 0])
    ax_pair = fig.add_subplot(grid[:, 1])

    shown = stream[:12]
    class_ids = sorted({item["class_id"] for item in shown})
    palette = plt.get_cmap("tab10")
    class_color = {cid: palette(i % 10) for i, cid in enumerate(class_ids)}
    xs = np.arange(1, len(shown) + 1)

    for y, label in [(1.0, "Abstract keep/pass"), (0.0, "Reusable script")]:
        for x, item in zip(xs, shown):
            face = class_color[item["class_id"]]
            if y == 1.0:
                ax_task.scatter(x, y, s=310, color=face, edgecolor="white", linewidth=1.2, zorder=3)
                ax_task.text(x, y, str(item["class_id"]), ha="center", va="center",
                             color="white", fontsize=8, weight="bold")
            else:
                ax_task.scatter(x, y, s=310, marker="s", color=face, edgecolor="white",
                                linewidth=1.2, zorder=3)
                ax_task.text(x, y, f"P{item['class_id']}", ha="center", va="center",
                             color="white", fontsize=7.5, weight="bold")
        ax_task.text(0.1, y, label, ha="right", va="center", fontsize=10, weight="bold")

    for x in xs:
        ax_task.plot([x, x], [0.18, 0.82], color=LIGHT, linewidth=0.8, zorder=0)
    ax_task.text(
        6.5,
        0.50,
        "same latent class stream",
        ha="center",
        va="center",
        fontsize=9,
        color=GRAY,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
    )
    ax_task.set_xlim(0.2, 12.8)
    ax_task.set_ylim(-0.55, 1.55)
    ax_task.set_xticks(xs)
    ax_task.set_xlabel("Stream position (first 12 of 60)")
    ax_task.set_yticks([])
    ax_task.set_title("a  One allocation problem, two decision frames", loc="left", weight="bold")
    for spine in ax_task.spines.values():
        spine.set_visible(False)

    jitter = np.linspace(-0.055, 0.055, len(SEEDS))
    for i, (u, t) in enumerate(zip(urn, tool)):
        ax_pair.plot([0 + jitter[i], 1 + jitter[i]], [u, t], color=LIGHT, linewidth=1.0, zorder=1)
        ax_pair.scatter(0 + jitter[i], u, s=27, color=BLUE, edgecolor="white", linewidth=0.5, zorder=2)
        ax_pair.scatter(1 + jitter[i], t, s=27, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=2)
    means = [np.nanmean(urn), np.nanmean(tool)]
    ax_pair.plot([0, 1], means, color=INK, linewidth=2.1, zorder=3)
    ax_pair.scatter([0, 1], means, s=105, color=[BLUE, ORANGE], edgecolor=INK, linewidth=1.2, zorder=4)
    ax_pair.text(0, means[0] - 8, f"{means[0]:.0f}%", ha="center", va="top", weight="bold", color=BLUE)
    ax_pair.text(1, means[1] - 8, f"{means[1]:.0f}%", ha="center", va="top", weight="bold", color=ORANGE)
    ax_pair.set_xticks([0, 1])
    ax_pair.set_xticklabels(["Abstract R0", "Script construction R3"])
    ax_pair.set_xlim(-0.35, 1.35)
    ax_pair.set_ylim(-5, 108)
    ax_pair.set_ylabel("First-sight commitments (%)")
    ax_pair.set_title("b  Same-information Haiku dissociation", loc="left", weight="bold")
    ax_pair.grid(axis="y", color="#E8EBEE", linewidth=0.8)
    ax_pair.text(
        0.5,
        -0.19,
        "24 paired canonical streams · A2 discloses N=8 in both frames",
        transform=ax_pair.transAxes,
        ha="center",
        fontsize=8.5,
        color=GRAY,
    )
    save(fig, "fig1_task_and_core_dissociation")


def figure_ladder() -> None:
    """Figure 2: paired seed-level framing ladder."""
    values = ladder_seed_values()
    labels = ["R0", "R1", "R2", "R2c", "R3"]
    descriptions = [
        "text\nkeep/pass",
        "tool-call\nkeep/pass",
        "declarative\nclaim",
        "code-required\nclaim",
        "full script\nconstruction",
    ]
    matrix = np.vstack([values[label] * 100 for label in labels]).T
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.4, 5.0), constrained_layout=True)
    for row in matrix:
        ax.plot(x, row, color="#CDD2D7", linewidth=0.9, alpha=0.85, zorder=1)
        ax.scatter(x, row, s=22, color="#A8AFB6", edgecolor="white", linewidth=0.4, zorder=2)
    means = np.nanmean(matrix, axis=0)
    colors = [BLUE, BLUE, PURPLE, ORANGE, ORANGE]
    ax.plot(x, means, color=INK, linewidth=2.4, zorder=3)
    ax.scatter(x, means, s=125, color=colors, edgecolor=INK, linewidth=1.2, zorder=4)
    for xi, mean in zip(x, means):
        ax.text(xi, min(mean + 7, 104), f"{mean:.0f}%", ha="center", va="bottom", weight="bold")

    ax.axvspan(2.5, 4.25, color=ORANGE, alpha=0.08, zorder=0)
    ax.annotate(
        "required code emission\nR2 → R2c: +67 points",
        xy=(3, means[3]),
        xytext=(2.35, 66),
        arrowprops={"arrowstyle": "->", "color": ORANGE, "linewidth": 1.3},
        color=ORANGE,
        fontsize=9,
        weight="bold",
        ha="center",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\n{desc}" for label, desc in zip(labels, descriptions)])
    ax.set_ylim(-4, 110)
    ax.set_ylabel("First-sight commitments (%)")
    ax.set_title("Requiring code reproduces full eager construction", loc="left", weight="bold")
    ax.grid(axis="y", color="#E8EBEE", linewidth=0.8)
    ax.text(
        0.0,
        -0.25,
        "Thin lines: 24 paired streams. Large points: pooled first-sight percentage. "
        "R2c keeps R2's payoff and removes correctness stakes.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=GRAY,
    )
    save(fig, "fig2_framing_ladder")


def figure_economic_elasticity() -> None:
    """Figure 3: R0, R2c, and exact hindsight optimum over visible charges."""
    summary = load_json(RUNS / "economic_surface_haiku" / "analysis_n24.json")
    charges = [0, 10, 24]
    budgets = [1, 3, 5]
    fig, axes = plt.subplots(1, 3, figsize=(11.1, 4.2), sharey=True, constrained_layout=True)

    for ax, budget in zip(axes, budgets):
        r0 = [summary[f"R0:B{budget}:K{k}"]["model_fs_hazard"] * 100 for k in charges]
        r2c = [summary[f"R2c:B{budget}:K{k}"]["model_fs_hazard"] * 100 for k in charges]
        opt = [summary[f"R0:B{budget}:K{k}"]["reference_fs_hazard"] * 100 for k in charges]
        ax.plot(charges, opt, marker="o", color=GRAY, linewidth=1.8, linestyle="--", label="Hindsight opt*")
        ax.plot(charges, r0, marker="o", color=BLUE, linewidth=2.2, label="Abstract R0")
        ax.plot(charges, r2c, marker="o", color=ORANGE, linewidth=2.2, label="Code-required R2c")
        ax.set_title(f"Build budget B={budget}", weight="bold")
        ax.set_xticks(charges)
        ax.set_xticklabels(["0\nbuild", "10\nselective", "24\nnever"])
        ax.set_xlabel("Visible build charge K")
        ax.set_ylim(-4, 106)
        ax.grid(axis="y", color="#E8EBEE", linewidth=0.8)
        if budget == 1:
            ax.set_ylabel("First-sight commitment hazard (%)")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    fig.suptitle(
        "Abstract allocation responds to price; code-required commitment does not",
        x=0.02,
        ha="left",
        weight="bold",
        fontsize=12,
    )
    fig.text(
        0.5,
        -0.035,
        "Haiku · 24 paired streams per cell · opt* is the exact hindsight net optimum, not an online policy",
        ha="center",
        fontsize=8.5,
        color=GRAY,
    )
    save(fig, "fig3_economic_elasticity")


def figure_capability_map() -> None:
    """Figure 4: abstract competence versus preservation during construction."""
    points = [
        ("Haiku 4.5", 28.0, 100.0, ORANGE, "o"),
        ("Opus 4.8", 4.0, 98.6, ORANGE, "o"),
        ("GPT-5.4-mini", 15.3, 85.0, PURPLE, "o"),
        ("GPT-5.6 Sol", 23.6, 38.9, GREEN, "o"),
        ("Qwen-14B base†", 75.0, 90.9, GRAY, "s"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    ax.fill_between([0, 100], [0, 100], [100, 100], color=ORANGE, alpha=0.06)
    ax.plot([0, 100], [0, 100], color="#9CA3AA", linestyle="--", linewidth=1.2)
    ax.plot([0, 100], [20, 100], color=GREEN, alpha=0)

    offsets = {
        "Haiku 4.5": (6, -8),
        "Opus 4.8": (6, -8),
        "GPT-5.4-mini": (7, -3),
        "GPT-5.6 Sol": (7, -3),
        "Qwen-14B base†": (-54, 12),
    }
    for name, x, y, color, marker in points:
        hollow = "†" in name
        ax.scatter(
            x,
            y,
            s=105,
            marker=marker,
            facecolor="white" if hollow else color,
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
        dx, dy = offsets[name]
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9, weight="bold")

    ax.text(65, 52, "cross-frame\npreservation", color=GREEN, ha="center", fontsize=10, weight="bold")
    ax.text(46, 73, "constructive\nsuppression", color=ORANGE, ha="center", fontsize=10, weight="bold")
    ax.text(83, 82, "policy eager\nin both frames", color=GRAY, ha="center", fontsize=9)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Abstract first-sight commitments (%)")
    ax.set_ylabel("Code-required/tool first-sight commitments (%)")
    ax.set_title("Online tool investment is difficult but achievable", loc="left", weight="bold")
    ax.grid(color="#ECEFF1", linewidth=0.8)
    ax.text(
        0.0,
        -0.15,
        "Near the diagonal: behavior is preserved across frames. Far above: code makes a selective model eager.\n"
        "† Qwen is descriptive only: its urn and final tool values come from different matched evaluation panels.",
        transform=ax.transAxes,
        fontsize=8.2,
        color=GRAY,
    )
    save(fig, "fig4_capability_map")


def figure_rl_transfer() -> None:
    """Figure 5: learned reserve across lexical, modality, and construction shifts."""
    categories = ["Abstract urn", "Vocabulary\nreskins", "Keep/pass\ntool calls", "Reusable\nscripts"]
    base = np.asarray([75, 89, 99, 90.9], dtype=float)
    rl = np.asarray([32, 19, 62, 94.3], dtype=float)
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for xi, b, r in zip(x, base, rl):
        ax.plot([xi, xi], [r, b], color=LIGHT, linewidth=5, solid_capstyle="round", zorder=1)
    ax.plot(x, base, color=GRAY, marker="o", linewidth=1.8, markersize=7, label="Base Qwen-14B")
    ax.plot(x, rl, color=GREEN, marker="o", linewidth=2.3, markersize=8, label="RL-final")
    for xi, value in zip(x, base):
        offset = -5 if value >= 95 else 4
        va = "top" if value >= 95 else "bottom"
        ax.text(xi, value + offset, f"{value:.0f}%", ha="center", va=va,
                color=GRAY, fontsize=9, weight="bold")
    for xi, value in zip(x, rl):
        offset = -6
        ax.text(xi, value + offset, f"{value:.0f}%", ha="center", va="top", color=GREEN, fontsize=9, weight="bold")

    ax.axvspan(-0.35, 1.35, color=GREEN, alpha=0.06)
    ax.axvspan(2.65, 3.35, color=ORANGE, alpha=0.06)
    ax.text(0.5, 8, "reserve acquired and lexically generalized", ha="center", color=GREEN, fontsize=8.5)
    ax.text(3, 8, "no activation", ha="center", color=ORANGE, fontsize=8.5, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 108)
    ax.set_ylabel("First-sight commitments (%)")
    ax.set_title("Reward-learned reserve remains context-bound", loc="left", weight="bold")
    ax.grid(axis="y", color="#E8EBEE", linewidth=0.8)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 0.98))
    ax.text(
        0.0,
        -0.20,
        "Pooled publication values; the script panel is the matched q8_0, n=24, EFR4 rerun. "
        "Panels differ in evaluation protocol and are not a single paired trajectory.",
        transform=ax.transAxes,
        fontsize=8.3,
        color=GRAY,
    )
    save(fig, "fig5_rl_transfer_boundary")


def keep_rate_by_occurrence(tag: str, max_position: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Pooled P(KEEP) by class_position (1st/2nd/... sighting of a color), across all 24 seeds'
    session.json transcripts for the given served Ollama tag. Positions beyond max_position are
    folded into the last bucket."""
    keeps = np.zeros(max_position, dtype=int)
    totals = np.zeros(max_position, dtype=int)
    for seed in SEEDS:
        path = RUNS / f"urn_{tag}_latest" / f"seed_{seed}" / "session.json"
        if not path.exists():
            continue
        session = load_json(path)
        for rec in session["transcript"]:
            pos = min(rec["class_position"], max_position) - 1
            totals[pos] += 1
            if rec["decision"] == "KEEP":
                keeps[pos] += 1
    return keeps, totals


def figure_reserve_policy() -> None:
    """Figure 6: the learned reserve policy's actual decision rule -- P(keep) by occurrence count,
    base vs RL-final, pooled over the paired 24-seed held-out urn eval."""
    max_position = 5
    positions = np.arange(1, max_position + 1)
    base_k, base_n = keep_rate_by_occurrence("qwen-rl-base-q8", max_position)
    rl_k, rl_n = keep_rate_by_occurrence("qwen-rl-urn-final", max_position)
    base_rate = 100 * base_k / base_n
    rl_rate = 100 * rl_k / rl_n

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    fig.subplots_adjust(top=0.90, bottom=0.24)
    ax.plot(positions, base_rate, color=GRAY, marker="o", linewidth=1.8, markersize=7,
             label="Base Qwen-14B")
    ax.plot(positions, rl_rate, color=GREEN, marker="o", linewidth=2.3, markersize=8,
             label="RL-final")
    # only callout the two positions that carry the story; 3+ have small, noisy per-bucket n
    ax.annotate(f"{base_rate[0]:.0f}%  ({base_k[0]}/{base_n[0]})", (positions[0], base_rate[0]),
                xytext=(8, 14), textcoords="offset points", ha="left", fontsize=9, color=GRAY,
                weight="bold")
    ax.annotate(f"{rl_rate[0]:.0f}%  ({rl_k[0]}/{rl_n[0]})", (positions[0], rl_rate[0]),
                xytext=(8, -18), textcoords="offset points", ha="left", fontsize=9, color=GREEN,
                weight="bold")
    ax.annotate(f"{base_rate[1]:.0f}%  ({base_k[1]}/{base_n[1]})", (positions[1], base_rate[1]),
                xytext=(0, 14), textcoords="offset points", ha="center", fontsize=9, color=GRAY,
                weight="bold")
    ax.annotate(f"{rl_rate[1]:.0f}%  ({rl_k[1]}/{rl_n[1]})", (positions[1], rl_rate[1]),
                xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9, color=GREEN,
                weight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels([str(p) for p in positions[:-1]] + [f"{max_position}+"])
    ax.set_xlabel("Occurrence count of this color within the stream", labelpad=10)
    ax.set_ylabel("P(KEEP) (%)")
    ax.set_xlim(0.5, max_position + 0.5)
    ax.set_ylim(0, 100)
    ax.set_title("The learned policy defers, then confirms", loc="left", weight="bold")
    ax.grid(axis="y", color="#ECEFF1", linewidth=0.8)
    ax.legend(frameon=False, loc="upper center")
    fig.text(
        0.02, 0.01,
        "Pooled decisions across the paired 24-seed held-out urn eval (seeds 2000-2023, no-announce,\n"
        "q8_0). Base keeps eagerly from the first sighting onward; RL-final passes on the first sighting\n"
        "and keeps once recurrence is confirmed on the second -- the reserve policy's decision rule made\n"
        "visible, not just its aggregate first-sight rate. Positions 3+ have small per-bucket n (3-7\n"
        "decisions) and are a noisy tail, not part of the claim.",
        fontsize=8.2, color=GRAY, va="bottom",
    )
    save(fig, "fig6_reserve_policy_shape")


def main() -> None:
    configure_style()
    figure_task_and_core()
    figure_ladder()
    figure_economic_elasticity()
    figure_capability_map()
    figure_rl_transfer()
    figure_reserve_policy()


if __name__ == "__main__":
    main()
