"""Extensible episode-statistics framework for the toolworld_v2 sweeps.

Reads one or more episodes.jsonl files (or run dirs containing one), runs a set
of ANALYZERS over each episode, then aggregates per (model, n, n_types) cell.

Adding a statistic later is a one-function change -- it does NOT touch the
loader, aggregator, or reporter:

    @analyzer
    def my_stat(ep: Episode) -> dict:
        return {"my_metric": <scalar>, ...}

Return a flat dict of named metrics per episode. Aggregation is type-driven:
  * bool      -> reported as a rate (true / n)
  * int/float -> reported as mean / median / min / max
  * str       -> reported as a category-count distribution
  * list      -> kept per-episode (shown with --per-episode), not aggregated
  * None      -> dropped from that episode's contribution (use for "N/A")

FIRST ANALYZER -- build_dynamics: how many times and WHEN the model tries to
fuse byproducts into the machine, so we can tell "kept trying" from "gave up".
A build attempt is a combine(a, b) where BOTH a and b are byproduct-type labels
(the only combine that can ever produce the machine). Each attempt's outcome is
read from its observation: "fuse into" = built, "already hold" = redundant
rebuild after success, otherwise = nothing happened.

Usage:
    python -m scripts.analyze_episode_stats runs/nt_sweep_XXXX
    python -m scripts.analyze_episode_stats runs/a/episodes.jsonl runs/b --json
    python -m scripts.analyze_episode_stats runs/nt_sweep_XXXX --per-episode
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# An attempt at/after this normalized episode time (0=first action, 1=last)
# counts as "still trying at the end" rather than "gave up earlier".
FINAL_STRETCH = 0.75


# --- episode model ------------------------------------------------------

@dataclass
class Episode:
    """Thin typed view over one JSONL row, with the derived helpers analyzers
    tend to need (action list, byproduct types, normalized action time)."""

    row: dict

    @property
    def model(self) -> str | None:
        return self.row.get("model")

    @property
    def n(self) -> int | None:
        return self.row.get("n")

    @property
    def n_types(self) -> int | None:
        return self.row.get("n_types")

    @property
    def budget(self) -> int | None:
        return self.row.get("budget")

    @property
    def solved(self) -> bool:
        return bool(self.row.get("solved"))

    @property
    def actions(self) -> list[tuple]:
        return [tuple(a) for a in self.row.get("actions", [])]

    @property
    def obs(self) -> list[str]:
        return self.row.get("obs", [])

    @property
    def total_actions(self) -> int:
        return self.row.get("total_actions", len(self.actions))

    @property
    def types(self) -> set[str]:
        return set(self.row.get("labels", {}).get("types", []))

    @property
    def machine(self) -> str | None:
        return self.row.get("labels", {}).get("machine")

    def time_frac(self, i: int) -> float:
        """Normalize an action index to [0, 1]; 0 = first action, 1 = last."""
        t = self.total_actions
        return 0.0 if t <= 1 else i / (t - 1)


# --- analyzer registry --------------------------------------------------

ANALYZERS: list[Callable[[Episode], dict]] = []


def analyzer(fn: Callable[[Episode], dict]) -> Callable[[Episode], dict]:
    ANALYZERS.append(fn)
    return fn


# --- shared extraction helpers ------------------------------------------

def build_attempts(ep: Episode) -> list[tuple[int, str, str, str]]:
    """All genuine tool-build attempts: combine(a, b) with both args byproduct
    types. Returns (action_index, a, b, outcome) where outcome is one of
    'built' | 'already_built' | 'nothing'."""
    types = ep.types
    obs = ep.obs
    out: list[tuple[int, str, str, str]] = []
    for i, a in enumerate(ep.actions):
        if len(a) < 3 or a[0] != "combine":
            continue
        x, y = a[1], a[2]
        if x in types and y in types:
            o = obs[i] if i < len(obs) else ""
            if "fuse into" in o:
                outcome = "built"
            elif "already hold" in o:
                outcome = "already_built"
            else:
                outcome = "nothing"
            out.append((i, x, y, outcome))
    return out


# --- analyzers ----------------------------------------------------------

@analyzer
def build_dynamics(ep: Episode) -> dict:
    attempts = build_attempts(ep)
    idxs = [i for i, _x, _y, _o in attempts]
    built = any(o == "built" for _i, _x, _y, o in attempts)
    build_idx = next((i for i, _x, _y, o in attempts if o == "built"), None)

    # Categorical verdict -- the headline "give up vs. keep trying" signal.
    # Only meaningful for non-builders; builders are their own bucket since
    # they (correctly) stop attempting once the machine exists.
    if built:
        verdict = "built"
    elif not idxs:
        verdict = "never_tried"
    elif ep.time_frac(idxs[-1]) >= FINAL_STRETCH:
        verdict = "persisted"      # still attempting in the final stretch
    else:
        verdict = "gave_up"        # stopped attempting well before the end

    m: dict = {
        "build_outcome": verdict,
        "built": built,
        "build_attempts": len(attempts),
        "build_attempts_distinct": sum(1 for _i, x, y, _o in attempts if x != y),
        "redundant_rebuilds": sum(1 for _i, _x, _y, o in attempts
                                  if o == "already_built"),
        "combine_total": sum(1 for a in ep.actions if a and a[0] == "combine"),
        "first_attempt_idx": idxs[0] if idxs else None,
        "first_attempt_frac": ep.time_frac(idxs[0]) if idxs else None,
        "last_attempt_idx": idxs[-1] if idxs else None,
        "last_attempt_frac": ep.time_frac(idxs[-1]) if idxs else None,
        "build_turn_idx": build_idx,
        "build_turn_frac": ep.time_frac(build_idx) if build_idx is not None else None,
        # list metric: kept per-episode (e.g. for plotting), never aggregated.
        "build_attempt_turns": idxs,
    }
    # How long the model went on AFTER its last build attempt without building
    # -- large gap = abandoned the build idea and ground out the rest.
    if idxs and not built:
        m["actions_after_last_attempt"] = (ep.total_actions - 1) - idxs[-1]
        m["frac_after_last_attempt"] = 1.0 - ep.time_frac(idxs[-1])
    return m


# --- aggregation --------------------------------------------------------

def aggregate(metric_rows: list[dict]) -> dict:
    """Summarize a list of per-episode metric dicts, dispatching on value type.
    Bool checked before int (bool is an int subclass)."""
    keys = {k for row in metric_rows for k in row}
    summary: dict = {}
    for k in keys:
        vals = [row[k] for row in metric_rows if row.get(k) is not None]
        if not vals or any(isinstance(v, list) for v in vals):
            continue
        if all(isinstance(v, bool) for v in vals):
            summary[k] = {"kind": "rate", "n": len(vals),
                          "true": sum(vals), "rate": sum(vals) / len(vals)}
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool)
                 for v in vals):
            summary[k] = {"kind": "num", "n": len(vals),
                          "mean": statistics.mean(vals),
                          "median": statistics.median(vals),
                          "min": min(vals), "max": max(vals)}
        else:
            counts: dict = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            summary[k] = {"kind": "cat", "n": len(vals), "counts": counts}
    return summary


# --- loading ------------------------------------------------------------

def load_episodes(paths: list[str]) -> list[Episode]:
    eps: list[Episode] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            p = p / "episodes.jsonl"
        if not p.exists():
            raise FileNotFoundError(p)
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("error") or "actions" not in row:
                continue  # skip failed / non-episode rows
            eps.append(Episode(row))
    return eps


# --- reporting ----------------------------------------------------------

# Build-dynamics keys first (the headline question), then anything analyzers
# add later, sorted. Unknown keys still print via the generic formatter.
_PREFERRED = [
    "build_outcome", "built", "build_attempts", "build_attempts_distinct",
    "redundant_rebuilds", "combine_total",
    "first_attempt_frac", "last_attempt_frac", "build_turn_frac",
    "actions_after_last_attempt", "frac_after_last_attempt",
    "first_attempt_idx", "last_attempt_idx", "build_turn_idx",
]


def _order(keys) -> list[str]:
    pref = [k for k in _PREFERRED if k in keys]
    return pref + sorted(k for k in keys if k not in _PREFERRED)


def _fmt(k: str, s: dict) -> str:
    if s["kind"] == "rate":
        return f"{k}: {s['rate'] * 100:.1f}% ({s['true']}/{s['n']})"
    if s["kind"] == "num":
        g = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
        return (f"{k}: mean={s['mean']:.2f} median={s['median']:.2f} "
                f"min={g(s['min'])} max={g(s['max'])} (n={s['n']})")
    parts = "  ".join(f"{v}={c}" for v, c in sorted(s["counts"].items(),
                                                    key=lambda kv: -kv[1]))
    return f"{k}: {parts} (n={s['n']})"


def print_text(report: dict, per_episode: bool) -> None:
    for (model, n, t), grp in report.items():
        print(f"\n=== model={model}  n={n}  T={t}   "
              f"({grp['n_episodes']} episodes) ===")
        summ = grp["metrics"]
        for k in _order(summ.keys()):
            print(f"  {_fmt(k, summ[k])}")
        if per_episode:
            print("  --- per episode ---")
            for i, row in enumerate(grp["per_episode"]):
                turns = row.get("build_attempt_turns")
                print(f"  ep{i:>2}: outcome={row.get('build_outcome'):<11} "
                      f"attempts={row.get('build_attempts')} "
                      f"turns={turns}")


# --- plotting (build attempts vs. time) ---------------------------------
# Lazy matplotlib import lives inside the functions so the analysis CLI works
# without matplotlib installed; only --plot pulls it in.

# Per-outcome marker styling, shared by both figures.
_OUTCOME_STYLE = {
    "built":         {"color": "seagreen",   "marker": "*", "s": 110, "label": "built (fused)"},
    "nothing":       {"color": "darkorange", "marker": "o", "s": 26,  "label": "no effect"},
    "already_built": {"color": "steelblue",  "marker": "s", "s": 26,  "label": "redundant rebuild"},
}
_OUTCOME_ORDER = ["built", "nothing", "already_built"]


def _short(model: str | None) -> str:
    if not model:
        return "?"
    for k in ("haiku", "sonnet", "opus", "fable", "gpt", "gemini"):
        if k in model:
            return k
    return model


def _cells_grid(eps: list[Episode]):
    """Return (cells, rows, cols) where rows = sorted (model, T), cols = sorted n.
    Cells maps (model, n, T) -> [Episode]."""
    cells: dict[tuple, list[Episode]] = defaultdict(list)
    for ep in eps:
        cells[(ep.model, ep.n, ep.n_types)].append(ep)
    rows = sorted({(m, t) for (m, _n, t) in cells}, key=lambda mt: (_short(mt[0]), mt[1] or 0))
    cols = sorted({n for (_m, n, _t) in cells}, key=lambda n: n or 0)
    return cells, rows, cols


def fig_attempt_raster(eps: list[Episode], out: Path) -> None:
    """One panel per (model, n, T) cell. Each row is an episode: a grey timeline
    from action 0 to its last action, build attempts marked along it (colored by
    outcome), and a tick at the end (green=solved, red=failed). Reads directly:
    attempts stop well before the timeline end => the model gave up building."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cells, rows, cols = _cells_grid(eps)
    max_eps = max((len(g) for g in cells.values()), default=1)
    nr, nc = max(len(rows), 1), max(len(cols), 1)
    fig, axes = plt.subplots(nr, nc, squeeze=False,
                             figsize=(3.4 * nc, 0.26 * max_eps * nr + 1.0 * nr + 0.6),
                             sharex=False)

    def sort_key(ep: Episode):
        at = build_attempts(ep)
        built = any(o == "built" for *_r, o in at)
        last = at[-1][0] if at else -1
        return (not built, -last)  # builders on top, then latest-last-attempt

    for r, (model, T) in enumerate(rows):
        for c, n in enumerate(cols):
            ax = axes[r][c]
            grp = sorted(cells.get((model, n, T), []), key=sort_key)
            if not grp:
                ax.set_axis_off()
                continue
            for y, ep in enumerate(grp):
                end = max(ep.total_actions - 1, 0)
                ax.hlines(y, 0, end, color="0.85", lw=2.0, zorder=1)
                for i, _x, _yb, o in build_attempts(ep):
                    st = _OUTCOME_STYLE[o]
                    ax.scatter(i, y, color=st["color"], marker=st["marker"],
                               s=st["s"], zorder=3, edgecolors="none")
                ax.scatter(end, y, marker="|", s=80, linewidths=1.6,
                           color="green" if ep.solved else "red", zorder=2)
            ax.set_ylim(-0.7, len(grp) - 0.3)
            ax.set_yticks([])
            ax.set_title(f"{_short(model)}  n={n}  T={T}", fontsize=9)
            if r == nr - 1:
                ax.set_xlabel("action index", fontsize=8)
            ax.tick_params(labelsize=7)

    handles = [Line2D([], [], color=st["color"], marker=st["marker"],
                      linestyle="none", markersize=8, label=st["label"])
               for st in (_OUTCOME_STYLE[o] for o in _OUTCOME_ORDER)]
    handles += [Line2D([], [], color="green", marker="|", linestyle="none",
                       markersize=9, label="episode end (solved)"),
                Line2D([], [], color="red", marker="|", linestyle="none",
                       markersize=9, label="episode end (failed)")]
    fig.suptitle("Build attempts over time (one row = one episode)",
                 y=1.0, fontsize=10, x=0.5, ha="center")
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_attempt_hist(eps: list[Episode], out: Path) -> None:
    """One panel per cell: histogram of build-attempt times over NORMALIZED
    episode time (0=start, 1=end), stacked by outcome. Shows whether attempts
    cluster early (give-up pattern) or spread to the end (persistence)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cells, rows, cols = _cells_grid(eps)
    nr, nc = max(len(rows), 1), max(len(cols), 1)
    fig, axes = plt.subplots(nr, nc, squeeze=False,
                             figsize=(3.2 * nc, 2.4 * nr + 0.8), sharex=True)
    bins = [i / 10 for i in range(11)]

    for r, (model, T) in enumerate(rows):
        for c, n in enumerate(cols):
            ax = axes[r][c]
            grp = cells.get((model, n, T), [])
            if not grp:
                ax.set_axis_off()
                continue
            by_outcome: dict[str, list[float]] = {o: [] for o in _OUTCOME_ORDER}
            for ep in grp:
                for i, _x, _yb, o in build_attempts(ep):
                    by_outcome[o].append(ep.time_frac(i))
            series = [by_outcome[o] for o in _OUTCOME_ORDER]
            ax.hist(series, bins=bins, stacked=True,
                    color=[_OUTCOME_STYLE[o]["color"] for o in _OUTCOME_ORDER])
            ax.axvline(FINAL_STRETCH, color="0.4", ls="--", lw=0.8)
            ax.set_title(f"{_short(model)}  n={n}  T={T}", fontsize=9)
            if r == nr - 1:
                ax.set_xlabel("normalized episode time", fontsize=8)
            if c == 0:
                ax.set_ylabel("# attempts", fontsize=8)
            ax.tick_params(labelsize=7)

    handles = [Line2D([], [], color=_OUTCOME_STYLE[o]["color"], marker="s",
                      linestyle="none", markersize=8,
                      label=_OUTCOME_STYLE[o]["label"]) for o in _OUTCOME_ORDER]
    handles.append(Line2D([], [], color="0.4", ls="--",
                          label=f"final-stretch ({FINAL_STRETCH})"))
    fig.suptitle("When build attempts happen (normalized time)", y=1.0,
                 fontsize=10)
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="episodes.jsonl files and/or run directories")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--per-episode", action="store_true",
                    help="also list each episode's outcome and attempt turns")
    ap.add_argument("--plot", action="store_true",
                    help="write build-attempts-over-time figures (raster + histogram)")
    ap.add_argument("--out", type=Path, default=None,
                    help="dir for --plot figures (default: first path's dir)")
    args = ap.parse_args()

    eps = load_episodes(args.paths)
    groups: dict[tuple, list[Episode]] = defaultdict(list)
    for ep in eps:
        groups[(ep.model, ep.n, ep.n_types)].append(ep)

    report: dict = {}
    for key in sorted(groups, key=lambda k: (str(k[0]), k[1] or 0, k[2] or 0)):
        grp = groups[key]
        rows = []
        for ep in grp:
            metrics: dict = {}
            for fn in ANALYZERS:
                metrics.update(fn(ep))
            rows.append(metrics)
        report[key] = {"n_episodes": len(grp), "metrics": aggregate(rows),
                       "per_episode": rows}

    if args.json:
        out = {f"model={m}|n={n}|T={t}": {"n_episodes": g["n_episodes"],
                                          "metrics": g["metrics"]}
               for (m, n, t), g in report.items()}
        print(json.dumps(out, indent=2, default=str))
    else:
        if not eps:
            print("No episodes loaded.")
            return
        print(f"Loaded {len(eps)} episodes across {len(report)} cells. "
              f"(final-stretch threshold = {FINAL_STRETCH})")
        print_text(report, args.per_episode)

    if args.plot and eps:
        first = Path(args.paths[0])
        outdir = args.out or (first if first.is_dir() else first.parent)
        outdir.mkdir(parents=True, exist_ok=True)
        fig_attempt_raster(eps, outdir / "fig_build_attempts_raster.png")
        fig_attempt_hist(eps, outdir / "fig_build_attempts_hist.png")


if __name__ == "__main__":
    main()
