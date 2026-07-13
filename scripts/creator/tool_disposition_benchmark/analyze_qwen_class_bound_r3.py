"""One-off analysis of the class-bound-r3-v1 Qwen panels (base-q8 vs RL-final), matching the
reporting format used for the GPT-5.4-mini/GPT-5.6 R3 rows in docs/tool-investment.tex Table~tab:models
(pooled first-sight among realized commitments, builds/seed, zero-commit seeds, mechanical validity).

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.analyze_qwen_class_bound_r3
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.creator.tool_disposition_benchmark.skirental_scorer import (
    actions_from_session, model_builds_from_actions, AUTHOR_ACTIONS,
)

RUNS = {
    "qwen-rl-base-q8": "runs/arm_a1_announce_qwen-rl-base-q8_latest_n-announced_class-bound-v1_efr4",
    "qwen-rl-urn-final": "runs/arm_a1_announce_qwen-rl-urn-final_latest_n-announced_class-bound-v1_efr4",
}


def analyze(run_dir: str) -> dict:
    seed_dirs = sorted(Path(run_dir).glob("seed_*"))
    n_seeds = len(seed_dirs)
    total_builds = 0
    first_sight_builds = 0
    lateness_vals = []
    zero_commit_seeds = 0
    malformed = unknown = refused = cross_class_refused = 0
    budget_trunc = cap_trunc = neither_trunc = 0
    per_seed_builds = []

    for sd in seed_dirs:
        session = json.loads((sd / "sessions.jsonl").read_text().splitlines()[0])
        slots = json.loads((sd / "stream.json").read_text())
        actions = actions_from_session(session, slots)
        builds = [a for a in actions if a["action"] in AUTHOR_ACTIONS]
        per_seed_builds.append(len(builds))
        if not builds:
            zero_commit_seeds += 1
        total_builds += len(builds)
        model_builds = model_builds_from_actions(actions)
        for cid, pos in model_builds.items():
            if pos is None:
                continue
            lateness = pos - 1
            lateness_vals.append(lateness)
            if lateness == 0:
                first_sight_builds += 1

        malformed += session.get("n_malformed_tool_calls", 0)
        unknown += session.get("n_unknown_tool_calls", 0)
        refused += session.get("n_refused_tool_calls", 0)
        cross_class_refused += session.get("n_cross_class_run_refused", 0)
        stopped_on_budget = session.get("stopped_on_budget")
        hit_cap = session.get("hit_cap")
        if stopped_on_budget:
            budget_trunc += 1
        elif hit_cap:
            cap_trunc += 1
        else:
            neither_trunc += 1

    n_commits = len(lateness_vals)
    return {
        "n_seeds": n_seeds,
        "total_builds": total_builds,
        "builds_per_seed": total_builds / n_seeds,
        "zero_commit_seeds": zero_commit_seeds,
        "first_sight_builds": first_sight_builds,
        "n_commits": n_commits,
        "first_sight_pct": 100 * first_sight_builds / n_commits if n_commits else float("nan"),
        "mean_lateness": sum(lateness_vals) / len(lateness_vals) if lateness_vals else float("nan"),
        "max_lateness": max(lateness_vals) if lateness_vals else None,
        "malformed": malformed,
        "unknown": unknown,
        "refused": refused,
        "cross_class_refused": cross_class_refused,
        "budget_trunc": budget_trunc,
        "cap_trunc": cap_trunc,
        "neither_trunc": neither_trunc,
        "per_seed_builds": per_seed_builds,
    }


def main() -> None:
    for tag, run_dir in RUNS.items():
        r = analyze(run_dir)
        print(f"\n===== {tag} ({run_dir}) =====")
        for k, v in r.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
