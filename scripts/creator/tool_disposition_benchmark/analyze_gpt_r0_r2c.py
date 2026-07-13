"""Locked analysis for docs/gpt-cross-family-r0-r2c-spec.md.

Reads one model's B=3/K=0 R0 and R2c panels, applies the preregistered R0
competence gate, and (when both panels exist) computes the paired framing test.
No network calls.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

from scripts.creator.tool_disposition_benchmark.economic_surface import (
    CANONICAL_SEEDS, commits_of, reference_builds, reference_net_score)

B = 3
K = 0
N_BOOT = 10_000


def _seed_dir(run_dir: Path, framing: str, seed: int) -> Path:
    return run_dir / framing / f"B_{B}" / f"K_{K}" / f"seed_{seed}"


def _load_panel(run_dir: Path, framing: str, seeds: tuple[int, ...]) -> list[dict]:
    rows = []
    for seed in seeds:
        d = _seed_dir(run_dir, framing, seed)
        session_path = d / "session.json"
        if not session_path.exists():
            continue
        row = json.loads(session_path.read_text())
        slots = json.loads((d / "stream.json").read_text())
        commits = commits_of(row)
        commit_label = "KEEP" if framing == "R0" else "CLAIM_SOLVER"
        fs = sum(t["decision"] == commit_label and t["class_position"] == 1
                 for t in row["transcript"])
        eligible_fs = sum(t["class_position"] == 1 for t in row["transcript"])
        lateness = [position - 1 for position in commits.values()]
        rows.append({
            "seed": seed,
            "commits": len(commits),
            "first_sight_commits": fs,
            "eligible_first_sight": eligible_fs,
            "fs_among_commits": fs / len(commits) if commits else None,
            "fs_hazard": fs / eligible_fs if eligible_fs else None,
            "mean_lateness": st.mean(lateness) if lateness else None,
            "collected": row["collected"],
            "unparsed": row["unparsed"],
            "turns": len(row["transcript"]),
            "actual_cost_usd": row.get("actual_cost_usd", 0.0),
            "termination": row.get("termination"),
            "slots": slots,
        })
    return rows


def _summarize(rows: list[dict]) -> dict:
    total_commits = sum(r["commits"] for r in rows)
    total_fs = sum(r["first_sight_commits"] for r in rows)
    total_eligible = sum(r["eligible_first_sight"] for r in rows)
    total_turns = sum(r["turns"] for r in rows)
    lateness = [r["mean_lateness"] for r in rows if r["mean_lateness"] is not None]
    return {
        "n_seeds": len(rows),
        "commits_per_seed": st.mean(r["commits"] for r in rows) if rows else None,
        "first_sight_among_commits": total_fs / total_commits if total_commits else None,
        "first_sight_hazard": total_fs / total_eligible if total_eligible else None,
        "mean_lateness": st.mean(lateness) if lateness else None,
        "mean_collected": st.mean(r["collected"] for r in rows) if rows else None,
        "unresolved_rate": (sum(r["unparsed"] for r in rows) / total_turns) if total_turns else None,
        "actual_cost_usd": sum(r["actual_cost_usd"] for r in rows),
        "per_seed": [{k: v for k, v in r.items() if k != "slots"} for r in rows],
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    weight = index - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def _paired_bootstrap(r0: list[dict], r2c: list[dict]) -> dict | None:
    r0_by_seed = {r["seed"]: r for r in r0}
    r2c_by_seed = {r["seed"]: r for r in r2c}
    seeds = sorted(set(r0_by_seed) & set(r2c_by_seed))
    diffs = [
        r2c_by_seed[s]["fs_among_commits"] - r0_by_seed[s]["fs_among_commits"]
        for s in seeds
        if r0_by_seed[s]["fs_among_commits"] is not None
        and r2c_by_seed[s]["fs_among_commits"] is not None
    ]
    if not diffs:
        return None
    rng = random.Random(0)
    boot = [st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(N_BOOT)]
    return {
        "n_paired_seeds": len(diffs),
        "mean_r2c_minus_r0_first_sight": st.mean(diffs),
        "bootstrap_95_ci": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
        "bootstrap_samples": N_BOOT,
        "bootstrap_seed": 0,
    }


def analyze(run_dir: Path, seeds: tuple[int, ...]) -> dict:
    r0 = _load_panel(run_dir, "R0", seeds)
    r2c = _load_panel(run_dir, "R2c", seeds)
    r0_summary = _summarize(r0)
    r2c_summary = _summarize(r2c)

    bayes = reference_builds(B, K, seeds=seeds)
    bayes_collected = [
        reference_net_score(B, K, r["seed"], r["slots"], builds=bayes[r["seed"]])
        for r in r0
    ]
    mean_bayes = st.mean(bayes_collected) if bayes_collected else None
    collected_ratio = (
        r0_summary["mean_collected"] / mean_bayes
        if r0_summary["mean_collected"] is not None and mean_bayes else None
    )
    EXTENDED_SEEDS = tuple(range(2000, 2024))  # seed-extension panel (2012-2023 added post hoc)
    canonical_panel = tuple(seeds) in (tuple(CANONICAL_SEEDS), EXTENDED_SEEDS)
    gate_complete = canonical_panel and len(r0) == len(seeds)
    gate_pass = bool(
        gate_complete
        and r0_summary["commits_per_seed"] >= 2.5
        and r0_summary["first_sight_among_commits"] <= 0.50
        and collected_ratio >= 0.90
    )
    gate = {
        "complete_canonical_panel": gate_complete,
        "mean_commits_at_least_2_5": (
            r0_summary["commits_per_seed"] >= 2.5
            if r0_summary["commits_per_seed"] is not None else None
        ),
        "first_sight_at_most_50pct": (
            r0_summary["first_sight_among_commits"] <= 0.50
            if r0_summary["first_sight_among_commits"] is not None else None
        ),
        "collected_at_least_90pct_bayes": (
            collected_ratio >= 0.90 if collected_ratio is not None else None
        ),
        "mean_bayes_collected": mean_bayes,
        "r0_collected_ratio_to_bayes": collected_ratio,
        "passed": gate_pass,
    }

    paired = _paired_bootstrap(r0, r2c)
    replication_complete = (
        canonical_panel and len(r2c) == len(seeds) and paired is not None
    )
    replication_pass = bool(
        gate_pass
        and replication_complete
        and r2c_summary["first_sight_among_commits"] >= 0.80
        and (r2c_summary["first_sight_among_commits"]
             - r0_summary["first_sight_among_commits"]) >= 0.30
        and paired["bootstrap_95_ci"][0] > 0
    )
    return {
        "run_dir": str(run_dir),
        "seeds": list(seeds),
        "R0": r0_summary,
        "R2c": r2c_summary,
        "competence_gate": gate,
        "paired_test": paired,
        "replication_complete": replication_complete,
        "replication_passed": replication_pass,
    }


def _selftest() -> None:
    assert _percentile([0.0, 1.0], 0.5) == 0.5
    rows0 = [{"seed": i, "fs_among_commits": 0.0} for i in range(12)]
    rows2 = [{"seed": i, "fs_among_commits": 1.0} for i in range(12)]
    paired = _paired_bootstrap(rows0, rows2)
    assert paired["mean_r2c_minus_r0_first_sight"] == 1.0
    assert paired["bootstrap_95_ci"] == [1.0, 1.0]
    print("analyze_gpt_r0_r2c self-test OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CANONICAL_SEEDS))
    parser.add_argument("--json-out")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
    else:
        if not args.run_dir:
            parser.error("--run-dir is required unless --selftest is used")
        result = analyze(Path(args.run_dir), tuple(args.seeds))
        print(json.dumps(result, indent=2))
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2))
