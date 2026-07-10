"""Economic response surface -- aggregate the 216-session run into cell/seed summaries and the
charge-aware regret comparison (`docs/economic-response-surface-spec.md` §4).

Two references (2026-07-09 reference repair, spec §2):
  - PRIMARY = the EXACT hindsight net-optimum (`hindsight_net_optimal_builds`) -- prior-free, no cap.
    All `reference_*` / `regret` fields below are measured against THIS. Regret vs it is >= 0 by
    construction, so "optimum" / "regret" language is honest.
  - SECONDARY = the online Bayesian (Dirichlet-prior) `ExactDP` policy (`reference_builds`), reported
    under `bayes_*` fields -- kept for its online *timing* pattern (it waits for a recurrence), NOT
    as the optimum. It is the OLD reference; at K=20 it reports negative net (dominated by
    never-building), which is exactly why it was demoted.

Primary outcomes (defined even when the optimum never commits):
  - first-sight commitment hazard = commits at a class's FIRST sighting / eligible first-sight
    decision turns (a decision turn on a class at class_position==1, with budget still left). Computed
    identically for the model (straight from its transcript) and for each reference policy (by
    replaying that policy's build positions through the same per-draw budget loop the live session
    uses), so the denominators mean the same thing.
  - commits per seed and zero-commit incidence.
  - net points (`raw - K*commits`) and regret vs. the hindsight optimum (optimum net - model net) --
    NOT the historical token-cost `exact_pistar_report` currency, kept separate from net-point regret.

Secondary: first-sight proportion among realized commitments, mean lateness, R2c code correctness
(diagnostic only, never scored).

This module only READS the run artifacts and the (cached) reference policy -- zero API cost.

  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.analyze_economic_surface
  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.analyze_economic_surface --selftest
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path

from scripts.creator.tool_disposition_benchmark.economic_surface import (
    FRAMINGS, BUDGETS, CHARGES, CELLS, CANONICAL_SEEDS, BAYES_CHARGES, net_score, commits_of,
    reference_builds, reference_net_score, hindsight_net_optimal_builds)

# The single "commit" decision label each framing writes into its transcript rows.
COMMIT_DECISION = {"R0": "KEEP", "R2c": "CLAIM_SOLVER"}

RUN_DIR = Path("runs/economic_surface_haiku")


def _cell_dir(framing: str, B: int, K: int, seed: int) -> Path:
    return RUN_DIR / framing / f"B_{B}" / f"K_{K}" / f"seed_{seed}"


def simulate_first_sight(slots: list[dict], builds: dict[int, int | None], B: int) -> tuple[int, int, int]:
    """Replay a *policy* (given as `{class_id: build_position_or_None}`) through the SAME per-draw
    loop the live sessions use (`urn_session.run_episode` / `run_episode_code_claim`): walk slots in
    order, skip already-committed classes (auto-solved, no decision), stop once budget is exhausted,
    and at every remaining slot count a decision turn. Returns
    `(eligible_first_sight_turns, first_sight_commits, total_commits)`:
      - eligible_first_sight_turns: decision turns landing on a class's FIRST sighting (class_position
        == 1) while budget remained -- the honest denominator for a first-sight hazard.
      - first_sight_commits: of those, how many the policy commits on.
      - total_commits: commits anywhere (<= B).
    Used for the reference policy here; the model's own numbers come straight from its transcript
    (which already records exactly these decision turns), and `_selftest` checks the two agree."""
    committed: set[int] = set()
    budget = B
    eligible_fs = fs_commits = total_commits = 0
    for s in sorted(slots, key=lambda z: z["slot_index"]):
        cid, pos = s["class_id"], s["class_position"]
        if cid in committed:
            continue
        if budget == 0:
            break
        # this is a decision turn on cid at position `pos`
        build_pos = builds.get(cid)
        commits_here = build_pos is not None and build_pos == pos
        if pos == 1:
            eligible_fs += 1
            if commits_here:
                fs_commits += 1
        if commits_here:
            committed.add(cid)
            budget -= 1
            total_commits += 1
    return eligible_fs, fs_commits, total_commits


def model_first_sight(row: dict, framing: str) -> tuple[int, int, int]:
    """Model's first-sight numbers straight from its transcript rows -- the transcript records one
    entry per decision turn (a turn only happens while budget>0 and the class isn't already
    committed), so `class_position == 1` entries ARE the eligible first-sight turns, with no need to
    re-simulate. Returns `(eligible_first_sight_turns, first_sight_commits, total_commits)`."""
    commit_label = COMMIT_DECISION[framing]
    eligible_fs = fs_commits = 0
    for t in row["transcript"]:
        if t["class_position"] == 1:
            eligible_fs += 1
            if t["decision"] == commit_label:
                fs_commits += 1
    return eligible_fs, fs_commits, len(commits_of(row))


def _hazard(fs_commits: int, eligible: int) -> float | None:
    return (fs_commits / eligible) if eligible else None


def _code_correctness(row: dict) -> tuple[int, int]:
    """R2c-only: (n_correct, n_tested) pooled over this session's graded claims. Diagnostic only."""
    tested = correct = 0
    for g in (row.get("code_grades") or {}).values():
        tested += g.get("n_tested", 0)
        correct += g.get("n_correct", 0)
    return correct, tested


def analyze_cell(framing: str, B: int, K: int, seeds=CANONICAL_SEEDS) -> dict:
    # SECONDARY online-Bayes DP: only where its ExactDP is tractable (spec §7); at K=10 it does not
    # finish, so bayes_* fields are None there and only the exact hindsight optimum is reported.
    bayes = reference_builds(B, K, seeds=seeds) if K in BAYES_CHARGES else None
    per_seed = []
    for seed in seeds:
        d = _cell_dir(framing, B, K, seed)
        p = d / "session.json"
        if not p.exists():
            continue
        row = json.loads(p.read_text())
        slots = json.loads((d / "stream.json").read_text())

        m_elig, m_fs, m_commits = model_first_sight(row, framing)
        m_net = net_score(row, K)

        # PRIMARY reference: exact hindsight net-optimum (prior-free, no cap).
        opt_builds = hindsight_net_optimal_builds(slots, B, K)
        r_elig, r_fs, r_commits = simulate_first_sight(slots, opt_builds, B)
        r_net = reference_net_score(B, K, seed, slots, builds=opt_builds)

        # SECONDARY reference: online Bayesian (Dirichlet) policy -- timing comparator, not optimum.
        if bayes is not None:
            bz_builds = bayes[seed]
            bz_elig, bz_fs, bz_commits = simulate_first_sight(slots, bz_builds, B)
            bz_net = reference_net_score(B, K, seed, slots, builds=bz_builds)
            bz_fs_hazard = _hazard(bz_fs, bz_elig)
        else:
            bz_elig = bz_fs = bz_commits = 0
            bz_net = bz_fs_hazard = None

        corr, tested = _code_correctness(row) if framing == "R2c" else (0, 0)
        lateness = [pos - 1 for pos in commits_of(row).values()]
        per_seed.append({
            "seed": seed,
            "model_eligible_fs": m_elig, "model_fs_commits": m_fs, "model_commits": m_commits,
            "model_fs_hazard": _hazard(m_fs, m_elig),
            "model_net": m_net, "reference_net": r_net, "regret": r_net - m_net,
            "ref_eligible_fs": r_elig, "ref_fs_commits": r_fs, "ref_commits": r_commits,
            "ref_fs_hazard": _hazard(r_fs, r_elig),
            "bayes_net": bz_net, "bayes_regret": (bz_net - m_net) if bz_net is not None else None,
            "bayes_eligible_fs": bz_elig, "bayes_fs_commits": bz_fs, "bayes_commits": bz_commits,
            "bayes_fs_hazard": bz_fs_hazard,
            "zero_commit": m_commits == 0,
            "fs_among_commits": (m_fs / m_commits) if m_commits else None,
            "mean_lateness": st.mean(lateness) if lateness else None,
            "code_correct": corr, "code_tested": tested,
        })

    n = len(per_seed)
    tot_elig = sum(s["model_eligible_fs"] for s in per_seed)
    tot_fs = sum(s["model_fs_commits"] for s in per_seed)
    r_tot_elig = sum(s["ref_eligible_fs"] for s in per_seed)
    r_tot_fs = sum(s["ref_fs_commits"] for s in per_seed)
    bayes_available = bayes is not None and n > 0
    bz_tot_elig = sum(s["bayes_eligible_fs"] for s in per_seed)
    bz_tot_fs = sum(s["bayes_fs_commits"] for s in per_seed)
    regrets = [s["regret"] for s in per_seed]
    bayes_regrets = [s["bayes_regret"] for s in per_seed] if bayes_available else []
    commits = [s["model_commits"] for s in per_seed]
    code_correct = sum(s["code_correct"] for s in per_seed)
    code_tested = sum(s["code_tested"] for s in per_seed)
    return {
        "framing": framing, "B": B, "K": K, "n_seeds": n,
        "model_fs_hazard": _hazard(tot_fs, tot_elig),
        "reference_fs_hazard": _hazard(r_tot_fs, r_tot_elig),
        "bayes_fs_hazard": _hazard(bz_tot_fs, bz_tot_elig) if bayes_available else None,
        "model_commits_per_seed": (sum(commits) / n) if n else None,
        "reference_commits_per_seed": (sum(s["ref_commits"] for s in per_seed) / n) if n else None,
        "bayes_commits_per_seed": (sum(s["bayes_commits"] for s in per_seed) / n) if bayes_available else None,
        "zero_commit_incidence": (sum(s["zero_commit"] for s in per_seed) / n) if n else None,
        "mean_net": (st.mean(s["model_net"] for s in per_seed)) if n else None,
        "mean_reference_net": (st.mean(s["reference_net"] for s in per_seed)) if n else None,
        "mean_bayes_net": (st.mean(s["bayes_net"] for s in per_seed)) if bayes_available else None,
        "mean_regret": (st.mean(regrets)) if n else None,
        "se_regret": (st.stdev(regrets) / n ** 0.5) if n > 1 else 0.0,
        "mean_bayes_regret": (st.mean(bayes_regrets)) if bayes_available else None,
        "se_bayes_regret": (st.stdev(bayes_regrets) / len(bayes_regrets) ** 0.5) if len(bayes_regrets) > 1 else 0.0,
        "code_correctness": (code_correct / code_tested) if code_tested else None,
        "per_seed": per_seed,
    }


def analyze_all(seeds=CANONICAL_SEEDS, charges=CHARGES) -> dict:
    """Analyze every framing x budget x charge cell that has a run directory. `charges` defaults to the
    original run grid; pass a superset (e.g. (0, 10, 20, 24)) to also fold in the K=10 wait cell -- the
    K=10 online-Bayes DP is intractable, so its `bayes_*` fields come back None (spec §7)."""
    out = {}
    for f in FRAMINGS:
        for B in BUDGETS:
            for K in charges:
                if _cell_dir(f, B, K, CANONICAL_SEEDS[0]).parent.exists() or K in CHARGES:
                    out[f"{f}:B{B}:K{K}"] = analyze_cell(f, B, K, seeds=seeds)
    return out


def _fmt_pct(x: float | None) -> str:
    return "  -  " if x is None else f"{x:5.0%}"


def print_report(summary: dict) -> None:
    charges = sorted({int(k.split(":K")[1]) for k in summary})
    print("\n==== ECONOMIC RESPONSE SURFACE: first-sight commitment hazard ====")
    print(f"     model  |  hindsight-optimum (opt*)  |  online-Bayes (bayes)   "
          f"(each block: K={'/'.join(str(k) for k in charges)})")
    for framing in FRAMINGS:
        print(f"\n  {framing}:")
        header = ("    B   " + "".join(f"K={K:<6d}" for K in charges) +
                  " | opt* " + "".join(f"K={K:<5d}" for K in charges) +
                  " | bayes " + "".join(f"K={K:<5d}" for K in charges))
        print(header)
        for B in BUDGETS:
            mrow = "".join(_fmt_pct(summary[f"{framing}:B{B}:K{K}"]["model_fs_hazard"]) + "  " for K in charges)
            orow = "".join(_fmt_pct(summary[f"{framing}:B{B}:K{K}"]["reference_fs_hazard"]) + "  " for K in charges)
            brow = "".join(_fmt_pct(summary[f"{framing}:B{B}:K{K}"]["bayes_fs_hazard"]) + "  " for K in charges)
            print(f"    {B}   {mrow} |    {orow} |     {brow}")

    print("\n==== net points and regret vs the hindsight optimum (mean over seeds) ====")
    print("     (optNet >= 0 always; bayesNet shown for contrast where computed -- it goes negative "
          "at K=20, i.e. the old reference was dominated by never-building; '-' = DP intractable)")
    print(f"  {'cell':14s} {'commits/seed':>12s} {'zeroCommit':>11s} {'net':>7s} {'optNet':>7s} "
          f"{'regret':>10s} {'bayesNet':>8s} {'code%':>6s}")
    for key in summary:
        c = summary[key]
        code = "" if c["code_correctness"] is None else f"{c['code_correctness']:5.0%}"
        bnet = "   -   " if c["mean_bayes_net"] is None else f"{c['mean_bayes_net']:8.1f}"
        print(f"  {c['framing']+' B='+str(c['B'])+' K='+str(c['K']):14s} {c['model_commits_per_seed']:12.2f} "
              f"{c['zero_commit_incidence']:10.0%} {c['mean_net']:7.1f} {c['mean_reference_net']:7.1f} "
              f"{c['mean_regret']:6.1f}\u00b1{c['se_regret']:<3.1f} {bnet:>8s} {code:>6s}")


def _selftest() -> None:
    """Deterministic checks on the two first-sight functions -- the part that's easy to get subtly
    wrong (denominator semantics, budget-exhaustion cutoff). No file I/O, no reference DP."""
    # Stream: class 0 at pos 1,2 (recurs), class 1 at pos1, class 2 at pos1,2. B=1.
    class_seq = [0, 1, 0, 2, 2]
    slots = [{"slot_index": i, "class_id": c, "class_position": class_seq[:i + 1].count(c)}
             for i, c in enumerate(class_seq)]

    # 1. reference-style replay: a policy that builds class 0 at its FIRST sighting (pos 1).
    elig, fs, tot = simulate_first_sight(slots, {0: 1, 1: None, 2: None}, B=1)
    # decision turns before budget dies: slot0(class0,pos1)->commit. Budget now 0 -> stop.
    # eligible first-sight turns seen: just slot0 (class0 pos1). fs commit there. 1 commit total.
    assert (elig, fs, tot) == (1, 1, 1), (elig, fs, tot)

    # 2. a WAIT policy: build class 0 at its SECOND sighting (pos 2), skip everything at first sight.
    elig, fs, tot = simulate_first_sight(slots, {0: 2, 1: None, 2: None}, B=1)
    # turns: slot0(c0p1,skip,eligFS),slot1(c1p1,skip,eligFS),slot2(c0p2,COMMIT->budget0 stop).
    # eligible first-sight turns = slot0,slot1 = 2; fs commits = 0; total commits = 1.
    assert (elig, fs, tot) == (2, 0, 1), (elig, fs, tot)

    # 3. NEVER policy (all None): every non-recurring first sighting is an eligible turn, 0 commits.
    elig, fs, tot = simulate_first_sight(slots, {0: None, 1: None, 2: None}, B=1)
    # budget never spent, so we walk all decisions: slot0(c0p1 FS), slot1(c1p1 FS), slot2(c0p2, not FS),
    # slot3(c2p1 FS), slot4(c2p2 not FS). eligible FS = 3 (c0p1,c1p1,c2p1); commits 0.
    assert (elig, fs, tot) == (3, 0, 0), (elig, fs, tot)

    # 4. model_first_sight from a transcript mirrors the same counting. Eager R0: KEEP at pos1.
    row = {"transcript": [{"class_position": 1, "decision": "KEEP"}], "kept": {"0": 1}, "collected": 3}
    assert model_first_sight(row, "R0") == (1, 1, 1), model_first_sight(row, "R0")
    # wait-y R2c: two first-sight skips then a delayed claim.
    row2 = {"transcript": [{"class_position": 1, "decision": "SKIP_SOLVER"},
                           {"class_position": 1, "decision": "SKIP_SOLVER"},
                           {"class_position": 2, "decision": "CLAIM_SOLVER"}],
            "claimed": {"5": 2}, "collected": 4}
    assert model_first_sight(row2, "R2c") == (2, 0, 1), model_first_sight(row2, "R2c")

    print("analyze_economic_surface self-test OK (first-sight replay: eager/wait/never denominators "
          "correct incl. budget-exhaustion cutoff; model transcript counting matches)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--charges", type=int, nargs="+", default=[0, 10, 20, 24],
                    help="charges to analyze; default folds in the K=10 wait cell (spec §7)")
    ap.add_argument("--json-out", default=str(RUN_DIR / "analysis.json"))
    args = ap.parse_known_args()[0]
    if args.selftest:
        _selftest()
    else:
        summary = analyze_all(charges=tuple(args.charges))
        print_report(summary)
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote machine-readable summary -> {out}")
