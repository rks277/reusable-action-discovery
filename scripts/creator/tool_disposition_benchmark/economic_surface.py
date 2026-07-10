"""Economic response surface -- cell spec, charge-aware net scoring, and reference-policy lookup
(`docs/economic-response-surface-spec.md`, which this module implements section-for-section).

Two framings (R0 free-text, R2c code-required claim) x three budgets B in {1,3,5} x three charges
K in {0, 20, 24} = 18 frame-cells, 12 canonical seeds each. Each collected/solved item earns +1;
each commit (KEEP/`claim_solver`) deducts K points once; PASS/skip earns 0 -- identical economics in
both framings, since both already reuse `urn_common._balls_collected` for raw collection counts.

REFERENCE (2026-07-09 repair, spec doc §2). The PRIMARY reference is now the EXACT hindsight
net-optimum (`hindsight_net_optimal_builds`) -- prior-free, generator-agnostic, no DP, no cap. It
replaced the belief-state `ExactDP` as the object regret/optimum claims are made against, because the
DP is Bayes-optimal only for a symmetric Dirichlet prior that does NOT match the fixed hot/trap
generator, and at K=20 its capped realization reported negative net (a "reference" provably dominated
by never-building, which the model beat -- see spec §2). The `ExactDP` policy is RETAINED as a
SECONDARY, explicitly-labelled "online Bayesian (Dirichlet-prior) policy" comparator -- useful for the
online *timing* pattern (it waits for a recurrence), NOT as the optimum. `reference_builds` still
computes that online-Bayes policy; K=0 uses cap=3 (the project's pre-existing validated cap for
uncharged economics), K=20 uses cap=10 (B in {1,3}) / cap=5 (B=5, per spec's documented B=5 caveat),
and K>=24 is the exact "never builds" proof (max class size across the 12 canonical streams is 23, so
K>=24 makes never-build strictly optimal for every class regardless of B). The online-Bayes DP is
expensive at full scale (minutes per (B,K) at higher cap) and result-INVARIANT across repeated calls,
so its results are cached to disk after the first computation; the hindsight optimum is cheap and
recomputed on demand.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.economic_surface   # self-test
"""
from __future__ import annotations
import json
from pathlib import Path

from scripts.creator.tool_disposition_benchmark.urn_common import UNIFORM, N, T, _balls_collected
from scripts.creator.tool_disposition_benchmark.exact_dp import ExactDP

FRAMINGS = ("R0", "R2c")
BUDGETS = (1, 3, 5)
CHARGES = (0, 20, 24)          # original run grid -- docs/economic-response-surface-spec.md §2
K_NEVER_THRESHOLD = 24         # proof: max class size across the 12 canonical streams (23) + 1

# The genuine wait-band charge added after the reference repair (spec §7): K=0 eager, K=10 wait,
# K=24 never. This is the charge set the paper figure uses (eager -> wait -> never); K=20 is retained
# in CHARGES/the analysis only as the historical near-never point.
FIGURE_CHARGES = (0, 10, 24)

# Charges at which the SECONDARY online-Bayes ExactDP is tractable enough to compute/cache (K=0 cap=3;
# K=20 the documented `_WAIT_CAP_FOR_B` caps; K>=24 the closed-form never-build proof). At K=10 the
# DP needs cap>=10 to avoid the forced-build artifact and does NOT finish (the same intractability that
# motivated demoting it), so the analysis reports only the exact hindsight optimum there.
BAYES_CHARGES = (0, 20, 24)

CELLS = tuple((f, b, k) for f in FRAMINGS for b in BUDGETS for k in CHARGES)
assert len(CELLS) == 18

CANONICAL_SEEDS = tuple(range(2000, 2012))
CANONICAL_STREAM_DIR = Path("runs/urn_haiku_n-announced")   # R0's cached dir -- source of truth for
                                                             # canonical stream hashes (spec doc §1)

# Largest cap the spec doc's calibration found tractable, per budget, for the K=20 ("wait") cell.
# K=0 always uses cap=3 (pre-existing project convention for uncharged economics); K>=24 never calls
# ExactDP at all (see module docstring).
_WAIT_CAP_FOR_B = {1: 10, 3: 10, 5: 5}

_REFERENCE_CACHE_DIR = Path("runs/economic_surface_reference")


def commits_of(row: dict) -> dict[int, int]:
    """R0/R1 use `kept`, R2/R2c use `claimed` -- same shape (`class_id -> class_position`), just a
    different key name per rung's own convention. Returns whichever is present."""
    d = row.get("claimed", row.get("kept"))
    if d is None:
        raise KeyError("row has neither 'claimed' nor 'kept'")
    return {int(k): v for k, v in d.items()}


def net_score(row: dict, K: int) -> float:
    """`raw_points - K * num_commits`. `raw_points` is `_balls_collected`'s "+1 per collected/solved
    item" count (R0's `collected` field / R2c's `collected` field -- same name, same meaning, both
    already computed by their own `run_episode*` functions); `num_commits` is the number of
    KEEP/`claim_solver` calls that actually registered (len of the commits dict), NOT the number of
    decision turns -- PASS/skip/unresolved turns cost nothing, matching the spec's economics."""
    return row["collected"] - K * len(commits_of(row))


def _reference_cache_path(B: int, K: int) -> Path:
    return _REFERENCE_CACHE_DIR / f"B{B}_K{K}.json"


def reference_builds(B: int, K: int, seeds=CANONICAL_SEEDS,
                     stream_dir: Path = CANONICAL_STREAM_DIR) -> dict[int, dict]:
    """Reference-policy builds per canonical seed: `{seed: {class_id: build_position_or_None}}`.
    K>=24 is hardcoded to "never builds" (proof, see module docstring) -- zero DP cost, exact for any
    B. K<24 calls `ExactDP.policy_builds` once per seed, reusing ONE `ExactDP` instance (and its
    memoized value-function cache) across all seeds in `seeds` -- the expensive part is filling the
    cache from the root, not re-walking it per stream. Cached to disk after the first computation
    since this is deterministic and expensive (minutes at the higher caps `_WAIT_CAP_FOR_B` uses)."""
    cache_path = _reference_cache_path(B, K)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if tuple(sorted(int(s) for s in cached)) == tuple(sorted(seeds)):
            return {int(s): {int(cid): pos for cid, pos in builds.items()}
                    for s, builds in cached.items()}

    if K >= K_NEVER_THRESHOLD:
        out = {}
        for seed in seeds:
            slots = json.loads((stream_dir / f"seed_{seed}" / "stream.json").read_text())
            out[seed] = {s["class_id"]: None for s in slots}
    else:
        cap = 3 if K == 0 else _WAIT_CAP_FOR_B[B]
        dp = ExactDP(u_hand=0.0, u_build=1.0 - K, u_reuse=1.0, N=N, T=T, B=B, alpha=1.0, cap=cap)
        out = {}
        for seed in seeds:
            slots = json.loads((stream_dir / f"seed_{seed}" / "stream.json").read_text())
            out[seed] = dp.policy_builds(slots)

    _REFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {str(s): {str(cid): pos for cid, pos in builds.items()} for s, builds in out.items()},
        indent=2))
    return out


def reference_net_score(B: int, K: int, seed: int, slots: list[dict],
                        builds: dict[int, int | None] | None = None) -> float:
    """Net score of ANY build policy (`{class_id: build_position_or_None}`) on one realized stream:
    `raw_collected - K * num_commits`. Works for either reference -- the primary hindsight optimum
    (`hindsight_net_optimal_builds`) or the secondary online-Bayes policy (`reference_builds`) --
    since both share the build-dict shape. `builds` lets a caller reuse an already-fetched policy
    instead of recomputing; if omitted it fetches the online-Bayes DP builds (expensive at higher
    caps -- prefer passing `builds` in a loop over seeds)."""
    if builds is None:
        builds = reference_builds(B, K, seeds=(seed,))[seed]
    commits = {cid: pos for cid, pos in builds.items() if pos is not None}
    collected = _balls_collected(slots, commits)
    return collected - K * len(commits)


def hindsight_net_optimal_builds(slots: list[dict], B: int, K: int) -> dict[int, int | None]:
    """EXACT realized net-optimum for the charge-aware net objective (`raw_collected - K*commits`),
    given the full realized stream. This is the PRIMARY economic-surface reference after the
    2026-07-09 reference repair (`docs/economic-response-surface-spec.md` §2): it is prior-free and
    generator-agnostic -- no Dirichlet assumption, no belief-state DP, no cap, so it carries none of
    the cap-artifact / prior-mismatch problems that made the old `ExactDP` reference report NEGATIVE
    net (a policy provably dominated by never-building) in the K=20 cells.

    Derivation. Building class `c` at its FIRST sighting captures every one of its `size_c`
    occurrences for net `size_c - K`; building it later is strictly worse (each skipped occurrence is
    a lost +1); not building it nets 0. Occurrences of distinct classes are disjoint, so the objective
    is separable and the optimum is: build, at first sight, the <=B classes with the largest strictly
    positive `size_c - K`. Because a hindsight optimum never has a reason to wait, every build is at
    `class_position == 1` -- so this drops straight into `simulate_first_sight` / `reference_net_score`
    with the same build-dict shape `ExactDP.policy_builds` returns.

    Returns `{class_id: 1 if built else None}`. Regret vs this reference is >= 0 by construction (no
    online or hindsight policy can beat the realized net-optimum), and its first-sight commitment
    count is exactly the number of classes still worth building at charge K, which falls monotonically
    as K rises -- the clean charge-aware co-movement signal the surface is testing for."""
    sizes: dict[int, int] = {}
    for s in slots:
        sizes[s["class_id"]] = sizes.get(s["class_id"], 0) + 1
    gains = sorted(((sizes[c] - K, c) for c in sizes if sizes[c] - K > 0), reverse=True)
    chosen = {c for _, c in gains[:B]}
    return {s["class_id"]: (1 if s["class_id"] in chosen else None) for s in slots}


def _selftest():
    """Small/synthetic-scale checks only -- the full N=8/T=60 canonical-stream computation is
    expensive at the higher `_WAIT_CAP_FOR_B` caps (minutes; see spec doc §2) and is NOT re-derived
    here. This validates the MECHANISM (net-score arithmetic, the K>=24 hardcode, cell-spec shape),
    not the locked calibration numbers themselves."""
    # 1. cell spec shape.
    assert len(CELLS) == 18
    assert set(f for f, b, k in CELLS) == set(FRAMINGS)
    assert set(b for f, b, k in CELLS) == set(BUDGETS)
    assert set(k for f, b, k in CELLS) == set(CHARGES)

    # 2. net_score arithmetic, R0-style ("kept") and R2c-style ("claimed") rows.
    row_r0 = {"kept": {"0": 1, "2": 2}, "collected": 4}
    row_r2c = {"claimed": {"0": 1, "2": 2}, "collected": 4}
    assert net_score(row_r0, K=0) == 4
    assert net_score(row_r0, K=1) == 4 - 1 * 2 == 2      # 2 commits, 1 point each deducted once
    assert net_score(row_r2c, K=1) == 2                   # identical economics, different key name
    assert commits_of(row_r0) == commits_of(row_r2c) == {0: 1, 2: 2}

    # 3. K>=24 hardcodes "never builds" -- exact for ANY B, on a small hand-built stream, no DP call.
    class_seq = [0, 0, 1, 2, 0, 2, 3, 3, 3]
    slots = [{"slot_index": i, "class_id": cid, "class_position": class_seq[:i + 1].count(cid)}
             for i, cid in enumerate(class_seq)]
    tmp_dir = Path("runs/_economic_surface_selftest_tmp")
    seed_dir = tmp_dir / "seed_9999"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "stream.json").write_text(json.dumps(slots))
    for B in (1, 3, 5):
        builds = reference_builds(B, 24, seeds=(9999,), stream_dir=tmp_dir)
        assert all(v is None for v in builds[9999].values()), (B, builds)
        assert reference_net_score(B, 24, 9999, slots, builds=builds[9999]) == 0.0
    import shutil
    shutil.rmtree(tmp_dir)
    (_REFERENCE_CACHE_DIR / "B1_K24.json").unlink(missing_ok=True)
    (_REFERENCE_CACHE_DIR / "B3_K24.json").unlink(missing_ok=True)
    (_REFERENCE_CACHE_DIR / "B5_K24.json").unlink(missing_ok=True)

    # 4. K<24 path: `reference_builds` end-to-end on a small synthetic case (N=8/T=60's own module
    #    constants are used for N/T internally, so this exercises the actual K=0/cap=3 code path,
    #    not a re-derivation) -- mechanism check, not the locked calibration numbers.
    small_dir = Path("runs/_economic_surface_selftest_small")
    small_seed_dir = small_dir / "seed_8888"
    small_seed_dir.mkdir(parents=True, exist_ok=True)
    small_seq = [0, 0, 0, 1, 2, 0, 0, 1, 2, 0]
    small_slots = [{"slot_index": i, "class_id": cid, "class_position": small_seq[:i + 1].count(cid)}
                   for i, cid in enumerate(small_seq)]
    (small_seed_dir / "stream.json").write_text(json.dumps(small_slots))
    want = ExactDP(u_hand=0.0, u_build=1.0, u_reuse=1.0, N=N, T=T, B=1, alpha=1.0, cap=3
                   ).policy_builds(small_slots)
    got = reference_builds(1, 0, seeds=(8888,), stream_dir=small_dir)[8888]
    assert got == want, (got, want)
    assert reference_net_score(1, 0, 8888, small_slots, builds=got) == _balls_collected(small_slots, {
        cid: pos for cid, pos in got.items() if pos is not None})
    (_REFERENCE_CACHE_DIR / "B1_K0.json").unlink(missing_ok=True)
    import shutil
    shutil.rmtree(small_dir)

    # 5. PRIMARY reference: exact hindsight net-optimum. Hand stream with known class sizes:
    #    class 0 size 5, class 1 size 3, class 2 size 1. Builds at first sight (position 1) only.
    hs_seq = [0, 1, 0, 2, 0, 1, 0, 1, 0]  # 0:x5, 1:x3, 2:x1
    hs_slots = [{"slot_index": i, "class_id": cid, "class_position": hs_seq[:i + 1].count(cid)}
                for i, cid in enumerate(hs_seq)]
    #   K=0, B=1: only the biggest class (0, size 5) -> net 5.
    b = hindsight_net_optimal_builds(hs_slots, B=1, K=0)
    assert {c: p for c, p in b.items() if p is not None} == {0: 1}, b
    assert reference_net_score(1, 0, 0, hs_slots, builds=b) == 5.0
    #   K=0, B=3: all three classes (sizes 5,3,1) at first sight -> net 5+3+1 = 9.
    b = hindsight_net_optimal_builds(hs_slots, B=3, K=0)
    assert {c: p for c, p in b.items() if p is not None} == {0: 1, 1: 1, 2: 1}, b
    assert reference_net_score(3, 0, 0, hs_slots, builds=b) == 9.0
    #   K=4, B=3: only class 0 has size>4 -> build just it, net 5-4 = 1 (never negative, unlike the
    #   old DP -- classes 1,2 with size<=4 are correctly declined rather than force-built).
    b = hindsight_net_optimal_builds(hs_slots, B=3, K=4)
    assert {c: p for c, p in b.items() if p is not None} == {0: 1}, b
    assert reference_net_score(3, 4, 0, hs_slots, builds=b) == 1.0
    #   K=10, any B: no class pays (max size 5 < 10) -> never build, net 0 (matches never-optimal).
    for B in (1, 3, 5):
        b = hindsight_net_optimal_builds(hs_slots, B=B, K=10)
        assert all(p is None for p in b.values()), (B, b)
        assert reference_net_score(B, 10, 0, hs_slots, builds=b) == 0.0

    print("economic_surface self-test OK (cell spec shape, net-score arithmetic for both R0/R2c "
          "row shapes, K>=24 hardcode exact on any B, K<24 online-Bayes ExactDP mechanism sane, "
          "hindsight net-optimum: top-B by size-K, first-sight builds, never negative)")


if __name__ == "__main__":
    _selftest()
