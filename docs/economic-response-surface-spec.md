# Economic Response Surface — Preregistration + Implementation Plan

**Status (2026-07-09): COMPLETE + REFERENCE REPAIRED. Full 216-session run done, mechanically clean, analyzed and plotted — see §6. The economic reference was subsequently repaired (§2 "Reference repair", §6.1/§6.2 rerun on the same 216 sessions, $0): the primary reference is now the EXACT hindsight net-optimum (prior-free, no cap), against which regret is ≥0 in every cell; the old Dirichlet `ExactDP` is retained only as a secondary online-timing comparator. The genuine "wait"-band cell (K=10) is now also DONE (§7): 72 sessions, ≈$1.2, mechanically clean — the optimum builds selectively (~33–57% first-sight), R0 waits (lateness 1.6–3.8), R2c stays pinned at 100%.**

## 1. Design

Compare two framings on the identical 12 canonical Haiku A2 streams (`runs/urn_haiku_n-announced/seed_2000..2011`):
- **R0** — abstract free-text keep/pass (`urn_session.run_episode`).
- **R2c** — real-problem code-required claim/skip (`claim_solver_code_session.run_episode_code_claim`).

This re-conflates modality with construction burden (the framing ladder kept those separate) — deliberate here, not an oversight: this surface tests economic sensitivity at the two most-contrasting endpoints, the same move the project's original core result (urn vs. tool) made.

Each collected/solved item earns +1; each KEEP/`claim_solver` deducts `K` points once; PASS/skip earns 0. R2c payoff stays unconditional on code correctness (correctness remains a diagnostic only, never scored — unchanged from R2c's own design).

Cross `B ∈ {1, 3, 5}` with `K ∈ {K_eager, K_wait, K_never} = {0, 20, 24}` → 9 cells × 2 framings = **18 frame-cells**, 12 seeds each = **216 sessions**.

## 2. Calibration (locked, 2026-07-09)

### Reference repair (2026-07-09, supersedes the DP-reference parts of this section)

The economic surface originally scored regret and "optimum" first-sight hazard against the
belief-state `ExactDP` policy π\* (symmetric Dirichlet(α=1) prior). **That reference was replaced as
the primary object** for two reasons a reviewer would catch:

1. **Prior mismatch.** π\* is Bayes-optimal only for a symmetric Dirichlet prior; the actual generator
   is fixed hot-count + trap-early (g=1). So π\* is *not* the true optimum for these streams, and
   "actual-optimum" language is not defensible (same caveat already flagged for Opus urn in
   `rl-phase1-results.md`).
2. **Negative realized reference net = provably dominated.** At K=20 the capped-DP realization had
   *negative* net in every cell (B=1/3/5: −7.2 / −22.2 / −27.3). Since "never build" nets exactly 0,
   the "optimal" reference was strictly worse than doing nothing there — and the model beat it in two
   cells (regret −1.8, −0.7). A reference the model can beat is not an optimum.

**Primary reference is now the EXACT hindsight net-optimum** (`economic_surface.hindsight_net_optimal_builds`):
building class *c* at first sight captures all `size_c` occurrences for net `size_c − K`, later is
strictly worse, not building is 0; occurrences are disjoint, so the optimum builds — at first sight —
the ≤B classes with the largest positive `size_c − K`. This is **prior-free, generator-agnostic, no
DP, no cap**. Consequences: regret vs it is **≥0 by construction** (no policy beats the realized
optimum); net is never negative (K=20: ≈+0.3; K=24: exactly 0, matching the never-build proof); and
the B=5/K=20 cap caveat below is **moot** for the primary reference.

**π\* (`ExactDP`) is retained only as a SECONDARY, explicitly-labelled "online Bayesian
(Dirichlet-prior) policy"** (`bayes_*` fields in the analysis) — useful for the online *timing*
pattern (it waits for a recurrence), never as the optimum. The cap notes below apply only to it.

**Empty "wait" band.** Under the true optimum, K=20 is essentially a *never-build* cell (net ≈0.3;
a class must recur >20 times to pay, and max class size is 23). The `K_wait=20` label was itself a
prior/cap artifact. On these streams (≈3 hot classes of size 13–23 + ≈4 traps of size 1–6 per seed),
a genuine online "wait" band sits near **K≈10** (hot classes pay, traps don't, and online you must
wait to tell them apart). Getting a real wait cell requires a new live run at a lower charge — see §7.

### Original DP-based calibration (now applies only to the secondary online-Bayes comparator)

`K_eager = 0` (given).

**`K_never = 24` — proof-backed, not DP-approximated.** Building a class can never net more than (occurrences from the build point onward) − `K`. The largest class size across all 12 canonical streams is 23 (verified directly from the stream files). So `K ≥ 24` makes "never build" strictly optimal for every class in every one of these streams, regardless of budget or belief state — an exact bound, independent of B, requiring no DP at all.

**Deviation from the original plan: `cap=None` is intractable at this scale.** `ExactDP` with `cap=None` at N=8, T=60, B=1 didn't finish in 5+ minutes; even `cap=8` scales combinatorially (cap=3: 0.6s → cap=5: 3.4s → cap=8: 25s+, roughly 2x per unit of cap) — reaching cap≈24 (needed to never trigger the forced-build artifact on the observed max recurrence of 23) would take years, not minutes. **`K_never` therefore uses the closed-form proof above, not a capped DP, in every cell** — a capped DP was directly observed to give a *wrong* answer at K=24 (41 spurious builds at B=5/cap=5, a pure cap artifact: the forced-build branch fires before the value comparison, and cap=5 is far below the actual recurrence counts it needs to dominate).

**`K_wait = 20`.** Found by bisection at B=3 using increasing cap values, cross-checked for convergence: cap=3 and cap=6 stayed flat/non-zero up to K=22 (both cap-biased — the forced-build artifact was still dominating); cap=8 first showed 0% first-sight at K=19–20; cap=10 pushed the crossover to K≤18. The exact smallest qualifying integer kept drifting down as cap increased and was not fully pinned down (same intractability shape as `K_never`, one level down) — **K=20 is a verified-safe, conservative choice** (confirmed 0% first-sight / 100% seed-commitment at both cap=8 and cap=10), not the literal minimum. The direction of the remaining bias is known and one-sided (capped DP can only overstate eagerness, never understate it), so this is a safe conservative pick, not an arbitrary one.

**Per-cell verification (not just B=3):**

| B | K=0 (eager) | K=20 (wait) | K=24 (never) |
|---|---|---|---|
| 1 | 0% first-sight | 0% first-sight | 0% first-sight |
| 3 | 42% first-sight | 0% first-sight (verified) | 0 builds (proof) |
| 5 | 95% first-sight | 27% first-sight (cap=5, likely overstated) | 0 builds (proof) |

**B=1 finding, not a defect:** budget scarcity alone already makes the reference policy maximally conservative even at `K=0` — the eager/wait/never axis collapses at B=1. Report this as-is rather than re-engineering B=1 to force three distinct labels; it's a real result about how budget and charge trade off.

**B=5/K=20 caveat:** 27% is measured at `cap=5` (the fastest tractable cap at B=5; `cap=8` did not finish in 5 minutes). Since capped DP overstates eagerness, the true value is very likely ≤25% (satisfying the intended "wait" bar), but this wasn't independently confirmed at a higher cap the way B=3 was. Treat the B=5/K=20 cell as "very likely wait, not independently confirmed" in any writeup.

## 3. Implementation

- **`economic_surface.py`** — cell spec (`FRAMINGS`, `BUDGETS`, `CHARGES`), net-score calc (`raw_points − K × commits`, reusing `_balls_collected` unchanged), reference-policy lookup (hardcoded "never" for K=24 per §2's proof; `ExactDP` cap=3 lookup for K=0/K=20 — cap=3 is the project's existing validated cap for the *standard, K=0-scale* comparisons already used elsewhere; the K=20 numbers in the table above are diagnostic-only calibration results, not what the runtime reference-policy computation uses for regret — see code comments for the exact boundary), artifact schema, self-tests.
- **`run_economic_surface.py`** — executes framing × B × K × seed, resume support, canonical stream-hash assertion per cell, explicit `tool_choice="auto"` for Haiku (not inherited implicitly), one global spend cap, `call_with_retry`'s existing transport-vs-model failure distinction.
- **Extend `ladder_parity_selftest.py`** — scripted R0/R2c choices across representative `(B,K)` cells asserting identical commitment positions, budgets, and now net scores.

## 4. Primary outcomes (defined even when the optimal never commits)

- First-sight commitment hazard (commits on first-sighting turns ÷ eligible first-sighting turns).
- Commitments per seed and zero-commit incidence.
- Net points and regret vs. the charge-aware reference (proof-based for K=24, DP-based for K=0/20).

Secondary: first-sight % among realized commitments, mean lateness, trap allocations, unresolved decisions, R2c code correctness (diagnostic only).

**Interpretation:** primary test is the framing × economic-cell interaction — does R0 co-move with the reference policy while R2c stays eager regardless? Any unresolved rate >10% invalidates that cell; mechanical repair may trigger reruns, behavioral outcomes may not.

## 5. Run order

1. Self-tests (this doc's §3 items) — $0.
2. Two-seed smoke across all 18 cells — explicitly checking per-cell token-cap truncation and prose-derailment, not just pass/fail (the exact failure modes that cost three debugging cycles building the framing ladder).
3. Full 12-seed run, $15 hard cap, uniform-seed-reduction fallback if projected cost exceeds it.

**Stop point for this pass: end of step 1. No live API calls without separate approval.**

---

## 6. Results (2026-07-09, all 216 sessions)

**Run:** 2 framings × B∈{1,3,5} × K∈{0,20,24} × 12 canonical seeds (2000–2011), A2, Haiku, on the byte-identical canonical streams. Two-seed mechanical smoke first (§5 step 2), then the full 12-seed panel.

**Mechanical validity (governs stop/go; behavioral outcomes never do):** all 216 sessions present and clean. **Zero token-cap truncation** in any cell — R0 output peaked at 252/512 tokens, R2c at 717/2500, wide headroom everywhere (the charge sentence did *not* provoke runaway per-turn K-arithmetic; R0 reasons about the charge in 2–4 sentences and ends cleanly). **0% unresolved decisions** in every cell (all R0 turns parsed via the `DECISION:` tag save one via the standalone-token fallback — still a valid parse, not a failure; all R2c turns clean single tool calls; zero `malformed_args`/`default`/`error`). No stream drift (`assert_canonical` gates every cell).

**Cost:** $0.62 (2-seed smoke, 36 sessions) + $2.91 (full run, 216 sessions incl. the 36 cached from the smoke) = **≈$2.9 total**, well under the $15 cap; the uniform-seed-reduction fallback never triggered. As usual in this codebase the a-priori EST guard was ~5× the measured cost.

### 6.1 First-sight commitment hazard — model vs. references (repaired reference, same 216 sessions)

Hazard = commits at a class's first sighting ÷ eligible first-sight decision turns (a decision turn on a class at `class_position==1` with budget remaining). Computed identically for the model (from its transcript), for the **hindsight optimum `opt*`** (primary), and for the **online-Bayes `bayes`** DP (secondary), by replaying each policy's build positions through the same per-draw budget loop.

| | K=0 (eager) | K=20 (opt≈never) | K=24 (never) |
|---|---|---|---|
| **R0** B=1 / 3 / 5 | 17 / 25 / 42% | 3 / 10 / 16% | 0 / 9 / 5% |
| **opt\*** B=1 / 3 / 5 | 33 / 61 / 83% | 2 / 2 / 2% | 0 / 0 / 0% |
| **bayes** B=1 / 3 / 5 | 0 / 29 / 89% | 0 / 0 / 14% | 0 / 0 / 0% |
| **R2c** B=1 / 3 / 5 | 100 / 95 / 100% | 100 / 97 / 100% | 100 / 97 / 98% |

- **The hindsight optimum `opt*` gives the honest charge gradient:** first-sight hazard high at K=0 (33/61/83%, rising with budget), ~2% at K=20, exactly 0% at K=24. This is the true "build the hot classes eagerly → build essentially nothing" collapse as the charge rises.
- **R0 is economically sensitive:** its first-sight hazard falls monotonically with the charge and co-moves with `opt*`. (R0 is *under*-eager relative to `opt*` at K=0 — 42% vs 83% at B=5, a bit too cautious when building-on-sight is optimal — and mildly *over*-eager at K=20/24 relative to `opt*`'s ~0–2%, but the point is it *moves with K*, unlike R2c.)
- **R2c is economically invariant:** pinned at ~100% first-sight across the entire charge axis. It commits on first sight even at K=24, where never-building is *provably* optimal — the same eagerness the framing ladder found, now shown to be insensitive to the visible economics rather than merely present.
- The `bayes` row is the old online-Dirichlet policy, kept for its online *timing* pattern (it waits, so its K=0 first-sight is lower than `opt*`'s despite building the same hot classes). It is **not** the optimum and is not what regret is scored against.

This is the pre-registered **framing × economic-cell interaction** (§4): R0 tracks the optimum surface while R2c does not. Per §4's interpretation rules, this is the "R2c remains eager in wait/never cells → code-triggered commitment is economically invariant" outcome, with R0 confirming the same charge is economically legible in the abstract framing.

### 6.2 Net points and regret vs. the hindsight optimum (mean over 12 seeds, ±SE)

Regret = **opt\*** net − model net (net = raw collected − K × commits). `opt*` net is the exact hindsight net-optimum (§2 reference repair): **≥0 in every cell** (K=20 ≈+0.3, K=24 exactly 0), so regret is **≥0 everywhere** — the model can no longer "beat the optimum". `bayes net` is the old Dirichlet DP's net, shown only for contrast: it goes **negative** at K=20 (dominated by never-building), which is exactly why it was demoted. Net-point currency, kept separate from the historical token-cost regret.

| cell | model net | opt\* net | regret vs opt\* | bayes net (contrast) |
|---|---|---|---|---|
| R0  B=1 K=0 / 20 / 24 | 8.8 / −5.4 / −11.0 | 19.0 / 0.3 / 0.0 | 10.2±1.6 / 5.8±0.5 / 11.0±0.9 | 12.8 / −7.2 / 0.0 |
| R0  B=3 K=0 / 20 / 24 | 36.9 / −21.6 / −34.2 | 49.5 / 0.3 / 0.0 | 12.6±2.6 / 21.9±2.1 / 34.2±2.2 | 38.4 / −22.2 / 0.0 |
| R0  B=5 K=0 / 20 / 24 | 51.2 / −41.1 / −60.3 | 55.9 / 0.3 / 0.0 | 4.7±0.9 / 41.4±5.1 / 60.3±6.0 | 52.4 / −27.3 / 0.0 |
| R2c B=1 K=0 / 20 / 24 | 9.8 / −10.2 / −14.2 | 19.0 / 0.3 / 0.0 | 9.2±1.9 / 10.6±1.9 / 14.2±1.9 | 12.8 / −7.2 / 0.0 |
| R2c B=3 K=0 / 20 / 24 | 32.3 / −27.5 / −38.1 | 49.5 / 0.3 / 0.0 | 17.2±2.3 / 27.8±2.2 / 38.1±1.7 | 38.4 / −22.2 / 0.0 |
| R2c B=5 K=0 / 20 / 24 | 50.2 / −49.8 / −69.9 | 55.9 / 0.3 / 0.0 | 5.8±2.0 / 50.2±2.1 / 69.9±2.1 | 52.4 / −27.3 / 0.0 |

R2c's invariance is expensive: in the high-charge cells its regret is uniformly larger than R0's at matched B (e.g. B=5/K=24: 69.9 vs 60.3; B=5/K=20: 50.2 vs 41.4), because it spends its whole budget on charged commits the optimum declines to make. Zero-commit incidence is 0% in every cell for both framings (neither ever runs a seed to completion without committing), so the entire signal lives in *when* and *whether-relative-to-optimal* they commit, not in abstention.

### 6.3 Calibration caveats, confirmed in the live data

After the reference repair (§2) the old DP-cap caveats no longer touch the primary result:

- **The hindsight optimum needs no cap and no prior**, so the former B=5/K=20 "cap=5, likely overstated" caveat is moot for `opt*` — its net and first-sight hazard are exact at every (B,K).
- **K=20 is a near-never cell, not a wait cell:** `opt*` first-sight is ~2% and its net is ≈+0.3 there (a class must recur >20 times to pay; max class size 23). So {0,20,24} separates **eager vs never**, not eager/wait/never. A genuine wait cell requires a lower charge (§7). Reported honestly rather than relabelled.
- The old-reference caveats (B=1 Dirichlet collapse; B=5/K=20 cap uncertainty) now apply only to the secondary `bayes` comparator, where they remain true but no longer gate any claim.

### 6.4 Scope note (as pre-registered)

This surface deliberately re-conflates modality (free-text vs. tool-call) with construction burden — the two endpoints the framing ladder separated. That is intentional (it mirrors the project's original urn-vs-tool contrast) and does not contradict the ladder's decomposition; it tests economic *sensitivity* at the two most-contrasting framings, not the intermediate rungs.

### 6.5 Artifacts

- `scripts/creator/tool_disposition_benchmark/economic_surface.py` — cells, charge-aware net score, **primary** reference `hindsight_net_optimal_builds` (exact, prior-free, no cap), and the **secondary** online-Bayes `reference_builds` (proof for K≥24, cached `ExactDP` for K∈{0,20}).
- `scripts/creator/tool_disposition_benchmark/run_economic_surface.py` — the driver (resume, stream-hash gate, provider-conditional `tool_choice="auto"`, spend cap).
- `scripts/creator/tool_disposition_benchmark/analyze_economic_surface.py` — per-cell/seed metrics → `runs/economic_surface_haiku/analysis.json`: `reference_*`/`regret` are vs the hindsight optimum (primary), `bayes_*` vs the online-Bayes DP (secondary). Seed-level points retained for seed-clustered bootstrap.
- `scripts/plot_economic_response_surface.py` → `figs/paper/fig_economic_response_surface.{png,pdf}` (overlay is the hindsight optimum `opt*`).
- Secondary online-Bayes DP cached in `runs/economic_surface_reference/`; sessions under `runs/economic_surface_haiku/<frame>/B_<B>/K_<K>/seed_<seed>/`.

---

## 7. The genuine "wait"-band cell: K=10 (DONE, 2026-07-09)

The repaired reference showed {0,20,24} separates **eager vs never**, not eager/wait/never — under the true optimum both K=20 and K=24 are essentially never-build. A **lower charge K=10** was added to recover a real intermediate cell: from the observed class-size structure (≈3 hot classes of size 13–23 + ≈4 traps of size 1–6 per seed), K=10 makes hot classes strictly worth building and traps strictly not, so the hindsight optimum builds exactly the ~3 hot classes at first sight while an eager policy would waste budget on traps.

**Run:** 2 framings × B∈{1,3,5} × K=10 × 12 canonical seeds = **72 sessions**, Haiku, on the byte-identical canonical streams (2-seed smoke first, then the full panel). **Cost: $0.18 (smoke) + $1.19 (full, incl. the 12 cached from smoke) = ≈$1.2 total.**

**Mechanical validity (n=72):** 0 unresolved decisions, 0 malformed/error/default/unknown decisions, no token-cap truncation (R0 peaked 234/512, R2c 620/2500). Clean.

**Result — K=10 is a genuine wait/selective cell** (first-sight hazard; regret vs the hindsight optimum, ±SE over 12 seeds):

| cell | model first-sight | opt\* first-sight | model net | opt\* net | regret | model lateness |
|---|---|---|---|---|---|---|
| R0  B=1 / 3 / 5 | 0 / 6 / 16% | 33 / 57 / 39% | 3.2 / 10.8 / −0.1 | 9.0 / 19.6 / 19.6 | 5.8±1.1 / 8.8±1.9 / 19.7±2.1 | 3.75 / 2.00 / 1.56 |
| R2c B=1 / 3 / 5 | 100 / 100 / 100% | 33 / 57 / 39% | −0.2 / 2.7 / 0.2 | 9.0 / 19.6 / 19.6 | 9.2±1.9 / 16.9±2.2 / 19.4±2.2 | 0.00 / 0.00 / 0.00 |

- Unlike K=20/24 (opt\* first-sight ~0–2%), at K=10 **the optimum genuinely builds selectively at first sight** (33/57/39%) — it commits to the hot classes that pay. This is the intermediate cell the original {0,20,24} axis lacked.
- **R0 waits:** first-sight ≤16%, lateness 1.56–3.75 (it builds on recurrence, after a class proves itself) — it responds to the charge by becoming cautious, in fact slightly *over*-cautious relative to the optimum.
- **R2c is economically invariant, again:** pinned at 100% first-sight, lateness 0, net ≈0, even though the optimum here is to build ~3 hot classes *selectively after confirming them*. The code framing commits on sight regardless.

**Note on the secondary online-Bayes comparator at K=10:** its `ExactDP` is intractable at K=10 (cap=10 does not finish), so only the model-vs-hindsight-optimum comparison is reported for this cell. This does not weaken the result — the hindsight optimum is exact and is the object regret is defined against. `analyze_economic_surface` sets the `bayes_*` fields to `None` at K=10 (guarded by `economic_surface.BAYES_CHARGES`) and computes the exact hindsight optimum as usual.

**Figure axis (done).** `fig_economic_response_surface.{png,pdf}` now plots the clean **eager → wait → never** axis `K ∈ {0, 10, 24}` (`plot_economic_response_surface.CHARGES`), dropping the redundant near-never K=20. `analyze_economic_surface.py` analyzes `{0, 10, 20, 24}` by default so `analysis.json` retains K=20 for the §6 tables while the figure uses the three-point axis. The K=10 column is a genuine intermediate: `opt*` builds selectively (33/57/39% first-sight across B), R0 waits (0/6/16%), R2c stays pinned at 100%.
