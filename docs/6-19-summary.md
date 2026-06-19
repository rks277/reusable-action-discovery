# ToolWorld — multi-rep precision pass + natural-build E (6-19)

A ToolWorld thread (the 6-18 work was WoodWorld; see `docs/6-18-summary.md`). Two
methodology changes to the region-sweep decomposition plus a multi-rep re-estimate that
tightens the headline numbers and *sharpens* the recognition inversion.

## TL;DR

1. **Multi-rep reruns sharpen the recognition inversion.** Re-running the 180-cell region
   grid at 3 reps (Haiku) / ≥2 reps (Sonnet) shrank SE ~1.8× and exposed a consistent
   1-rep bias: the single-rep estimates **understated C/solve/E and overstated R** for both
   models. **R: Haiku 0.857 ≫ Sonnet 0.583 > Opus 0.36** — Sonnet's drop *widened* the gap.
2. **Natural-build E.** Redefined the exploitation factor as **E_natural = P(solved |
   spontaneously built)**, computed from the region sweep itself, making `Z = C·R·E`
   fully self-consistent from one run (no forced-build fork needed).
3. **Build-path Z = C·R·E: Sonnet 0.289 > Haiku 0.239 > Opus 0.178** — Sonnet leads by
   being balanced; Opus is bottlenecked by its recognition collapse.
4. **Open:** Opus is still 1-rep; the matched cross-model set needs Opus at 2–3 reps.

---

## 1. Natural-build E

Previously the exploitation factor came from a *forced-build* efficiency sweep (inject the
correct combine for every episode → measures pure exploitation, free of recognition
selection; headline E Haiku 0.52 / Sonnet 0.76 / Opus 0.68). We added a second definition
computed purely from the region sweep — **E_natural = P(solved | spontaneously built)** —
conditioning on the episodes that built the machine on their own.

It reads **lower** than the forced-build E (it charges the build's budget cost to the solve
attempt) and is confounded with recognition (built episodes are self-selected). But it is
**fully self-consistent** with C and R from the same run:

```
C · R · E_natural = P(held both) · P(built | held both) · P(solved | built)
                  = P(solved & built)          (the build-path solve probability, exactly)
```

A single-model back-analysis on the original 1-rep Haiku run gave E_natural = 0.43 ± 0.055
(35/81 built-and-solved), vs the forced-build 0.52 — the gap is the build's budget tax plus
selection. New panels use E_natural; the forced-build fork is on hold (runnable later if a
de-confounded exploitation estimate is wanted).

Scripts: `scripts/plot_efficiency_natural_panels.py` (3-model) and
`scripts/plot_efficiency_natural_haiku.py` (single-model); `plot_solve_decomp_panels.py`
now takes only the 3 region dirs and computes E the natural way.

## 2. Multi-rep reruns

The headline region sweeps were **1 rep/cell**. Re-ran the full 180-cell grid
(T∈[2,10] × N≤20, grind-calibrated budget, run-to-solve):

- **Haiku at 3 reps** — `runs/haiku_region_sweep_p180_r3_Nle20_20260618_172222/`, 540 eps,
  0 errored, ~$47.
- **Sonnet at ≥2 reps** — `runs/sonnet_region_sweep_p180_r3_Nle20_20260618_180413/`. Started
  as 3-rep, stopped at the 2nd rep (cost/time), resumed with `--reps 2` → **uniform ≥2
  reps/cell** (mean 2.52, 453 eps, ~$30). The runner is *point-major* (each cell runs all
  its reps before the next cell), so capping reps uniformly is done by killing mid-run and
  resuming with a lower `--reps`, **not** by stopping at an episode count.
- **Opus** is still 1 rep (`runs/opus_region_sweep_p200_r1_Nle20_20260615_220600/`).

All panel loaders were fixed to **average over reps per cell** (they were silently
last-rep-wins, which would have wasted the extra reps); validated they reproduce the
published 1-rep headlines exactly. All panel scripts gained `--outdir`. New figures live in
**`figs/toolworld/3rep_naturalE/`**; the old `figs/toolworld/` 1-rep figures are kept.

## 3. Results

**Pooled headlines, multi-rep vs old 1-rep** (binomial SE; ✶ = moved > 1 old SE):

| factor | Haiku (3-rep) | old 1-rep | Sonnet (≥2-rep) | old 1-rep |
|---|---|---|---|---|
| **R** = P(built \| held both) | 0.857 ± 0.019 | 0.87 ± 0.035 | **0.583 ± 0.028 ✶** | 0.64 ± 0.045 |
| **C** = P(held both) | 0.620 ± 0.021 ✶ | 0.52 ± 0.037 | 0.682 ± 0.022 | 0.63 ± 0.036 |
| **solve** = P(solved) | 0.257 ± 0.019 | 0.21 ± 0.030 | 0.395 ± 0.023 ✶ | 0.33 ± 0.035 |
| **E_nat** = P(solved \| built) | 0.449 ± 0.029 | 0.43 ± 0.055 | 0.728 ± 0.033 ✶ | 0.65 ± 0.056 |

**Consistent 1-rep bias:** the single-rep estimates understated C/solve/E and overstated R
for *both* models. SE shrank ~1.7–1.9× (≈√reps), as expected.

**The recognition inversion sharpens, not weakens.** With the better data:

- **R (recognition): Haiku 0.857 ≫ Sonnet 0.583 > Opus 0.36** — Sonnet's drop from the
  (overestimated) 1-rep 0.64 *widened* the Haiku–Sonnet gap.
- **C (gathering): Sonnet 0.68 ≈ Opus 0.68 > Haiku 0.62** — orders with capability.
- **E_natural (exploitation): Sonnet 0.73 ≈ Opus 0.73 > Haiku 0.45** — the mirror of R.
- **Build-path Z = C·R·E: Sonnet 0.289 > Haiku 0.239 > Opus 0.178** — self-consistent
  (Sonnet 0.68·0.583·0.728 = 0.289 = 131 built-and-solved / 453). Sonnet wins by being
  *balanced* (no weak stage); Opus is last, bottlenecked by its recognition collapse despite
  the best gathering and exploitation.

## Caveats

- **Opus is still 1-rep** (R=0.36, SE ~0.05); Haiku/Sonnet are multi-rep. The R inversion is
  far too wide to be threatened by this, but since multi-rep *lowered* Sonnet's R by ~0.06,
  Opus's point could shift too — the C and Z orderings among the lower-precision legs are
  within noise until Opus is re-run. **Next:** Opus at 2–3 reps to finish the matched set.
- E here is **natural-build** (confounded with recognition; charges the build's budget). The
  de-confounded forced-build E remains available via the efficiency fork (on hold).
- 540-episode Haiku run flagged **1 builder** hitting the no-progress safeguard (counts as
  built; negligible, but raise `NO_PROGRESS_WINDOW` if re-running).

## Artifacts

| kind | path |
|---|---|
| Haiku 3-rep region | `runs/haiku_region_sweep_p180_r3_Nle20_20260618_172222/` (540 eps, ~$47) |
| Sonnet ≥2-rep region | `runs/sonnet_region_sweep_p180_r3_Nle20_20260618_180413/` (started 3-rep, resumed `--reps 2`; 453 eps, ~$30) |
| Opus 1-rep region (unchanged) | `runs/opus_region_sweep_p200_r1_Nle20_20260615_220600/` |
| natural-E scripts | `scripts/plot_efficiency_natural_panels.py`, `scripts/plot_efficiency_natural_haiku.py` |
| decomp (region-dirs-only, natural E) | `scripts/plot_solve_decomp_panels.py` |
| loader reps-averaging fix | `scripts/plot_{recognition,held_both,solve,efficiency_solve}_panels.py` (+ `--outdir`) |
| multi-rep panels | `figs/toolworld/3rep_naturalE/fig_{recognition,industry,solve,efficiency_natural,solve_decomp}_panels_Nle20.*` |
| old 1-rep panels (kept) | `figs/toolworld/fig_*_panels_Nle20.*` |

---

# Open-source extension: qwen2.5 ladder (1.5B / 3B / 7B), both worlds (6-19)

Separate thread: extending the capability axis **below Haiku** with an open-weight
**Qwen2.5 ladder** (1.5B, 3B, 7B), served locally via Ollama on a rented **Lambda A10**
(24 GB), driven over the existing `ollama` provider in `lomekwi/raw_chat.py`. Region sweeps
match the modern configs, so the three rungs drop straight into the cross-model panels to the
left of Haiku (size increases rightward). The full box run (3 rungs × both worlds, plus Opus
WoodWorld) finished overnight; everything is pulled and panels rebuilt.

## TL;DR

1. **The qwen ladder is the "weak-model floor," and it's monotone within-family** — across
   1.5B→3B→7B every metric rises with size, sitting at or below the Claude tiers.
2. **ToolWorld: the whole qwen ladder solves ≈0** (1.5B 0.00, 3B 0.01, 7B 0.00) vs Haiku 0.21 /
   Sonnet 0.33 / Opus 0.29. Recognition climbs with size (0.00→0.04→0.35) but never converts —
   **E_natural = 0** for all three (0/4 and 0/34 built-and-solved). The sub-Haiku regime can't
   exploit a built machine at N≤20.
3. **ToolWorld recognition is non-monotone over the full axis:** rises across qwen
   (0.00→0.04→0.35), **peaks at Haiku 0.87**, then *falls* across Claude tiers (Sonnet 0.64 →
   Opus 0.36) — the known recognition inversion, now with the rising left tail the qwen rungs add.
4. **WoodWorld: solve rises monotonically with size** (0.00→0.09→0.30→0.77→0.80→0.84) — the grind
   escape-hatch keeps it >0 where ToolWorld is flat-0. Recognition broadly rises too (qwen-7B 0.51
   edges *above* Haiku's 0.31 dip; Sonnet 0.63 < Opus 0.79) — the mirror of ToolWorld's inversion.
5. **Infra:** panel scripts are model-count-agnostic and now lay **qwen on a top row, Claude on
   a bottom row** (2×3 here); the ladder is just more dirs in size order. WoodWorld gained a
   natural-efficiency panel (P(solved | built axe)) to match ToolWorld.

## Results

**ToolWorld** — region sweep, T∈[2,10] × N≤20, 1 rep, 180 eps each
(`runs/qwen2.5:{1.5b,3b,7b}_region_sweep_p180_r1_Nle20_*`). Claude legs are the **1-rep**
region runs (the only set with all three Claude models — Opus has no multi-rep run), so these
match `figs/toolworld/` 1-rep headlines, not `3rep_naturalE/`.

| factor | qwen-1.5B | qwen-3B | qwen-7B | Haiku | Sonnet | Opus |
|---|---|---|---|---|---|---|
| **R** = P(built \| held both) | 0.00 | 0.04 | 0.35 | 0.87 | 0.64 | 0.36 |
| **C** = P(held both) (industry) | 0.04 | 0.61 | 0.54 | 0.52 | 0.63 | 0.68 |
| **E_nat** = P(solved \| built) | — | 0.00 | 0.00 | 0.43 | 0.65 | 0.73 |
| **solve** = P(solved) | 0.00 | 0.01 | 0.00 | 0.21 | 0.33 | 0.29 |

(qwen-1.5B held both ingredients in only 8/180 cells → E_nat undefined, 0 builds.)

**WoodWorld** — `iso_recipe` / nohint / N10-90, 1 rep, 81 eps each; Opus `iso_recipe` was run
6-19 (`runs/opus_woodworld_region_iso_recipe_mult1.2_r1_nohint_N10-90_20260619_034109`, 81/81).

| factor | qwen-1.5B | qwen-3B | qwen-7B | Haiku | Sonnet | Opus |
|---|---|---|---|---|---|---|
| **R** = P(built \| held ingredients) | 0.02 | 0.23 | 0.51 | 0.31 | 0.63 | 0.79 |
| **C** = P(held ingredients) (curiosity) | 0.72 | 0.80 | 0.98 | 1.00 | 0.98 | 1.00 |
| **E** = P(solved \| built axe) (natural) | 0.00 | 0.07 | 0.35 | 0.87 | 0.96 | 0.98 |
| **solve** = P(solved) | 0.00 | 0.09 | 0.30 | 0.77 | 0.80 | 0.84 |

**WoodWorld R broadly rises with capability** (qwen 0.02→0.23→0.51, Haiku 0.31 dips, Sonnet
0.63 < Opus 0.79) — the mirror of ToolWorld's recognition *inversion*; solve rises
monotonically across the entire ladder. **Natural efficiency E = P(solved | built axe) also
rises monotonically** (0.00→0.07→0.35→0.87→0.96→0.98) — and unlike ToolWorld, where qwen's E
stayed 0 even at 7B, here qwen-7B exploits a built axe ~35% of the time (the grind-friendlier
world: gathering progress carries the build through).

## Caveats / open

- **Opus WoodWorld** completed **81/81** (cost-capped at $40 but reached full coverage); it ran
  far faster than estimated (~15–20 min at conc-12 via the API, not 1–2 h).
- ToolWorld qwen `solve` / `efficiency_natural` panels are **near/all-zero surfaces** — the real
  result, not a render bug (the whole sub-Haiku ladder fails to exploit at N≤20).
- Panels are at **1 rep/cell** (qwen and the matched Claude legs); no multi-rep precision pass.
- ToolWorld Claude legs use mixed sampling density (Haiku p100, Sonnet/Opus p200, qwen p180) —
  all pooled by the same KDE, fine for the surface but not identical coverage.

## Artifacts

| kind | path |
|---|---|
| qwen ToolWorld regions (1.5B/3B/7B) | `runs/qwen2.5:{1.5b,3b,7b}_region_sweep_p180_r1_Nle20_*/` (180 eps each) |
| qwen WoodWorld regions (1.5B/3B/7B) | `runs/qwen2.5:{1.5b,3b,7b}_woodworld_region_iso_recipe_mult1.2_r1_nohint_N10-90_*/` (81 eps each) |
| Opus WoodWorld region | `runs/opus_woodworld_region_iso_recipe_mult1.2_r1_nohint_N10-90_20260619_034109/` (81 eps) |
| ToolWorld panels (order: 1.5B→3B→7B→Haiku→Sonnet→Opus) | `figs/toolworld/qwen/fig_{recognition,efficiency_natural,industry,solve}_panels_Nle20.*` |
| WoodWorld panels (same order; 2 rows) | `figs/woodworld/qwen/fig_woodworld_{recognition,curiosity,efficiency,solve}_qwen2.5-1.5b_qwen2.5-3b_qwen2.5-7b_haiku_sonnet_opus_nohint_N10-90.*` |
| panel scripts made variadic + non-claude labels | `scripts/plot_{recognition,efficiency_natural,held_both,solve}_panels.py`, `scripts/plot_woodworld_panels.py` |
| local serving routing | `lomekwi/raw_chat.py` (`gemma`→google added; `ollama` path); `scripts/sweep_config.py` (`OSS_MODELS`) |
