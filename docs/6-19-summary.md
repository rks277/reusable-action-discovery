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

# Open-source extension: qwen2.5-7B (both worlds, 6-19)

Separate thread: extending the capability axis **below Haiku** with open-weight models.
First rung is **qwen2.5:7b**, served locally via Ollama on a rented **Lambda A10** (24 GB),
driven over the existing `ollama` provider in `lomekwi/raw_chat.py`. Region sweeps match the
modern configs so qwen drops straight into the cross-model panels.

## TL;DR

1. **qwen-7B is a clean "weak-model floor."** It gathers and even builds at mid rates, but
   converts that into solves far worse than any Claude tier — and the two worlds disagree on
   *how* it fails.
2. **ToolWorld: qwen solve = 0.00 and E_natural = 0.00.** It builds the machine ~34× (R=0.35,
   ~Opus-level recognition) but **never once turns a build into a solve**, and solves nothing
   overall at N≤20.
3. **WoodWorld: qwen solve = 0.30** (vs Haiku 0.77 / Sonnet 0.80) — much weaker, but *not*
   zero. It gathers ingredients fully (C≈0.98) and recognizes/builds at 0.51 (between Haiku
   and Sonnet). The grind escape-hatch is what keeps WoodWorld solve > 0 where ToolWorld is 0.
4. **Infra:** the panel scripts are now model-count-agnostic; adding more rungs is a re-run.

## Results

**ToolWorld** — region sweep, T∈[2,10] × N≤20, 1 rep, `runs/qwen2.5:7b_region_sweep_p180_r1_Nle20_20260618_231413` (180 eps).
Claude legs are the **1-rep** region runs (the only set with all three models — Opus has no
multi-rep run), so these match `figs/toolworld/` 1-rep headlines, not `3rep_naturalE/`.

| factor | **qwen-7B** | Haiku | Sonnet | Opus |
|---|---|---|---|---|
| **R** = P(built \| held both) | 0.35 | 0.87 | 0.64 | 0.36 |
| **C** = P(held both) (industry) | 0.54 | 0.52 | 0.63 | 0.68 |
| **E_nat** = P(solved \| built) | **0.00** | 0.43 | 0.65 | 0.73 |
| **solve** = P(solved) | **0.00** | 0.21 | 0.33 | 0.29 |

**WoodWorld** — `iso_recipe` / nohint / N10-90, 1 rep, `runs/qwen2.5:7b_woodworld_region_iso_recipe_mult1.2_r1_nohint_N10-90_20260618_235345` (81 eps). **3-model** (Haiku, Sonnet, qwen) — Opus was never run on `iso_recipe`.

| factor | **qwen-7B** | Haiku | Sonnet |
|---|---|---|---|
| **R** = P(built \| held ingredients) | 0.51 | 0.31 | 0.63 |
| **C** = P(held ingredients) (curiosity) | 0.98 | 1.00 | 0.98 |
| **solve** = P(solved) | 0.30 | 0.77 | 0.80 |

## Caveats / open

- **Only the 7B rung so far.** 3B and 1.5B are still running on the box (tmux `sweep`,
  7B→3B→1.5B, ToolWorld then WoodWorld each); they slot in as additional panels via the same
  commands once pulled.
- **WoodWorld has no Opus** on `iso_recipe` (an ~$25–35 / ~1–2 h Anthropic-API run if wanted).
- ToolWorld qwen `solve`/`efficiency_natural` panels are **all-zero surfaces** — the real
  result, not a render bug.
- Panels are at **1 rep/cell** (qwen and the matched Claude legs); no multi-rep precision pass.

## Artifacts

| kind | path |
|---|---|
| qwen ToolWorld region | `runs/qwen2.5:7b_region_sweep_p180_r1_Nle20_20260618_231413/` (180 eps) |
| qwen WoodWorld region | `runs/qwen2.5:7b_woodworld_region_iso_recipe_mult1.2_r1_nohint_N10-90_20260618_235345/` (81 eps) |
| ToolWorld qwen panels (order: qwen→Haiku→Sonnet→Opus) | `figs/toolworld/qwen/fig_{recognition,efficiency_natural,industry,solve}_panels_Nle20.*` |
| WoodWorld qwen panels (order: qwen→Haiku→Sonnet) | `figs/woodworld/qwen/fig_woodworld_{recognition,curiosity,solve}_*_nohint_N10-90.*` |
| panel scripts made variadic + non-claude labels | `scripts/plot_{recognition,efficiency_natural,held_both,solve}_panels.py`, `scripts/plot_woodworld_panels.py` |
| local serving routing | `lomekwi/raw_chat.py` (`gemma`→google added; `ollama` path); `scripts/sweep_config.py` (`OSS_MODELS`) |
