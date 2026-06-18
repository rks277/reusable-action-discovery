# Toolworld figures, grouped by run

Figures are sorted into one subfolder per source run. **Note:** the plotting scripts
still write to the `figs/toolworld/` root (paths are hardcoded), so regenerated figures
land loose and need re-sorting into the matching folder below.

| Folder | Source run dir | Contents |
|---|---|---|
| `_analytic/` | *(none — from `sweep_config` cost model)* | budget-vs-N curve, build-vs-grind region map |
| `haiku_region_p30_r10_Nle40/` | `runs/haiku_region_sweep_p30_r10_20260615_161842/` | headline P(built\|solved), 300 eps (30 pts × 10 reps), N≤40 |
| `haiku_region_p100_r1_Nle20/` | `runs/haiku_region_sweep_p100_r1_Nle20_20260615_172029/` | Haiku density (built, ±gaussian) + P(built\|solved), full 180-cell N≤20 |
| `sonnet_region_p200_r1_Nle20/` | `runs/sonnet_region_sweep_p200_r1_Nle20_20260615_175252/` | Sonnet density (±gaussian1, gaussian1.5) + P(built\|solved), 180-cell |
| `opus_region_p200_r1_Nle20/` | `runs/opus_region_sweep_p200_r1_Nle20_20260615_220600/` | Opus density (±gaussian) + P(built\|solved), 180-cell |
| `haiku_efficiency_Nle20/` | `runs/haiku_efficiency_sweep_Nle20_20260616_160808/` | tool-exploitation efficiency: redundant continuation actions after a forced build — raw count + normalized by remaining budget; 93 held-both continuations, N≤20 |
| `sonnet_efficiency_Nle20/` | `runs/sonnet_efficiency_sweep_Nle20_20260616_162343/` | same efficiency map for Sonnet; 113 held-both continuations, N≤20 |
| `opus_efficiency_Nle20/` | `runs/opus_efficiency_sweep_Nle20_20260616_163925/` | same efficiency map for Opus; 123 held-both continuations (1 cell, T=7/N=19, killed mid-spiral and reconstructed as all-redundant), N≤20 |
| `_diffs/` | *(cross-run: two run dirs each)* | Haiku−Sonnet, Opus−Haiku, Opus−Sonnet build-propensity diffs |
| `_panels/` | *(cross-run: three squares)* | three-model panels: gathering/held-both (`fig_held_both_panels_Nle20`), recognition, region solve-rate; efficiency-waste (`fig_efficiency_panels_Nle20`); efficiency solve-rate-with-tool (`fig_efficiency_solve_panels_Nle20`); build-path decomposition `Z=E·R·C` (`fig_solve_decomp_panels_Nle20`), N≤20 |
| `_playground/` | `runs/{haiku,opus,sonnet}_playground_expt_T6_n12_*/` | playground build-demo recovery experiment, per model (n=12, T=6) |

Three-way headline (N≤20, full coverage): P(built\|solved) Haiku 0.95 > Sonnet 0.80 > Opus 0.60.
