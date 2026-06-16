# Figures, grouped by run

Figures are sorted into one subfolder per source run. **Note:** the plotting scripts
still write to `figs/` root (paths are hardcoded), so regenerated figures land loose and
need re-sorting into the matching folder below.

| Folder | Source run dir | Contents |
|---|---|---|
| `_analytic/` | *(none — from `sweep_config` cost model)* | budget-vs-N curve, build-vs-grind region map |
| `haiku_region_p30_r10_Nle40/` | `runs/haiku_region_sweep_p30_r10_20260615_161842/` | headline P(built\|solved), 300 eps (30 pts × 10 reps), N≤40 |
| `haiku_region_p100_r1_Nle20/` | `runs/haiku_region_sweep_p100_r1_Nle20_20260615_172029/` | Haiku density (built, ±gaussian) + P(built\|solved), full 180-cell N≤20 |
| `sonnet_region_p200_r1_Nle20/` | `runs/sonnet_region_sweep_p200_r1_Nle20_20260615_175252/` | Sonnet density (±gaussian1, gaussian1.5) + P(built\|solved), 180-cell |
| `opus_region_p200_r1_Nle20/` | `runs/opus_region_sweep_p200_r1_Nle20_20260615_220600/` | Opus density (±gaussian) + P(built\|solved), 180-cell |
| `_diffs/` | *(cross-run: two run dirs each)* | Haiku−Sonnet, Opus−Haiku, Opus−Sonnet build-propensity diffs |

Three-way headline (N≤20, full coverage): P(built\|solved) Haiku 0.95 > Sonnet 0.80 > Opus 0.60.
