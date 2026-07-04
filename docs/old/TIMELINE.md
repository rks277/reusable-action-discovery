# Repository Timeline

Everything in this repo, grouped by creation date (git first-added date for tracked
files; filesystem birth time for untracked ones and run directories). This is a
**reading guide / manifest** — no files were moved, so all imports and paths are intact.

The project studies *reusable action discovery*: when an LLM agent in an obfuscated
toolworld chooses to **build** a reusable machine vs. **grind** (brute-force) the
locked doors, and how that choice scales (inversely) with model capability.

---

## 2026-06-01 — Initial toolworld engine + first budget sweep

The original environment, runners, validators, and the first budget-sweep data.

**Core engine & infra**
- `scripts/toolworld_v2.py` — the obfuscated toolworld environment (N doors, T byproduct types, build-the-machine vs. grind); the `run()` coroutine every sweep calls.
- `lomekwi/obfuscation.py` — label-obfuscation scheme (the "letter" scheme; replay is coupled to it).
- `lomekwi/raw_chat.py` — raw chat/LLM transport helper.
- `lomekwi/__init__.py` — package init.

**Runners (first generation)**
- `scripts/run_toolworld.py` — single-episode runner.
- `scripts/run_toolworld_llm.py` — LLM-driven episode runner.
- `scripts/run_toolworld_sweep.py` — first sweep harness.
- `scripts/run_budget_sweep.py` — budget sweep (T3, n8).
- `scripts/run_budget_sweep_newmodels.py` — budget sweep re-run on newer models.

**Analysis / validation**
- `scripts/analyze_toolworld.py` — episode analysis.
- `scripts/analyze_budget.py` — budget-sweep analysis; source of the canonical `built` predicate.
- `scripts/replay_toolworld.py` — replay episodes (requires `DEFAULT_SCHEME == "letter"`).
- `scripts/validate_toolworld.py`, `scripts/validate_toolworld_v2.py` — environment self-checks.
- `scripts/_smoke_new_models.py` — smoke test for new model IDs.

**Data**
- `runs/budget_sweep_T3_n8_b36_20260531_185546/` — first budget-sweep run (episodes + headline/outcome figs).
- `runs/budget_sweep_newmodels_T3_n8_b36_20260601_133952/` — new-models budget sweep.

**Repo meta**
- `README.md`, `.gitignore`, `.env.example`, `requirements.txt`, `abstract.pdf`
  *(README/abstract content dates to 06-01; files touched again 06-12 — see below).*

---

## 2026-06-11 — (n, T) sweeps + shared sweep config

Generalized the sweeps to vary the number of doors and byproduct types.

- `scripts/sweep_config.py` — central budget/cost model: `_grind_cost`, `_build_cost`, grind-calibrated `budget_for(n)`, `OBFUSCATION_SCHEME="letter"`, concurrency, hints. Imported by every later sweep.
- `scripts/run_nt_sweep.py` — sweep over (n, T).
- `scripts/run_budget_arr_sweep.py` — array/grid budget sweep.
- `scripts/run_sonnet_opus_budget_sweep.py` — Sonnet+Opus budget sweep.
- `scripts/analyze_nt_sweep.py` — (n, T) sweep analysis.
- `scripts/analyze_episode_stats.py` — per-episode statistics.

---

## 2026-06-12 — Array-sweep analysis, model-output inspection, paper scaffolding

- `scripts/analyze_budget_arr_sweep.py` — analysis for the array budget sweep.
- `scripts/print_model_outputs.py` — dump raw model outputs for inspection.
- `runs/budget_sweep_combined_T3_n8_b36/` — combined budget-sweep dataset (episodes + figs in png/pdf/svg).
- `README.md`, `abstract.pdf` — paper scaffolding updated.

---

## 2026-06-13 — Build sweeps + budget-curve plots + MDP baseline

Shifted focus to *build propensity* and an analytic MDP reference.

- `scripts/run_sonnet_build_sweep.py` — Sonnet build sweep (T3, n12; swept budget, stop-on-build).
- `scripts/plot_build_sweep.py` — build-sweep plotting.
- `scripts/plot_build_attempts_by_budget.py` — build attempts vs. budget.
- `scripts/plot_budget_curves.py` — budget-curve figures.
- `runs/sonnet_build_sweep_T3_n12_20260613_003611/` — Sonnet build-sweep data.
- `runs/opus_build_sweep_T3_n12_20260613_212256/` — Opus build-sweep data.
- `runs/mdp_toolworld_D12_T3/` — MDP toolworld baseline data.

---

## 2026-06-14 — Haiku/Opus build sweeps, MDP solver, cross-model compare, first writeup

- `scripts/mdp_toolworld.py` — analytic MDP model of the toolworld.
- `scripts/plot_mdp_trajectories.py` — MDP trajectory plots.
- `scripts/run_haiku_build_sweep.py` — Haiku build sweep.
- `scripts/run_opus_build_sweep.py` — Opus build sweep.
- `scripts/plot_build_sweep_compare.py` — Haiku/Sonnet/Opus comparison.
- `docs/inverse-scaling-agentic-confabulation.md` — first writeup of the inverse-scaling / confabulation finding.
- `runs/haiku_build_sweep_T3_n12_20260614_190439/` — Haiku build-sweep data.

---

## 2026-06-15 — Region maps (T×N), commitment probe, recognition-latency, diffs, writeups

The most recent and largest layer: 2-D build/grind region geometry, the
density/heatmap maps over (T, N), a commitment probe, and supporting writeups.

**Budget geometry & region maps**
- `scripts/plot_budget_vs_n.py` — grind-calibrated budget B vs. N with build-cost overlays.
- `scripts/plot_build_vs_grind_region.py` — green/grey build-vs-grind region map; boundary N_b(T).
- `scripts/run_haiku_region_sweep.py` — model-generic region sweep over (T, N) (`--model`, `--n-points`, `--reps`, `--n-hi`); produces the Haiku/Sonnet region datasets.
- `scripts/plot_haiku_region_heatmap.py` — region heatmaps in 3 metrics (built|solved, built, built&solved) with griddata / gaussian / rbf smoothing.
- `scripts/plot_region_diff.py` — Haiku−Sonnet build-propensity diff map.

**Commitment probe** *(untracked — not yet committed)*
- `scripts/run_commitment_probe.py`, `scripts/analyze_commitment_probe.py`, `scripts/plot_commitment_probe.py`.

**Other analysis**
- `scripts/analyze_paired_build_losses.py` — paired build-loss analysis.
- `scripts/analyze_recognition_latency.py` — recognition-latency decomposition.

**Datasets (runs/)**
- `commitment_probe_T3_n12_20260615_121244/`
- `haiku_region_sweep_smoke_20260615_161712/`
- `haiku_region_sweep_p30_r10_20260615_161842/` — 300-ep P(built|solved) heatmap.
- `haiku_region_sweep_p100_r1_Nle20_20260615_172029/` — Haiku density map (now full 180-cell coverage).
- `sonnet_region_sweep_smoke_20260615_175231/`
- `sonnet_region_sweep_p200_r1_Nle20_20260615_175252/` — Sonnet full-coverage density map.

**Figures (figs/toolworld/)** — all generated 06-15
- `fig_budget_vs_n`, `fig_build_vs_grind_region` — budget geometry.
- `fig_haiku_region_build_rate`(`_Nle20`), `fig_sonnet_region_build_rate_Nle20` — P(built|solved) heatmaps.
- `fig_haiku_density_built_Nle20`(`_gaussian1`), `fig_sonnet_density_built_Nle20`(`_gaussian1`, `_gaussian1.5`) — build-propensity density maps.
- `fig_haiku_minus_sonnet_density_Nle20` — capability-inverse diff map.
- `runs/fig_build_compare_haiku_sonnet_opus`, `runs/fig_recognition_latency`, `runs/fig_recognition_decomposition` — cross-model / latency figures.

**Writeups**
- `docs/why-opus-builds-less-experiments.md` — why Opus builds less (experiment notes).
- `docs/6-15-summary.md` — running summary as of 06-15.
- `docs/TIMELINE.md` — this file.
