# Why does Opus build the machine less often than smaller models?

*Experiment menu for diagnosing the inverse-scaling-in-discovery result (Haiku 0.97 > Sonnet 0.81 > Opus 0.73 build proportion). The goal is to identify the **cause** before scaling to more models/environments. Most of these reuse the existing run transcripts — every episode stores `actions`, `agent_texts`, `obs`, `build_turn`, `stopped_reason`, `usage`, and `labels` (which records the exact winning `recipe`), and worlds are paired by seed across models.*

## Candidate causes (the targets each experiment discriminates)

1. **Confabulation / budget pathology** — derails into fabricated state, stops reading the env, never executes the real build (the documented `opus-confabulates-mechanics-not-lore` mode).
2. **Over-exploration / overthinking** — deliberates, hedges, explores more options, never commits to the simple combine.
3. **Discovery is fine, execution isn't** — finds the recipe but doesn't build (note `built_machine` ≠ `solved`; it may chase the full solve and skip building).
4. **Over-literal rule inference** — invents constraints ("keys expire") that make it abandon a valid build path.
5. **Refusals / malformed actions** — wastes effective actions on unparsed/refused turns.

## Tier 1 — existing transcripts, no new runs

- **(A) Paired-seed trajectory diff *(highest value; started — `scripts/analyze_paired_build_losses.py`)*.** On every (budget, seed) where a smaller model built and Opus didn't — *same world, perfectly controlled* — diff the trajectories. Because `labels.recipe` records the exact winning type pair and holdings are monotonic, we can rigorously decompose each Opus loss into: **never gathered both ingredients** (gathering failure), **gathered both but never tried the winning combine** (discovery/commitment gap — the interesting case), or **tried the winning pair without holding both** (sequencing/execution slip). Plus exploration breadth (distinct combine pairs tried) and `stopped_reason`.
- **(B) Discovery-vs-execution decomposition.** Generalize (A)'s decomposition across *all* Opus non-builders (not just paired losses). A high "gathered-not-tried" rate means it's not a discovery deficit at all.
- **(C) Population-scale confab detection.** Apply the agent_text-vs-obs diff method across all Opus non-builders: what fraction derail into confabulation, and does confab onset *precede* the missed build? Turns the n=1 worst case into a population statistic.
- **(D) Action-economy comparison.** Distinct actions, redundant re-examines, combine attempts, `noop_total`, `refusals`, `unparsed`, and wasted-action fraction (actions after confab onset). Compares the Haiku/Sonnet/Opus distributions; separates "burned budget confabulating" (1) from "exploring" (2) from "malformed actions" (5).
- **(E) Time-to-first-combine.** Does Opus *delay* committing to combine actions vs the smaller models? Tests over-exploration (2).

## Tier 2 — small targeted probes (a handful of episodes, not sweeps)

- **(F) Oracle-recipe ablation.** Re-run failed Opus seeds with the recipe given explicitly. Builds now → exploration/commitment problem, not discovery. Still fails → execution or confab.
- **(G) Teacher-forced build.** Replay a failed Opus transcript to the point it held both ingredients, then force the build action. Confirms mechanically whether the build was available and simply not taken (cause 3).
- **(H) Confab-suppression on the same failed seeds — MOOT given (A)/(E).** *Originally:* add a grounding line ("trust only the literal OBS; never assume an action succeeded") and re-run the failed seeds; a build-rate jump would show confab is causal. *Superseded:* (A)/(E) found **0 confabulation** in the build sweep (safeguard never fired), so there is no confab to suppress here — this experiment no longer addresses the gap. Kept for the record; not the commitment probe (see (J)).
- **(I) Effort/thinking knob.** Lower Opus effort or disable thinking on failed seeds. Build-rate rise → over-exploration/overthinking (2) implicated.
- **(J) Commitment probe (the live next experiment).** A one-time, mid-episode nudge fired when Opus holds the ingredients but hasn't built — pointing it at the *combine mechanic it isn't using*, **without naming the recipe pair** (that distinction is what keeps it separate from (F) oracle-recipe and from (H) anti-confab grounding). E.g. "You hold several distinct components and have actions left; combining the right pair can build a tool — have you tried `combine` on what you hold?" Three conditions on the gathered-not-tried losses: baseline-rerun (variance control) · commit-nudge · neutral-nudge ("plenty of actions left, keep going", same trigger — rules out "any interruption helps"). Metric: recognition rate `P(built | held both)` per condition. commit-nudge ≫ baseline≈neutral → the recognition gap is a **closable commitment failure**; commit-nudge ≈ baseline → deeper than a prompt.

## Recommendation / order of attack

Start with **A, B, C** on existing data (free). *Prior (written before the findings):* B+C will likely show Opus derailed into confabulation rather than failing to discover. **This prediction was wrong** — see FINDINGS: there is no confabulation in the build sweep; the gap is a *recognition* failure (holds the ingredients, never combines them). The decisive follow-up is therefore **(J) the commitment probe**, not the originally-planned (H) grounding: if a nudge toward the combine mechanic recovers Opus, the recognition gap is a closable commitment failure rather than inability.

---

## FINDINGS

### (A) Paired-seed build losses — done (`scripts/analyze_paired_build_losses.py`)

49 Opus non-builders, *all* on worlds a smaller model built (every loss is winnable). Head-to-head: **Haiku beat Opus 48–4, Sonnet beat Opus 34–19.** Decomposing each loss against its own recorded recipe:

- **82% (40/49): Opus held BOTH winning ingredients and never issued the winning combine.**
- 18% (9/49): never gathered both ingredients.
- 0%: sequencing slips. **0%: confabulation derail — the `no_progress` safeguard never fired.** (Confabulation is the earlier *non-*`stop_on_build` budget sweep's runaway; it does not drive the build-sweep gap.)

The 82% splits by budget into two regimes: **b≤40 (17 cases): genuine `out_of_budget` deficit** (Opus gathers/commits too slowly; smaller models built in ~15 actions on the identical world); **b≥40 (32 cases): `solved`** — Opus opened all 12 doors by brute force *without building*, every time while holding the recipe. The latter is a metric artifact of `stop_on_build` (it conflates "couldn't build" with "didn't need to").

### (E) Recognition vs gathering — done (`scripts/analyze_recognition_latency.py`)

Build rate factorizes exactly: **build = gathering × recognition.** Gathering is flat; recognition carries the whole inverse-scaling signal.

| model | gathering P(held both) | **recognition P(built \| held both)** | build |
|---|---|---|---|
| Haiku | 0.99 | **0.98** | 0.97 |
| Sonnet | 1.00 | **0.81** | 0.81 |
| Opus | 0.95 | **0.77** | 0.73 |

Holding the *exact* winning components, Haiku commits 98% of the time, Opus only 77% — monotone-inverse in capability. **Under the project's definition (discovery = recognizing the tool can be made), this recognition gap *is* the discovery failure**: Opus doesn't recognize/commit even while holding the ingredients. Recognition *latency* among recognizers is only modestly worse for Opus (mean 16 vs 13 actions; figs show Opus's ECDF shifted right), so the gap is *whether* it commits, not *how fast*. Recognition rate also does **not** improve with budget for Opus (sags to ~0.60 at b=2560) while Haiku saturates at 1.00.

Figures: `runs/fig_recognition_decomposition.{png,pdf}` (decomposition bars + recognition-rate vs budget), `runs/fig_recognition_latency.{png,pdf}` (recognition-latency + first-combine ECDFs).

**Revised headline:** the inverse-scaling is real and lives in **recognition**, not gathering or confabulation — the larger model, holding the winning components, systematically fails to recognize/commit to the build. **Metric caveat:** report recognition rate P(built | held both), or build rate at b≤40, to avoid the solve-without-build confound.

### (J) Commitment probe — done (`scripts/run_commitment_probe.py`, `scripts/analyze_commitment_probe.py`)

Re-ran the 40 b≤640 paired-loss seeds under three conditions (run `runs/commitment_probe_T3_n12_20260615_121244`, 120 episodes, $13.13, 0 errors). Recognition rate `P(built | held both)`:

| condition | recognition | build rate |
|---|---|---|
| baseline-rerun | **0.72** | 0.57 |
| commit-nudge | **0.91** | 0.75 |
| neutral-nudge | **0.52** | 0.42 |

commit − baseline = **+0.19**; commit − neutral = **+0.39**. Per-budget, commit ≥ baseline at every budget and neutral ≤ baseline at every budget.

**Conclusions:**
1. **The recognition gap is largely closable.** A one-line commitment nudge (no recipe leak) lifts Opus from 0.72 to 0.91 — near Haiku's 0.98. So the deficit is capability-induced **under-recognition, not inability**.
2. **Original "failures" were mostly stochastic per-attempt misses.** baseline-rerun (0.72) ≈ the original Opus base rate (0.77), so re-running the failed cells recovers them at the base rate — the baseline-rerun control was essential to see this.
3. **Recognition is elicitation-sensitive in *both* directions.** The neutral nudge *hurts* (0.52 < 0.72) — it is emphatically **not** "any interruption helps." The commit effect is content-specific (the +0.39 commit−neutral gap is the cleanest causal estimate).

**Implication for the project:** "Opus is worse at discovery" must be qualified to "worse at *spontaneously* recognizing/committing to the build without a nudge" — and that gap is prompt-elicitable, a caveat for any cross-family / generality claims.

![Commitment probe](../runs/commitment_probe_T3_n12_20260615_121244/fig_commitment_probe.png)

Figure (`scripts/plot_commitment_probe.py` → `runs/commitment_probe_T3_n12_20260615_121244/fig_commitment_probe.{png,pdf}`): recognition + build rate per condition with Wilson 95% CIs (left), recognition vs. budget (right).

### Next

**(F) oracle-recipe** (separate recognizing-*that*-it-can-build from recognizing-*which*-pair) and the **cross-family arm** (GPT-5 / Gemini — does the recognition gap reproduce off the Claude family?). **(H) confab-suppression remains moot** — (A)/(E) found no confabulation to suppress.
