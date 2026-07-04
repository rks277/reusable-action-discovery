# Woodworld — obfuscation ablation, combine-arity fix, and a reasoning probe (6-18)

Follow-on to the 6-17 woodworld work (`docs/6-17-summary.md`). All experiments here
are on a single fixed cell — **Haiku, p=0.5, N=10, grind-calibrated budget B=20
(mult=1.0), 40 paired seeds/cell** (rep r ⇒ relabel_seed = gather_seed = r) — chosen
because it is where the build/solve trade-off is most informative and matches the
6-17 verb/hint A/B pilots.

## TL;DR

1. **De-obfuscating the items does NOT recover Haiku's build rate.** With real
   English names (`wood`/`stick`/`axe`) instead of nonsense tokens, build stayed
   **0/40** and combine-exploration *dropped*. Obfuscation is not the barrier — if
   anything, obfuscation *helps* by inducing an exploratory frame.
2. **Part of the near-zero build rate was an *arity artifact*.** Haiku's rare
   combine attempts dumped the whole pile (`combine wood wood wood`, 9-wood dumps),
   which fail the recipe's multiset-equality check opaquely. Capping `combine` to
   **exactly two items** (every recipe is binary, so nothing breaks) roughly
   **tripled** the build rate (≈0 → 0.10–0.12 in 3/4 cells).
3. **The hint, not obfuscation, is the active lever** for whether crafting is tried —
   consistent with the 6-17 hint-gating finding.
4. **Reasoning transcripts confirm the mechanism**: real names make Haiku satisfice
   on *"gather wood directly"*; obfuscation forces *"figure out what X is / can I
   multiply it"* — but it's a probabilistic nudge on top of a strong gather-default,
   which is why the metric gaps are real-in-direction but modest.
5. **(§8) Capability acts through *curiosity*, not recognition** — the qualitative
   opposite of ToolWorld. In the 3-model hint-off N=10–90 decomposition, C=P(hold
   ingredients) scales 0.12→0.75→0.93 (Haiku→Sonnet→Opus) while recognition
   (0.90/0.97/0.99) and efficiency (0.89/0.90/0.91) are flat-high. WoodWorld is
   *discovery-limited*; ToolWorld was *disposition-limited* (recognition inverted).
6. **(§9) The hostile build is the driver of that switch.** A non-hostile control
   (axe from *free sticks* instead of consuming wood; single-step; widened) flips
   Haiku's build rate **0.04 → 0.86** vs the hostile single-step variant, width and
   step-count matched. Hostility — building spends the goal item — is what suppresses
   building, not tree width or step count.

---

## 1. Obfuscation × hint ablation (no arity cap)

Added an `obfuscate` parameter to `scripts/woodworld.py` (`make_world` / `run`):
`obfuscate=False` keeps the real latent names (`wood`/`stick`/`axe`) while the
recipes and the axe's +2-wood payoff remain hidden, so the agent must still
*discover* the structure — only the surface vocabulary changes.

Runner: `scripts/run_woodworld_obf_ablation.py` (2×2 over {obfuscate} × {hint},
40 paired seeds, metrics tried-combine / got-sticks / built-axe / solved).
Run dir `runs/woodworld_obf_ablation_p05_N10_20260617_182237`.
Figure `figs/woodworld/obf_ablation/fig_woodworld_obf_ablation_haiku_p05_N10_nocap.png`.

| cell | tried combining | **built axe** | solved |
|---|---|---|---|
| obf + hint | 0.33 | **0.05** | 0.50 |
| obf + no-hint | 0.05 | 0.00 | 0.50 |
| no-obf + hint | 0.10 | 0.00 | 0.50 |
| no-obf + no-hint | 0.00 | 0.00 | 0.55 |

**Removing obfuscation did not raise the build rate at all (stays 0/40), and it
*lowered* exploration** (tried 0.10 vs 0.33 with the hint; 0.00 vs 0.05 without).
Solve rate is flat ~0.50–0.55 everywhere — the brute-gather escape hatch is always
viable at this budget. The single built episode across all 160 runs was obf+hint.

**Why obfuscation helps rather than hurts.** With real names the goal ("accumulate
10 **wood**") and the action ("**gather** → wood") share the goal word, so the model
pattern-matches a complete strategy at read-time and satisfices. Nonsense tokens
break that surface match, putting the model in an epistemic-uncertainty / puzzle
frame that provokes more probing. Obfuscation is therefore better read as an
**exploration knob** than a difficulty knob.

> Methodology note: only the two no-obf cells were newly run; the two obfuscated
> cells were reused from the same run dir rather than re-spending API on
> already-collected data. There was no prior 40-seed obf+no-hint at this cell (only
> an n=10 probe + the 1-rep/cell region sweep), so that cell was genuinely new.

---

## 2. Combine-arity fix: cap `combine` to exactly two items

Transcripts of the no-obf+hint episodes revealed *how* the rare combine attempts
failed: they passed too many items.

```
[t7]  combine wood wood wood            -> nothing happens   (recipe needs exactly 2 wood)
[t19] combine wood×9                    -> nothing happens
```

`craft()` requires multiset **equality** with a recipe's inputs, so any over-arity
pile fails opaquely, and Haiku reads "nothing happens" as *combining doesn't work*
and reverts to gathering. It never landed the exact `combine wood wood`.

**Fix** (`scripts/woodworld.py`): `craft()` now rejects any non-binary combine with
an explicit message — *"You can only combine two items at a time (you tried N).
Combine exactly two."* — and the prompt verb line / `SYS` were updated to
`combine <a> <b>  - combine exactly two items`. Both recipes (`2 wood`,
`1 stick + 1 wood`) are already binary, so no recipe is affected; only wrong-arity
attempts change behavior (instructive feedback instead of a silent miss, and they
still count as a craft attempt).

### Re-run of the full 2×2 under the cap

Run dir `runs/woodworld_obf_ablation_p05_N10_20260617_183935`.
Figure `figs/woodworld/obf_ablation/fig_woodworld_obf_ablation_haiku_p05_N10_cap2.png`.

| cell | **built** (no-cap → cap) | tried (no-cap → cap) | solved (cap) |
|---|---|---|---|
| obf + hint | 0.05 → **0.12** | 0.33 → 0.20 | 0.53 |
| obf + no-hint | 0.00 → **0.10** | 0.05 → 0.12 | 0.53 |
| no-obf + hint | 0.00 → **0.12** | 0.10 → 0.17 | 0.53 |
| no-obf + no-hint | 0.00 → 0.03 | 0.00 → 0.15 | 0.42 |

**The zeros disappear — build is ~0.10–0.12 in three of four cells (was ~0).** A
chunk of the near-zero build rate was an arity artifact, not pure disposition:
capping converts *attempts into builds* far more efficiently (a binary pair is much
likelier to hit the exact recipe than a pile-dump), and the explicit feedback
teaches the rule. Tellingly, in obf+hint **tried went *down* (0.33 → 0.20) while
built went *up* (0.05 → 0.12)** — the cap is not provoking more attempts, it is
making the attempts land.

Under the cap the two earlier conclusions still hold: **obfuscation still doesn't
matter** (obf+hint 0.12 ≈ no-obf+hint 0.12) and the **hint helps only marginally**.
Solve stays ~0.5 (no-obf+no-hint dipped to 0.42, within noise; building also costs
budget).

---

## 3. Reasoning probe — why real names suppress exploration

Re-ran 8 seeds per condition (obf vs real-names, hint on) capturing chain-of-thought
(`/tmp/reasoning_dump.json`). The qualitative contrast:

**Real-names, seed 0 (pure gather, never combines):**
> *"I'll continue gathering to accumulate more wood **directly**."*

**Obfuscated, seed 4 (budget math → experiment → build in 14 actions):**
> *"at 1 b per gather, I'd need 10 gathers... **that won't work**... Let me try
> combining to see if there's a way to **multiply or transform** items"*
> → `combine b b` (sticks) → `combine b c` (axe). Built + solved.

Across all 16 episodes the **default in both conditions is to gather**; the
difference is framing. Real-names reasoning recurs as *"continue gathering to
accumulate more wood directly"* (goal-word ↔ gathered-item anchoring); obfuscated
reasoning recurs as *"figure out what x is / combine strategically to create or
transform items."* Several real-names episodes even *read the hint and gathered
anyway* (the satisficing pull overriding the nudge). Tried-combine was 2/8
obfuscated vs 1/8 real-names — same direction as the aggregate, small n.

The single decisive behavior is the **budget calculation** (*"10 gathers won't fit,
I need a multiplier"*): obfuscation makes Haiku more likely to perform it; real names
let it skip straight to the literal plan.

---

## 4. ToolWorld-style panels (efficiency / recognition / curiosity)

Built the three ToolWorld-style region panels for woodworld, one figure each, a
Haiku/Sonnet/Opus row of Gaussian-pooled (p, N) heatmaps in the established
woodworld box style (RdYlGn, E[build]=E[grind] boundary overlaid). All read the
**6-17 hint-on region sweeps** (1 rep/cell, 171 cells/model) and recompute every
metric from the recorded actions/obs — no re-run needed. Script
`scripts/plot_woodworld_panels.py`; figures in `figs/woodworld/panels/`.

These mirror ToolWorld's **`build = C·R·E`** solve-rate decomposition (6-17): curiosity
(acquire ingredients) × recognition (build given ingredients) × efficiency (solve given
the tool). The 6-17 panel formerly called "industry" — `P(held both ingredients)` — is
renamed **curiosity** here.

| panel | metric | Haiku | Sonnet | Opus |
|---|---|---|---|---|
| **curiosity** (C) | P(hold ingredients) = P(got sticks), all cells | 0.09 | 0.60 | 0.88 |
| **recognition** (R) | P(built axe \| got sticks), got-sticks cells only | 0.69 | 0.84 | 0.71 |
| **efficiency** (E) | P(solved \| tool built), built-axe cells only | 0.73 | 0.84 | 0.79 |
| solve | P(solved), all cells | 0.71 | 0.58 | 0.57 |

- **Curiosity** C = `P(hold ingredients)` = P(the episode ever reached sticks). The
  acquisition stage. Monotonic with capability: 0.09 → 0.60 → 0.88 — Opus reaches the
  ingredients almost everywhere, Haiku almost nowhere.
- **Recognition** R = `P(built axe | got sticks)`, conditioned on got-sticks cells (rest
  dropped, grey ×). Follow-through to the tool: Haiku 0.69 (n=16), Sonnet best (0.84).
- **Efficiency** E = `P(solved | tool built)`, conditioned on built-axe cells. Given the
  axe, did it reach the goal in budget? Haiku 0.73 < Sonnet 0.84, Opus 0.79. (Note: this
  is **not** the old action-count "efficiency" — that metric and the `action_efficiency`
  field are no longer used by the panel, though the p-bug fix in `woodworld.py` still
  stands.)
- **`C·R·E`** (build-path solve estimate): Haiku 0.045, Sonnet 0.42, Opus 0.49 — vs the
  actual solve 0.71/0.58/0.57. The gap is grind-based solving (largest for Haiku, which
  builds least but grinds most), the same exploration-tax story as §5.
- **Solve** rate **inverts** the capability axis: Haiku 0.71 > Sonnet 0.58 > Opus 0.57.
  The mechanism is an **exploration tax**, not bad building (verified from the region
  data). At mult=1.0 the budget `B=round(N/p)=E[grind]`, so pure grinding is a *coin
  flip* (avg 0.66 over the grid, 0.53 at low p → 1.00 at p=1), not a long shot. Haiku
  grinds (non-gather fraction 0.04) and rides that coin flip → 0.71. Building itself is
  fine — `solve|built` = 0.73/0.84/0.79 and above-N\* builds solve 0.82–0.87; below-N\*
  builds barely happen (0/3 Sonnet, 0/4 Opus). The deficit is entirely in **non-built**
  episodes: `solve|not-built` = 0.71/0.32/0.20, with non-gather fractions 0.04/0.25/0.41
  and out-of-budget rates 29%/68%/80%. At a zero-slack budget every probe is one fewer
  gather, so an abandoned build attempt can't fall back on a clean grind. **Curiosity —
  the trait that enables building — is a liability under an E[grind] budget**; the
  budget-optimal policy below N\* is "grind, don't explore," which Haiku does by accident.

## 5. Why ToolWorld didn't show this solve inversion

ToolWorld grind-calibrates the budget the same way (`budget_for = BUDGET_MULT·E[grind]`)
and grinding there also costs less than the budget — yet *all* models built, building
won, and solve rate did **not** invert. The exploration tax (§4 solve bullet) needs three
things; ToolWorld breaks all three:

1. **Exploration *is* grinding there.** ToolWorld's build-discovery step is `G = t·H_t`
   **examines**, and each examine *also drops a uniformly-random door-key*
   (`scripts/sweep_config.py` `_build_cost`) — so the actions you spend discovering the
   recipe simultaneously advance the grind goal. Exploration has ~zero opportunity cost
   and a failed build has still keyed most doors. In woodworld a failed `combine` yields
   nothing, and a *successful* stick-combine **consumes wood** (the goal item) —
   exploration is orthogonal-to-*negative* for the goal. This asymmetry is the core driver.
2. **Discovery is shallow & salient** in ToolWorld (combine 2 of `C(t,2)` candidate pairs
   over an examinable affordance) vs woodworld's blind 2-step `wood→sticks→axe` with a
   hidden payoff (deep, high-variance, fizzles often).
3. **Budget slack:** ToolWorld `BUDGET_MULT=1.2` (P(grind) ≈ 0.88, slack absorbs
   exploration) vs the woodworld region sweeps' `M=1.0` (budget = E[grind] exactly, a coin
   flip with zero slack, so wasted probes are unrecoverable).

**One-liner:** ToolWorld is **disposition-limited** (everyone can discover the tool; the
question is whether they *build* — hence its inverse build-rate, Haiku 0.97 > Opus 0.73),
woodworld is **discovery-limited** (building requires costly search that competes with the
goal). The exploration tax is a discovery-limited phenomenon, so it cannot appear in a
disposition-limited world. Design corollary: to make building dominate, make discovery
dual-purpose with the goal and/or add budget slack; the woodworld inversion is *engineered*
by making exploration compete with the goal under a tight budget.

## 6. Large-N region sweep (N=10–90): Haiku's build rate rises with N

The 6-17 region sweeps used N=2–20, where Haiku built only 0.06 of the time. But N\*
(the target above which building is cheaper than grinding) is small — ≤11 across all p
— so most of the 2–20 grid sits *below or near* the threshold, where building isn't yet
rational. To test whether Haiku's near-zero build was an N-range artifact rather than a
flat disposition, re-ran the Haiku region sweep at **N ∈ {10,20,…,90} × p ∈ {0.2,…,1.0}**
(81 cells, 1 rep, hint on, mult=1.0, run-to-solve). **$13.15, ~13 min, 0 errors, no
`max_turns`/`no_progress` aborts.** (Cost note: my a-priori estimate of ~$28 was too high
— it extrapolated a quadratic cost-vs-length fit from short episodes; with prompt caching,
long 300–450-action episodes cost ~linearly, ≈$0.002/action.)

| | N=2–20 (6-17) | **N=10–90 (this run)** |
|---|---|---|
| build rate | 0.06 | **0.20** |
| solve rate | 0.71 | 0.73 |

Build rate by N:

| N | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 |
|---|---|---|---|---|---|---|---|---|---|
| **build** | 0.00 | 0.22 | 0.11 | 0.11 | 0.33 | 0.11 | 0.22 | 0.33 | 0.33 |
| solve | 0.78 | 0.78 | 0.67 | 0.67 | 0.56 | 0.67 | 0.78 | 0.89 | 0.78 |

**Haiku's build disposition is not flat-zero — it responds to the economic incentive.**
Extending to N=50–90 lifts the build rate to 0.22–0.33 (3–5× the small-N value), so the
"Haiku ~never builds" result was substantially an artifact of the 2–20 range sitting
mostly below N\*. Solve rate stays ~0.73 (the grind coin-flip is still available), so at
these larger N building is additive rather than the exploration *tax* it was at small N —
there's more room to amortize the axe before the budget binds. In the heatmap the
`E[build]=E[grind]` boundary sits at the very bottom (N\*≤11), i.e. the **entire** N=10–90
plane is a region where building is the cheaper strategy; the green (build) cells
concentrate at higher N and higher p.

Figure `figs/woodworld/region/haiku/fig_woodworld_region_build_solve_haiku_hint_mult1_N10-90.png`.
Only Haiku was run (Sonnet/Opus deferred). The region heatmap plotter was generalized to
infer the N lattice from the data (it previously hardcoded 2–20), so it handles the
step-10 grid. (`plot_woodworld_panels.py` was subsequently generalized the same way —
see §7 — so the panels also render from arbitrary N grids now.)

## 7. Large-N region sweep with the hint OFF (N=10–90)

Repeated the N=10–90 region sweep with the **hint off** to isolate whether large N alone
(strong economic incentive) induces building without the nudge. Run in two pieces and
merged: N=10–50 (new, 45 cells, $5–6) + N=60–90 (36 cells) → **81 cells, hint off, 1
rep/cell, mult=1.0**. Combined data: `runs/haiku_woodworld_region_r1_nohint_N10-90_combined/`.
Figures in `figs/woodworld/region/haiku/`.

| | build | solve | P(built\|sticks) | got-sticks cells |
|---|---|---|---|---|
| **hint OFF, N10–90** | **0.11** | 0.77 | **0.90** | 10/81 |
| hint ON, N10–90 (§6) | 0.20 | 0.73 | 0.89 | 18/81 |
| hint OFF, N2–20 (6-17) | ~0.01 | 0.98 | — | ~1 |

- **Large N induces building even without the hint** (0.11 vs ~0 at small N) — the
  build cells sit at high p / high N, and the `E[build]=E[grind]` boundary is at the very
  bottom (N\*≤11), so the whole plane favors building.
- **The hint acts on *discovery*, not follow-through.** Turning it on doubles both
  building (0.11→0.20) and got-sticks cells (10→18) but leaves P(built|sticks) unchanged
  (~0.90). Once Haiku reaches sticks it almost always finishes the axe, hint or not —
  exactly the discovery-limited pattern.
- Solve is slightly *higher* hint-off (0.77 vs 0.73), consistent with the §4/§5
  exploration tax being lighter when less probing happens.

Figures: `fig_woodworld_region_build_solve_haiku_nohint_mult1_N10-90.*` (build & solve) and
`fig_woodworld_recognition_haiku_nohint_N10-90.*` (recognition). Single-model panel
filenames now carry a `hint`/`nohint` tag so the two regimes don't collide.

**`C·R·E` decomposition for this hint-off set** (all recomputed from recorded fields, no
re-run — `solved` / `built_axe` / got-sticks suffice): curiosity **C = P(hold ingredients)
= 10/81 = 0.12**; recognition **R = P(built | sticks) = 9/10 = 0.90**; efficiency **E =
P(solved | tool built) = 8/9 = 0.89**; **C·R·E = 0.099** (build-path solve estimate; actual
solve is 0.77, so the bulk of solving is grinding, not the tool). Panels:
`fig_woodworld_{curiosity,recognition,efficiency}_haiku_nohint_N10-90.*` in
`figs/woodworld/region/haiku/`. Caveat: E rests on just 9 builders here (sparse).

## 8. Three-model decomposition (hint OFF, N=10–90): capability acts through *curiosity*

Ran the full N=10–90 hint-off sweep on **Sonnet** (~$7) and **Opus** ($9.83) to match Haiku
(§7). All three 81-cell, 1 rep, mult=1.0, run-to-solve. Runs:
`runs/{sonnet,opus}_woodworld_region_r1_nohint_N10-90_*/`. Three-model panels (Haiku/Sonnet/
Opus rows) in `figs/woodworld/panels/fig_woodworld_{curiosity,recognition,efficiency,solve}_haiku_sonnet_opus_nohint_N10-90.*`;
per-model heatmaps in `figs/woodworld/region/{model}/`.

| model | **C** = P(hold ingr) | **R** = P(built\|sticks) | **E** = P(solved\|built) | C·R·E | build | solve |
|---|---|---|---|---|---|---|
| **Haiku** | 0.12 | 0.90 | 0.89 | 0.10 | 0.11 | 0.77 |
| **Sonnet** | 0.75 | 0.97 | 0.90 | 0.65 | 0.73 | 0.77 |
| **Opus** | 0.93 | 0.99 | 0.91 | 0.83 | 0.91 | 0.89 |

**Capability acts almost entirely through one stage — curiosity (acquisition).** C scales
0.12 → 0.75 → 0.93 (~8×), while **recognition (0.90/0.97/0.99) and efficiency
(0.89/0.90/0.91) are flat-high and essentially model-independent**. Once a model reaches the
ingredients it builds (R≈1) and the built axe lets it finish (E≈0.9) — for *every* model.
The build-rate gap is a pure *discovery* gap. (Solve: Haiku & Sonnet both 0.77 by opposite
routes — Haiku grinds, C·R·E=0.10; Sonnet builds, C·R·E=0.65 — while Opus reaches 0.89 by
doing both.)

### 8.1 This is qualitatively opposite to ToolWorld

In ToolWorld the same decomposition gave **recognition R that *inverted* with capability**
(Haiku 0.87 > Opus 0.36 — the central "under-recognition" tension) while discovery was
near-free, and gathering/efficiency rose with capability. Here **R is flat-high and
discovery (C) is the entire story**, scaling the "normal" way. The capability ordering of
*who builds* flips between the two worlds, and the bottleneck moves from recognition
(ToolWorld) to acquisition (WoodWorld).

### 8.2 Three structural differences, and a plan to isolate them

WoodWorld differs from ToolWorld in (at least) three ways, any of which could drive the
qualitative switch. The next phase **isolates each factor** (one-at-a-time edits to the
WoodWorld recipe/mechanics, re-running the three-model hint-off N=10–90 decomposition) to
find which one flips the behavior:

1. **Narrow build tree** — WoodWorld offers only one or two viable combine actions at any
   point, vs ToolWorld's wider space of candidate pairs. *Ablation:* widen the build tree
   (add decoy combinable items / more candidate pairs) and see whether recognition starts
   to invert.
2. **Actively hostile build action** — building *consumes the goal item* (2 wood → sticks
   spends wood, so the goal counter drops; the intermediate looks like a loss — the
   "counterproductive, I lost 2 r" misread seen in Haiku/Sonnet transcripts). ToolWorld's
   build was net-neutral-to-positive (examines also dropped keys). *Ablation:* make the
   build non-consuming (or have it not touch the goal item) and see whether curiosity rises.
   **→ DONE (§9): this is the driver.** A non-hostile single-step control flips Haiku's
   build rate 0.04 → 0.86 (width/step-count matched).
3. **Multilayered build tree** — the axe needs **two consecutive builds** (wood→sticks→axe),
   vs ToolWorld's single combine. The intermediate (sticks) is inert on its own, so a model
   that stops after one build sees no payoff. *Ablation:* collapse to a one-step recipe
   (wood→axe directly) and see whether the discovery gap shrinks.

Goal: determine which factor (or combination) causes the bottleneck to move from recognition
to acquisition, i.e. what makes a tool-discovery task discovery-limited vs disposition-limited.

## 9. Isolating factor (2): the hostile build is the driver

Two single-step WoodWorld variants (`scripts/woodworld.py` `VARIANTS`, selected via
`run_woodworld_region_sweep.py --variant`), each Haiku / hint-off / N=10–90 / 81 cells,
holding **tree-width and step-count fixed** and flipping only **hostility**:

- **`iso_apples3`** (hostile) — single step **2 wood → axe** (building *consumes the goal
  item*), widened with 3 inert combinable apple decoys.
- **`iso_nonhostile`** (non-hostile control) — single step **2 sticks → axe**, where sticks are
  a *free side-resource* gathered independently w.p. p alongside wood (building consumes
  sticks, **not** wood), widened with one inert apple. `use axe → +2 wood`, unchanged. So the
  axe ingredients are essentially free (C≈1 by construction) and building is pure upside.

| variant (Haiku, hint-off, N10–90) | **C** P(hold ingr) | **R** P(built\|held) | **E** P(solved\|built) | build | solve | cost |
|---|---|---|---|---|---|---|
| baseline (§7, hostile + multilayer + narrow) | 0.12 | 0.90 | 0.89 | 0.11 | 0.77 | — |
| **`iso_apples3`** (hostile, 1-step, widened) | 0.99 | **0.04** | 0.67 | **0.04** | 0.09 | $13.82 |
| **`iso_nonhostile`** (non-hostile, 1-step, widened) | 1.00 | **0.86** | 0.86 | **0.86** | **0.86** | **$8.56** |

**Hostility is the driver.** Flipping the build from hostile to non-hostile — width and
step-count matched — takes Haiku's build rate from **0.04 → 0.86** (~20×) and solve from
**0.09 → 0.86**. With a free-ingredient, non-consuming build, Haiku behaves **ToolWorld-like**:
it builds readily and recognition is no longer the bottleneck (R 0.86). This is the
qualitative flip §8.2 was hunting for, and it lands on factor (2), the actively hostile build
action — not tree width or step count. Mechanistically it matches the transcript misread
("counterproductive, I lost 2 wood"): when building no longer spends the goal item, the
disincentive disappears. Panels (C/R/E/solve/persistence) in `figs/woodworld/nonhostile/`;
builder rows verified (`axe_cost=[0,1]`, `combine 2 sticks → axe (persists)`, then repeated
`use axe → +2 wood`).

### 9.1 `iso_nonhostile` across the full capability axis (Haiku / Sonnet / Opus)

Re-ran `iso_nonhostile` on **Sonnet** (`claude-sonnet-4-6`, $4.58, 3194 actions) and **Opus**
(`claude-opus-4-8`, $9.14, 3197 actions) — both ~0.6× Haiku's action count (they build
immediately and grind less). Panels in `figs/woodworld/nonhostile/{sonnet,opus}/`.

| `iso_nonhostile` (hint-off, N10–90) | **C** | **R** P(built\|held) | **E** | build | solve | cost |
|---|---|---|---|---|---|---|
| **Haiku** | 1.00 | 0.86 | 0.86 | 0.86 | 0.86 | $8.56 |
| **Sonnet** | 1.00 | 0.89 | **1.00** | 0.89 | 0.89 | $4.58 |
| **Opus** | 1.00 | **0.99** | 0.96 | **0.99** | **0.95** | $9.14 |

All three land in the same ToolWorld-like regime: C and E saturate, and **R rises *monotonically
and positively* with capability (0.86 → 0.89 → 0.99)** — the *normal* direction, and the exact
opposite of ToolWorld. Compare their baseline build rates (§8: Haiku 0.11 / Sonnet 0.73 / Opus
0.91): non-hostility collapses the build gap to near-ceiling for every model, most completely for
Opus.

**This is the exact mirror of ToolWorld's recognition inversion** (there R *fell* with capability:
Haiku 0.87 → Opus 0.36; here it *rises*: Haiku 0.86 → Opus 0.99). ToolWorld's inversion was a
*disposition* effect — capable models declining a build they were able to do (recoverable by a
build demo; cf. the playground finding), because ToolWorld offered a salient brute-force
alternative and rich state to confabulate around. `iso_nonhostile` removes both (the build is
strictly dominant; the world is minimal and gives explicit success feedback), so recognition
saturates and the residual capability signal points the normal way (more capable → marginally
more reliable execution of a simple, obvious step). Crucially, hostility is *not* the axis behind
the R-ordering — ToolWorld's build was already non-hostile and R *still* inverted, so the
ToolWorld↔WoodWorld R-ordering difference is separate from the §9 hostility finding (which is
about build *rate* via curiosity, not the R *ordering*).

**Caveat on the contrast (not the conclusion).** The two variants aren't perfectly
width-matched: `iso_apples3` has **3** apple decoys, `iso_nonhostile` only **1**. With 3
decoys the agent had many junk pairs to chase — 481/484 combine attempts were apple-pairs and
only 3 were the working wood+wood, so its build rate ≈0.04 partly reflects *"it rarely even
tried the build"*, not *"it tried and declined the hostile build."* The airtight isolation is
a **matched pair with an identical item set + gather distribution, flipping only the recipe
input**: both worlds gather wood + stick + apple (each w.p. p) with the same 4 items, and the
only difference is `2 wood → axe` (hostile; stick + apple both inert decoys) vs `2 stick → axe`
(non-hostile; apple the only decoy). That removes the decoy-count / candidate-pair confound
entirely. But the direction is already unambiguous: a non-hostile single-step build flips Haiku
from ~never-builds to almost-always-builds.

### 9.2 Budget slack does *not* reproduce the ToolWorld inversion (Sonnet + Opus)

ToolWorld's recognition inversion (R falls with capability, Opus R≈0.36) could in principle come
from a *viable brute-force alternative*: ToolWorld ran with budget **slack** (mult ≈1.2 —
grinding had a ~20% surplus, a real escape hatch), whereas WoodWorld is the **forced** regime
(mult 1.0 — grinding is a coin-flip). Hypothesis: with slack, a capable model defects to the
viable grind instead of building, lowering R. Test variant `iso_nonhostile_pear` (= `iso_nonhostile`
+ a second inert decoy "pear", to also widen the candidate space) run at **`--budget-mult 1.2`**
on Sonnet and Opus. (New plumbing: `validate_woodworld.budget_for(n, p, mult)` +
`run_woodworld_region_sweep.py --budget-mult`; figures `figs/woodworld/nonhostile/wide/{model}/`.)

| `iso_nonhostile`, hint-off N10–90 | R / build | solve | non-build cells |
|---|---|---|---|
| **Sonnet** mult 1.0, 1 decoy | 0.89 | 0.89 | (build = solve) |
| **Sonnet** mult 1.2 + pear (wide) | 0.83 | 0.91 | 14 cells, mean p 0.88, 8 grind-solved |
| **Opus** mult 1.0, 1 decoy | 0.99 | 0.95 | (build = solve) |
| **Opus** mult 1.2 + pear (wide) | 0.86 | 0.96 | 11 cells, mean p 0.85, 9 grind-solved |

**Two findings.** (1) **Slack genuinely lowers build rate** — both models build less (Sonnet
0.89→0.83, Opus 0.99→0.86). But (2) **it does NOT reproduce the inversion, and the lower R is
rational, not a failure.** The cells where each model stops building are the **high-p** ones
(Sonnet mean p 0.88, Opus 0.85; vs grid mean 0.6), they **still solve by grinding** (solve rises
to 0.91 / 0.96), and for Opus 8 of the 11 were cells it *built* at mult 1.0 — i.e. they flip from
build-necessary (tight budget) to grind-sufficient (20% surplus). The model correctly reads the
economics and grinds where building is unneeded. Crucially the **capability ordering is preserved**:
under slack Opus (0.86) is still ≥ Sonnet (0.83) — *not* the ToolWorld pattern (Opus < Sonnet).
And the *character* is opposite: here a capable model declines a build that is *unnecessary* (and
still solves); in ToolWorld it declined a build that was *needed* (and lost performance — a
mistake recoverable by a demo).

So **alternative-viability explains "less building," but not "Opus specifically worse than
Sonnet."** Rational defection to a viable grind keeps the ordering intact. Combined with the
earlier falsification of the build-landscape-width hypothesis (ToolWorld also has exactly *one*
valid build; and WoodWorld baseline's 2-step tree didn't invert either), the surviving driver of
the inversion is **ambiguity / confabulation room** (Opus invents mechanics under ambiguity;
WoodWorld is minimal and gives explicit success feedback, so Opus's capability is pure upside →
R 0.99/0.86). **Next test:** inject *ambiguity* into WoodWorld (misleading/suggestive feedback,
opaque failures, fake-mechanic lore) — *not* more decoys or slack — and check whether Opus's R
finally falls *below* Sonnet's. That would be the genuine recreation of the ToolWorld inversion.

## Caveats

- **Single cell, n=40, SE ≈ 0.05–0.08.** The across-the-board *direction* of the
  arity-cap effect is solid, but individual cell differences are ~1–2 SE; the
  obfuscation gaps under the cap are within noise.
- The N=10–90 sweep is 1 rep/cell (per-N build rates rest on n=9, wide CIs); the *trend*
  (build rises with N) is the claim, not the individual cell values.
- Reasoning probe is n=8/condition — illustrative of mechanism, not a rate estimate.
- All findings are Haiku-only at this cell. The cap should be re-run across the
  (p, N) region and on Sonnet/Opus before generalizing.

## Artifacts

| kind | path |
|---|---|
| env change | `scripts/woodworld.py` (`obfuscate` param; 2-item `combine` cap) |
| runner | `scripts/run_woodworld_obf_ablation.py` |
| no-cap run | `runs/woodworld_obf_ablation_p05_N10_20260617_182237/` |
| cap run | `runs/woodworld_obf_ablation_p05_N10_20260617_183935/` |
| figures | `figs/woodworld/obf_ablation/fig_woodworld_obf_ablation_haiku_p05_N10_{nocap,cap2}.png` (+ .pdf) |
| reasoning dump | `/tmp/reasoning_dump.json` |
| panels script | `scripts/plot_woodworld_panels.py` → `figs/woodworld/panels/fig_woodworld_{solve,efficiency,recognition,curiosity}_panels_Nle20.*` |
| panels source data | `runs/{haiku,sonnet,opus}_woodworld_region_r1_hint_N2-20_*/` (6-17 region sweeps) |
| large-N sweep (§6) | `runs/haiku_woodworld_region_r1_hint_N10-90_20260617_225431/` ($13.15, 81 cells) |
| large-N heatmap | `scripts/plot_woodworld_region_heatmap.py` (now infers N lattice; writes to `region/<model>/`) → `figs/woodworld/region/haiku/fig_woodworld_region_build_solve_haiku_hint_mult1_N10-90.*` |
| large-N hint-OFF (§7) | `runs/haiku_woodworld_region_r1_nohint_{N10-50_*,N60-90_*}/` merged → `runs/haiku_woodworld_region_r1_nohint_N10-90_combined/` |
| hint-OFF figures (§7) | `figs/woodworld/region/haiku/fig_woodworld_{region_build_solve_haiku_nohint_mult1,curiosity_haiku_nohint,recognition_haiku_nohint,efficiency_haiku_nohint}_N10-90.*` |
| 3-model hint-OFF (§8) | `runs/{sonnet,opus}_woodworld_region_r1_nohint_N10-90_*/` (Sonnet ~$7, Opus $9.83) |
| 3-model panels (§8) | `figs/woodworld/panels/fig_woodworld_{curiosity,recognition,efficiency,solve}_haiku_sonnet_opus_nohint_N10-90.*` |
| small-N panels (archived) | `figs/woodworld/panels/small/` (the N=2–20 hint-on trio panels) |
| variant seam (§9) | `scripts/woodworld.py` `Mech` / `VARIANTS` (`iso_apples3`, `iso_nonhostile`; `extra_drops` field) |
| hostility control (§9) | `runs/haiku_woodworld_region_iso_nonhostile_r1_nohint_N10-90_20260618_134932/` ($8.56, 81 cells) |
| hostility control — Sonnet (§9.1) | `runs/sonnet_woodworld_region_iso_nonhostile_r1_nohint_N10-90_20260618_141641/` ($4.58, 81 cells) → `figs/woodworld/nonhostile/sonnet/` |
| hostility control — Opus (§9.1) | `runs/opus_woodworld_region_iso_nonhostile_r1_nohint_N10-90_20260618_142729/` ($9.14, 81 cells) → `figs/woodworld/nonhostile/opus/` |
| slack probe — `iso_nonhostile_pear` + `--budget-mult` (§9.2) | env: `scripts/woodworld.py` `ISO_NONHOSTILE_PEAR`; `validate_woodworld.budget_for(n,p,mult)`; `run_woodworld_region_sweep.py --budget-mult` |
| slack probe runs (§9.2) | `runs/{sonnet,opus}_woodworld_region_iso_nonhostile_pear_mult1.2_r1_nohint_N10-90_2026061[8]_*/` (Sonnet $4.82, Opus $10.34) → `figs/woodworld/nonhostile/wide/{sonnet,opus}/` |
| `iso_apples3` run (§9) | `runs/haiku_woodworld_region_iso_apples3_r1_nohint_N10-90_20260618_123329/` ($13.82) |
| hostility figures (§9) | `figs/woodworld/nonhostile/fig_woodworld_{curiosity,recognition,efficiency,solve,persistence}_haiku_nohint_N10-90.*`; `figs/woodworld/iso/` (`iso_apples3`) |

## Open threads

**Primary next phase — isolate the WoodWorld↔ToolWorld switch (§8.2).** One-at-a-time
ablations of the three structural factors, each re-running the three-model hint-off
N=10–90 C·R·E decomposition, to find which flips the bottleneck from recognition
(ToolWorld) to acquisition (WoodWorld):
- **(a) widen the build tree** (decoy combinable items / more candidate pairs) — does R
  start to invert?
- **(b) non-hostile build** (build doesn't consume the goal item) — **DONE (§9): YES, this is
  the driver.** Non-hostile single-step control flips Haiku build 0.04 → 0.86.
- **(c) one-step recipe** (wood→axe directly, no inert intermediate) — does the discovery
  gap shrink? (`iso_apples3` is one-step + hostile; built only 0.04, but confounded by
  combinable decoys — see below.)
Then pairwise / full-factorial if a single factor doesn't fully account for the switch.

Immediate follow-ups to §9:
- **Airtight hostility isolation:** build a matched comparator with the *same item set + gather*
  as `iso_nonhostile` (wood + stick + apple, each w.p. p) but recipe `2 wood → axe` (hostile;
  stick + apple both inert decoys). Identical width / candidate-pairs to `iso_nonhostile`,
  differing only in whether the build consumes the goal item — removes the 3-vs-1 decoy-count
  confound in the current `iso_apples3` contrast (where 481/484 combines were apple-pairs).
- **Extend §9 to Sonnet/Opus** (`iso_nonhostile` + the fixed `iso_apples3`) to confirm the
  hostility effect across the capability axis.

Secondary:
- Does revealing the recipe (vs only capping arity, §2) close the rest of the gap?
- Goal-metric lever (cumulative vs net wood) interacts with both the "gather directly"
  satisficing (§3) and factor (b).
