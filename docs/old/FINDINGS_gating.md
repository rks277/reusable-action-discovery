# Gating resolves the tool-wood discrepancy — and recreates the capability inversion

*tool-wood-discrepancy, 2026-06-19. Haiku (`claude-haiku-4-5`) + Sonnet (`claude-sonnet-4-6`),
nohint, 1 rep/cell, T = 2..8 × N = 2..20 (133 cells/sweep).*

## The question

ToolWorld and WoodWorld disagree on how **recognition** — `P(built the tool | ever held the
build ingredients)` — moves with model capability:

- **ToolWorld:** recognition *falls* with capability — Haiku 0.86 > Sonnet 0.58 > Opus 0.36.
- **WoodWorld (iso recipe):** recognition *rises* with capability — Haiku 0.31 < Sonnet 0.63 < Opus 0.79.

A chain of experiments narrowed the structural cause. Making WoodWorld a **coupon collector**
(goal = collect all N distinct kinds, matching ToolWorld's "open N distinct doors") did *not*
reproduce ToolWorld's high Haiku recognition — it stayed floored at **0.01** (stick+stick recipe)
and **0.08** (two-distinct-resource recipe). So *distinct goal items* and *recipe shape* are not
the driver. CoT analysis pointed at the remaining gap: in the coupon world Haiku frames combining
as "create new wood types" and never conceives of a reusable tool, because nothing in the world
is an **obstacle** that demands one.

This experiment isolates that last variable: **gating**.

## The gated env

`gated_woodworld.py` adds ToolWorld's obstacle structure to the coupon world, holding everything
else at the WoodWorld value:

- **Goal:** collect all N distinct kinds of wood. Each kind `w_i` is **locked inside container
  `c_i`**; the N containers are **listed from turn 1** (like ToolWorld's doors).
- **`examine <container>`** (grind/search): drops 1 uniformly-random **key** (coupon-collector
  over the N keys) + 1 uniformly-random tree-resource (the build ingredient). Both prob 1.
- **`combine <a> <b>`** → axe. `same` recipe = stick + stick; `distinct` recipe = two distinct
  resource types (ToolWorld-shaped). Look-alike numbered family obfuscation; the agent must
  search which pair combines.
- **`use <axe> <container_i>`** → yields `key_i` — the key for *that* container (**directed
  tool**, like `use machine on door_i`). The axe persists.
- **`use <key_i> <container_i>`** → opens `c_i`, revealing wood `w_i` (if the key matches).
- **Budget** = `round(1.2·(N·H_N + N))` == ToolWorld's exact grind cost (coupon over keys + N
  opens). Building stays **optional**, never forced by the budget.

This makes the grind bypass a coupon-collector over keys, and the build path a directed tool
(build once, then `axe→key`, `key→open` per container) — structurally identical to ToolWorld.

## Result 1 — gating is the dominant lever (2×2: gating × recipe, Haiku)

Recognition `P(built axe | held ingredients)`:

| recipe | **not gated** (coupon) | **gated** (new) |
|---|---|---|
| same (stick + stick) | 0.01 | **0.40** |
| distinct (r0 + r1)   | 0.08 | **0.72** |

*ToolWorld Haiku reference: 0.86.*

- Gating lifts recognition by **+0.39** (same recipe) and **+0.64** (distinct recipe).
- Recipe shape is a real but **secondary** lever (+0.32 within the gated world): note that
  `gated + same` (0.40) already far exceeds `coupon + distinct` (0.08).
- **`gated + distinct` (0.72) approaches ToolWorld (0.86)** — it *is* ToolWorld reskinned in wood,
  serving as a positive control that validates the env.

Gating also fixes **tool-use**, not just building: in the plain coupon world 7/8 builders never
used the axe (`use_axe_count = 0`); in the gated world **89–93% of builders used it** (same 33/37,
distinct 55/59). The visible locked target + directed tool make "build → use on container → open"
the obvious next step.

**Conclusion:** ToolWorld's high Haiku recognition comes from its **obstacle structure** —
visible locked targets the agent must act on, plus a tool *directed* at a specific lock — not from
distinct goal items (the coupon already had those) and not primarily from recipe shape.

Figures: `tool-wood-discrepancy/figs/fig_gated_haiku_same_*`, `tool-wood-discrepancy/figs/fig_gated_haiku_distinct_*`.

## Result 2 — the chain-of-thought mechanism (why gating builds ~30× more)

Comparing `gated + same` vs `coupon + same` (recipe identical, so the *only* difference is gating):

| metric | coupon + same | gated + same |
|---|---|---|
| episodes that try ≥1 combine | 36% | **98%** |
| total combine attempts | 215 | **1602** |
| self-pair combines reached (what the recipe needs) | ~1 | **239** |
| episodes whose CoT uses tool / open / unlock / locked language | **3%** | **100%** |

The visible **locked containers reframe the task from "accumulate items" to "overcome locked
obstacles" from turn 0.** Haiku plans for the tool before it has any ingredients:

> "First, I need to examine containers to understand what keys/tools I need… I'll need to find or
> **create keys/tools to unlock the containers**." *(turn 0, T=2 N=7)*

> "I haven't collected any wood yet — the items are still in the containers. **I need to figure out
> how to unlock the containers.** Let me try combining the items I have **to see if that creates a
> key or tool**." *(turn 4, T=2 N=4)*

In the coupon world `gather` directly hands over the goal wood, so there is no obstacle to reason
about; the leftover resources are ignored as noise and combining is rarely tried at all.

**Causal chain:** visible locked obstacle → "I need a tool/key to open these" (tool-seeking frame)
→ search the combine space (incl. the self-pair) → find the recipe → build. Gating supplies the
*obstacle that primes tool-seeking*; recipe shape only governs whether that primed search succeeds.

## Result 3 — gating recreates the ToolWorld capability inversion

`gated + distinct`, full capability ladder:

| model | recognition | build rate | solve rate | builders used axe |
|---|---|---|---|---|
| **Haiku** (`claude-haiku-4-5`)   | **0.72** | 0.44 | 0.20 | 93% (55/59) |
| **Sonnet** (`claude-sonnet-4-6`) | **0.35** | 0.21 | 0.16 | 61% (17/28) |
| **Opus** (`claude-opus-4-8`)     | **0.28** | 0.22 | 0.47 | 93% (27/29) |

**Recognition falls monotonically with capability under gating: Haiku 0.72 > Sonnet 0.35 >
Opus 0.28** — the same direction as ToolWorld (Haiku 0.86 > Sonnet 0.58 > Opus 0.36), and Opus
lands right next to ToolWorld's 0.36. The plain coupon world could not show this: Haiku was
floored at 0.01–0.08, leaving no room for an inversion.

Note Opus's **solve** rate is the *highest* of the three (0.47 vs 0.16/0.20) even though it
recognizes the tool *least*: it under-builds but grinds the key-coupon efficiently and opens
doors by brute force. That is the capability-axis signature — the stronger model solves *without*
recognizing the reusable tool.

The two results compose:

1. **Gating supplies the floor.** The obstacle structure is what gets any model to recognize the
   build (coupon 0.08 → gated 0.72 for Haiku).
2. **The capability inversion sits on top of that floor.** Once gating engages tool-seeking, the
   *more capable* model recognizes the build *less* — consistent with the confabulation /
   over-reasoning story (the stronger model talks itself out of the literal stick+stick recipe and
   proposes wrong combines; see `toolworld-recognition-inversion-artifact`).

This is strong evidence that the inversion is a property of the **gated/obstacle structure**, not
of distinct goal labels.

Figures: `tool-wood-discrepancy/figs/fig_gated_{sonnet,opus}_distinct_*`.

## Caveat — solve rate is budget-limited, not a recognition artifact

Solve rate is low (Haiku 0.16 same / 0.20 distinct; Sonnet 0.16); most episodes end
`out_of_budget`. Recognition is measured at **build time**, so it is unaffected. Solve is depressed
because the directed exploit costs **2 actions per container** (`axe→key`, then `key→open`), so the
build path busts the grind-calibrated budget at small/mid N — the same budget bind ToolWorld shows
at low N. Builders recognize *and* use the tool, then run out of budget before opening all N
containers. To make solve track recognition, widen the budget multiplier or let the keys that
`examine` already drops open containers in one action (currently underused vs the directed axe path).

## Reproduce

```bash
# Haiku, both recipe modes
PYTHONPATH=. python tool-wood-discrepancy/run_gated_sweep.py --recipe same     --conc 12 --max-cost 15
PYTHONPATH=. python tool-wood-discrepancy/run_gated_sweep.py --recipe distinct --conc 12 --max-cost 15
# Sonnet, distinct
PYTHONPATH=. python tool-wood-discrepancy/run_gated_sweep.py --recipe distinct --model claude-sonnet-4-6 --conc 12 --max-cost 50
# plot any run (recognition + solve heatmaps; --prefix auto-detects "gated")
PYTHONPATH=. python tool-wood-discrepancy/plot_coupon_heatmaps.py tool-wood-discrepancy/runs/<dir>
```

| sweep | recognition | build | solve | cost | wall |
|---|---|---|---|---|---|
| Haiku gated + same     | 0.40 | 0.28 | 0.16 | ~$13.5 | ~25 min |
| Haiku gated + distinct | 0.72 | 0.44 | 0.20 | ~$13.0 | ~25 min |
| Sonnet gated + distinct| 0.35 | 0.21 | 0.16 | ~$12.3 | ~45 min |

Files: `gated_woodworld.py` (env), `run_gated_sweep.py` (sweep), `budget_coupon.py`
(`budget_for_gated`), `plot_coupon_heatmaps.py` (heatmaps).

## Result 4 — recipe DEPTH costs recognition, and the cost is T-gated

A deeper recipe tree (`r0+r1 → part`, then `part+part → axe`) tests whether recognition depends
on how many combine-steps the tool requires. New env in the main folders
(`scripts/gated_woodworld.py` `recipe_mode=two_layer`, `scripts/run_gated_woodworld_sweep.py`,
`scripts/plot_gated_woodworld_heatmap.py`; figs `figs/gated-woodworld/`). Haiku, N=10..30 ×
T=2..6, nohint, 1 rep (105 eps, ~$17.7).

Headline recognition `P(built_axe | held_base)` = **0.53** vs the single-step gated Haiku 0.72 —
a deeper tree roughly halves the leak. The two-step decomposition + the per-T break localize it:

| T | step-1 `P(part\|base)` | recognition `P(axe\|base)` | step-2 `P(axe\|part)` |
|---|---|---|---|
| 2 | 1.00 | **0.86** | 0.86 |
| 3 | 0.86 | 0.48 | 0.56 |
| 4 | 1.00 | 0.68 | 0.68 |
| 5 | 0.89 | 0.47 | 0.53 |
| 6 | 0.62 | **0.06** | 0.10 |

- **The penalty is almost entirely a T (decoy-count) effect, not a depth effect per se.** At T=2
  (one candidate pair) the two-layer build is as easy as single-step (0.86); recognition then
  falls monotonically to **0.06 at T=6** as the `C(T,2)` step-1 search grows.
- **Both layers leak, and both worsen with T.** Step-1 discovery stays high until T=6 (0.62);
  step-2 follow-through (`part+part→axe`, which involves no decoys) *also* falls with T
  (0.86→0.10) — more decoys mean more budget burned reaching two parts and more wrong combines
  with junk, so fewer agents that make one part go on to complete the axe.
- Every axe-builder used the axe (51/51). Solve 0.41 (budget-limited as before).

So depth matters mainly by *amplifying the recipe-search cost* (`C(T,2)` over a longer plan),
not as an intrinsic "two steps is hard" penalty. Figure:
`figs/gated-woodworld/fig_gatedwood_haiku_two_layer_T2-6_N10-30_recognition_solve.png`.

**Capability axis on the two-layer recipe — the inversion holds, and it's a step-1 failure:**

| two-layer gated | recognition `P(axe\|base)` | step-1 `P(part\|base)` | step-2 `P(axe\|part)` | solve | P(built\|solved) |
|---|---|---|---|---|---|
| Haiku  | **0.53** | 0.89 | 0.60 | 0.41 | 0.96 |
| Sonnet | **0.18** | **0.31** | 0.59 | **0.80** | 0.23 |

Recognition still falls with capability (full-grid Haiku 0.53 > Sonnet 0.18). And Sonnet's *solve*
is the highest of the two (0.80) precisely because it **abandons the hard 2-step recipe and
grinds** the key-coupon: only 23% of its solves are tool-built (65/84 are pure grind), vs Haiku's
96% — the same grind-vs-build signature as single-step Opus.

**Three-model C·R·E decomposition (FULL grid T2-6 × N10-30, all three complete; 105 cells each):**

| two-layer | curiosity `P(base)` | step-1 `P(part\|base)` | step-2 `P(axe\|part)` | recognition `P(axe\|base)` | efficiency `P(solv\|axe)` | solve |
|---|---|---|---|---|---|---|
| Haiku  | 0.91 | **0.89** | 0.60 | **0.53** | 0.75 | 0.41 |
| Sonnet | 0.99 | 0.31 | 0.59 | 0.18 | 1.00 | **0.80** |
| Opus   | 0.90 | 0.28 | 0.69 | 0.19 | 0.78 | 0.51 |

The deeper recipe shows the recognition inversion lives **almost entirely in step-1 discovery**:
- **Step-1 discovery INVERTS** (Haiku 0.89 ≫ Sonnet 0.31 ≈ Opus 0.28): the weaker model stumbles
  onto `r0+r1→part` far more often (high combine-propensity / exploration). This single subpart
  carries the whole inversion.
- **Everything downstream does NOT invert:** step-2 follow-through is ~flat (0.60/0.59/0.69),
  efficiency is high for all (0.75/1.00/0.78), and curiosity is uniformly high (~0.9, Sonnet peaks).
- Net recognition `P(axe|base) ≈ step1 × step2` = Haiku **0.53** ≫ Sonnet 0.18 ≈ Opus 0.19 — the
  single-step world's clean monotonic inversion (0.72>0.35>0.28) softens to "Haiku ≫ {Sonnet,Opus}"
  because the stronger models tie at the floor of step-1 discovery.
- **Solve peaks at Sonnet (0.80)**, not monotonic (Haiku 0.41, Opus 0.51): the middle model grinds
  the key-coupon most cleanly, while Opus's noop/efficiency losses at high N pull it below. Solve is
  a grind-skill signal, not a recognition signal.

So the build-vs-grind capability story decomposes cleanly: **stronger models are worse at
spontaneously DISCOVERING the reusable tool (step-1) but as good or better at everything downstream**
(completing the build, exploiting it, grinding the goal). Capability-axis panel:
`figs/gated-woodworld/fig_gatedwood_two_layer_metric_panels.png`. Per-model heatmaps:
`figs/gated-woodworld/fig_gatedwood_{haiku,sonnet,opus}_two_layer_*`. **Cost note:** Opus is a poor
fit for this regime (~60s/call, full-budget grind at high N), so the full Opus grid cost ~$127 over
several hours — most of it in the high-N corner.

## Open next step

**ToolWorld nohint control — DONE, and it overturns the hint hypothesis.** Reran ToolWorld
3-model with `--no-hint` (full 180 cells each). Removing the hint barely moved ToolWorld: solve
Haiku 0.26→0.19, Sonnet 0.40→0.34, Opus 0.28→0.28; recognition ~flat (Haiku 0.86→0.83, Sonnet
0.58→0.62, Opus 0.37→0.28). **ToolWorld's solve rank stays Sonnet > Opus > Haiku with or without
the hint** — so the hint is NOT why gated and ToolWorld differ.

Cell-matched (T2-8 × N2-20), both nohint, the picture is:
- **Recognition is isomorphic and inverted in both worlds:** ToolWorld 0.87/0.71/0.28, gated
  0.72/0.35/0.28 (Haiku>Sonnet>Opus). This is the robust headline signal.
- **Solve flips, and it's real (not hint, not grid):** ToolWorld Sonnet 0.41 > Opus 0.29 > Haiku
  0.26; gated Opus 0.47 > Haiku 0.20 > Sonnet 0.16.

The flip's driver is **obfuscation + solve-conflation**: ToolWorld byproducts are *distinct*
tokens (`y`,`k`), gated resources are a *look-alike* family (`v1,v2,…`), so the build path is
less discoverable in gated. That suppresses build attempts most for the would-be builder
(combines/ep Sonnet 7.7→3.4, build 0.53→0.21) — hurting Sonnet's solve — while the pure-grinder
Opus (P(built|solved)≈0.4) is unaffected and even grinds to a *higher* solve in gated. **Solve
conflates tool-recognition with brute-force grind skill and is not a clean cross-world disposition
signal; recognition is.** Future cross-world comparisons should hold *obfuscation* fixed (both
look-alike or both distinct), not just the hint. See `gated-vs-toolworld-hint-confound`.
