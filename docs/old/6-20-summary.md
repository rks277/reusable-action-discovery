# Summary — gating, the recognition inversion, and what "solve" hides (6-22)

This session resolved a long-standing puzzle in the build-vs-grind work: **why does
recognition** — `P(built the tool | ever held the build ingredients)` — **fall with model
capability in ToolWorld but rise in WoodWorld?** The answer turns out to be structural, and
along the way we learned that the *solve* metric is not a clean cross-world signal at all.

Detailed gating writeup (tables, figures, repro commands): [`FINDINGS_gating.md`](FINDINGS_gating.md).
Prior session: [`6-19-summary.md`](6-19-summary.md).

---

## 1. The setup

- **ToolWorld:** open N locked doors. `examine(door)` → a random key (coupon over N keys) + a
  random byproduct type; combine 2 distinct byproducts → machine; `use machine on door_i` →
  key_i (directed); `use key_i on door_i` → opens. Recognition **falls** with capability:
  Haiku 0.86 > Sonnet 0.58 > Opus 0.36.
- **WoodWorld (iso recipe):** accumulate wood; gather (stochastic) gives wood; stick+stick→axe;
  use axe → +wood. Recognition **rises** with capability: Haiku 0.31 < Sonnet 0.63 < Opus 0.79.

Earlier work falsified bypass-availability, hostility, and build-search-cost as the reason for
the divergence, and established the ToolWorld inversion is a *genuine* recognition failure (the
stronger model proposes wrong combines / confabulates the recipe), not an interface artifact.

## 2. What we ruled out

**Distinct goal items are not the driver.** We rebuilt WoodWorld as a coupon collector (goal =
collect all N distinct *kinds* of wood, the direct analog of ToolWorld's distinct doors). Haiku
recognition stayed floored at **0.01** (stick+stick recipe). Making the goal distinct did not
make WoodWorld behave like ToolWorld.

**Recipe shape is only a minor lever.** Switching the coupon recipe to two distinct resources
(ToolWorld-shaped) lifted Haiku recognition only 0.01 → **0.08**. And tellingly, 7/8 of the rare
builders *never used the axe* — they built the tool and walked away, because nothing in the world
signalled it was the path to the goal.

## 3. The driver: GATING (obstacle structure)

We added ToolWorld's obstacle structure to the coupon world — each wood kind locked in a visible
container, a directed tool (`use axe on container → that container's key → open`) — holding
everything else at the WoodWorld value. The 2×2 (Haiku recognition `P(built axe | held
ingredients)`):

| recipe | not gated (coupon) | **gated** |
|---|---|---|
| same (stick+stick) | 0.01 | **0.40** |
| distinct (r0+r1) | 0.08 | **0.72** |

Gating is the dominant lever (+0.39 to +0.64). `gated+distinct` (0.72) approaches ToolWorld
(0.86). Gating also fixes tool-**use**: builders who used the axe rose from ~12% (coupon) to
**89–93%** (gated).

**Mechanism (chain-of-thought).** The visible locked containers reframe the task from "accumulate
items" to "overcome locked obstacles" *from turn 0*. Tool-seeking language in CoT: **3% of coupon
episodes → 100% of gated episodes**; episodes attempting ≥1 combine: 36% → 98%; self-pair combines
reached: ~1 → 239. Gating supplies the *obstacle that primes tool-seeking*; recipe shape only
governs whether the primed search succeeds.

**Conclusion:** ToolWorld's high recognition comes from its obstacle structure — visible locked
targets + a tool directed at a specific lock — not from distinct goal labels and not primarily
from recipe shape.

## 4. The capability inversion is recreated by gating

Running the gated+distinct world up the capability ladder:

| gated+distinct | recognition | solve |
|---|---|---|
| Haiku | **0.72** | 0.20 |
| Sonnet | **0.35** | 0.16 |
| Opus | **0.28** | 0.47 |

Recognition falls **monotonically** with capability (Haiku > Sonnet > Opus), landing next to
ToolWorld's Opus (0.36). The plain coupon world could *not* show this — Haiku was floored at
0.01–0.08, leaving no room for an inversion. So the inversion is a property of the **gated
structure**: gating supplies the floor, and the capability inversion (stronger model
confabulates / over-reasons past the literal recipe) sits on top of it.

## 5. Recipe depth — a deeper tree, and where it leaks

We made the axe a **two-layer** build (`r0+r1 → part`, then `part+part → axe`; Haiku, N=10..30 ×
T=2..6). Headline recognition `P(axe | held base)` = **0.53** (vs single-step 0.72), decomposed
into step-1 discovery and step-2 follow-through:

| | recognition | step-1 `P(part\|base)` | step-2 `P(axe\|part)` | solve |
|---|---|---|---|---|
| Haiku | 0.53 | 0.89 | 0.60 | 0.41 |
| Sonnet | 0.18 | **0.31** | 0.59 | 0.80 |

- **Depth's cost is really a decoy (T) effect, not "two steps are hard."** For Haiku, recognition
  is 0.86 at T=2 (≈ single-step) and collapses to 0.06 at T=6 as the `C(T,2)` search grows.
- **The inversion holds on the deeper recipe (Haiku 0.53 > Sonnet 0.18), and Sonnet's deficit is
  purely step-1 discovery** (0.31 vs Haiku 0.89); once Sonnet makes a part, its follow-through
  (0.59) matches Haiku's (0.60).
- **Sonnet "solves" most (0.80) by abandoning the tool and grinding** — only 23% of its solves are
  tool-built (vs Haiku 96%). Same grind-vs-build signature as single-step Opus.

Figures (3-panel step-1 / step-2 / solve): `figs/gated-woodworld/fig_gatedwood_{haiku,sonnet}_two_layer_*`.

## 6. Solve is not a clean cross-world signal (and the hint was a red herring)

We noticed solve rankings *flip* between worlds: ToolWorld Sonnet > Opus, but gated Opus > Sonnet.
First hypothesis was a **hint confound** (ToolWorld region sweeps run `hint=True`; all gated runs
are nohint). We ran the apples-to-apples control: **ToolWorld 3-model, full 180 cells, `--no-hint`.**

Removing the hint barely changed ToolWorld:

| ToolWorld | solve (hinted→nohint) | recognition (hinted→nohint) |
|---|---|---|
| Haiku | 0.26 → 0.19 | 0.86 → 0.83 |
| Sonnet | 0.40 → 0.34 | 0.58 → 0.62 |
| Opus | 0.28 → 0.28 | 0.37 → 0.28 |

ToolWorld's solve rank stays Sonnet > Opus > Haiku regardless of the hint, and recognition is
essentially hint-independent. **The hint was ruled out.** The solve flip is a *genuine
non-isomorphism*, cell-matched (T2-8 × N2-20), both nohint:

- **Recognition is isomorphic across worlds and inverted in both:** ToolWorld 0.87/0.71/0.28,
  gated 0.72/0.35/0.28 (Haiku>Sonnet>Opus).
- **Solve flips and it's real:** ToolWorld Sonnet 0.41 > Opus 0.29 > Haiku 0.26; gated Opus 0.47 >
  Haiku 0.20 > Sonnet 0.16.

**Driver = obfuscation + solve-conflation.** ToolWorld byproducts are *distinct* tokens (`y`,`k`);
gated resources are a *look-alike* family (`v1,v2,…`), so the build path is less discoverable in
gated. This suppresses build attempts most for the would-be builder (Sonnet combines/ep 7.7→3.4,
build 0.53→0.21, solve 0.41→0.16) while the pure-grinder Opus (≈40% of solves toolless) is
unaffected and even grinds to a higher solve. **Solve conflates tool-recognition with brute-force
grind skill; recognition is the clean signal.**

## 7. Synthesis

- **Recognition** — the disposition to build a reusable tool — is driven by **obstacle structure
  (gating)**, is **robust to the hint**, and **inverts with capability** (Haiku > Sonnet > Opus)
  in every gated world we built, matching ToolWorld. This is the real, isomorphic finding.
- The original ToolWorld-rises / WoodWorld-falls discrepancy traces to **structure, not goal
  semantics**: ungated, fungible-goal WoodWorld floors recognition; adding gating both lifts it
  and reproduces the capability inversion.
- **"Solve" is a contaminated metric** across worlds — it mixes recognition with grinding skill,
  and is sensitive to build-discovery difficulty (obfuscation). Cross-world comparisons must hold
  obfuscation fixed and should lead with recognition, not solve.

## 8. Open threads

- **Opus on the two-layer recipe** (not yet run; ~$40–50, ~60–90 min) — would extend the
  depth × capability surface and test whether Opus's step-1/step-2 split mirrors Sonnet's.
- A clean **obfuscation-matched** cross-world run (ToolWorld with a look-alike byproduct family, or
  gated with distinct resource tokens) would isolate the obfuscation effect on solve directly.
- Why the *stronger* model confabulates past a literal recipe under gating remains the deep open
  question (the mechanism behind the inversion itself).

## 9. Artifacts produced this session

- **Code (main folders):** `scripts/gated_woodworld.py` (gated env; `recipe_mode` =
  same/distinct/two_layer; `tokens` obfuscation), `scripts/run_gated_woodworld_sweep.py`,
  `scripts/plot_gated_woodworld_heatmap.py` (3-panel step-1/step-2/solve for two_layer);
  `scripts/run_haiku_region_sweep.py` gained a `--no-hint` flag.
- **Exploratory envs:** `tool-wood-discrepancy/` (coupon-collector + single-step gated envs,
  budgets, plots).
- **Figures:** `figs/gated-woodworld/` (two-layer), `tool-wood-discrepancy/figs/` (single-step
  gated + coupon).
- **Runs:** `runs/{model}_gatedwood_two_layer_*`, `runs/{model}_region_sweep_nohint_p180_*`,
  `tool-wood-discrepancy/runs/{model}_gated_*`.
- **Docs:** [`FINDINGS_gating.md`](FINDINGS_gating.md) (full gating writeup).
