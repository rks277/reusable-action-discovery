# CREATOR fork — what we've tried (summary, 2026-06-22)

## The question

Does ToolWorld's **recognition inversion** — spontaneous tool-building *falling* with
model capability (Haiku 0.86 > Sonnet 0.58 > Opus 0.36) — replicate in a **language-native,
non-gridworld** setting? We forked the CREATOR tool-creation dataset
([external/CC.jsonl](../external/CC.jsonl), 2,047 math word-problems; each carries a
`question`, a reference `tool`, a `solution`, and a ground-truth `answer`) and ran it across
the Claude trio (Haiku 4.5 / Sonnet 4.6 / Opus 4.8). The original CREATOR paper never swept
model size, so this is also the first capability-axis run of these questions.

Design docs: [docs/CREATOR-fork-plan.md](CREATOR-fork-plan.md) (mods + v1/v2 specs),
[docs/lit-review-tool-creation-disposition.md](lit-review-tool-creation-disposition.md)
(prior work), [docs/FINDINGS_gating.md](FINDINGS_gating.md) (the gating result we leaned on).
Code: `scripts/creator/`. Figures: `figs/creator/`.

## Bottom line

**The inversion does not replicate in CREATOR — and we now understand why.** v1 (a choice
between answering inline and building a tool) makes recognition *rise* with capability
(0.56 → 0.70 → 0.80) because CREATOR is the **un-gated regime**: inline arithmetic directly
produces the answer, so "build a tool" is optional, free, and obvious — it measures
code-factoring *style*, which improves with capability. v2 (decouple grind from reward via
held-out generalization) *flattens* recognition to ≈0.49 / 0.49 / 0.51 but still shows no
inversion — because mandating `solve()` and scoring offline measures **generalization
competence, not a recognition disposition**: the agent faces no choice, no discovery, and
never experiences the decoupled reward. **The inversion needs interactive tool-discovery
under cost/ambiguity, which a single-shot word-problem QA format structurally cannot
provide.**

## Reference points (gridworlds)

Recognition = `P(built the tool | had the opportunity)`:

| Setting | Haiku | Sonnet | Opus | Direction |
|---|---|---|---|---|
| ToolWorld | 0.86 | 0.58 | 0.36 | **inverts** |
| Gated WoodWorld (distinct) | 0.72 | 0.35 | 0.28 | **inverts** |
| Un-gated WoodWorld (iso) | 0.31 | 0.63 | 0.79 | **rises** |

FINDINGS_gating localized the inversion to **step-1 discovery** of a *non-obvious* tool
**under an obstacle that demands one** (visible locked containers; grind = costly key-coupon;
the strong model confabulates a wrong recipe and grinds instead of building).

## v1 — choice prompt, partial-information chain (DONE, ~$31)

**Design.** Single interactive episode per question. One required value is stripped from the
`question`; neutral prompt offers *"answer directly, OR define a reusable function — your
choice; if something is missing, ask."* If the model asks, a scripted user supplies the value
and it continues. All three axes fall out of one nested chain mirroring ToolWorld's
`acquire → build → solve`:

| Metric (v1) | def | Haiku | Sonnet | Opus | direction |
|---|---|---|---|---|---|
| **Curiosity** | P(asks for missing value) | 0.89 | 0.74 | 0.92 | U-shaped (Sonnet dips) |
| **Recognition** | P(built tool \| asked) | 0.56 | 0.70 | 0.80 | **rises** |
| **Efficiency** | P(correct \| asked & built) | 0.56 | 0.56 | 0.67 | flat→up |
| **Solve** | P(correct) | 0.49 | 0.44 | 0.57 | U-shaped |
| **Grind G** | P(correct & not via tool path) | 0.21 | 0.15 | 0.08 | falls |

**Decomposition.** `Solve = C·R·E + G` holds **exactly** (the chain telescopes:
C·R·E = P(ask & build & correct) = the tool path; G = grind = correct-via-inline). Tool path
rises 0.27/0.29/0.49; grind falls 0.21/0.15/0.08 — as capability rises, *how* a model solves
shifts from grinding to tooling. Figure: [figs/creator/fig_metric_lines_creator.png](../figs/creator/fig_metric_lines_creator.png).

**Findings (1)** — recognition **rises**, the opposite of ToolWorld. **(2)** — curiosity and
recognition both scale *opposite* to their gridworld namesakes (they're different constructs:
ask-vs-confabulate ≠ environmental exploration; code-factoring style ≠ combine-discovery).
**(3)** — grinding *falls* with capability here, the opposite of gated/ToolWorld where the
strong model grinds *more*.

## The gating connection — v1 is the un-gated regime

v1 recognition (0.56/0.70/0.80) is nearly identical to un-gated WoodWorld (0.31/0.63/0.79) —
both rise. The inversion requires three things v1 lacks: (a) an **obstacle** that demands the
tool (inline solves it; a function is never *needed*), (b) **non-obvious discovery** (the
computation is obvious from the prompt), (c) a **cost** to grinding. Strip all three and only
code-factoring style remains, which rises with capability. The grind-direction flip confirms
the diagnosis.

## v2 — held-out generalization (decouple grind from reward; ~$18, stopped near-complete)

**Idea (from the gating insight).** Gating's essential move is that the grind path *no longer
directly advances the reward*. v2 implements that: the model is told up front to write
`solve(<inputs>)` that *generalizes*, and we score it on **held-out input values** (golds from
the reference solution). Hardcoding the shown instance fails held-out → grind; a correctly
generalized tool passes → recognition. Single-turn, announced ("your `solve` will be re-run on
other values").

| Metric (v2) | Haiku | Sonnet | Opus | (v1 recognition) |
|---|---|---|---|---|
| **Recognition** = P(generalizes) | 0.49 | 0.49 | 0.51 | 0.56 / 0.70 / 0.80 |
| **Grind** = P(shown-correct & ¬gen) | 0.13 | 0.15 | 0.12 | — |
| Solve (shown) | 0.57 | 0.61 | 0.61 | — |
| has_solve | 1.00 | ~1.00 | ~1.00 | — |

(Haiku complete n=1719; Sonnet n=1531 ≈89%, Opus n=1486 ≈86% when stopped — recognition was
flat and stable.) Figure: [figs/creator/fig_creator_heldout.png](../figs/creator/fig_creator_heldout.png)
(v2 metrics + dashed v1-recognition overlay).

**Result: flattening, not inversion.** Held-out scoring *compresses* the v1 rise to ≈flat —
Sonnet/Opus lose ~0.2–0.3 of their v1 recognition because much of it was **non-generalizing
factoring** (cosmetic/overfit `solve`s that pass the shown instance but fail other inputs).
This is the predicted "decoupling = floor, no inversion" outcome.

**Identity wrinkle (noted honestly):** `shown_correct` runs the model's own output;
`generalizes` runs our `solve()` harness. ~5% of episodes generalize but the model's own
printed answer was wrong, so `Solve(shown) = Recognition + Grind` is off by ~0.05. The clean
fix is to define `Solve := P(shown_correct OR generalizes)`.

## Why v2 still didn't give us the inversion — what the agent actually sees

Per episode the model sees only: a fixed system prompt instructing it to write
`solve(...)` that generalizes, and a user turn with the **fully-specified** problem plus the
**exact signature** `solve(sides, length, fence_cost)`. One shot. Three fatal gaps for
studying recognition:

1. **We mandated the tool** → no choice between grind and build → no recognition *decision*;
   we measure generalization *competence* (which rises/saturates with capability).
2. **We handed it the signature** → zero discovery; the inversion lives in step-1 discovery.
3. **The agent never experiences the decoupling** → it's *told* one sentence but never sees a
   held-out failure or any feedback; "grind doesn't pay" is invisible to it, purely in our
   offline scorer.

So a single-shot "write `solve()` for this fully-specified word problem" cannot exhibit
recognition-as-disposition or the inversion, regardless of scoring. v1 at least preserved a
*choice* (so it measured a disposition — but un-gated, so it rose); v2 removed the choice to
force the reward and thereby deleted the very decision we were studying.

## v3 — visible variant-batch + pure-announced budget (DONE, $17)

**Idea.** Restore the recognition *decision* v2 deleted: show the model **N=8
structurally-identical word problems at once** (all values given, one shared quantity
blanked = Curiosity gate), **ask only for the answers** (no tool cue), and announce a
small **~400-token budget** (perceived scarcity, *not* hard-enforced). Grind = solve each
inline; build = write one function and apply it across the batch. Keeps v1's C·R·E chain
(ask → built|ask → correct|built). Code: `scripts/creator/creator_batch.py`,
`creator_runner_batch.py`, `run_creator_batch_sweep.py`, `analyze_creator_batch.py`,
`plot_creator_batch.py`.

**The budget is the weak point, by design.** We established that *no* budget on external
behavior binds CREATOR's grind, because the grind lives in **reasoning**: an action budget
is gamed by thinking-then-submitting; an output-token budget (even announced) doesn't bite
when only answers are emitted; a wall-clock budget is unperceivable, capability-confounded,
and *penalizes* building (mental grind is fast, building is slow). A binding budget needs
the work **externalized** (hard arithmetic requiring code, or mandatory shown work) — which
reconstructs ToolWorld's "must act." v3 tests only the *pure-announced* nudge (perceived
scarcity); a binding budget is still open.

**One bug caught + fixed before the run:** sequential value-substitution blanked the wrong
number when a resampled value collided with another input's original; fixed with single-pass
span-based rendering (collision-rejecting).

**Final results** (`runs/creator_batch_20260622_010104/`, 611 items × 3, figure
[figs/creator/fig_creator_batch.png](../figs/creator/fig_creator_batch.png)):

| Model | Curiosity (ask) | Recognition (built\|ask) | Efficiency (all\|ask&built) | Solve (all 8) | Grind |
|---|---|---|---|---|---|
| Haiku  | 0.85 | **0.12** | 0.10 | 0.37 | 0.36 |
| Sonnet | 0.53 | **0.12** | 0.29 | 0.37 | 0.36 |
| Opus   | 0.90 | **0.21** | 0.35 | 0.52 | 0.45 |

**Read: no inversion, and recognition is *low*.** Un-cued spontaneous build rate is
0.12 / 0.12 / 0.21 — flat-to-slightly-rising, never inverting, and **far below v1's
0.56 / 0.70 / 0.80**. The gap is the explanation: v1's high recognition was largely the
explicit *"you may define a reusable function"* cue; remove the cue (v3) and spontaneous
building collapses to ~0.12, and the **pure-announced budget barely lifts it** (the K=8
pilot showed ~0; at scale ~0.12–0.21). Solve is almost entirely **grind** (grind ≈ solve:
0.36/0.36/0.45) — building contributes essentially nothing to solving. Curiosity is again
U-shaped (0.85/0.53/0.90, Sonnet under-asks). So a non-binding announced budget does not
create the obstacle recognition needs — consistent with the prediction and with the
boundary finding. Cost $17.

## Conclusion & open paths

The inversion is a property of **interactive tool-discovery under cost/ambiguity** (ToolWorld /
gated WoodWorld). CREATOR's static QA format provides none of that, so:

- **(A) Rebuild CREATOR as interactive/choice-based** — no tool cue, no signature; a multi-step
  session where building is optional, grinding is available but costly, and the agent must
  discover that a reusable tool pays. Effectively re-creating ToolWorld's structure in language
  — substantial, and may still be null.
- **(B) Report the boundary finding** — v1 (un-gated → rises) + v2 (decoupling alone → flattens,
  no inversion) together show the inversion needs interactive discovery-under-cost that static
  word-problem QA structurally cannot provide. Test the inversion OOD only in environments with
  real tool discovery.

Leaning **(B)** as the truthful headline; (A) is a large build at real risk of still-null.

## Naming note

We keep the canonical axes **Curiosity / Recognition / Efficiency**; the lesson is that the
CREATOR *operationalizations* drifted to adjacent constructs (ask-vs-confabulate; code-factoring
style; execution correctness). The goal going forward is better CREATOR analogies that capture
the *same* construct as the gridworld axes — not renaming the axes.

## Artifacts & costs

- **Code:** `scripts/creator/` — v1 (`creator_exec`, `creator_ablation`, `creator_runner`,
  `run_creator_sweep`, `analyze_creator`, `plot_metric_lines_creator`); v2 (`creator_heldout`,
  `creator_runner_heldout`, `run_creator_heldout_sweep`, `analyze_creator_heldout`,
  `plot_creator_heldout`). Held-out payloads cached at `runs/creator_heldout_payloads_M5.json`
  (parallel build ~2 min cold, instant on hit).
- **Runs:** v1 merged at `runs/creator_all/`; v2 at `runs/creator_heldout_2026062{1,2}_*/`
  (Haiku complete; Sonnet/Opus partial, stopped).
- **Figures:** `figs/creator/fig_metric_lines_creator.png` (v1, with grind),
  `figs/creator/fig_creator_heldout.png` (v2 + v1 overlay).
- **Memory:** `creator-cre-fork`, `creator-axes-dissociate-relabel`.
- **Cost:** v1 ≈ $31 (full trio); v2 ≈ $18 (Haiku full + Sonnet/Opus ~87%). Total ≈ $49.
