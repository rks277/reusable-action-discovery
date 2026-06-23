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

**UPDATE (v4/v5 — we built that interactive version, and it changed the conclusion).** Giving the
model a **real executor tool** (v4) lifts spontaneous recognition off the floor (0.006 → ~0.9) and
routes solving *through* the C·R·E chain. Making the tool **costly** + keeping a Curiosity gate +
**hard arithmetic** (v5) finally **produces a recognition inversion** in the open→Claude range: the
weak model leans on the tool (R≈0.85–0.99), the Claude frontier declines it and overconfidently
hand-grinds (Sonnet 0.53, Opus 0.56). **BUT the cross-family sweep (Qwen 7/14/72B, Claude trio,
GPT-5-mini/5/5.5) shows the decline is NOT a capability law** — it's a **per-family, per-version
disposition**: Qwen never declines at any size (R 0.9–0.99, 7B→72B); Claude declines at Sonnet/Opus;
GPT: mini 0.62, gpt-5 0.52 (declines like the Claude frontier), gpt-5.5 0.78 (recovers). So the
overconfident tool-decline appears in *some* models (gpt-5, Sonnet, Opus) and is absent/fixed in
others (Qwen all sizes, gpt-5.5) — tracking neither scale nor a single family cleanly. (gpt-5-nano
was run but is excluded from the plot: its very low R 0.25 is a *different* mode — cost-instruction
over-compliance + incompetent bare-answer grinding, not capable overconfidence.) See v4 / v5 below.

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

## v3b — recognition = "wrote any Python" (no rerun; + dropped-instruction Haiku)

Redefined Recognition from `reused()` (a `def` applied across the batch) to **wrote any Python at
all** (`creator_batch.wrote_code`). Two regimes:
- **With** the v3 SYS line "Show your calculations as Python code": recognition is a *compliance
  ceiling* — 0.95/0.98/0.95 (offline-recomputed on the v3 trio run); grind→~0, variance shifts to
  Efficiency (0.43/0.60/0.57).
- **Dropping** that SYS line (Haiku full, `runs/creator_batch_20260622_095934/`): spontaneous
  code-use = **0.006** (3/536); solve 0.49 = all grind. So "wrote Python" is prompt-pinned
  floor↔ceiling (0.006 ↔ 0.95) with no headroom for an inversion.
- **Why it floors:** grinding is the *cheaper* path here — measured build > grind in every
  meterable currency (forced-code medians: out-tok Haiku 658 > 265, Sonnet 476 > 96, Opus 257 >
  155; elapsed also higher). Grind's real work lives in the forward pass (uncharged); code only
  *adds* visible tokens. Opposite of ToolWorld, where grinding = many metered world-actions, so a
  budget bites. ⇒ no token/time/announced budget can favor building in CREATOR; only forcing
  externalized work can.

## v4 — template + N value-rows + a FREE `evaluate` tool (FULL TRIO DONE)

The first variant to give the model a **real executor** (text-action loop, like ToolWorld):
one word problem stated with variable names, then N input rows (one shared value withheld →
Curiosity gate); the model may emit code, we run it and return stdout, then it answers.
Recognition = **chose to call the executor** (a real action, not a coding-style guess). Files:
`scripts/creator/creator_eval_tool.py`, `run_creator_eval_sweep.py`. Pilot: Haiku, 200 problems ×
N ∈ {5,20,100}, `runs/creator_eval_20260622_102602/`.

Result (patched-detector rerun, `runs/creator_eval_20260622_110648/`): Curiosity ~0.84,
**Recognition ~0.92 and DEAD FLAT across N** (0.92/0.91/0.92), Efficiency ~0.67, Solve ~0.56,
**Grind ~0.03–0.08** — first CREATOR variant where solving flows through the C·R·E path, not
around it. Solve is also flat across N (~0.56): once you build, scaling 5→100 rows is free. The
reliability mechanism is confirmed: tool-users emit ~all answers (97% coverage) and get *more*
accurate as N grows; hand-grinders only emit 54–66% (they give up partway) at ~33% accuracy. So a
free executor lifts spontaneous recognition off the floor (0.006 → ~0.92) — giving an actual tool,
not just permission to write code, is the lever. NB: the *original* pilot run
(`…_102602/`) showed a fake recognition *decline* 0.83→0.69 entirely due to the detection bug
below; the rerun with the patched executor flattens it.

> ⚠️ **TODO / known limitation — native tool-call detection.** The pilot used a custom
> `EVALUATE:\n```python` text convention. Models frequently ignore it and emit their **trained-in
> native tool-call syntax** (`<invoke name="execute"><parameter name="code">…`), which the
> original detector missed → those episodes were misscored as "grind," faking a recognition
> *decline* with N (0.83→0.69; the native-format share among "grinders" rose 54%→68%). **Fixed for
> now (option 1):** `_eval_block` now also parses the native format and executes it (re-scored
> recognition is flat ~0.90). **Still to do (option 2, the proper fix):** give the model a *real*
> tool via the Anthropic native tool-use API instead of a regex-parsed text convention — `RawChat`
> is text-only, so this needs a tool-use path added. The patched-detector **Haiku rerun is done**
> (`…_110648/`): recognition flattened to ~0.92 and Solve rose at high N (0.48→0.56), confirming
> the original decline was the bug. The **full trio ran on the patched executor** (Haiku/Sonnet/
> Opus below). Remaining option-2 work: a *real* native tool-use API would remove the residual
> ~7 native-format misses per model; the current patch handles the rest. (Sonnet N=100 left at 17
> episodes — re-run if its N=100 cell is needed.)

**Full trio, same items/seeds** (Haiku `…_110648/`, Sonnet `…_115833/` [N=100 only 17, partial],
Opus `…_123750/`). C·R·E at the two N levels where all three are complete:

| | Curiosity | Recognition | Efficiency | Solve |
|---|---|---|---|---|
| **N=5**  Haiku / Sonnet / Opus | 0.86 / 0.41 / 0.81 | 0.92 / 0.96 / 0.90 | 0.66 / 0.73 / 0.70 | 0.56 / 0.39 / 0.57 |
| **N=20** Haiku / Sonnet / Opus | 0.84 / 0.40 / 0.75 | 0.91 / 0.97 / 0.89 | 0.67 / 0.72 / 0.71 | 0.55 / 0.37 / 0.57 |

Opus is N-complete (0.90/0.89/0.95 Recognition at N=5/20/100; Curiosity 0.81/0.75/0.73; Solve
0.57/0.57/0.59). Three findings:

1. **No clean recognition inversion in the raw numbers — Recognition is high for all three
   (0.89–0.97).** Curiosity is **U-shaped** (Haiku 0.84, Sonnet **0.40**, Opus ~0.78), the *same*
   Sonnet under-asking dip as v1. Solve tracks Curiosity (Sonnet's 0.38 is its asking failure;
   Recognition+Efficiency are its highest). So the dominant cross-model axis is **epistemic
   (asking), not dispositional (building)**, in the un-gated direction.

2. **But there IS a genuine, weak inversion at the top — Opus's tool-decline.** Decomposing
   non-builders into artifact vs real: Haiku 0 real declines (43 non-builds = 36 clarification
   loops + 7 native-format misses), Sonnet 0, **Opus 20 real hand-grinds** (skips the free tool and
   computes by hand). Artifact-corrected build *disposition*: Haiku ~1.00, Sonnet ~1.00, **Opus
   ~0.95**. Opus is the **only** model that declines the executor — the predicted overconfidence
   mechanism (most-capable model trusts its own arithmetic). It is **N-gated**: declines peak at
   **N=20 (10), then N=5 (8), and nearly vanish at N=100 (2)** — Opus opts out only where grinding
   feels tractable, and concedes at scale. It is **partially calibrated but not safe**: 4 of 20
   declines crater the whole batch (mean frac-correct 0.80). That is real overconfidence cost no
   other model exposes itself to.

3. **Haiku's 0.92 < Sonnet 0.97 Recognition is a harness artifact, not a disposition gap.** Haiku's
   sub-Sonnet non-builders are all clarification loops (ask for a *second* value/formula after the
   withheld one is supplied; the single-supply harness scripts only one, so Haiku dead-ends with
   zero answers, solve 0.0). Flip side of its high Curiosity.

**Why N=20 shows the most inversion, and how to amplify (next experiment).** The effect lives where
P(decline | capable) × Cost(decline) is maximal: N=5 → grinding succeeds (no cost); N=100 → even
Opus concedes (no decline); **N≈20 → tempting but unreliable**. To amplify the overconfident-decline
inversion: (a) **deceptively-hard arithmetic** — same simple structure, but large/messy/decimal
numbers so it *looks* one-pass-easy (capable model declines) yet *is* error-prone (declines crater);
(b) **drop the Curiosity gate** to remove the asking U-shape + Haiku's over-ask artifact and isolate
build-vs-grind; (c) **fine N sweep {8,12,16,24,32}** with high item count to find the decline peak
(current signal is only 20 cases). Risk: if arithmetic is *obviously* hard, even Opus builds and the
effect collapses (the N=100 failure mode) — the trick is keeping it *perceptually* easy.

### v4-hard — amplification run (DONE: N=20, 400 items/model)

Built the amplifier (`creator_eval_hard.py`, `run_creator_eval_hard_sweep.py`,
`analyze_creator_eval_hard.py`): same simple-formula facade, inputs resampled to large/messy
values (`hard_resample`), **no Curiosity gate** (all values shown → Recognition = pure
build-vs-grind). Runs: Haiku `…_130849/` (400), Sonnet+Opus `…_133837/` (400 each, concurrency 10).

| model | n | recog (P tool) | declines | rate | crater\|dec | solve |
|---|---|---|---|---|---|---|
| Haiku  | 400 | 0.88 | 0  | 0.000 | —    | 0.55 |
| Sonnet | 400 | 0.99 | 3  | 0.007 | 0.67 | 0.56 |
| Opus   | 400 | 0.89 | 25 | **0.062** | **0.48** | 0.56 |

**Inversion is now clean & monotonic in decline rate: 0.000 → 0.007 → 0.062** (Opus declines the
free executor ~9× Sonnet, ∞× Haiku). Vs easy baseline (Opus N=20: 10 declines, crater 0.20):
- **Decline *rate* barely moved (5.0%→6.2%)** — big numbers don't deter Opus; it gauges "can I do
  this in-head?" by *formula simplicity*, not digit count (matches the decline problem-type profile:
  F=ma, P=VI, fuel=rate·dist, pool volume — simple-looking formulas). The facade held.
- **Crater *cost* more than doubled (0.20→0.48)** — declining on `8689×6922×5108` craters ~half the
  time. Amplification landed on the *consequence* of overconfidence, not its frequency.
- **Power improved** (25 declines / 12 craters vs 10 / 2).
- **Honest limit:** aggregate Solve is flat (~0.56) — declines are only ~6% of episodes; the other
  ~94% use the tool and succeed regardless of model, so the inversion is real & clean *in the
  decline tail* but a minority behavior that doesn't move overall solve. Amplifying the *rate*
  (not just cost) needs *more* perceptually-trivial problems (more temptation), not harder ones —
  Opus's ~5-6% decline rate is stubbornly calibrated.

**Decline problem-type profile (from `…_123750/`):** Opus declines on simple-looking formulas
(force, power, parallel-capacitance, fuel, work-hours, ETA, volume). Craters split into (a) careless
slips on trivial formulas (`Volume=L×W×2`, unverified) and (b) deceptively-iterative formulas it
assumed closed-form (`arrangements`→factorial/loop, `infection prob`→conditional). Hard numbers
punish (a); multi-step formulas punish (b).


---

**Continued in [docs/6-23-summary.md](6-23-summary.md)** — Lever 1 (costed tool), the canonical v5 condition, the full cross-family sweep (Qwen / Claude / GPT-5 / GPT-5.4 / Gemini), the AA capability axis, curiosity mechanics, conclusions, and artifacts.
