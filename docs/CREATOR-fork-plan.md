# CREATOR Fork: Opening Curiosity, Recognition & Efficiency on a Fixed Question Set

> Design note (2026-06-20). How to modify the CREATOR methodology — **keeping the
> original questions** — so it measures the project's three disposition axes
> (Curiosity, Recognition, Efficiency) instead of only end-task accuracy.
> Companion to [lit-review-tool-creation-disposition.md](lit-review-tool-creation-disposition.md).

## Why CREATOR, and why fork it

CREATOR ([2305.14318](https://arxiv.org/abs/2305.14318)) is the one major prior
system that does **genuine tool *creation*** (not selection): the model writes its
own reusable Python function and then uses it. We have the dataset locally at
[external/CC.jsonl](../external/CC.jsonl) — 2,047 records.

But as imported it cannot measure disposition. The lit review's verdict applies
exactly here: CREATOR's Creation stage "explicitly instruct[s] LLMs ... to create
tools" for **every** instance — creation is *forced*, so the recognition axis is
clamped. Its native C·R·E profile:

- **Curiosity** — absent. Every problem is fully observed in the prompt; nothing
  to discover.
- **Recognition** — **forced**. Generative ("abstract a reusable function"), which
  is the *same cognitive act* as ToolWorld's combine/build recognition — but
  mandated, so disposition is invisible.
- **Efficiency** — weak. Single-shot, only the final answer is graded; there is no
  trace, no reuse, no redundancy to count.

The goal of the fork: recover gradable C, R, and E **without changing the
questions** (they are already validated, have ground-truth answers, executable
reference tools, and an abstract `utility` statement).

### The enabling asset: per-record fields

Each record carries both the question-specific answer **and** the general
capability, which is what lets us grade the *shape* of a model's output, not just
correctness:

| Field | Role in the fork |
|---|---|
| `question` | The fixed natural-language problem (unchanged). |
| `utility` | Plain-English statement of the *general* capability → grader reference for tool **generality**. |
| `tool` | Reference reusable function (docstring, typed args, return) → reference for **recognition quality**. |
| `args` / `return` / `constant` | Interface spec → drives the **partial-information ablation** (which value to strip) and the **parsimony** grade (which constants are legitimate). |
| `solution` | The call-site that composes the tool with question-specific arithmetic → reference for **compositional efficiency**. |
| `answer` | Ground-truth numeric answer → automatic scoring. |

### A useful coincidence: CREATOR's stages pre-figure C·R·E

CREATOR's native pipeline is **Creation → Decision → Execution → Rectification**.
Most of the fork is *un-clamping stages that already exist*, plus two wrapper
protocols (a batch/session wrapper for amortization, an information-withholding
wrapper for discovery):

- Creation ≈ **Recognition** (un-force it)
- Decision ≈ recognition / tool-selection
- Execution + Rectification ≈ **Efficiency** + latent **Curiosity** (the test/debug
  loop is an opportunity to probe before trusting)

## Recognition — un-force the Creation stage

Native CREATOR mandates creation, clamping the exact disposition ToolWorld
measures. Recover it:

1. **Spontaneous-vs-forced prompt (core disposition metric).** Replace the
   "create a tool" instruction with a neutral two-affordance prompt:
   *"Answer directly, or define a reusable function and then answer — your choice."*
   Score `P(builds)` per question. This is the `P(builds | could profit)` metric
   the lit review identifies as unmeasured anywhere, now on a fixed question set.

2. **Amortization incentive (does R respond to payoff?).** The dataset has
   tool-sharing families (the interest tool serves Alice *and* Bob; a perimeter
   tool is reusable). Serve a *batch* of related questions, tell the model up front
   it will face N of them, and test whether build-disposition rises with N — the
   direct analogue of WoodWorld's "build rises with economic incentive," and a
   clean test of payoff recognition (open question #4 in the lit review).

3. **Generality grading (recognition *quality*, not just presence).** Using
   `utility` + the reference `tool`, grade whether the created tool abstracts at the
   utility level (general `polygon_perimeter`) or overfits to the question (a
   function that just returns 5000). Overfit-but-correct = recognition failure even
   when the answer is right. Separates "built a tool" from "built the *right* tool."

4. **Generative-vs-discriminative contrast (same questions, two R modes).** Pool
   all reference tools across the dataset + distractors into a menu. One condition
   forces *selection* (discriminative R, WildToolBench-like); the other forces
   *creation* (generative R). Identical questions. This is the cleanest test of
   whether the ToolWorld inversion is **generative-specific** — the
   double-dissociation prediction (generative R inverts with capability,
   discriminative R scales normally), on one corpus.

## Curiosity — withhold information so the model must act to acquire it

Native CREATOR is fully observed; inject uncertainty so discovery becomes
necessary:

1. **Partial-information ablation (ask-vs-confabulate).** Strip one required
   quantity from the question (drop "$100 per meter," or the side count). The
   `args` / `solution` fields say exactly what was removed and what the gold
   clarification is. Score whether the model **asks** vs. **confabulates a value
   and proceeds**. Direct confabulation probe; ties to the project's
   Opus-confabulation findings. Curiosity here is *epistemic* (probe the user),
   not exploratory.

2. **Rectification-as-curiosity (does it test before trusting?).** Turn on the
   executor and measure whether the model *runs* its tool / inspects output /
   checks an edge case before committing, vs. submitting blind. The reference tools
   and solutions are real executable Python, so this is a live trace. Disposition
   to verify = curiosity, gradable as probing actions before the final answer.

3. **Hidden-signature / docs variant.** Don't hand the model the tool spec — put it
   behind a `lookup_docs` action it must query (mirrors CREATOR's real
   "documentation" framing and Alita's brainstorming). Curiosity = how much of the
   available API surface it probes before writing the solution.

## Efficiency — make it multi-step and amortized so a trace exists

Native CREATOR is single-shot with only final-answer grading, so there is no
redundancy to count. Add a session and/or grade the artifact:

1. **Batch reuse protocol (the TroVE metric, on known-reusable items).** Run the
   tool-sharing families as a session and measure `P(reuse | tool already built)` —
   call count, tokens, redundant re-creation. Recreates the reuse-failure finding
   (TroVE reused learned functions in 3/3,201) in a controlled setting where reuse
   is *provably* possible, making it a real efficiency signal rather than an
   accuracy proxy.

2. **Tool parsimony / amortizability.** Grade the created tool for hardcoded
   constants vs. correct parameterization — the `constant` field says which
   constants are legitimate. A tool that bakes in the question's numbers is
   inefficient (non-amortizable) even if correct. Efficiency-as-reusability of the
   artifact.

3. **Rectification-loop step count.** With the executor on, count rounds-to-correct
   / redundant tool edits — the static analogue of ToolWorld's ~26 redundant
   post-build actions.

4. **Compositional efficiency.** On multi-call questions (Alice/Bob), check whether
   the model parameterizes / loops vs. copy-pastes — the parallel-vs-redundant
   structure WildToolBench grades directly.

## C·R·E coverage after the fork

| Axis | Native CREATOR | After fork | Primary mechanism |
|---|---|---|---|
| **Curiosity** | absent | epistemic (ask-under-uncertainty) + verify-before-trust | partial-info ablation; executor probing trace |
| **Recognition** | forced (generative) | spontaneous build-disposition, generality, payoff-response, generative-vs-discriminative | neutral prompt; amortization batch; generality grade; menu contrast |
| **Efficiency** | weak (answer only) | reuse rate, parsimony, step count, composition | batch session; artifact grading; executor trace |

## Build order (most signal per unit effort)

Two pieces unlock the most:

1. **Neutral batch harness** — serve tool-sharing families as a session with a
   no-instruction prompt and a real executor. In one protocol this yields **R**
   (spontaneous build + amortization response), **E** (reuse rate + step count),
   and the parsimony grade — reusing only existing fields.
2. **Partial-information ablation** — semi-automatable from `args` (drop a required
   arg, score ask-vs-confabulate) — opens **C** and doubles as a confabulation
   probe.

The single most valuable *finding* would come from the generative-vs-discriminative
contrast (Recognition #4): whether the ToolWorld **recognition inversion**
(Haiku 0.95 > Sonnet 0.80 > Opus 0.60) reproduces on language-native questions.
That is the out-of-distribution replication that turns the inversion from a
gridworld result into a model property — with WildToolBench-style discriminative
selection as the control that should scale *normally* if the inversion is
generative-specific.

## Open implementation questions

- **Reuse-family grouping.** How to cluster the 2,047 records into
  tool-sharing families — by reference-tool name/signature similarity, by `utility`
  embedding, or by hand for a curated subset? Determines the amortization and reuse
  protocols.
- **Generality / parsimony grader.** LLM-judge against `utility` + `tool`, static
  AST checks for hardcoded constants vs. `constant`, or both? Needs a small
  human-labeled calibration set.
- **Ask-vs-confabulate scoring.** Single-turn (does the model emit a clarifying
  question?) vs. interactive (simulate a user that supplies the stripped value).
  The interactive version is closer to WildToolBench but needs a user simulator.
- **Executor sandboxing.** Running model-written Python for the curiosity/efficiency
  traces needs a sandbox; the reference `tool` + `solution` are already executable
  and can seed the harness / golden traces.

## Caveats

- The C·R·E mapping is the project's interpretive overlay; CREATOR does not frame
  itself in these terms.
- Stripping information (Curiosity mods) changes the *inputs* but not the
  *questions' identity* — answers may become under-determined by design; that is
  the point (it forces an ask), but those items can't be scored on `answer`.
- Generative-vs-discriminative results hinge on a fair distractor menu; a too-easy
  or too-hard menu confounds the discriminative condition.

---

## v1 results (3-model sweep, 2026-06-21) — and why recognition did NOT invert

Built and ran the single-interactive-run design above (scripts/creator/, full 1,502
strippable Q × Haiku/Sonnet/Opus, 1 rep, ~$31). Metrics via the chain
ask→build→correct, with the identity **Solve = C·R·E + G** (G = grind = correct
without the tool path; holds exactly):

| | Curiosity P(ask) | Recognition P(built\|ask) | Efficiency P(correct\|ask&built) | Solve | Grind G |
|---|---|---|---|---|---|
| Haiku  | 0.89 | 0.56 | 0.56 | 0.49 | 0.21 |
| Sonnet | 0.74 | 0.70 | 0.56 | 0.44 | 0.15 |
| Opus   | 0.92 | 0.80 | 0.67 | 0.57 | 0.08 |

**The ToolWorld recognition inversion did NOT replicate — recognition RISES
(0.56→0.70→0.80).** This **revises the "CREATOR recognition = the same generative act
as ToolWorld's combine" claim** made earlier in this doc: it is not the same act.

The reason is structural, and it lines up exactly with `docs/FINDINGS_gating.md`:
**v1 is the *un-gated* regime.** Un-gated WoodWorld recognition also rises with
capability (0.31/0.63/0.79) — nearly identical to CREATOR v1 (0.56/0.70/0.80) — while
*gated* WoodWorld and ToolWorld invert (0.72/0.35/0.28 and 0.86/0.58/0.36). FINDINGS
localized the inversion to **step-1 discovery** of a *non-obvious* reusable tool under
**an obstacle that demands one**; the stronger model is worse at spontaneously
discovering it. v1 has none of that: no obstacle (a word problem is solved inline; a
function is never *needed*), no discovery (the computation is obvious once the value is
supplied), and the tool was explicitly *cued* ("you may define a function"). That
leaves only **code-factoring style**, which rises with capability. The grind direction
confirms it: in v1 grind *falls* with capability (weak models default to inline), the
**opposite** of gated/ToolWorld where the strong model grinds *more* (escapes a tool it
failed to recognize). v1 is therefore the correct **un-gated control**, not a failed
replication.

## v2: Gated CREATOR — recover the constructs (and test for the inversion)

Goal: re-operationalize C/R/E so each measures its *gridworld* construct (not the
adjacent easy one v1 drifted to), by adding the **gating analogue** — a visible
obstacle that demands a reusable tool, a tool that is non-obvious to discover, and an
always-available grind bypass. Same questions; new harness. v1 stays as the un-gated
control, so the v1↔v2 contrast isolates "gating structure" on identical items.

### The obstacle: a visible variant-batch (recurrence = the lock)

For each base question, generate **K isomorphic numeric variants** (K≈6) by resampling
the `solution`'s `name = value` initializations and recomputing each gold answer by
executing the (value-substituted) reference `tool`+`solution` via the existing
`creator_exec`. Present **all K at once, un-cued** — "Solve these problems; end each
with `ANSWER_i: <number>`" — never mentioning functions, tools, or reuse. The K visible
problems are the analogue of ToolWorld's N visible locked containers: the *volume is
the obstacle*. Re-deriving each inline is the **grind bypass** (always works, so
building stays optional); writing one parameterized function and calling it K times is
the **directed tool**. Building must NOT be forced — keep K modest so inline is still
feasible (disposition, not budget).

### Faithful metrics + the step-1/step-2 decomposition (mirrors two-layer gated)

- **Recognition** = `P(builds a reusable parameterized tool applied across ≥2 variants
  | facing the batch)`. Detect via AST: a single `FunctionDef` *called ≥2× with
  different argument bindings* — distinguishing genuine recurrence-recognition from K
  inline computations or K copy-pasted hardcoded snippets. Decompose, per FINDINGS
  Result 4:
  - **step-1 discovery** `P(conceives a reusable abstraction at all)` — did it spot the
    recurrence and define a general tool. *Hypothesis: the inversion lives here.*
  - **step-2 apply** `P(all K correct | conceived)` — did the recognized tool actually
    parameterize + execute across variants. *Expected flat/high across models.*
- **Curiosity** = discovery-by-action, not asking: hide a needed formula/constant (or a
  small helper "reference library" salted with distractors) behind an `inspect`/`lookup`
  action the model must *choose* to call. `P(probes the resource before solving)` —
  the WoodWorld-exploration analogue. (Optional arm; recognition is the headline.)
- **Efficiency** = parsimonious exploitation: among recognizers, `P(reuses the one tool
  across all K vs. rebuilds/re-derives)` + redundant-call/recompute count — the direct
  analogue of ToolWorld's "Haiku over-acts post-build."
- **Grind G** closes `Solve = C·R·E + G` as before (correct-via-inline).

### Expected results (the v1→v2 predictions FINDINGS_gating implies)

| Axis | v1 (un-gated, observed) | v2 (gated, predicted) |
|---|---|---|
| Recognition | rises 0.56/0.70/0.80 | **inverts / stops rising**, driven by step-1 discovery |
| Grind G | falls 0.21/0.15/0.08 | **rises** with capability (strong model grinds the batch inline) |
| Efficiency | flat-ish 0.56/0.56/0.67 | flat-high; weak models do more redundant rebuilding |
| Curiosity (discovery) | n/a (was ask-curiosity, U-shaped) | **rises** with capability (exploration axis) |

If recognition inverts under v2 while v1 (same questions, un-gated) rises, that is the
clean within-CREATOR demonstration that **the inversion is a property of gated/obstacle
structure**, reproducing FINDINGS_gating in a language-native, non-gridworld setting.

### Implementation (new files under `scripts/creator/`, reuse v1 modules)

- `creator_variants.py` — `make_variants(item, K)`: regex-substitute the `solution`
  init RHS values (resample in a sensible range around the original, avoid degenerate/
  zero), execute the modified reference solution per variant to get gold, skip variants
  whose reference errors. Reuses `creator_exec.extract_code` / `_parse_answer`.
- `creator_runner_v2.py` — one un-cued batch episode over the K variants; AST-based
  step-1 (reusable FunctionDef called ≥2× with distinct args) and step-2 (per-variant
  execute+check) detection; optional `inspect` action for the curiosity arm.
- `run_creator_sweep_v2.py` / `analyze_creator_v2.py` — same async/JSONL/conditioning
  patterns; emit recognition + step1/step2 + efficiency + curiosity + grind.
- Plot: extend `plot_metric_lines_creator` (or a v2 sibling) — overlay v1 vs v2
  recognition, and a step-1/step-2 panel like `fig_gatedwood_two_layer_metric_panels`.

### Open v2 questions

- **K and the build/grind balance** — too large forces building (becomes a budget bind,
  not a disposition); too small kills the amortization payoff. Calibrate so inline is
  feasible but tedious (mirror FINDINGS' "building optional, never forced by budget").
- **Batch-visible vs sequential-hidden** — showing all K up front makes recurrence
  obvious (weakens the *discovery* character but keeps factor-vs-grind); revealing
  variants one-by-one better tests spontaneous recognition but muddies "should I build
  before I know more are coming." Default to batch-visible (volume = visible obstacle,
  matching gating's turn-0 visible containers); pilot both.
- **Hold obfuscation fixed** (FINDINGS open-next-step): solve conflates grind-skill with
  recognition, so report **recognition** (and step-1) as the clean cross-regime signal,
  not solve.

---

## v3 (IMPLEMENTED): visible variant-batch + pure-announced token budget

The v2 held-out run (above, in the v1-results section) measured generalization competence,
not a recognition disposition — the agent had no *choice* (we mandated `solve()`), no
discovery, and never *experienced* the decoupled reward. v3 fixes the choice: present **N=8
structurally-identical word problems at once** (all values shown; one shared quantity blanked
= Curiosity gate), **ask only for the answers** (no tool cue), and **announce a ~400-token
budget** (perceived scarcity, NOT hard-enforced). Grind = solve each inline; build = one
function applied across the batch. v1's C·R·E chain is preserved: `ask → built|ask →
correct|built`. This supersedes the v2-spec "variant-batch as obstacle" idea above — same
batch, but with answers-only + an announced budget instead of held-out scoring.

**Code (all in `scripts/creator/`):** `creator_batch.py` (`make_batch` = variant gen +
single-pass span-based NL rendering + withhold last init var; `reused()` = a function applied
in a loop/comprehension or called ≥2×), `creator_runner_batch.py` (announced-budget prompt,
ask→supply turn, `ANSWER_i` parse, score from stated answers), `run_creator_batch_sweep.py`
(parallel disk-cached batch build, async trio launcher), `analyze_creator_batch.py`,
`plot_creator_batch.py`. ~611 usable items (need ≥2 inputs each a single clean occurrence in
the question; ~30% of the dataset).

**The budget is the unresolved core.** No budget on *external behavior* binds CREATOR's grind,
because the grind is in **reasoning**: an **action/tool-call budget** is gamed by think-then-
submit; an **output-token budget, even announced,** doesn't bite under answers-only (cheap to
emit N numbers); a **wall-clock budget** is unperceivable by the model, capability-confounded
(slower model → less budget), and backwards (mental grind is fast, building is slow). A binding
budget requires **externalizing the work** (arithmetic hard enough to force code, or mandatory
shown work) — which reconstructs ToolWorld's "must act in a world you can't compute your way
through." v3 tests only the **pure-announced** nudge (perceived scarcity, no enforcement); a
*binding* budget is the open design problem.

**Final results** ($17; `runs/creator_batch_20260622_010104/`, 611 items × 3; figure
`figs/creator/fig_creator_batch.png`): Recognition `P(built|ask)` = **0.12 / 0.12 / 0.21**
(Haiku/Sonnet/Opus) — low, flat-to-slightly-rising, **no inversion**, and far below v1's
0.56/0.70/0.80. Curiosity 0.85/0.53/0.90 (U-shaped); Efficiency(all|ask&built) 0.10/0.29/0.35;
Solve(all 8) 0.37/0.37/0.52; Grind 0.36/0.36/0.45 ≈ solve (building contributes ~nothing). The
v1→v3 recognition collapse shows v1's high recognition was mostly the explicit function *cue*;
un-cued spontaneous building is ~0.12, and the **pure-announced budget barely lifts it** (K=8
pilot ≈0). A non-binding announced budget does not create the obstacle recognition needs —
matching the prediction. A *binding* externalized-work budget remains the open lever.

**Open (for the budget design):** a *binding* externalized-work budget — e.g. require shown
computation and budget total output, or force code execution and budget executed-work — vs.
accepting the boundary finding that CREATOR (solvable by pure reasoning) can't host the
obstacle the inversion needs.
