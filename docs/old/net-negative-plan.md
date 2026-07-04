# Plan: The Capability-Graded Verify/Rewrite Spiral — Why Iterative Code Goes Net-Negative, and Predicting the Optimal Code-Budget

*Title note: earlier framing led with "capability-indexed"; corrected 2026-06-29 to lead with the
spiral/iteration axis, since capability-spanning and budget-allocation are largely covered by To-Call
([2605.00737](https://arxiv.org/abs/2605.00737)). See the positioning correction below.*

*Draft 2026-06-27 (rev. 2).* Builds on [AIME 2026 — Script-Budget Disposition](aime-disposition.md),
[Tool-Disposition Benchmark](disposition-bench.md), and the
[Related-Work / Direction Map](budget-disposition-related-work.md). This is a research plan, not
results — nothing here has been run.

## Thesis (one sentence)

> The marginal value of the k-th code/script attempt on a problem is a **predictable function of
> (model capability × problem computation-necessity × token-budget state)**; models mis-allocate
> code because they act on *perceived* rather than *true* marginal value, capability-dependently —
> and from this surface we can **predict the optimal code budget** that maximizes solve rate.

Two design commitments came out of working through the idea and should not be lost:

1. **The control variable is the *budget/retry depth*, not a per-call yes/no.** One script rarely
   hurts; harm is the 5th rewrite (Haiku's `bug_paths_v2…v5` spiral). So the predictor's target is a
   **marginal-value curve `u(k)`**, and the optimal per-problem allowance `n*` is where it crosses
   zero. The session budget cap is **`B* = Σ_p n*(p, m)`** — *composed*, not grid-searched.
2. **Labels need both-way runs; features must not.** `u(k)`/`n*` are measured at *training* time by
   actually running code. The whole bet is that `n*` is **predictable at test time from code-free
   features** (capability scalar + text necessity proxy + budget state). If it isn't, the predictor
   degenerates to a lookup table — so that predictability is tested *first* (E4a), as a gate.

This reframes our budget result (now attributed to *awareness*, per BATS) as one consequence of a
deeper, predictable mis-allocation. **The axis no prior paper owns is not capability** (To-Call spans
3B–120B) **but the iterative-code retry-DEPTH and the verify/rewrite spiral** — which atomic-call
work (To-Call) and forced-refinement work (self-refine) cannot produce. Capability enters as a
*moderator* of the spiral, not as our novel contribution.

## Scope — what generalizes, and what doesn't (read this before believing the title)

A *general, deployable* "any model + any task → optimal budget" predictor would need breadth (many
models × task types × difficulties × domains) we **cannot** collect. We do not claim it. The
generality of this work lives at three different levels, and each claim is scoped to what its data
can support:

| layer | claim scope | what backs it |
|---|---|---|
| **Mechanism** (capability × necessity × awareness; spiral-not-code; over-tooling gap) | **General** — explanatory; should transfer; resolves the literature tensions | the capability law (E1) + transcripts (R4) + E0 |
| **Predictor of B\*** (E4/E5) | **Within contest math, across a capability ladder** — "the mechanism is *actionable*," validated on held-out models/problems (E4a OOD). NOT a universal oracle | a *deep, single-domain* dataset (feasible), not a broad one |
| **Benchmark** (instrument) | the **bridge to generality** — lets *others* test the mechanism/predictor in new domains | it's the transform + metric, not the data |

So: **promise generality of the *mechanism*; demonstrate predictability *within a deep single-domain
dataset*; ship the *benchmark* as the path others use to extend it.** "Predictable" here means
"generalizes to held-out models and held-out problems *within this regime*" (the E4a test) — not
"works on any task." If even the within-regime predictor fails (E4a), we fall back to the descriptive
mechanism, which stands on its own.

## Why this is publishable (positioning)

| Prior work | What they did (verified) | What they leave for us |
|---|---|---|
| To-Call ([2605.00737](https://arxiv.org/abs/2605.00737)) | necessity/utility/affordability; per-instance utility estimators that beat always-call; **6 models 3B–120B**; **budget allocation across instances** (top-K under a K-call budget). Tool calls **atomic** (one web search/instance) | **per-MODEL estimators (no capability-generalizing predictor); ATOMIC calls (no retry-DEPTH / iteration); web-search only (no code → no spiral).** NOT "capability-blind" and NOT "no budget" — see correction note below. |
| BATS ([2511.17006](https://arxiv.org/abs/2511.17006)) | budget *awareness* (not size) drives tool-use efficiency | no per-retry prediction; no code; no capability axis |
| Not-All-Reasoners ([2410.01748](https://arxiv.org/abs/2410.01748)) | **single-shot** code helps, **small** models most | only single-shot (no iteration/spiral); contradicts our "iterative all-code hurts the weak" → reconciled by protocol (single-shot vs iterative) |
| Self-refine / Reflexion / Self-Debug | accuracy improves with refinement/debug **rounds** (a retry-depth axis) | not under a shared budget; not *disposition* (forced refinement, not the model choosing); no capability-graded spiral |
| Understanding-TIR ([2508.19201](https://arxiv.org/abs/2508.19201)) | tools **strictly expand** capability (feasibility) | feasibility ≠ realized solve rate under budget |
| Overthinking scaling ([2604.10739](https://arxiv.org/abs/2604.10739), [2508.13141](https://arxiv.org/abs/2508.13141)) | optimal *reasoning tokens* vs difficulty | the *tool/code-budget* analog open |
| TIR-inefficiency ([2604.05404](https://arxiv.org/abs/2604.05404)) | over-use ↔ lower correctness (descriptive) | no mechanism, no prediction |

> **Correction (2026-06-29, verified against the paper).** Earlier drafts called To-Call
> "capability-blind" with "no optimal budget." **Both are false:** it spans 3B–120B and it allocates
> a K-call budget to top-K instances. So **"capability-indexed" and "budget allocation" are NOT our
> novelty** — To-Call has versions of both. What it lacks (atomic web-search calls): the
> **code-iteration / retry-DEPTH axis** and the **spiral**, which cannot exist in an atomic-call
> setting. Position accordingly.

**Our defensible differentiators (post-correction):** (1) **code-execution with ITERATION** — the
retry-depth axis (`n*` = how many attempts), absent from To-Call's atomic calls; (2) the
**capability-graded verify/rewrite spiral** (the mechanism: weak models over-verify, net-negative
even in isolation) — impossible in an atomic setting; (3) **single-shot vs iterative** as the knob
that reconciles 2410.01748 (single-shot, code-helps-weak) with us (iterative, code-hurts-weak);
(4) *disposition* (does the model self-regulate) vs a forced/trained controller. **Weaker/shared
with prior work (do NOT lead with these):** capability-spanning (To-Call has it), budget allocation
(To-Call's top-K), retry-depth-helps (self-refine), interior compute optimum (overthinking lit). The
honest clean slice is **the spiral mechanism + its capability-grading in an iterative-code setting.**

## Contributions (target) — labeled by scope of claim

1. **[GENERAL claim — THE HEADLINE] The capability-graded verify/rewrite spiral, and single-shot vs
   iterative as the reconciling knob.** With *iterative* code access, weak models over-verify/rewrite
   (spiral) — net-negative *even in isolation* (R4), worsening as capability drops; strong models
   treat code as deadweight. This reconciles **2410.01748 (single-shot → code helps the weak)** with
   **our result (iterative → code hurts the weak)** via the protocol knob (single-shot vs iterative),
   not a contradiction. The mechanism — spiral, perceived > true marginal value, awareness-not-tightness
   as the lever (E0) — is the explanatory core and the part that transfers. *(NB: "capability matters
   for tool decisions" is not itself novel — To-Call spans 3B–120B. The novelty is the spiral and its
   grading in an **iterative-code** setting, which atomic-call work cannot produce.)*
2. **[WITHIN-REGIME claim] The retry-DEPTH / optimal-code-budget is predictable.** An interpretable
   model of the marginal-value curve `u(k | capability, necessity, budget-state)` → `n*` and
   `B* = Σ n*`; used as a gate/cap it beats the model's own disposition, always-code, never-code, and
   a **capability-blind ablation** (capability feature removed — an internal ablation, not a claim
   that To-Call is capability-blind). **Novelty vs To-Call is the retry-DEPTH dimension** (`n*` = how
   many attempts on a problem) — they predict per-instance call/no-call and allocate a K-call budget
   across instances, but have no notion of depth (atomic calls). Scoped to **contest math across a
   capability ladder**, validated on **held-out models and problems** (E4a); evidence the mechanism is
   exploitable, **not** a universal oracle.
3. **[GENERALIZATION BRIDGE] A tool-disposition benchmark (instrument; possible standalone paper).** An
   MCP-delivered transform turning standard math benchmarks into option-not-obligation tool-use tasks,
   with a standardized framing + cost metric (see next section). This is the artifact that lets *others*
   test the mechanism/predictor in new domains — i.e., the path to generality we are *not* walking
   ourselves.

## The instrument: a tool-disposition benchmark

The experiments need many problems across a wide difficulty range; that infrastructure *is* a
contribution. Precise novelty claim (don't overstate): **there is no general, tool-agnostic transform
turning standard math benchmarks into a tool/code-*disposition* benchmark with a standardized
disposition-and-efficiency metric.** Closest priors: OptimalThinkingBench
([2508.13141](https://arxiv.org/abs/2508.13141)) (option-not-obligation, but *thinking tokens*) and
To-Call (web-search, method not benchmark).

Design requirements (two are first-class, not footnotes):
- **Framing standardization.** Disposition is extremely prompt-sensitive (our CREATOR-MCP result:
  biased prompt drove Qwen-7B 0%→100% tool use). The benchmark must fix a neutral "option" framing
  *and report `∂disposition/∂framing`* as an axis, not hide it.
- **Cost metric.** Use/extend PTE ([2604.05404](https://arxiv.org/abs/2604.05404)); no ad-hoc cost.
- **MCP is plumbing, not the pitch** — it makes tools tool-agnostic and easy to attach; the
  contribution is the transform + metric.

## Operational definitions (lock these first)

- **Isolated baselines** (per problem × model, single-turn): `a_hand` (tools off) and `a_code`
  (code-forced). Have `byhand_eval.py`; add a `code_forced_eval` twin.
- **Computation-necessity** `C(p) = a_code − a_hand` (per-model `C_m`, or averaged). High C = code
  genuinely helps a hand-solver (AIME-I P9, the 6⁶ simulation). *Used as a label, not a feature —
  see leakage note.*
- **Marginal value of the k-th script** `u(k | p, m) =` change in expected solve contribution from
  allowing the k-th attempt vs the (k−1)-th. `u(1)` ≥ 0 usually; `u(k≥2)` separates productive
  debugging from spiral.
- **Optimal per-problem allowance** `n*(p, m) =` last k with `u(k) > 0` under the token shadow-price
  (so `n*` shrinks as the budget tightens). `n*=0` ⇒ don't code; `n*=1` ⇒ one shot, no rewrites;
  `n*≥2` ⇒ productive iteration warranted.
- **Optimal code budget** `B*(m, set, T) = Σ_p n*(p, m)`, subject to token budget T.
- **Net-negative retry:** any allowed attempt with `u(k) ≤ 0` (the thing the cap should clip). Weak
  models' natural retry depth ≫ `n*` (the spiral); self-limiting models have natural depth ≈ `n*`.
- **Perceived vs true value gap:** perceived = the model's disposition (did it write/retry); true =
  `u(k)`. Mis-allocation = retrying where `u(k) ≤ 0` (and vice-versa). Capability-dependent.

> **Label-vs-feature (the central methodological point).** `u(k)`/`n*` require both-way runs and are
> the *training labels*. Test-time **features must be code-free**: capability scalar `c_m` (once per
> model), `code_competence_m` (once per model — separates "shouldn't code" from "can't code"),
> text/LLM-judge **necessity proxy** (no execution; judge from a *different* cheap model → no
> leakage), and budget state. **Do not feed true `C(p)` as a feature** (it is the label averaged →
> leakage); instead predict from the proxy and *show the fit recovers C*.

## Experiment sequence (with decision gates)

### E0 — Premise lock: announce-vs-enforce *(do this first)*

**Q:** Is the operative variable awareness (announced budget) or tightness (enforced cap)?

**Why Haiku, not Sonnet (corrected).** The test needs a model whose *natural* script count is far
**above** a tight announced value, so self-limiting is attributable to awareness and the effect has
dynamic range. Sonnet self-limits to ~2, so announcing it a tight number ≈ its natural disposition —
no perceived squeeze, ~no dynamic range, and announce≈natural can't be told apart from "it would do
that anyway." Haiku naturally writes ~12 (all-code thrash) and its *optimal* is ~5 (the mix peak,
8/15), so announcing "5" is a genuine below-natural constraint near its optimum. Haiku's script
enforcement is *also* non-binding here (its **token** cap binds before a 15-script cap), so the clean
isolation I originally wanted from Sonnet is preserved on Haiku too — for free.

**Setup (Paper I):**
- **Primary — Haiku, 2×2: announce ∈ {5, 15} × enforce ∈ {5, 15}.** Load-bearing cell
  **announce-5 / enforce-15** (told ~optimal, *not* forced): `--announce-budget 5 --budget 15`. The
  earlier "told 5 but allowed to write more" worry *is* the measurement: will Haiku obey an unenforced
  stated limit?
- **Null-control — Sonnet, announce ∈ {5, 15} at enforce 15.** Awareness should do ~nothing where
  natural ≈ announce.

**Predictions:**
- *Awareness is the lever:* Haiku announce-5/enforce-15 curbs scripts toward ~5, kills the thrash, and
  recovers mix-like solve (~8/15) **without enforcement**, vs announce-15/enforce-15 → thrash (~2/15);
  enforce adds little beyond announce (announce-5/enforce-5 ≈ announce-5/enforce-15).
- *Capability interaction:* large effect for Haiku, ~zero for Sonnet — itself a result.
- *Awareness fails (weak instruction-following):* Haiku ignores the "5" and writes ~12 anyway → the
  lever needs a capability floor (also a finding).
- *Null:* no announce effect anywhere → the budget bump was noise; re-plan.

**Gate:** awareness confirmed → predictor inputs are belief-state (announced), not the actual cap, and
the E4b gate/cap can be *delivered as an announced budget*. Null/tightness → re-plan. Cost in
[Sequencing](#sequencing--cost-approval-gated--no-reps-run-without-sign-off).

### E1 — Capability ladder: the law

**Q:** Is "all-code hurts the weak, deadweight for the strong" a monotone capability law, and where
does it cross zero?
**Setup:** 3-arm grid (all-code / mix / by-hand), Claude Haiku→Sonnet→Opus **+** OSS ladder (Qwen
3B/7B/14B/72B, Llama 8B/70B) via vLLM on the Oracle A10 box. AIME I+II. Capability proxy = isolated
`a_hand` (continuous).
**Metric:** penalty `Δ = solve(all-code) − solve(by-hand)` and net-negative-retry rate vs `a_hand`.
**Predictions:** `Δ` rises (less negative) as `a_hand` falls, crossing zero at low capability — below
it code helps (recovers 2410.01748), above it hurts (our regime).
**Falsification:** `Δ` flat in capability → no law → fall back to the descriptive over-use story.

### E2 — Computation-necessity × budget: the switch, as a dose-response

**Q:** Does the value of a code budget scale with how many problems genuinely need code?
**Setup:** label all 30 AIME problems by `C`; build synthetic 15-problem sets with code-necessary
fraction ∈ {0, 0.2, 0.4, 0.6}; mix vs by-hand per fraction × model. (Needs a custom-set loader.)
**Prediction:** `mix − by-hand` gap rises monotonically with the fraction, ≈0 at 0 (Paper II already
shows this tie); net-negative rate falls with `C`.
**Falsification:** gap doesn't track the fraction → necessity isn't the switch.

### E3 — Retry-depth sweep: the marginal curve *(the data generator for E4/E5)*

**Q:** What is `u(k)` — and is the weak-model harm code itself or unbounded iteration?
**Setup:** add `--max-runs-per-script` / `--max-rewrites` to `session_state.py` (per-problem attempt
counter). Sweep the cap ∈ {1, 2, 3, ∞} per problem × model (Claude + OSS). The solve-vs-depth curve
*is* `u(k)`.
**Predictions:** capping at 1 **recovers** weak-model all-code toward mix (harm = process thrash,
reconciling 2410.01748 → "*unbounded code-iteration* hurts weak models"); productive depth scales
with `code_competence_m` (Haiku ~1 then spiral; Opus 2–3 productive). This sweep yields the labels
`u(k)`/`n*` for E4.

### E4 — Predict `n*` (optimal retry depth) and validate generalization *(the headline)*

**E4a — the make-or-break OOD test (run before anything else in E4):** fit `n̂*` from **code-free
features only** (`c_m`, `code_competence_m`, text necessity proxy, budget state) on the E1–E3 corpus;
predict held-out problems via **leave-one-dataset-out** and held-out models via
**leave-one-model-out**. Report AUC/accuracy of `n̂*`.
- High OOD → necessity/retry-depth is text-predictable → the gate and `B*` generalize. Proceed.
- Chance-level → necessity is irreducibly empirical → **drop the gate/scaling-law application**; keep
  E1 (the law) + the mechanism as the paper.

**E4b — the gate/cap (deployable framing = sequential escalation):** the model hand-solves first
(default), then the gate decides whether to **escalate to code**, and each *additional* retry is
decided online from the previous attempt's failure signal (does the error look fixable?) — never
running code preemptively. Equivalent to enforcing `n̂*` per problem.
**Ablation:** session solve/coverage/cost under the gate vs (a) model's own disposition, (b)
always-code, (c) never-code, (d) **capability-blind** gate (features minus `c_m` = To-Call-style).
**Predictions:** capability-indexed `n̂*` beats capability-blind OOD; the cap **recovers the weak
model's solve rate** toward the frontier and Pareto-dominates the extremes.
**Falsification:** capability adds no OOD power over To-Call-style features → framing collapses to
To-Call → fall back to E0 as a standalone "awareness in the code-budget setting" result.

### E5 — Capstone: optimal budget `B*(T, c, d)` read off the surface *(within-regime, NOT a universal law)*

**Scope note:** this is the **within-regime** demonstration (contributions §2), not a universal
scaling law. We show `B*` is a *readout of the mechanism* across our sampled (T, capability,
necessity) cells and validate it on held-out cells *inside contest math* — we do **not** claim a
functional form that transfers to other domains.
**Q:** Within our regime, is `B*(T, c, d)` predictable from the fitted surface rather than searched?
**Method:** from the fitted `u(k)`/`n̂*` surface, compute **`B*(T, c, d) = Σ_p n̂*(p)`** under token
budget `T` (the token shadow-price shrinks `n*` as `T` tightens). Validate against a **sparse**
held-out grid of (T, c, d) points — not a full brute-force surface (which at σ≈1.5 would be noisy and
expensive).
**Why this is the right framing:** for self-limiting models `solve(B)` is flat because natural depth
≈ `n*`; the cap helps iff natural depth > `n*`, so `B*` prediction = predicting the natural-minus-`n*`
gap (large for Haiku, ~0 for Sonnet, modest for Opus) — the inverse-capability law on the retry axis.
**Claim:** "within this regime our predictor *predicts* the optimal budget on held-out cells" ≫ "we
grid-searched it" — and weaker than "here is a universal optimal-budget law."

## The predictive model (detail)

- **Unit:** one decision point (problem × model × budget-state); for retries, the k-th attempt.
- **Target:** `u(k)` (marginal value) → `n*` (integer optimal depth) → `B* = Σ n*`. Also the
  per-problem `Δutility = a_code − a_hand` for the binary necessity sub-question.
- **Features (code-free):** `c_m`, `code_competence_m`, text/LLM-judge necessity proxy, remaining &
  announced budgets, problems-left; for online retries, the prior attempt's failure signal.
- **Form:** interpretable (logit / shallow GBM with SHAP), **explicit `c_m × necessity` interaction**
  (that coefficient *is* the capability law); report partial-dependence heatmap over capability ×
  necessity → `u`/`n*`.
- **Eval:** OOD via LODO (new problems) + LOMO (held-out capability); calibration; **gating/cap
  uplift** on held-out models; capability-indexed vs capability-blind.

## Figures / deliverables

1. **F1 (E0):** announce-vs-enforce 2×2 — awareness is the lever.
2. **F2 (E1):** penalty `Δ` vs capability `a_hand`, zero-crossing — *the law*; overlay 2410.01748
   (weak) vs our (strong) regimes.
3. **F3 (E2):** mix−by-hand gap vs code-necessary fraction — *the switch*.
4. **F4 (E3):** solve vs retry-depth cap {1,2,3,∞} per model — `u(k)`; weak-model recovery at depth 1.
5. **F5 (E4a):** OOD `n̂*` accuracy from code-free features — *does it generalize* (the pivot).
6. **F6 (E4b):** gating Pareto (solve vs cost) recovering the weak model; indexed vs blind.
7. **F7 (E5):** `B*(T, c, d)` surface predicted vs held-out grid — *the scaling law*.
8. **Mechanism overlay:** perceived (disposition) vs true (`u(k)`) over capability × necessity.

## Risks & fallbacks

- **Necessity/retry-depth not text-predictable (E4a fails).** The whole *application* (gate, `B*`)
  dies; the descriptive law (E1) + mechanism survive as the paper. Test it first and cheaply.
- **`n*` harder to predict than binary necessity.** Predicting a curve's zero-crossing > predicting a
  sign. Mitigate with sequential escalation (predict `n*=0 vs ≥1` first, decide further retries
  online from failure signals) rather than committing to a depth up front.
- **"Theory" overclaim at small N.** Frame as *empirical capability law + validated predictor*; the
  OSS ladder upgrades "3 Claudes" to "a law" — keep it in scope.
- **OSS models differ (tokenizer, weak code).** Measure each model's `code_competence_m`; a model
  that can't write working code is a degenerate point, report as such.
- **AIME contamination / leakage.** 2026 set is fresh; for OOD hold out by *model* and by *dataset*,
  not just problem.
- **Crowding by To-Call.** Neutralized by the capability axis + retry/iteration setting + the
  optimal-budget result; if capability adds nothing (E4 falsified), fall back to E0.
- **Benchmark validity (framing).** Standardize the neutral framing and report framing-sensitivity as
  an axis rather than a confound.

## Sequencing & cost (approval-gated — no reps run without sign-off)

| Step | What | Needs | Rough cost | Gate |
|---|---|---|---|---|
| E0 | announce×enforce: Haiku 2×2 + Sonnet null-ctrl, Paper I, 5 reps | built | **~$24** (30 runs) | premise must hold |
| E1 | capability ladder, 3 arms | vLLM box (have it) | low $, high compute | law must be monotone |
| E3 | retry-depth sweep {1,2,3,∞} (= E4/E5 data) | `--max-runs-per-script`/`--max-rewrites` (~20 LOC) | ~$15 | — |
| E2 | necessity dose-response | custom-set loader | ~$30 | gap must track fraction |
| E4a | OOD predictability of `n*` (code-free features) | analysis only | ~$0 | **must generalize, else drop application** |
| E4b | gate/cap ablation | replay + small online-gate harness | ~$15 | indexed > blind |
| E5 | derive & validate `B*(T,c,d)` | analysis + sparse grid | ~$20 | predicted ≈ held-out |
| — | benchmark transform (instrument) | MCP transform + framing/cost metric | infra | (parallel; possible standalone paper) |

**E0 cost detail** (Paper I; per-run estimates at the 200k cap — runs log only `spent_tokens`, so
these assume ~55k output/run + ~15% cache uplift): Haiku ~**$0.50**/run, Sonnet ~**$1.45**/run.

| Component | cells × reps | runs | $/run | subtotal |
|---|---|---|---|---|
| Haiku 2×2 (announce {5,15} × enforce {5,15}) | 4 × 5 | 20 | $0.50 | **$10** |
| Sonnet null-control (announce {5,15} × enforce 15) | 2 × 5 | 10 | $1.45 | **$14.50** |
| **Total** | | **30** | | **~$24.50** |

Variants: **key contrast only** (Haiku announce{5,15}×enforce15 + Sonnet same, 5 reps) = 20 runs,
~**$19.50**; **both papers** ≈ 2× (~$49). Note the cost is dominated by the *Sonnet control*, and
Haiku's expected effect is huge (≈6 problems, 2↔8), so **3 reps likely suffice for Haiku** — and a
staged option is to run Haiku first (~$5–10), confirm the awareness effect, then decide if the Sonnet
null-control is worth the $14.50.

**Recommended first moves:** **E0** (locks the premise you're confident in) → **E3** (tiny code
change; produces the marginal-value data E4/E5 need *and* resolves the 2410.01748 contradiction), with
**E1's OSS ladder in parallel** on the GPU box. Then **E4a is the pivot** — if `n*` isn't predictable
from code-free features, we stop the application and ship the law. Decision gates make "go back to
low-hanging fruit" a defined branch, not a scramble.
