# GPT Cross-Family R0–R2c Replication — Preregistration + Implementation Plan

**Status (2026-07-10): BOTH GPT PANELS COMPLETE — HETEROGENEOUS WITHIN-FAMILY RESULT.** GPT-5.4-mini
passes the preregistered replication criterion; GPT-5.6 Sol passes the abstract competence gate but
does not show a reliable R0→R2c framing shift. The initial GPT-5.4-mini R0
calibration used reasoning effort `low`, but the paired R2c request established that this snapshot
rejects function tools with `low` in Chat Completions. Both arms are now locked to `none`; the
original R0 smoke is retained only as excluded calibration provenance. All four final-config
12-seed panels are complete and mechanically clean. Official model pages confirm the exact subjects as
**`gpt-5.4-mini-2026-03-17`** and **`gpt-5.6-sol`**. This is the minimum cross-family breadth replication required by
`paper-structure-outline.md` §9.2: paired R0 versus R2c on a non-Claude family with demonstrated
abstract allocation competence.

## Results at a glance

First-sight values are realized first-sight commitments divided by all realized commitments.
Paired differences use the seed as the independent unit.

| Model | R0 first-sight | R2c first-sight | Paired R2c−R0 (95% CI) | Mean lateness R0→R2c | R0 gate | Preregistered outcome | Final-config cost |
|---|---:|---:|---:|---:|---|---|---:|
| `gpt-5.4-mini-2026-03-17` | 16.7% (6/36) | 87.9% (29/33) | **+75.8 pp** [+54.5, +93.9] | 2.08 → 0.42 | Pass | **Replication passed** | $0.254 |
| `gpt-5.6-sol` | 19.4% (7/36) | 27.8% (10/36) | **+8.3 pp** [−11.1, +27.8] | 1.14 → 1.11 | Pass | **Boundary: no reliable shift** | $1.477 |

**Bottom line:** GPT-5.4-mini strongly replicates the code-framing suppression effect, while
competence-qualified GPT-5.6 Sol preserves its reserve policy in both frames. All 48 final-config
sessions were mechanically clean with 0 unresolved decisions. Combined final-config API cost was
**$1.730**.

**Scope lock:** R0 and R2c only; A2; `N=8`, `T=60`, `B=3`, `K=0`; canonical seeds 2000–2011.
No R1/R2/R3, capability calibration, hand-solving calibration, `a_hand`/`a_script` calibration,
economic charge sweep, additional vocabulary, retraining, or naturalistic tasks.

**Cost-safety lock:** `[[no-auto-reps]]`. Each exact model × arm begins with exactly one serial
seed under an approved hard unit cap. Report actual cost and obtain fresh approval before any
multi-seed paid run. Batch concurrency remains 1. The approved batch cap is used verbatim, with no
headroom. If the first completed unit exceeds its approved per-unit budget by more than 1.5×, stop
and re-approve.

---

## 1. Question and estimand

The experiment asks:

> When a GPT model demonstrates an online reserve policy in an abstract allocation problem, does
> requiring it to construct code suppress that policy and restore eager first-sight commitment?

The paired conditions have identical streams, information, budget, and guaranteed payoff:

- **R0 — abstract allocation.** Free-text `KEEP`/`PASS` decisions over colored balls
  (`urn_session.run_episode`). Keeping a color collects the current and every future ball of that
  color.
- **R2c — code-required claim.** Tool-call `claim_solver`/`skip_solver` decisions over the
  isomorphic real-problem stream (`claim_solver_code_session.run_episode_code_claim`).
  `claim_solver` requires a non-empty Python `code` argument and grants the same current-plus-future
  payoff as KEEP.

R2c has **no hand-solving option**. Code correctness is deliberately payoff-irrelevant and may be
graded only as a post-hoc diagnostic. Therefore GPT-5.6's ability to hand-solve the underlying
numeric tasks cannot determine the primary result, and no task-difficulty or solve-rate calibration
is required.

This is a test of **framing suppression conditional on demonstrated R0 competence**, not a test of
whether every GPT model possesses the underlying allocation policy.

---

## 2. Locked design

### 2.1 Subjects

1. **Smaller GPT point:** dated snapshot `gpt-5.4-mini-2026-03-17`.
2. **Newest frontier GPT point:** `gpt-5.6-sol` (the frontier Sol tier, rather than Terra or Luna).

The two points provide a within-family analogue of Haiku and Opus while adding one non-Claude
family. Results are reported by model, never averaged into a single GPT effect.

### 2.2 Environment

- Canonical streams:
  `runs/urn_haiku_n-announced/seed_{2000..2011}/stream.json`.
- Seeds: 2000–2011, paired across R0 and R2c.
- `N=8` distinct types, disclosed in both arms (A2).
- `T=60` arrivals.
- `B=3` irreversible commitments.
- `K=0`; scarcity comes from the commitment budget, not an added point charge.
- Uniform-hard eight-family pool, `g=1`, `MAG=100`, inherited from the canonical streams.
- Provider-default sampling temperature, held fixed across arms. Do not introduce `temp=0` in only
  one arm.
- OpenAI reasoning effort: **`none` in both arms**, recorded in every session. This matches the
  no-extended-thinking Claude protocol and is required because GPT-5.4-mini's Chat Completions
  endpoint rejects function tools when `reasoning_effort=low`.
- R2c `tool_choice="auto"` to match the published Haiku R2c protocol. A missing tool call is an
  unresolved decision, not permission to hand-solve. Do not switch to `"required"` after seeing
  behavioral outcomes.

### 2.3 Session count and staged order

Full target per qualified model:

- 12 R0 sessions.
- 12 paired R2c sessions.
- 24 sessions/model; 48 sessions if both GPT models qualify.

Run order is deliberately asymmetric:

1. Zero-cost self-tests.
2. One serial R0 calibration/smoke for the exact model and config.
3. One serial R2c calibration/smoke for the exact model and config.
4. Report mechanics, token usage, and actual dollars; obtain approval.
5. Complete the locked 12-seed R0 panel serially.
6. Apply the preregistered R0 competence gate.
7. Only if the model passes the gate, obtain approval and complete its paired 12-seed R2c panel
   serially.

The calibration seed may count toward the 12 only if it used the final locked prompt, model ID,
reasoning setting, token limits, tool choice, and harness code and passed every mechanical gate.
If a mechanical repair changes any of those, discard the smoke from analysis and rerun that seed
once under the repaired configuration. Behavioral outcomes never justify a rerun.

---

## 3. Mandatory truncation

Every R0 and R2c session stops making API calls immediately when all `B=3` commitments have been
spent:

- R0: `urn_session.run_episode` breaks when `budget_left == 0`.
- R2c: `claim_solver_code_session.run_episode_code_claim` breaks at the same point.
- The remaining tail is scored analytically from the fixed stream and committed types via
  `_balls_collected`; no post-budget model behavior is required for first-sight, lateness, or total
  collected payoff.

There is no full-stream arm and no `--full-stream` override in this experiment.

A model that retains unused budget must continue until `T=60`; stopping it early would censor the
reserve policy. A safety stop caused by a token, turn, time, or dollar cap makes the session
**mechanically incomplete and invalid**. Its unobserved tail must not be analytically imputed unless
the model had already exhausted all three commitments.

---

## 4. Outcomes and preregistered interpretation

### 4.1 Primary outcomes

Computed per seed, with the seed—not turns or commitments—as the independent unit:

1. **First-sight commitment proportion:** realized commitments made at class position 1 divided by
   all realized commitments.
2. **Mean commitment lateness:** mean `class_position - 1` over realized commitments.
3. **Paired framing differences:** R2c minus R0 first-sight and R0 minus R2c lateness on the same
   canonical seed.

Report pooled numerators/denominators and seed-clustered bootstrap 95% confidence intervals. Preserve
all seed-level values.

### 4.2 R0 competence gate

A model qualifies as a suppression test only if its complete 12-seed R0 panel satisfies all three:

1. Mean commitments per seed ≥2.5, ruling out apparent caution caused by blanket abstention.
2. First-sight commitment proportion ≤50%.
3. Mean balls collected ≥90% of the same-information online Bayesian comparator on the identical
   streams.

This gate is assessed once on the full preregistered R0 panel. Do not repeatedly add seeds until a
model crosses it. A failure means **abstract competence was not demonstrated in this benchmark**;
it is not evidence for or against framing suppression, and the paid R2c batch is not run.

### 4.3 Replication criterion

For a model that passes R0, call the framing dissociation replicated if:

- R2c first-sight commitment is at least 30 percentage points above R0; and
- R2c first-sight commitment is at least 80%; and
- the paired difference has the preregistered positive direction under a seed-clustered bootstrap
  95% interval; and
- both arms pass all mechanical-validity gates.

Failure to clear this bar is reported as a boundary condition. Do not rerun, alter tool choice,
increase reasoning, change prompts, or add seeds to recover the expected result.

### 4.4 Secondary and diagnostic outcomes

- Commitments per seed and zero-commit incidence.
- Balls/auto-solved problems collected.
- Regret versus the labelled same-information online Bayesian comparator, secondary because its
  symmetric-Dirichlet prior is misspecified for the fixed generator.
- Unresolved-decision rate and resolution categories.
- Input, cached-input, output, and reasoning tokens per turn/session.
- Wall-clock time and actual dollars per session.
- R2c submitted-code length and optional post-hoc correctness. Correctness never affects payoff,
  validity, inclusion, or the replication verdict.

---

## 5. Mechanical-validity gates

These gates may trigger an implementation repair and one clean rerun. Behavioral results may not.

- Exact model ID and provider response metadata recorded.
- Canonical stream assertion passes; R0 and R2c class-ID sequences are byte-identical per seed.
- Exact prompt/config hashes match within each arm across seeds.
- No transport error survives retries.
- Unresolved decisions ≤10% in each model × arm panel.
- R2c has no unknown tools, multiple decisive tools, or malformed/missing `code` arguments.
- No session hits its token, turn, time, or unit-dollar safety cap before budget exhaustion or
  natural `T=60` completion.
- Usage metadata is present on every successful turn.
- Truncation occurs only after the third valid KEEP/claim, and no later API call is made.

The one-seed smoke must inspect the raw transcript, not merely the aggregate report. In R2c, confirm
that `tool_choice="auto"` still yields one valid decision tool call and that hidden reasoning does
not consume the completion budget before the tool call.

---

## 6. Paid-run safety and approval points

### 6.1 Harness defects found and repaired before Approval A

The pre-implementation `run_economic_surface.py`:

- knows prices only for Claude, so GPT usage is reported as `$0`;
- treats every non-Claude model as “local,” which also changes concurrency and R2c tool choice;
- uses an uncalibrated hardcoded `EST=0.10`;
- gates only the start of a session and has no per-session dollar circuit breaker;
- defaults to concurrency greater than 1; and
- cannot directly analyze a non-Haiku run directory.

`RawChat.chat()` also gave GPT reasoning models extra completion headroom and configurable reasoning
effort while `RawChat.chat_tools()` did neither. The implementation checklist below records the
repairs; the zero-cost tests and dry runs passed after they landed.

### 6.2 Required enforcement

Before any paid call:

1. Confirm the exact API IDs and official prices for input, cached input, output, and any separately
   billed reasoning tokens.
2. Add accurate GPT pricing or an equivalent provider-returned dollar ledger. Unknown price must be
   a hard error, never `$0`.
3. Add a per-session token/turn/dollar cap. If reached before a valid terminal point, persist the
   partial transcript as invalid and stop.
4. Add an actual-dollar global cap checked serially before starting each new session.
5. Force `--conc 1` for calibration and every batch.
6. Set the global cap exactly to the user-approved amount.

Approval sequence for each exact model:

- **Approval A:** one R0 seed, serial, under an explicit unit cap.
- Report actual cost.
- **Approval B:** one R2c seed, serial, under an explicit unit cap.
- Report actual cost and projected 12-seed R0 cost.
- **Approval C:** finish the R0 panel serially under the exact approved cap.
- Report the competence-gate result and projected R2c cost.
- **Approval D:** only for a qualified model, finish the R2c panel serially under the exact approved
  cap.

If the first completed unit exceeds the approved per-unit budget by >1.5×, stop. Do not continue and
revise the estimate after additional units.

---

## 7. Implementation checklist

No live run begins until all boxes below are complete.

- [x] Confirm exact GPT model IDs from the official model pages:
      `gpt-5.4-mini-2026-03-17` and `gpt-5.6-sol`.
- [x] Separate provider classification from the current `MODEL_KEY not in CLAUDE` “local” heuristic.
- [x] Lock R2c `tool_choice="auto"` explicitly for this replication.
- [x] Give OpenAI `chat()` and `chat_tools()` the same reasoning-effort handling and sufficient,
      recorded completion headroom; lock both GPT arms to `none` after the tool endpoint rejected
      the initial `low` smoke.
- [x] Add exact GPT cost accounting; reject unknown pricing. Prices verified 2026-07-10:
      GPT-5.4-mini $0.75/M input, $0.075/M cached input, $4.50/M output; GPT-5.6 Sol
      $5/M input, $0.50/M cached input, $6.25/M cache write, and $30/M output below
      272K input, with the documented 2× input and 1.5× output long-context tier.
- [x] Add per-session and global spend circuit breakers.
- [x] Add a first-unit >1.5× circuit breaker.
- [x] Keep all execution serial (`conc=1`).
- [x] Parameterize analysis by run directory/model instead of reading
      `runs/economic_surface_haiku` unconditionally.
- [x] Add the locked competence-gate and paired-bootstrap analysis
      (`analyze_gpt_r0_r2c.py`) before observing GPT data.
- [x] Emit config, prompt, stream, requested/resolved model metadata, provider-returned model IDs,
      and token/cost fields into every session.
- [x] Run all zero-cost self-tests:
  - `economic_surface`
  - `run_economic_surface --selftest`
  - `claim_solver_code_session --selftest`
  - `ladder_parity_selftest`
- [x] Dry-run both exact model configurations without an API key or network call.
- [x] Stop for Approval A; user approved the single GPT-5.4-mini R0 calibration.

Expected artifact layout:

```text
runs/economic_surface_gpt-5.4-mini-2026-03-17/{R0,R2c}/B_3/K_0/seed_<seed>/
runs/economic_surface_gpt-5.6-sol/{R0,R2c}/B_3/K_0/seed_<seed>/
```

Each seed directory must contain the canonical `stream.json`, complete `session.json`, and sufficient
config metadata to reproduce the request settings exactly.

### 7.1 Approval log

**GPT-5.4-mini initial Approval A — EXCLUDED AFTER MECHANICAL REPAIR (2026-07-10).**

- Model: `gpt-5.4-mini-2026-03-17`; R0; seed 2000; reasoning effort `low`; serial.
- Approved hard global cap: $1.00; unit circuit breaker: $0.50.
- Actual cost: **$0.0078585** over 9 turns.
- Mechanical validity in isolation was clean — exact provider model returned on all 9 turns, usage
  present, 0 unresolved decisions, canonical stream hash matched, and the run truncated immediately
  after the third KEEP — but it is excluded because the paired arm cannot use the same `low` setting.
- Calibration behavior (not a gate decision): 0/3 first-sight keeps, all three keeps on second
  sighting (mean lateness 1.0), 50 balls collected.
- One pre-API attempt failed because the declared `openai` dependency was not installed; it made no
  provider request and cost $0. The dependency was installed, the identical approved command was
  retried once, and the stale partial artifact was removed.
- The first R2c request at `low` was rejected before generation and cost $0:
  Chat Completions requires function-tool requests on this snapshot to use the Responses API or
  `reasoning_effort=none`.
- **Repair:** lock both arms to `none`; do not mix endpoints or reasoning settings.
- Replacement final-config R0 calibration (`none`): **$0.01003275**, 12 turns, exact provider model
  returned throughout, usage present, 0 unresolved decisions, and clean truncation after the third
  KEEP. It made 0/3 first-sight keeps, committing at class positions 2, 3, and 4 (mean lateness 2.0)
  and collected 47 balls.
- Final-config R2c calibration (`none`): **$0.0071364**, 5 turns, exact provider model returned
  throughout, usage present, 0 unresolved/malformed/unknown decisions, and clean truncation after
  the third claim. It made 3/3 claims at first sight (mean lateness 0.0) and collected 23
  auto-solved problems.
- The one-seed paired contrast (R0 0% vs R2c 100% first-sight among commitments) is calibration-only;
  the locked analyzer correctly refuses to mark competence or replication complete before all 12
  canonical seeds exist.
- Approval C R0 panel: **COMPLETE**, 12/12 canonical seeds, total **$0.12827925** including the
  final-config calibration seed; all sessions budget-truncated, exact snapshot returned, and
  0 unresolved decisions.
- R0 competence gate: **PASSED** — 3.0 commitments/seed, 16.7% first-sight among commitments
  (8.7% eligible-turn hazard), mean lateness 2.08, and 35.58 balls/seed = 92.6% of the
  same-information online-Bayesian comparator.
- Approval D was granted only after the R0 competence gate passed.
- Approval D R2c panel: **COMPLETE**, 12/12 canonical seeds, total **$0.1253796** including the
  final-config calibration seed; exact snapshot returned and 0 unresolved/malformed/unknown
  decisions. Eleven seeds exhausted the budget and truncated; seed 2010 made zero claims and
  therefore validly ran the full 60-turn stream.
- The first Approval D pass paused after seed 2010 cost $0.0543378, above the 1.5× calibrated
  threshold. Fresh approval was obtained before running seed 2011 alone.
- Final preregistered result: **replication passed**. R0 made 6/36 commitments at first sight
  (16.7%; eligible-turn hazard 8.7%; mean lateness 2.08). R2c made 29/33 realized commitments at
  first sight (87.9%; eligible-turn hazard 59.2%; mean lateness 0.42), a pooled +71.2-point gap.
  Across the 11 seeds with defined first-sight proportions in both arms, the paired mean difference
  is +75.8 points, seed-bootstrap 95% CI [+54.5, +93.9].
- Heterogeneity is retained, not hidden: R2c seed 2007 waited (0/3 first-sight; lateness 4.33), and
  seed 2010 made zero commitments. The result supports a strong distributional framing shift, not
  an absolute per-seed law.
- Final-config API cost: **$0.25365885** ($0.12827925 R0 + $0.1253796 R2c). Including the excluded
  initial `low` R0 calibration, total billed GPT-5.4-mini cost was **$0.26151735**.

**GPT-5.6 Sol — COMPLETE (2026-07-10).**

- R0 competence gate: **passed** — 3.0 commitments/seed, 19.4% first-sight among commitments,
  mean lateness 1.14, and 37.83 balls/seed = 98.5% of the online-Bayesian comparator.
- R2c: 3.0 commitments/seed, 27.8% first-sight among commitments, mean lateness 1.11, and
  36.92 auto-solved problems/seed.
- Paired R2c−R0 first-sight difference: **+8.3 points**, seed-bootstrap 95% CI
  **[−11.1, +27.8]**.
- Preregistered replication criterion: **failed** (R2c is below 80%, the gap is below 30 points,
  and the interval includes zero).
- Mechanical validity: clean — exact provider model returned throughout all 24 sessions,
  0 unresolved decisions, and every session truncated after budget exhaustion.
- Interpretation: GPT-5.6 Sol is a competence-qualified boundary condition. It expresses reserve
  in both abstract and code-required frames under the locked `reasoning_effort=none` protocol.
  Do not turn the contrast with GPT-5.4-mini into a monotonic capability or scaling claim.
- API cost: **$1.4766065** ($0.56811425 R0 + $0.90849225 R2c).

**Combined GPT spend:** **$1.73026535** for all final-config panels; **$1.73812385** including the
excluded GPT-5.4-mini `low` calibration. Rejected pre-generation requests cost $0.

---

## 8. Locked command shape

The implemented commands and their scientific arguments are locked:

```bash
# One serial R0 calibration unit.
PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.run_economic_surface \
  --model <exact-model-id> --seeds 2000 --cells R0:3:0 --conc 1 \
  --reasoning-effort none --cap-usd <approved-global-cap> \
  --unit-cap-usd <approved-unit-cap>

# One serial R2c calibration unit.
PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.run_economic_surface \
  --model <exact-model-id> --seeds 2000 --cells R2c:3:0 --conc 1 \
  --reasoning-effort none --cap-usd <approved-global-cap> \
  --unit-cap-usd <approved-unit-cap>

# Approved R0 panel completion; cached seed 2000 is reused only if configuration-identical and valid.
PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.run_economic_surface \
  --model <exact-model-id> --seeds 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 \
  --cells R0:3:0 --conc 1 --reasoning-effort none --cap-usd <approved-panel-cap> \
  --unit-cap-usd <approved-unit-cap>

# Approved R2c panel completion, only after the full R0 competence gate passes.
PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.run_economic_surface \
  --model <exact-model-id> --seeds 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 \
  --cells R2c:3:0 --conc 1 --reasoning-effort none --cap-usd <approved-panel-cap> \
  --unit-cap-usd <approved-unit-cap>
```

---

## 9. Reporting language

Allowed:

- “GPT-5.4-mini demonstrated abstract reserve and became substantially more eager when code
  construction was required.”
- “GPT-5.6 Sol demonstrated abstract reserve but did not show a reliable framing shift, defining a
  competence-qualified boundary condition.”
- “The Claude framing dissociation replicated/did not replicate in the tested GPT model.”
- “The smaller/newer GPT model did not pass the abstract competence gate, so it could not test
  suppression.”
- “Results are heterogeneous across tested models.”

Disallowed:

- “GPT models universally fail at tool investment.”
- “The effect scales with capability.”
- “A model that failed R0 disproves the framing effect.”
- “R2c code was correct, therefore construction ability caused the result.”
- Any pooled Claude+GPT average presented as a model-family law.

The final paper updates `paper-structure-outline.md` §9.2 and the project-review canvas regardless of
outcome. Nulls and boundary conditions are retained exactly as measured.
