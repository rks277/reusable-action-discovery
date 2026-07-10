# GPT Cross-Family R0–R2c Replication — Preregistration + Implementation Plan

**Status (2026-07-10): SPECIFIED, NOT RUN.** No live API calls are authorized by this document.
The intended subjects are **GPT-5.4-mini** and **GPT-5.6**, subject to confirming their exact API
model IDs before any call. This is the minimum cross-family breadth replication required by
`paper-structure-outline.md` §9.2: paired R0 versus R2c on a non-Claude family with demonstrated
abstract allocation competence.

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

1. **Smaller GPT point:** `gpt-5.4-mini`.
2. **Newest frontier GPT point:** GPT-5.6; replace this label with the exact API ID returned by the
   provider before the first smoke. Do not silently substitute another model.

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
- OpenAI reasoning effort: one fixed value across both arms, recorded in every session. The intended
  default is `low` unless the exact model rejects it.
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

### 6.1 Why the current driver must not be used unchanged

`run_economic_surface.py` currently:

- knows prices only for Claude, so GPT usage is reported as `$0`;
- treats every non-Claude model as “local,” which also changes concurrency and R2c tool choice;
- uses an uncalibrated hardcoded `EST=0.10`;
- gates only the start of a session and has no per-session dollar circuit breaker;
- defaults to concurrency greater than 1; and
- cannot directly analyze a non-Haiku run directory.

`RawChat.chat()` gives GPT reasoning models extra completion headroom and a configurable reasoning
effort, while `RawChat.chat_tools()` currently does neither. That arm asymmetry must be repaired
before the smoke.

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

- [ ] Confirm exact GPT-5.4-mini and GPT-5.6 API IDs.
- [ ] Separate provider classification from the current `MODEL_KEY not in CLAUDE` “local” heuristic.
- [ ] Lock R2c `tool_choice="auto"` explicitly for this replication.
- [ ] Give OpenAI `chat()` and `chat_tools()` the same reasoning-effort handling and sufficient,
      recorded completion headroom.
- [ ] Add exact GPT cost accounting; reject unknown pricing.
- [ ] Add physically enforced per-session and global spend caps.
- [ ] Add a first-unit >1.5× circuit breaker.
- [ ] Keep all execution serial (`conc=1`).
- [ ] Parameterize analysis by run directory/model instead of reading
      `runs/economic_surface_haiku` unconditionally.
- [ ] Emit config, prompt, stream, and model hashes plus token/cost fields into every session.
- [ ] Run all zero-cost self-tests:
  - `economic_surface`
  - `run_economic_surface --selftest`
  - `claim_solver_code_session --selftest`
  - `ladder_parity_selftest`
- [ ] Dry-run command construction without an API key or network call.
- [ ] Stop for Approval A.

Expected artifact layout:

```text
runs/economic_surface_gpt-5.4-mini/{R0,R2c}/B_3/K_0/seed_<seed>/
runs/economic_surface_<exact-gpt-5.6-id>/{R0,R2c}/B_3/K_0/seed_<seed>/
```

Each seed directory must contain the canonical `stream.json`, complete `session.json`, and sufficient
config metadata to reproduce the request settings exactly.

---

## 8. Locked command shape

The final commands may gain safety flags during implementation, but their scientific arguments are
locked:

```bash
# One serial R0 calibration unit.
PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.run_economic_surface \
  --model <exact-model-id> --seeds 2000 --cells R0:3:0 --conc 1 \
  --cap-usd <approved-global-cap> --unit-cap-usd <approved-unit-cap>

# One serial R2c calibration unit.
PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.run_economic_surface \
  --model <exact-model-id> --seeds 2000 --cells R2c:3:0 --conc 1 \
  --cap-usd <approved-global-cap> --unit-cap-usd <approved-unit-cap>

# Approved R0 panel completion; cached seed 2000 is reused only if configuration-identical and valid.
PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.run_economic_surface \
  --model <exact-model-id> --seeds 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 \
  --cells R0:3:0 --conc 1 --cap-usd <approved-panel-cap> \
  --unit-cap-usd <approved-unit-cap>

# Approved R2c panel completion, only after the full R0 competence gate passes.
PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.run_economic_surface \
  --model <exact-model-id> --seeds 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 \
  --cells R2c:3:0 --conc 1 --cap-usd <approved-panel-cap> \
  --unit-cap-usd <approved-unit-cap>
```

`--unit-cap-usd` is a required implementation item and does not exist at specification time.

---

## 9. Reporting language

Allowed:

- “GPT-5.4-mini/GPT-5.6 demonstrated abstract reserve and became more eager when code construction
  was required.”
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
