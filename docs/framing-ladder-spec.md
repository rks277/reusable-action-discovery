# Framing-Ladder Preregistration (Haiku, four rungs)

**Status (2026-07-09): RUN COMPLETE.** All four rungs measured on the identical 6 paired seeds
(2000–2005), mechanically clean (0 unresolved decisions in every rung). See §7 for results. Two
harness bugs were found and fixed mid-run (§3.1) before the numbers below should be trusted — read
that section before citing anything here.
This is the locked spec for the plan `economic_framing_surface_f7ac96ba` ("Four-Rung Framing
Decomposition"). It supersedes that plan's prose where the two documents differ — the changes below
came out of an adversarial review of the plan itself (2026-07-09) and are pre-registered *before*
any new rung is run. Deviations found after running get logged here, not silently edited in.

---

## 1. Estimand and causal limit (read before interpreting any result)

**Call this a coarse framing ladder, not a factorial decomposition.** A true factorial design would
vary modality and content-realism independently (2×2: {text, tool-call} × {abstract, real-problem})
plus isolate construction as its own factor. This ladder is a *linear chain* of four points, each
adjacent step changing more than one thing at once. It licenses exactly one class of claim: **the
earliest adjacent step with a large (≥30pp) first-sight jump is the primary suppressor**, not a
decomposed effect size for each factor.

**What each step actually bundles:**
- **R0→R1** (free-text → tool-call, same abstract balls): the cleanest step — only decision modality
  changes. Genuinely close to a controlled comparison.
- **R1→R2** (abstract tool-call → real-problem tool-call): bundles real coding-problem *content* with
  the *action labels* (`keep`/`pass` → `claim_solver`/`skip_solver`, introducing the word "solver").
  These are separable (see §4, the optional abstract `claim_solver` control) but R1→R2 alone cannot
  separate them.
- **R2→R3** (declarative claim → full script construction): bundles **three** distinct things, not
  two as the original plan stated:
  1. **construction burden** — R3 requires emitting and running code; R2 requires a zero-argument
     tool call only.
  2. **solve/investment conflation** — in R3 the current-problem answer and the future-reuse
     investment are the same action (`write_script` both answers now and pays forward); in R2 they
     are already separated (`claim_solver` invests without solving anything itself this turn,
     `skip_solver` scores zero and moves on).
  3. **hand-solving escape valve (newly identified, 2026-07-09 review)** — R3 gives the model a third
     option absent from every earlier rung: solve the current problem by hand, with no investment
     decision at all. R2's only two actions are `claim_solver` (invest) and `skip_solver` (score
     zero, no partial credit). A large R2→R3 jump could therefore reflect the model reaching for
     hand-solving *specifically because R2 offers no such option*, independent of any allocation
     reasoning. This axis is not tested by anything in the current ladder and is not separable
     post-hoc — flag it, do not adjudicate it, in any R2→R3 write-up.

**Consequence:** if the data show no single adjacent step clearing 30pp, report "cumulative/multifactor
interaction" (per the original plan's own fallback) and do not attempt to further attribute the
R2→R3 step without the deferred construction-only rung (`docs/online-tool-investment-plan.md`-style
future work; not in scope for this pass).

---

## 2. Rung definitions (unchanged from the original plan)

Fixed protocol: Haiku (`claude-haiku-4-5`) only; A2 (`N=8` disclosed, `T=60`, `B=3`); identical
uniform-hard eight-family streams (`n_hot=3`, `hot_share=.85`, `g=1`, `MAG=100`); seeds `2000–2005`
(six paired seeds; cached R0/R3 restricted to the same six for the primary comparison, full 12
reported for endpoint context — see §5).

- **R0 — abstract text decision (cached).** `urn_session.py`, free-text `DECISION: KEEP/PASS`.
- **R1 — abstract tool-call decision (new Haiku runs; harness ready).** `urn_tool_session.py`,
  zero-argument `keep`/`pass` tool calls, `--tool-choice required`.
- **R2 — real-problem declarative claim (harness not yet built).** `claim_solver_session.py`: real
  numeric problem text/inputs from the same stream; exactly two zero-argument tools, `claim_solver`
  (spends one of `B` claims, scores the current problem correct, auto-scores every future problem of
  that hidden type) and `skip_solver` (scores zero, advances). No code, no answer submission.
- **R3 — full reusable-script construction (cached).** Historical Haiku A2 results from
  `arm_a1_announce.py`: optional `write_script`, `run_script`, and hand-solving over the same streams.

---

## 3. Unresolved-decision handling (2026-07-09 fix, applies to R0/R1/R2)

**What changed.** A code check during this review found `urn_session.py` (R0) was silently masking
transport/API failures: on an exception it substituted the literal string `"DECISION: PASS ...
[error ...]"`, which `parse_decision` then read as an ordinary tag-parsed PASS — indistinguishable
from a real decision, and *not* counted in `unparsed`. `urn_tool_session.py` (R1) already flagged
transport failures via `unparsed`, but collapsed them into the same `"default"` bucket as "the model
chose not to call a tool" — a different failure mode with a different implication for the ladder.

**The fix (`urn_common.call_with_retry`, shared by R0 and R1, inherited by R2 when built):**
- A transport/API exception is retried up to `TRANSPORT_RETRIES=2` extra times (linear backoff)
  before being treated as failed. This is retried because it is an infra failure carrying no
  information about the model's disposition.
- A **model** decision failure (no tool call made, unknown tool name, no KEEP/PASS token in free
  text) is still **never retried** — retrying that would mean re-prompting the model, which breaks
  the no-retry contract that keeps the text-vs-tool-call modality comparison fair (documented
  previously in `urn_tool_session.py`).
- The two failure modes are now tagged distinctly in the transcript: `how="error"` (transport,
  retries exhausted) vs. `how="default"`/`"unknown"`/`"both"` (genuine model-side failure). Both are
  counted in the same `unparsed` total, so the existing rung-level falsification bar is unaffected —
  but a reader can now tell *which* kind of failure occurred by reading the transcript.
- **Explicitly rejected:** invalidating an entire session on any single unresolved turn. At `T=60`
  decisions/seed and only six seeds, one incidental transport blip would cost an entire paired seed
  for free — a worse trade than the bias it prevents. The existing rung-level bar (**>10% unresolved
  decision turns invalidates that rung**, from the original plan) stands as the confirmatory
  threshold; this fix makes what's inside that 10% legible, it does not tighten it to zero.

**Verified (2026-07-09):** both `run_episode` (R0) and `run_episode_tool` (R1) tested against fake
clients that (a) fail once then succeed — retry absorbs it silently, `unparsed` stays 0 — and (b)
fail permanently — flags `how="error"` and increments `unparsed` for every affected turn. See
`urn_common._selftest()` for the retry-primitive-level test.

### 3.1 Two more bugs found and fixed during the real run (2026-07-09)

**Bug A — `tool_choice` silently dropped on Anthropic.** `lomekwi/raw_chat.py`'s `chat_tools` never
passed `tool_choice` through to the Anthropic API call, so a caller's `"required"` was silently
ignored and Haiku always ran under unenforced `"auto"`. Fixed (translates to Anthropic's
`{"type": "any"}`/`{"type": "auto"}`), verified live. **This surfaced a second-order confound worse
than the bug itself:** once actually enforced, `tool_choice="required"` turned out to suppress ALL
deliberation text for Haiku — 0/29 real R1 decision turns had any content under enforcement, even
though the prompt explicitly invited brief reasoning. R0 gets substantial deliberation; R1 under
enforcement got none — conflating "modality" with "presence of a thinking window." Resolution:
`tool_choice` now defaults **per provider**, not globally — `"required"` for local/Ollama models
(where it fixes a real format-lock failure, unrelated to Haiku) and `"auto"` for Claude models
(Haiku never needed forcing on R1's abstract content — 0 unparsed either way — and `"auto"` restores
deliberation). `--tool-choice` remains overridable per invocation.

**Bug B — R2's prompt let real problem content win.** Under corrected `"auto"`, R1 was clean, but R2
hit two successive, genuinely distinct failure modes, both now fixed:
- *Hand-solving derailment* (23% unresolved on one batch): showing real, solvable arithmetic inside
  an elaborate ball-game-style narrative (the original `render_system_claim`, paragraph-parallel to
  `render_system`) let Haiku's compute-first reflex win — full step-by-step modular-arithmetic traces
  in the transcript, no tool call ever reached, `max_tokens` exhausted mid-computation. Happened on
  the FIRST decision turn of some sessions, so not an attention-dilution-over-a-long-session effect.
  Fix: **the prompt wording no longer mirrors R0/R1's narrative** (that parallelism was never the
  falsifiable requirement — only the decision *mechanics* are, and those are independently verified
  by `ladder_parity_selftest.py`). `render_system_claim` is now terse and states the non-solving
  constraint directly; the per-problem message repeats "(decide only — do NOT compute an answer)" at
  the point of temptation on every turn, not just once upfront.
- *Token-budget truncation* (11% unresolved after the prompt fix, concentrated in one seed): once
  hand-solving stopped, Haiku instead did legitimate, on-task reasoning — restating its running tally
  of problem types seen so far before each decision — but re-derived the FULL cumulative history from
  scratch every turn rather than referencing prior context economically, growing linearly with turn
  count until it hit the 512-token cap mid-reasoning, before ever emitting a tool call (confirmed:
  `output_tokens` pinned at exactly 512 across every affected turn). Fix: raised R2's `max_tokens` to
  1500 (R0/R1 keep 512 — their content doesn't grow this way) and added one clause asking for brief,
  relevant-only reasoning instead of a full re-list. Verified: 0 turns hit the new cap; 0/18 turns
  unresolved across all 6 seeds afterward.

Both bugs were caught by re-checking mechanical validity after seeing an unexpected number
(unresolved rate, not first-sight/lateness) — consistent with the stop rule: never intervene on a
behavioral outcome, always intervene on a mechanical-validity failure.

---

## 4. Additional experiment on deck (not yet run)

**Abstract `claim_solver` control**, inserted conceptually between R1 and R2: same abstract balls as
R0/R1, but the decision is elicited via `claim_solver`/`skip_solver` tool calls instead of
`keep`/`pass`. This separates action-*label* semantics (does calling something "claim a solver" vs.
"keep a ball" shift behavior on its own?) from the R1→R2 step's simultaneous change to real-problem
content. Estimated cost $1–3, only worth running once R2 exists and only if it fits under the $15 cap
already committed to R1+R2.

---

## 5. Confirmatory analysis (unchanged from the original plan, restated for completeness)

For each seed × rung: first-sight fraction among realized keeps/claims/builds; mean lateness;
commitments per seed and zero-commit incidence (never silently dropped); trap allocations;
malformed/unknown/multiple/no-call/error decisions; native task outcome as a secondary diagnostic.

1. Verify cached endpoints on seeds 2000–2005: R0 first-sight ≤50%; R3 ≥85%.
2. Compute adjacent first-sight changes R1−R0, R2−R1, R3−R2.
3. A large discontinuity, defined **before seeing data**, is **≥30 percentage points**.
4. Earliest large adjacent change localizes the primary suppressor (R0→R1 = modality; R1→R2 =
   real-problem/coding semantics, confounded with action-label wording per §1; R2→R3 = construction
   burden, solve/investment conflation, **or** the hand-solving escape valve per §1 — do not
   adjudicate among these three without the deferred construction-only rung). No single ≥30pp step:
   report cumulative/multifactor interaction.
5. Report both the full 12 cached R0/R3 seeds (endpoint stability context) and the six matched seeds
   (primary paired comparison) — see original plan §"Deliverables."
6. Seed-clustered bootstrap intervals and all seed-level points; with six seeds, emphasize magnitude
   and consistency over p-values.

**Falsification criteria (unchanged):** >10% unresolved decision turns in R1 or R2 invalidates that
rung. R2 must be byte-structurally parallel to R0/R1 apart from problem rendering and tool
names — **verified** (2026-07-09) by `ladder_parity_selftest.py`: identical scripted commit/decline
sequence on an identical hand-built stream produces identical commitment positions, collected
totals, budgets, and turn counts across all three rungs. This test exercises skip, first-sight
claim, delayed claim, auto-solve, and budget exhaustion — it does not (and cannot) verify parity on
the live-model failure paths (malformed/unknown/error), which are covered separately by each
module's own self-test instead.

---

## 6. Run order, cost, and stopping

1. **DONE (2026-07-09).** Self-test R2 transitions (skip, first-sight claim, delayed claim,
   auto-solve, budget exhaustion) and the R0/R1/R2 cross-rung parity check — see §3.
2. **DONE (2026-07-09).** Smoke, then full 6-seed runs of R1 and R2, seeds 2000–2005. This took four
   passes, not one — §3.1's two bugs were each caught by an unexpected unresolved-decision rate on
   an early pass, fixed, and rerun, per the stop rule (mechanical validity, never behavioral outcome,
   triggers a stop). Final pass on both rungs: 0/18 unresolved decisions each, byte-identical streams
   across all four rungs on all 6 seeds — see §7 for results.

**Cost note (corrected three times, 2026-07-09):** the original "$7–10 expected, $15 cap" estimate
was never a measurement (it was the sum of the harnesses' `EST` spend-guard constants, carried over
uncalibrated from `arm_a1_announce.py`'s much heavier R3 sessions — see prior correction for detail),
and it also didn't anticipate needing four passes to reach mechanical validity (§3.1). Actual total
spend across every pass, including the three discarded-and-rerun ones, plus the n=12 extension
(below): **≈$1.57**. All of it is far under the original estimate and nowhere near the $15 cap, which
remains unchanged as a safety ceiling.

---

## 7. Results (locked run, 2026-07-09; extended to n=12, 2026-07-09)

All four rungs, identical seeds, A2 (N=8 disclosed), post-§3.1 fixes. R0/R3 were already cached out
to 12 seeds at zero marginal cost; R1/R2 were extended from 6 to 12 seeds (seeds 2006–2011, $0.23 +
$0.13 = $0.36) specifically to firm up the noisiest part of the n=6 result (the R0/R1/R2
comparisons) — the confirmatory R2→R3 finding was never in doubt at n=6, this was about precision on
the null result, not about re-checking the headline. Seeds 2006–2011 passed the same mechanical
checks as before (byte-identical streams across all four rungs, 0/18 unresolved decisions in R1 and
R2 both) before being trusted.

| | R0 (text) | R1 (tool-call) | R2 (declarative claim) | R3 (script construction) |
|---|---|---|---|---|
| first-sight (n=12) | 28% (10/36) | 17% (6/36) | 39% (14/36) | 100% (34/34) |
| lateness (n=12) | 1.194 | 2.167 | 0.833 | 0.000 |
| first-sight (n=6, for reference) | 28% (5/18) | 22% (4/18) | 33% (6/18) | 100% (17/17) |

**Adjacent first-sight deltas at n=12:** R1−R0 = **−11pp**, R2−R1 = **+22pp**, R3−R2 = **+61pp**.
(At n=6 these were −6pp / +11pp / +67pp.)

**Reading, against the pre-registered ≥30pp bar (§5):** unchanged from n=6 — only R2→R3 clears it.
Doubling the sample moved the R1→R2 delta up (11pp → 22pp), which is worth tracking if this is
extended further, but it still falls short of the pre-registered bar, and R0's own first-sight
percentage is IDENTICAL at n=6 and n=12 (28%) despite drawing from a different pool of realized
commitments (10/36 vs 5/18) — a reassuring internal consistency check, not just an extension.
**The entire jump to eager, open-loop behavior remains concentrated at the final step — full script
construction — not at any earlier one.** Per §1's pre-registered limit, this localizes the suppressor
to the R2→R3 step as a whole, but **cannot** further attribute it among that step's three bundled
factors (construction burden / solve-investment conflation / the hand-solving escape valve
identified in §1) without the deferred construction-only rung. That follow-up experiment remains the
one piece of information this ladder cannot produce on its own, and is now the most information-dense
next step available (see the project's stop/continue review — the economic response surface is the
other flagship candidate for remaining time).

**Caveat:** n=12 per rung is still small for the R0/R1/R2 comparisons specifically (R3's own gap over
any of them is large enough not to need more precision). Treat the R0/R1/R2 ranking as a firmer null
than at n=6, not a fully resolved one — the R1→R2 delta's movement toward (but not across) the 30pp
bar under a 2x sample is the one number in this table worth re-checking if the ladder is extended
again.

---

## 8. R2c — the construction-only rung, run (2026-07-09)

The one open question §7 left — which of R2→R3's three bundled factors (construction burden /
solve-investment conflation / hand-solving escape valve) drives the jump — now has an answer for one
of the three. `claim_solver_code_session.py` (R2c) takes R2's binary claim/skip loop **unchanged**
(same budget, streams, current-plus-future payoff, absence of a hand-solving option — verified by
extending `ladder_parity_selftest.py`, which now checks all four rungs at once) and adds exactly the
one thing R2 doesn't have: `claim_solver` now requires a real `code` argument (`def solve(inputs:
dict) -> float`) instead of being a zero-argument call.

**Deliberate design choice, at the user's explicit direction:** payoff is unconditional on code
correctness — calling `claim_solver` with any non-empty code scores current-plus-future exactly like
R2, regardless of whether the code works. The question this rung answers is narrowly "does the model
become eager just because it has to write code," not "does a correctness incentive make it eager" —
gating payoff on correctness would reintroduce a version of solve/investment conflation, which is
exactly what R2c is built to avoid. Code correctness is measured anyway, but strictly as a post-hoc
diagnostic (`grade_claimed_code`, reusing `creator_exec._run`/`_parse_answer` and
`creator_heldout.has_solve` — the same sandboxed-execution path `session_state.op_run_script` uses,
not a reimplementation) that never touches scoring. The one new mechanical wrinkle this required:
`claim_solver` is no longer zero-argument, so R0–R2's "no malformed-arguments failure mode" property
doesn't carry over — `resolve_code_claim_decision` adds a fifth outcome, `how="malformed_args"`
(missing/empty/non-string `code`), which does not register a claim and counts as unresolved, same as
any other unparsed turn.

**Result (12 seeds, 2000–2011, A2, mechanically clean — 0/36 unresolved, byte-identical streams to
every other rung):**

| | R1 (tool-call) | R2 (declarative claim) | **R2c (code required)** | R3 (script construction) |
|---|---|---|---|---|
| first-sight | 17% (6/36) | 39% (14/36) | **100% (36/36)** | 100% (34/34) |
| lateness | 2.167 | 0.833 | **0.000** | 0.000 |

**R2c matches R3 exactly.** Requiring code, alone — with no correctness stakes, no hand-solving
option, and the identical guaranteed payoff structure R2 already has — is sufficient to produce full,
first-sight eagerness. This answers §7's open question for the construction-burden factor
specifically: **construction burden alone accounts for the entire R2→R3 jump.** Solve/investment
conflation and the hand-solving escape valve were never needed to explain it; R2c isolates
construction burden from both and still reproduces R3's eagerness in full.

**The code-correctness diagnostic adds a second, unplanned finding.** Only 2% of tested instances
(6/392) were correct, and 31/36 claims were "degenerate" (code ran, 0% correct across every instance
of its type). Reading the actual code (not just the aggregate number) shows this is not junk or
placeholder-stuffing: Haiku writes real, structurally-correct solutions to the *specific instance it
saw* — e.g. a correct xorshift implementation — but hardcodes that instance's incidental parameters
(shift amounts, repeat counts) as literals instead of reading them from `inputs`, so the code fails on
every other instance of the same type with different parameters. It solves the problem in front of it
and does not generalize, even though generalizing was the explicit, stated point of `claim_solver`.
That is a second, independent piece of evidence for the same underlying disposition the whole ladder
is about: presented with a concrete problem, the model optimizes for the problem in front of it, not
for the future-reuse structure it was explicitly told the action serves.

**Cost:** $0.02 (2-seed smoke) + $0.10 (remaining 10 seeds) = **$0.12** for all 12 seeds — self-tests
and the parity extension caught the new `malformed_args` mechanic before any live call, so unlike R1/R2
this rung needed only one pass.
