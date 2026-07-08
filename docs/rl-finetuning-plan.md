# RL fine-tuning plan (GRPO) — alternative to SFT bridge

Companion to `qwen-finetune-transfer-plan.md` (the SFT attempt this supersedes as the active
approach — read it first for the two SFT results and why both are stalled) and
`online-tool-investment-plan.md` (headline claim, §0 Orientation for notation). GPU box mechanics
live in `docs/box-setup.md`.

## Status (2026-07-07)

**PLANNED, not started.** No training run, no code beyond what's described here. This doc exists to
scope the approach before spending GPU time on it.

Now the **leading candidate**, not just a fallback: the corpus-fix retrain (see
`qwen-finetune-transfer-plan.md`, "Corpus-fix retrain result") and the format-only Adapter F ablation
both ruled out corpus content and urn-interference as the cause of the tool-eval collapse, leaving a
structural DAgger-compounding-error explanation that RL's on-policy rollouts sidestep for free (see
that section's item 2). The competing option is a hand-rolled iterative DAgger loop (item 1) — cheaper
to build but doesn't remove the token-level-imitation mechanism that produced the original repetition
collapse.

**Revised into two phases (2026-07-07)** after two pieces of new evidence changed the plan (see
"Phase 1" below for the full reasoning): (1) the pre-FT base model is already cleanly tool-competent
(no legibility issues at all, just eager), meaning the tool-calling *degeneracy* we've been chasing is
something SFT training introduced, not a base-model limitation — and (2) RL directly on the urn avoids
essentially the entire tool-calling-format engineering problem (no JSON tool-call schema exists in that
framing at all), while still testing a sharper, previously-untested question: does a policy the model
*discovers* via reward-seeking transfer better than one it was *imitation-trained* on (the SFT π\*
demos), independent of whether the tool-calling channel itself is fragile. **Phase 1 (below) is RL on
urn, from the base model — build this first, cheap and low-risk.** The tool-framing RL infra originally
specced (now labeled Phase 2, further down) is only needed if Phase 1's zero-shot tool-transfer eval is
inconclusive specifically because of channel fragility rather than policy quality.

## Phase 1 (2026-07-07): RL on urn, from base model

### Why this first, not RL-on-tools directly

- **The base model doesn't need fixing.** The pre-FT Qwen2.5-Coder-14B tool A2 baseline (12 seeds,
  `qwen-finetune-transfer-plan.md` Result 3 table) shows clean legibility — 0 malformed calls, 0
  unknown-tool errors, no reported collapse of any kind — at 88% first-sight / 0.125 lateness / regret
  2934±324. It's eager, not broken. Every degeneracy we've spent this investigation on (silence after
  first success, raw-JSON-without-wrapper reversion, hallucinated script names) shows up only in the
  **fine-tuned** checkpoints, never in the untouched base model. That's evidence the SFT training
  itself is what damages tool-calling robustness, not something inherent to the model or the task.
- **Urn-only RL sidesteps the tool-calling engineering problem entirely, not just partially.**
  `urn_session.py` parses decisions from plain text via a `DECISION: KEEP/PASS` regex through
  `client.chat()` — no `tools=` schema, no JSON tool-call syntax, no `<tool_call>` wrapper, no
  malformed/unknown-tool/raw-JSON-reversion failure modes, because none of those failure *shapes*
  exist in this framing. Nothing analogous to Phase 2's Decision 1 (vLLM tool-call bypass) is needed.
- **Sharper hypothesis than "does the policy transfer."** SFT already installs a near-optimal urn
  policy via **imitation** of π\*'s demonstrated trajectory (`qwen-finetune-transfer-plan.md` Result 2:
  7% first-sight, 101% of π\* balls) — and that still doesn't transfer to tools. Phase 1 asks whether a
  policy the model **discovers itself** under its own reward signal, rather than behavior-clones,
  generalizes differently — a mechanism-of-installation question the SFT experiments never isolated,
  independent of whether final urn-side behavior looks the same either way.
- **Reward simplified from the original spec** (per review — the earlier draft's `n_malformed` /
  `n_unknown` / `n_raw_json_reversion` penalty terms were pre-emptive reward-hacking guardrails, not
  something actually observed as necessary; those failure modes are tool-framing-specific anyway and
  don't apply here). The reward was originally `1 - regret / regret_eager_reference` (both terms
  pi*-relative) — **superseded 2026-07-07, see "Reward, revised" below: it's now just raw balls
  collected, no reference policy at all.**

### Reward, revised (2026-07-07): pi* dropped from the reward entirely

The original reward (`1 - regret/regret_eager_reference`) shipped, and a first pilot
(`runs/rl_urn_smoke`, 7 outer steps) ran on it — but its reward history
(`[0.558, -1.937, 0.010, 0.634, <corrupted>, 0.097, 0.097]`) never showed a clean trend, and
investigating why surfaced a problem with the reward's *design*, not just a numerical bug.
**`runs/rl_urn_smoke` is VOID — do not resume it; every subsequent run uses a fresh output directory.**

What happened, in order:

1. One episode hit `regret_eager ≈ 1e-13` (eager was already ~optimal on that stream) and
   `regret/regret_eager` blew up to ~1.57e15, dominating that step's batch mean and one GRPO group's
   advantage scaling. Patched with a `REGRET_EAGER_EPS` guard — which turned out to be a symptom, not
   the root cause.
2. Per-episode reward data showed `regret`/`regret_eager` frequently large-magnitude *negative* —
   meaning π* (`exact_dp.ExactDP`, `alpha=1` symmetric-Dirichlet belief, `cap=3`) was **losing to
   trivial baselines**, not just occasionally landing on a near-zero-regret stream.
3. Root cause (a structural mismatch, not a bug): π* is exactly Bayes-optimal for a world where the
   N-type rate vector is drawn *once* from a **symmetric Dirichlet(α)** — a smooth, continuous,
   single-draw-per-stream belief (`docs/old/same-info-optimal-dp.md` §2). The benchmark's actual
   generator (`stream_builder.build_stochastic_stream`) instead hard-assigns a **fixed discrete
   two-point split** every seed (`n_hot` types at a fixed `hot_share/n_hot` rate, the rest at a fixed
   `trap_share/(N-n_hot)` rate — only the family↔role *assignment* is randomized per seed). Being
   Bayes-optimal for the wrong prior family gives no guarantee against the true distribution; no `alpha`
   reparametrizes a continuous Dirichlet into an exact discrete two-point mixture, which is exactly why
   a partial `alpha` sweep (0.02–1.0) came back non-monotonic with no clean fix.
4. Measured the damage directly (`ExactDP` vs `pi_star.wait_k_builds(k=2)` — hand-solve a type's first
   sighting, build on its confirmed 2nd-sighting repeat, in temporal order, budget-limited):

   | condition | pi\* balls/seed | wait2 balls/seed | pi\* − wait2 | pi\* win rate |
   |---|---|---|---|---|
   | pilot default (`guarantee_trap_early=1.0`, every seed forced to have an early trap) | 39.8 | 43.0 | **−3.2** (z=−13.9) | 52% |
   | genuinely unconditioned i.i.d. sampling (~39% natural early-trap rate) | 45.2 | 45.5 | **−0.31** (z=−2.6) | 79% |

   (π* wins most *individual* seeds but loses big on the ~39% of seeds that draw an early trap, which
   drags its mean below wait2's.) **π*, as built, is not a valid "can't be beaten with the same info"
   reference against this benchmark's actual bimodal generator** — using it as both the reward's value
   term and its normalizing denominator meant the reward signal was untrustworthy independent of the
   epsilon bug.
5. **Fix: the reward no longer references any policy at all.** `rl_train.py::compute_advantages`
   already does exact group-relative advantage normalization — `(r - mean(r_group)) / std(r_group)`
   within each G-sized group of rollouts sharing one stream — so the reward's *absolute scale* never
   reaches the gradient, only its ordering within a group does. The `1 - regret/regret_eager` rescaling
   was solving a problem GRPO's own baselining already solves. Reward is now simply
   ```
   reward = balls collected
   ```
   the urn's literal, stated objective (`rl_reward.episode_reward`) — no DP, no reference policy, no
   normalization, zero dependency on `exact_dp`/`skirental_scorer`/π*. This removes the entire class of
   reference-policy-miscalibration bugs from the training loop, not just the epsilon case, and it's
   cheaper (no DP value-table calls per episode). π*/wait2/eager/clairvoyant remain valid and useful for
   **evaluation** — `rl_urn_pilot.py` prints the batch's eager/wait2 baselines alongside `mean_reward`
   each step purely for interpretability — but they no longer drive the gradient, which also keeps a
   "did RL discover reserve-then-build behavior" result from being close to tautological.

### Infra (simpler than Phase 2 — reuses existing pieces almost unchanged)

- **Rollout generation: reuse `urn_session.py` as-is.** It already supports sampling-mode decoding
  (`--temp`), already runs at `conc=8` against Ollama, and needs zero new serving-layer code — this is
  the single biggest simplification versus Phase 2 (no vLLM, no tool-call bypass, no hot-swap API).
- **Avoid the ~5-minute-per-step Ollama resync tax by batching rollouts per policy iteration, not per
  gradient step.** Ollama still needs a full merge → GGUF convert → `ollama create` cycle to pick up an
  updated adapter (same ~5 min measured in the SFT pipeline). Rather than paying that after every
  small gradient update (Phase 2's vLLM-hot-swap design), collect a **large batch of rollouts across
  many seeds under one fixed checkpoint** (an "outer step": ~20-30 seeds × G=4 rollouts each ≈ 80-120
  short episodes), then do the GRPO update(s) on that whole batch, then resync once. This amortizes the
  resync cost well and lets the pilot reuse the *exact* Ollama/GGUF pipeline already built and proven
  throughout this project — no new serving infra at all.
- **Log-probs / masking: reuse `train_lora.py`'s `build_example()` unchanged.** It masks on
  `role == "assistant"` generically; urn sessions (plain `user`/`assistant` turns, no `tool_calls`, no
  `role="tool"` turns) are actually a *simpler* case for it than the tool sessions Phase 2 targets.
- **New code needed:** just the GRPO loss/step itself (group-relative advantage from the regret-based
  reward, KL vs. a frozen base-model reference, backward pass through the local HF/PEFT model) and the
  outer-step orchestration loop (collect batch via `urn_session.py` → score via the existing regret
  report → GRPO update → merge/GGUF/`ollama create` → repeat). No new files needed in the
  tool-calling stack at all.

### Checkpointing / resumability (required, not optional)

Every outer step must leave the run in a state that survives an interruption — a dropped SSH session,
a crashed process, or the ephemeral box itself needing to be torn down mid-run (per `box-setup.md`,
these boxes have **no persistent disk**; the project's own standing lesson from 2026-07-03 is raw run
data getting lost when a box was released before syncing). Concretely:

- **After every outer step, save to a fixed path on the box, overwritten in place** (not accumulated
  as N separate snapshots — only the latest resumable state matters):
  - the LoRA adapter weights (`runs/rl_urn_pilot/checkpoint/adapter/`),
  - optimizer state (needed to resume AdamW momentum/variance properly — resuming cold loses this and
    isn't equivalent to an uninterrupted run),
  - a small `manifest.json`: last completed outer-step number, the reward-per-step history so far, and
    the RNG/seed state for the next batch of seeds to draw.
- **The orchestration script checks for this checkpoint on startup.** If present, resume from the
  recorded step number using the saved adapter/optimizer state instead of reinitializing from the base
  model at step 0. This makes a crash or disconnect mid-pilot cost at most the current in-flight outer
  step, not the whole run.
- **Split what gets synced back to the laptop by size, consistent with [[feedback-defer-before-large-pulls]]:**
  the `manifest.json` (reward history + step count, a few KB) is safe to pull after every step for live
  progress visibility and costs nothing. The adapter/optimizer checkpoint itself (likely hundreds of
  MB, possibly approaching 1GB with optimizer state included) is **not** auto-pulled after every step —
  it only needs to leave the box (a) at the end of the pilot, or (b) if the box is being deliberately
  released before the pilot finishes, matching the existing "rsync `runs/` back before releasing the
  box" convention — not as a per-step habit.
### Episode length: T=60 (not T=20) — this is not just an eval-matching preference

An earlier draft of this doc proposed short T=20 training episodes for iteration speed, "mirroring the
SFT urn demos' length" — that claim was wrong (never fact-checked against the actual corpus generator)
and the underlying idea is actively harmful here, not just a convention mismatch. `phase3_demos.py`
documents this explicitly (`N_RANGE, T_RANGE = (6, 8), (60, 80, 100)`):

> B relative to N collapses pi\*'s own optimal policy toward eager (e.g. T=20,B=2,N=6 -> 100% k=1
> commits), which would make the "treatment" corpus indistinguishable from the eager control —
> defeating the whole point of Phase 3. T>=60 with B/N roughly 0.25-0.35 reliably reserves.

At short horizons, **the exact-DP optimum itself degenerates to 100% eager** — there isn't enough
remaining runway left for waiting to pay off. Training Phase 1 at T=20 wouldn't just create a
train/eval mismatch, it would train against a reward signal where the optimal policy has no reason to
reserve at all, which would make the whole experiment meaningless for the question it's asking. The
SFT urn corpus never uses T=20 for exactly this reason (`T_RANGE = (60, 80, 100)`), and the urn eval
harness itself (`urn_session.py`) runs at `T=60` (`N, T, B, MAG, G = len(UNIFORM), 60, 3, 100, 1.0`).
Phase 1 training uses **T=60** (matching both), not T=20.

### Time estimate — measured (2026-07-07), superseding the pre-registered guess below

A real outer step (25 seeds × G=4 = 100 episodes, on the smoke-test box) measured:

| stage | measured |
|---|---|
| resync (merge LoRA → GGUF convert → `ollama create`) | ~190-220s (~3.3-3.7 min) |
| rollout collection (100 episodes, `conc=8`) | ~100-180s (~1.7-3 min) |
| GRPO update (100 per-example forward/backward passes) | ~53s |
| **per outer step, total** | **~6-8 min** |
| **pilot (5-8 outer steps)** | **~30-65 min of box time** |

Both resync and rollout came in well under the original estimate below; the GRPO update is now cheap
too (see OOM note). Original pre-measurement estimate, kept for the reasoning trail:

| stage | basis | estimate |
|---|---|---|
| rollout collection (~80-120 T=60 urn episodes at `conc=8`) | urn sessions are single-call-per-turn (vs. tool sessions' multi-call-per-problem, which take ~75-90s/episode at T=60 per this session's own `diag_temp0` timing); urn's per-turn cost is much lower (short text, no code generation) | **15-30 min** |
| GRPO update (forward+backward, a few minibatch steps over the batch) | T=60 episodes tokenize ~3x longer than the T=20 estimate this superseded; still far shorter than `mechanics_bridge`'s 60-*problem* tool sessions (each problem = several tool-call turns, not one) | **5-12 min** |
| resync (merge LoRA → GGUF convert → `ollama create`) | measured directly in the SFT pipeline logs (`run_format_only_training.sh` stage 5-6) — unaffected by episode length | **~5 min** |
| **per outer step, total** | sum of above | **~25-45 min** |
| **pilot (5-8 outer steps)** | | **~2-6 hours of box time** |

This matches [[no-auto-reps]] practice: propose, measure the first unit of cost, then scale.

**OOM bug found and fixed (2026-07-07):** the first several full-scale GRPO update attempts OOM'd at
~77-79GB despite the model+optimizer footprint being only ~15GB. Root cause: `prepare_model_for_kbit_
training` upcasts every non-4bit param (norm weights, LoRA A/B, biases) to fp32 for training stability
(standard QLoRA practice) — but `Qwen2RMSNorm`'s `weight * hidden.to(input_dtype)` and LoRA's `base_out
+ lora_B(lora_A(x))` both promote the *whole* downstream computation, including Q/K/V, to fp32 via
ordinary type promotion. fp32 Q/K/V disqualifies PyTorch's flash/mem-efficient SDPA kernels (bf16/fp16
only), silently forcing the "math" backend, which materializes a full `(seq_len, seq_len)` score matrix
in fp32 — confirmed to scale near-quadratically with sequence length (1378 tokens → 65.9GB reserved) and
blow past 79GB by ~3-4K tokens, exactly the episodes seen in practice. `train_lora.py`'s proven, default
`--backend unsloth` path never hits this (Unsloth manages LoRA compute dtype internally); `rl_train.py`'s
`load_policy_model` mirrored the untested `load_hf` fallback path instead, which inherited the bug.
Fixed in `load_policy_model` by casting all fp32 non-4bit params back to bf16 after PEFT wrapping —
verified via isolated repro up to 3671 tokens and then at full scale (100-episode batch, max 4356
tokens) with flat ~11GB memory and no OOM.

### What the pilot (5-8 steps) can and can't show — don't expect convergence

**5-8 outer steps is a feasibility/direction check, not enough data to reach anything close to
optimal.** This needs to be explicit going in, not discovered after the fact. The reason isn't episode
count so much as credit-assignment density:

- SFT reached near-optimal urn performance (`qwen-finetune-transfer-plan.md` Result 2: 7% first-sight,
  101% of π\* balls) from only 170 sessions — but imitation learning gets **dense, per-token
  supervision**: every one of a session's 60-100 KEEP/PASS decisions is directly labeled with π\*'s
  exact correct action. That's 60-100 pieces of direct feedback per session.
- GRPO gets **one scalar reward per entire episode** (the session's final regret) — 60-100x less
  feedback per unit of data than SFT got. The model has to infer which of ~60 decisions in an episode
  drove a good or bad outcome from a single end-of-episode number, which is fundamentally more
  sample-hungry than being told the right action at every step. 5-8 gradient updates is also just a
  small absolute number for any policy-gradient method — real GRPO/PPO runs typically need tens to
  hundreds of updates before showing a converged behavior shift, even with a clean reward like this one.
- The base-model warm start (deliberately not starting from an SFT checkpoint, to keep "did RL discover
  this cleanly" uncontaminated by imitation) costs sample efficiency on top of this — no head start on
  the policy at all, purely reward-driven exploration from scratch.

**So the pilot's actual success criterion is: does reward trend upward across the 5-8 steps, clearly
above the noise floor** (this is also why the "reward signal variance" pre-pilot risk check matters —
if per-group variance swamps any real signal, 5-8 steps won't show a trend even if the direction is
right) — not whether the resulting policy matches π\*. A clear upward trend is the evidence needed to
justify a **separate, larger convergence run**, proposed and costed on its own after seeing the pilot's
actual learning curve rather than guessed now — rough order of magnitude, scaling from the pilot's
measured per-step cost, plausibly 30-50+ outer steps (~15-35+ hours of box time), sharpened once real
data exists. If the pilot shows no trend (or a downward one), that's also a real result — it would
mean either the reward/gradient wiring has a bug, or this reward signal needs more shaping (e.g. G>4,
or a per-decision rather than per-episode signal) before more box time is worth spending.

### Falsifiable check: did RL-on-urn preserve the base model's tool-calling legibility?

The hope stated when this phase was proposed — that training only ever touching the urn framing
shouldn't degrade tool-calling — is plausible but **not guaranteed and not yet checked**: a LoRA
adapter updates the same shared weight matrices (attention/MLP projections) that both the urn and tool
generation paths flow through, so there's no architectural guarantee that urn-only gradient updates
leave tool-calling behavior untouched, even though the training data never shows the model a single
tool call. This needs to be verified empirically, not assumed. Concretely: after Phase 1 training,
run the **same zero-shot tool eval already built** for the SFT experiment, and read it in the same
two-stage order this project always has —

1. **Legibility first (necessary, not sufficient):** `n_malformed_tool_calls`, `n_unknown_tool_calls`,
   raw-JSON-reversion count — compare directly against the pre-FT baseline's clean numbers (0, 0, not
   measured but implicitly 0 given no collapse was ever reported for that row). If these numbers are
   still clean, RL-on-urn didn't damage tool-calling mechanics; if they've degraded toward what the SFT
   checkpoints showed, that's a real, useful negative result (RL isn't automatically safer for the
   channel just because it never trained on it).
2. **Only then, policy transfer:** first-sight / lateness against the pre-FT baseline's 88% / 0.125 —
   did RL-on-urn's reserve policy transfer better than SFT-on-urn's did (100% first-sight / 0.000
   lateness, i.e. no transfer at all)?

### Pilot v2, take 1 (2026-07-08): exploration insufficiency + step-size diagnosis

With the reward fixed (see "Reward, revised" above), the pilot was rerun fresh (`runs/rl_urn_pilot_v2`,
base model, 8 outer steps planned) — stopped after step 7 on user request once the reward curve showed
no upward trend. **This run is also considered a diagnostic dead end, not a result to read at face
value — like `rl_urn_smoke`, it surfaced a design problem rather than a clean pass/fail.**

- **Reward history (7 steps):** `[36.42, 37.64, 36.61, 35.63, 36.51, 35.95, 34.62]` — flat, converging
  *down* to almost exactly the eager baseline (34.56) by the final step, not up toward wait2 (~42-44).
  `n_zero_advantage_groups` bounced between 1 and 5 (of 25 groups) across steps — noisy, not cleanly
  monotonic, but consistently a meaningful fraction of the batch.
- **Root-caused via a direct rollout diagnostic** (8 rollouts, temperature=0.7, one fixed stream,
  against the final checkpoint): 6/8 rollouts were byte-identical (`KEEP` on the first three distinct
  colors — pure eager), 0/8 hit a parse failure (replies were coherent, on-topic reasoning, not
  garbled). **The model re-converged onto the base model's own pre-existing eager disposition rather
  than discovering anything new** — not a new local optimum, just insufficient exploration diversity to
  escape the starting policy's already-low-entropy bias toward eager.
- **Compounding step-size problem, found on follow-up:** each outer step is exactly ONE
  `optimizer.step()` (100 `backward()` calls accumulate into a single update — not 100 updates), so a
  7-8 step pilot takes only 7-8 total weight updates, ever. AdamW's per-parameter step size is
  ~bounded by `lr` regardless of gradient magnitude (moment-normalized), so at the original `lr=1e-5`
  the ceiling on cumulative movement after 7 steps is ~7e-5 per weight — likely too small to shift
  behavior at all within this step budget, independent of whether the exploration problem above is
  fixed. Even a working reward signal would plausibly look flat at this LR/step-count combination.
- **Temperature sanity check** (same checkpoint, same fixed stream, 8 rollouts each): raising sampling
  temperature increases behavioral diversity without breaking output parsing —

  | temp | distinct patterns (of 8) | dominant-mode share | reward range | unparsed |
  |---|---|---|---|---|
  | 0.7 | 3 | 6/8 | 37–44 | 0 |
  | 1.0 | 6 | 3/8 | 21–37 | 0 |
  | 1.2 | 4 | 4/8 | 23–49 | 0 |

  1.2 produced the widest, most informative spread — two rollouts found genuinely good deviations
  (48-49 balls, near clairvoyant, via wait-then-commit patterns like `KpppKK`) alongside two clearly
  worse ones (23), exactly the two-sided spread GRPO's group-relative advantage needs. Parsing stayed
  clean (0 unparsed) at every temperature tested.
- **Fix attempted (`rl_train.py`):** `RL_LR` raised `1e-5 → 3e-5` (still 3x below SFT's `1e-4`, tuned
  for a different, denser supervision signal); added `N_EPOCHS=3` — reuse each collected batch for 3
  separate `optimizer.step()`s instead of 1, multiplying total weight updates for free (no extra
  rollout collection, the expensive part of an outer step) rather than only via a bigger LR. Reference
  log-probs are policy-independent (frozen base model), so they're computed once per batch and reused
  across epochs rather than recomputed. Rollout temperature raised `0.7 → 1.2` for the same run.

### Pilot v3 (2026-07-08): multi-epoch reuse without an importance-ratio clip caused KL blowup

Relaunched fresh from base model (`runs/rl_urn_pilot_v3`) with the fix above. Temperature=1.2 worked as
intended — `n_zero_advantage_groups` dropped to 0/25 at step 0 (vs 1-5/25 before) and stayed low, and
`group_std` rose (~5.7-6.3 vs ~4-6 before), confirming real per-group behavioral diversity. **But
`mean_kl` exploded — `-1.03, -5.49, -16.63, -30.81, -56.82` over 5 steps, accelerating each step with
no sign of leveling off — while `mean_reward` stayed flat/noisy (37.58, 37.63, 36.05, 35.75, 38.39),
not tracking the divergence at all.** Killed at step 5 on user request rather than let it run to
completion; checkpoint/manifest left in a clean (non-corrupt) state but not usable as a result.

**Root cause, not just "LR too high":** the loss (`-(advantage * seq_logp)`) is only a valid gradient
estimator when the policy being updated is the one that generated the sampled actions (the on-policy
assumption). Reusing one rollout batch across `N_EPOCHS=3` violates this with nothing to catch it:
epoch 2+ recomputes `seq_logp` under the already-shifted post-epoch-1 weights but reapplies the SAME
advantage computed from the original (pre-any-update) policy, with no mechanism to reduce an example's
gradient contribution once its probability has already moved. Every epoch pushes further in the same
direction on the identical 100 examples — exactly the failure mode PPO's clipped importance-ratio
surrogate exists to prevent, which this implementation doesn't have. Raising `RL_LR` to 3e-5 didn't
cause this but multiplied it, since the instability is about *uncorrected repeated pushes*, and a
bigger LR just makes each push bigger. The flat reward alongside exploding KL is the tell: purposeful,
reward-driven divergence should track reward upward; this looked like unconstrained drift on
increasingly stale advantage estimates instead.

**Fix (2026-07-08): reverted `N_EPOCHS` to 1**, keeping `RL_LR=3e-5` and temperature=1.2. This isolates
whether the LR bump alone (without the unsafe multi-epoch reuse) is enough, before considering the
more invasive alternative (implementing a real PPO-style clipped surrogate — caching each example's
log-prob at sampling time, computing an importance ratio each epoch, and clipping it — which would be
the textbook-correct way to make multi-epoch reuse safe, deferred for now as more code than a quick
pilot iteration warrants). The `grpo_step` epoch loop is left in place parameterized by `n_epochs`
(just defaults to 1) so re-enabling multi-epoch later, if a proper clip is added, is a small change.
**Next pilot restarts from the base model in a fresh directory (`runs/rl_urn_pilot_v4`)**, not resumed
from `rl_urn_pilot_v3`'s checkpoint (destabilized, not a clean starting point).

### Pilot v4 (2026-07-08): stable, but reward still flat — two more root causes found

`N_EPOCHS=1` fixed the instability: `mean_kl` stayed small and bounded (`0.000, -0.143, -0.629, -1.457,
...`) through 5 steps, roughly 20x smaller than v3's runaway at the same step count, and
`n_zero_advantage_groups` stayed low (0-2/25). But `mean_reward` still didn't move
(`37.43, 36.47, 36.53, 35.18, 36.54`) — killed at step 5 on user request to redirect rather than let it
run out the clock on a config already showing the same flat pattern. Two further issues diagnosed:

1. **Reverting `N_EPOCHS` to 1 gave up most of the intended step-size fix, not just the unsafe part of
   it.** The original plan was LR×3 *and* epochs×3 together (~10x more cumulative movement than the
   original run). Dropping epochs back to 1 left only the LR×3 — cumulative movement after 8 steps is
   ~3x the original run's, not ~10x. A flat curve at this point was expected, not a new failure.
2. **The loss uses a raw SUM of per-token log-probs, and episode length is confounded with the exact
   behavior being trained.** `urn_session.run_episode` stops generating decision turns the moment the
   keep budget is exhausted (`if budget_left == 0: break`) — an eager episode (commits all 3 keeps
   almost immediately) produces a short transcript; a reserve/wait episode (keeps deciding for longer
   before committing) produces a much longer one. Since `_seq_logprob` summed log-probs over the whole
   transcript, longer (reserve-like) episodes got systematically larger-magnitude gradients than
   shorter (eager-like) ones at the *same* advantage value — noise correlated with the thing being
   learned, not a length-neutral signal.

**Fixes applied (`rl_train.py`):**
- `_seq_logprob` changed from a raw sum to a length-normalized **mean** over labeled tokens — removes
  the length confound. Verified against an independent `log_softmax`-based reference computation
  (exact match to 1e-4) before relaunching.
- `RL_LR` raised `3e-5 → 6e-5` — a second, deliberately incremental step (6x the original 1e-5, still
  ~40% below SFT's 1e-4), kept separate from the `N_EPOCHS` fix so this change is legible on its own.
  Explicitly a *different* risk axis than the v3 instability: that was a specific compounding pathology
  (now removed) from reapplying stale advantages across epochs; this is the plain, still-live reason
  `RL_LR` was originally kept low — single-episode reward is sparse and noisy, so a bigger step still
  means trusting a noisier gradient direction more, just not compounded 3x per outer step anymore.
  `MAX_GRAD_NORM` clipping remains as a floor-level safety net regardless of LR.

**Next pilot (`runs/rl_urn_pilot_v5`) restarts from the base model in a fresh directory**, not resumed
from `rl_urn_pilot_v4`.

### Pilot v5 (2026-07-08): both fixes applied, ran clean to completion, still flat — root cause is credit-assignment density, not tuning

`_seq_logprob` switched from a raw sum to a length-normalized mean (verified against an independent
`log_softmax` reference computation before relaunching); `RL_LR` raised `3e-5 → 6e-5`. Ran the full 8
steps to completion without intervention (first pilot to do so) — single process, no OOM, no crash.
`mean_kl` stayed small and bounded throughout (note: not comparable in magnitude to v2-v4's numbers,
since the length-normalization changed the metric's scale) and `n_zero_advantage_groups` stayed low
(0-1/25 across steps) — the exploration and stability fixes all held.

**Final reward history (8 steps): `[37.89, 35.62, 36.76, 36.52, 36.46, 37.69, 35.5, 37.36]`** — flat,
oscillating in the same 35-38 range as every prior attempt, no trend toward wait2 (~42-44) anywhere in
the run. This is now the fourth consecutive pilot (v2 with the pi*-based reward already ruled out;
v2-restart, v3, v4, v5 all reward-fixed and progressively de-bugged) to show no reward improvement,
despite independently fixing: the reward's reference-policy dependency, insufficient exploration
diversity, step-size, multi-epoch instability, and a length confound in the loss. **Ruling out tuning
as the remaining explanation** — see the conversation's live discussion for the full reasoning, summarized
here:

**Root cause: the loss applies ONE scalar advantage uniformly across every token in the episode, with
no per-decision or per-timestep credit propagation** (no critic/value function — GRPO deliberately
omits one). Reward genuinely depends on only a handful of pivotal decisions (a first-sighting
KEEP/PASS call on a color that turns out to matter) out of the ~3-13+ decisions in an episode; most
other decisions are easy/low-information and don't actually determine the outcome. Applying the same
flat advantage to all of them dilutes the gradient signal for the pivotal decisions with noise from the
rest, and from pure environmental luck (which stream got drawn) unrelated to the model's choices. This
is the standard weakness of vanilla per-episode REINFORCE/GRPO relative to actor-critic methods (PPO
with GAE, TD-learning, the family used for genuinely sparse-reward tasks like checkpoint-based racing-
game RL) — those work with sparse *reward events* specifically because a learned value function
propagates credit backward through every intervening timestep via bootstrapping, not because sparse
reward is inherently easy to learn from with a flat, uncredited scalar. GRPO's design assumption (a
whole generated completion is graded as one homogeneous unit — its origin is single-shot math/code
RLHF) also doesn't fit a multi-turn *sequential decision* trajectory like this one as well as it fits
a single-shot completion.

This also reframes the SFT-vs-RL comparison the phase was designed around: SFT reached near-optimal
urn performance from only 170 sessions specifically *because* it had dense, per-decision supervision
(`qwen-finetune-transfer-plan.md` Result 2) — deliberately not replicated here so RL's disposition
would be *discovered*, not imitated (see "Why this first" above) — and that choice is exactly what
cost the sample efficiency. Fixing this for real would mean either (a) constructing a per-decision
advantage from this project's existing scoring machinery (π*/wait2/`value_of_builds` can already say,
in hindsight, whether an individual KEEP/PASS was good) instead of one flat episode-level scalar, or
(b) accepting that vanilla per-episode GRPO on this task needs a training budget several orders of
magnitude larger than a pilot (successful sparse-reward RL with a *proper* critic still typically runs
into the millions of environment steps) — both open, unstarted, and outside a quick-pilot scope.

**Status at end of session (2026-07-08): Phase 1 GPU box work paused here, box being released.**
Model checkpoints/GGUF artifacts from all pilots were NOT pulled off the box before release (all
runs were flat/diagnostic dead-ends, not validated results, and the artifacts are large — the merged
model alone is ~28-29GB per run, regenerated fresh each step, pure intermediate output; the LoRA
adapter+optimizer checkpoints are ~800MB each). Preserved instead: full `pilot.log`s and
`manifest.json`s for `rl_urn_smoke`/v2/v3/v4/v5 (`runs/box_archive_2026-07-08/`, ~6MB total) and the
rollout-diversity diagnostic script (`scripts/box/diag_rl_pilot_rollouts.py`). If Phase 1 resumes, it
restarts from the base model on a fresh box — nothing here needs to survive except this doc's record
of what was tried and why it didn't work yet.

**Next step spec'd, not started: `docs/rl-ppo-credit-assignment-spec.md`** — adds per-decision credit
assignment (a lightweight critic + PPO-clipped surrogate) instead of GRPO's flat per-episode advantage,
targeting the root cause diagnosed above.

## Phase 2 (tool framing) — only if Phase 1 is inconclusive because of channel fragility

Everything below this point (through "Infra spec") was the original tool-framing RL design, from
before Phase 1 was added. Kept as the plan for **if** Phase 1's zero-shot tool eval comes back
ambiguous specifically because the tool-calling channel itself is fragile (as opposed to a clean
policy-transfer answer either way) — not a default next step after Phase 1 succeeds or fails cleanly.

### Why RL instead of SFT

The SFT approach (`qwen-finetune-transfer-plan.md`) hit two problems, one scientific and one
engineering:

1. **Design A (pure-transfer redesign)** cleanly separated "did the policy transfer from the urn"
   from "was it taught directly," but the tool-eval side is now blocked: every checkpoint collapses
   into a malformed-output loop from problem 2 onward, because every SFT training session (urn,
   anchor, mechanics bridge) is synthesized to be well-formed on every turn. None of them contain a
   turn that fails and gets corrected, so the model has no learned behavior for the conversational
   state the eval harness's retry-nudge (`driver.py: FORMAT_REMINDER`) puts it in the instant a real
   tool call fails to parse. See that doc's Result 2 and "Working hypothesis."

2. More fundamentally, SFT never lets the model discover the reserve-then-build policy under its
   *own* reward signal in the *target* framing — it only imitates π\*-labeled demonstrations in the
   urn and hopes the abstraction transfers to tools. Negative or blocked transfer can't distinguish
   "the policy didn't transfer" from "the tool-calling channel itself is too fragile to carry it."

RL sidesteps both:

- **No transfer question.** Training directly in the tool framing with reward = a function of
  regret/lateness installs the policy *in* the target framing — nothing needs to transfer from a
  different modality. (The urn framing remains available as a comparison arm, not a prerequisite.)
- **On-policy exposure to the retry-nudge state, for free.** RL rollouts run the actual `driver.py`
  loop, including `FORMAT_REMINDER` after a malformed turn. The model will occasionally emit a bad
  tool call during training, get nudged, and either recover (rewarded if the episode goes on to
  solve problems) or not (penalized). That's exactly the "error → correction → recovery" exposure
  flagged as untried in the SFT plan's open follow-ups — RL gets it as a side effect of sampling
  on-policy, instead of requiring hand-synthesized malformed-turn examples.

### Algorithm: GRPO

GRPO (Group Relative Policy Optimization), not PPO:

- Reward here is a clean scalar computed at episode end from the existing scorer (regret vs. π\*,
  balls collected) — no reward model needed, so PPO's extra machinery buys nothing.
- No critic/value network to train — smaller memory footprint on the single GPU that already needed
  a cuDNN backward-pass workaround (`train_lora.py`'s `enable_cudnn_sdp(False)`) to survive long
  sessions in plain SFT.
- Advantage is computed within a group of G rollouts sampled from the same starting condition (same
  seed/stream): `A_i = (r_i - mean(r_group)) / std(r_group)`. This is the standard credit-assignment
  substitute for a value function when reward is episode-level rather than per-token.

### Reward design

Reuse the existing scorer, not a new metric. **Start minimal, same principle as Phase 1**: begin with
`reward = 1 - regret/regret_eager` alone, from the same episode-end computation the eval harness
already produces. Only add penalty terms (e.g. `-0.1 * n_malformed` for malformed/no-tool-call turns,
or the `n_raw_json_reversion` term discussed in "Infra spec" below) if a pilot run actually shows
reward hacking — these were originally drafted pre-emptively, which is backwards; add complexity when
it's empirically justified, not before.

### Rollout generation

- Environment = the existing `run_session()` in `driver.py`, unchanged, run in **sampling mode**
  (temperature > 0) instead of eval's greedy/near-greedy decoding.
- Per training prompt (a seed/stream instance), sample a group of **G rollouts** (G=4–8) at the
  current checkpoint. This is the expensive part: each rollout is a full session — up to 60
  problems, up to a 200k-token cap — so a gradient step costs G× that. **Use T=60** (full length),
  not a shorter T — same reasoning as Phase 1's episode-length fix: at short horizons π\*'s own
  optimum degenerates toward eager (`phase3_demos.py`'s explicit note on `T_RANGE`), so training at a
  short T would teach against a reward signal with no reason to reserve, defeating the point.
- Serving for rollouts should be **vLLM, not Ollama** — RL needs high-throughput sampling of many
  rollouts, not the few careful greedy generations eval needs, and Ollama's tool-call parsing was
  already a source of friction in the SFT pipeline (`qwen-finetune-transfer-plan.md` §Serving: "vLLM
  +hermes is a dead end for Qwen-Coder tool-calling" — that parser problem applies equally to
  RL-time sampling, so it needs a working forced/parseable tool-call path in vLLM, not the GGUF→
  Ollama route used for SFT eval).

### Loss / masking

Reuse the SFT collator's masking logic (`train_lora.py: make_collator`, prefix-diffed against the
tokenizer's chat template) but apply it to **log-probs** instead of NLL labels: sum/average
log-prob only over assistant-generated spans (including tool_call spans — those are the decisions),
excluding injected user turns (`FORMAT_REMINDER`, next-problem prompts) and tool-result turns. The
existing template-diffing approach is directly reusable — no new masking logic needed.

### Guardrails

- **KL penalty against the SFT/base checkpoint** (standard in GRPO) — without it, reward hacking
  toward a degenerate-but-high-reward policy (e.g. spamming `submit_answer` on garbage, or
  exploiting an edge case in the regret computation) is a real risk with a coarse episode-level
  reward.
- **Keep the malformed-output penalty non-trivial.** Otherwise GRPO could "solve" the collapse by
  learning to avoid tool calls in ambiguous states entirely, rather than actually recovering from
  them — a shortcut that would look like success on the malformed-output metric while failing the
  actual task.

### Infra

Don't reach for TRL's `GRPOTrainer` as-is — it assumes single-turn prompt→completion, not this
multi-turn tool-loop-with-env-injected-turns shape. Custom GRPO loop, reusing `driver.py` for rollouts
(sampling mode) and the assistant-token masking logic for per-token log-probs, hand-rolling the
group-relative advantage + PEFT gradient step — consistent with the rest of this pipeline (SFT trainer
and eval driver are both hand-rolled already). (A multi-turn-agentic-RL library, e.g. `verifiers` or
OpenPipe's ART, was considered but not evaluated — the spec below is concrete enough to build directly,
revisit only if the custom loop hits a wall a library would have solved for free.)

LoRA config (rank/alpha/target modules/dropout) can likely stay identical to the SFT run
(`train_lora.py`: `r=32, α=64, dropout=0.05`, all seven linear target modules) — GRPO updates the
same adapter weights, just with a policy-gradient loss instead of NLL.

### Infra spec (2026-07-07)

Two design decisions resolve the two biggest open blockers from the earlier draft without new
external infra:

**Decision 1 — bypass vLLM's tool-calling feature entirely, don't fix it.** The documented blocker
(`vLLM+hermes` silently drops tool calls under `tool_choice="auto"` for Qwen2.5-Coder) is a
*server-side output-parsing* bug. We don't need vLLM's parser to work at all:

1. Render the exact prompt locally via `tokenizer.apply_chat_template(messages, tools=TOOL_SCHEMAS(),
   add_generation_prompt=True, tokenize=False)` — the identical call `train_lora.py`'s `build_example`/
   `verify_template` already make, so the rendering is byte-identical to what the model was trained on
   (no new template-consistency risk).
2. Send that rendered string to vLLM's plain `/v1/completions` endpoint (not `/v1/chat/completions`)
   — vLLM never invokes its own tool-call parser this way, so the drop-silently bug never triggers.
3. Parse the raw completion text with `raw_chat.py`'s existing `_extract_untagged_tool_call` regex
   extractor — already proven against Ollama's raw output, reusable unchanged.

Net effect: `driver.py` needs **zero changes** for RL rollouts. Only `lomekwi/raw_chat.py`'s vLLM
branch changes (the Ollama/OpenAI/Anthropic branches are untouched) — `chat_tools()`'s public
interface stays the same, so `run_session()` is oblivious to which serving path is underneath. This
fully resolves the old open-follow-up #3 without debugging vLLM's parser.

**Decision 2 — RL never touches Ollama/GGUF during training.** Ollama needs a full merge → GGUF
convert → `ollama create` cycle to pick up a new adapter (~5 min measured in the SFT pipeline logs) —
intolerable per-step cost across dozens of GRPO steps. vLLM supports hot-loading a LoRA adapter
straight from a HF-format adapter directory via its runtime `/v1/load_lora_adapter` API, no
reconversion. So: stay entirely in the vLLM+HF/PEFT stack for the whole RL run; only produce a
GGUF/Ollama build of the *final* checkpoint, to eval it the same way as the SFT checkpoints for a
like-for-like comparison.

### Reward function (concrete)

Start minimal, same as Phase 1: `reward = 1 - regret/regret_eager_reference`, via the exact call
`arm_a1_announce.py` already makes at eval time — `exact_pistar_report(slots, costs, B, N, T, UNIFORM,
model_builds_from_actions(actions_from_session(row, slots)), dp=dp)["regret"]` — no new scoring logic,
just called per-rollout instead of per-eval-seed. Only add penalty terms if a pilot run actually shows
reward hacking:

- `n_malformed_tool_calls` / `n_unknown_tool_calls` are already fields on `run_session()`'s return row
  — cheap to add if needed, no new code.
- `n_raw_json_reversion` would be **new** but mechanical: the mechanics-rebalance summary script
  (`run_format_only_mechbal_training.sh`) already hand-writes this exact detector inline
  (`TOOL_NAME_RE`, scanning assistant `content` for a tool-name-shaped JSON fragment with no real
  `tool_calls` attached). If it turns out to be needed, extract it into a reusable function (proposed:
  `skirental_scorer.count_raw_json_reversions(transcript)`) so both the SFT summary scripts and any RL
  reward call the same code — worth deduplicating out of the copy-pasted shell-embedded Python blocks
  regardless of whether this specific term ends up needed, since it's the sharpest failure signature
  the mechanics-rebalance result surfaced (it *rose* under a fix that improved everything else).

### Rollout generation (concrete)

`rl_rollout.py` (new): for a given seed/stream and the current adapter, run **G parallel
`driver.run_session()` calls unchanged** at `temperature=0.7` (matches the serving default already
used at eval, keeps enough diversity within a group for the advantage to be informative) against the
Decision-1 vLLM client. Since `run_session()` and `chat_tools()` don't change shape, this is a thin
wrapper: `asyncio.gather` over G calls with the same `SessionState`-constructing code
`arm_a1_announce.py` already uses (same `build_stochastic_stream`/`slots_to_problems`), just discarding
nothing — every rollout's full `row` (including `transcript`, `slots`, `meta`) is kept for the reward
+ log-prob steps.

### Log-probs + GRPO loss (concrete)

`rl_train.py` (new):

1. For each rollout's `transcript`, call `build_example(transcript, tokenizer, tools=TOOL_SCHEMAS())`
   — **reused verbatim from `train_lora.py`**, no new masking logic. This returns `input_ids`/`labels`
   where `labels != -100` marks exactly the assistant-generated token spans (the policy's decisions),
   already excluding role-declaration prefixes, user turns, and tool-result turns — the same
   prefix-diffed masking validated by `verify_template`'s regression guard.
2. Teacher-force `input_ids` through the local HF/PEFT model (the same one `train_lora.py: load_hf`
   builds) to get per-token log-probs; sum over positions where `labels != -100` → one scalar
   sequence log-prob per rollout.
3. Do the same forward pass through a **frozen reference copy** (the SFT checkpoint RL starts from —
   see "Pilot scope" below) for the KL term.
4. Group-relative advantage within each G-sized group (same seed/stream):
   `A_i = (r_i - mean(r_group)) / std(r_group)`.
5. Loss: `-mean_i(A_i * seq_logprob_i) + β * KL(policy || reference)`, backward, optimizer step
   (reuse `train_lora.py`'s optimizer/scheduler setup for consistency with the SFT run).

### Sync loop (concrete)

After each GRPO step: save the updated LoRA adapter to disk (small, fast — adapter-only, not a merge),
call vLLM's `/v1/load_lora_adapter` to hot-swap the serving adapter, continue to the next rollout
batch. GGUF/Ollama are not involved anywhere in this loop.

### New files

- `scripts/creator/tool_disposition_benchmark/rl_reward.py` — the reward formula above, plus the
  extracted `count_raw_json_reversions` detector (dedup'd out of the shell-embedded summary scripts).
- `scripts/creator/tool_disposition_benchmark/rl_rollout.py` — G-way parallel rollout collection via
  unmodified `driver.run_session()`.
- `scripts/creator/tool_disposition_benchmark/rl_train.py` — GRPO loss/step: `build_example`-based
  log-probs, KL vs. frozen reference, adapter save + vLLM hot-swap trigger.
- `lomekwi/raw_chat.py` — vLLM branch only changes (Decision 1); Ollama/OpenAI/Anthropic paths
  untouched.
- `scripts/box/run_rl_pilot.sh` — pipeline script mirroring the SFT box scripts' shape
  (stage/status-file/log pattern already established in `run_format_only_mechbal_training.sh`): stand
  up vLLM with `--enable-lora`, run N capped GRPO steps, periodically eval a checkpoint.

### Pilot scope (concrete numbers, for sign-off before any GPU time)

- N=8/T=60 — full length, not a shortened episode (see "Episode length" under Phase 1: short horizons make pi*'s own optimum degenerate toward eager, which would defeat the point here too).
- G=4 rollouts per group.
- ~20-30 GRPO steps, hard-capped.
- Tool arm only — skip the urn comparison arm for the pilot.
- **Init/reference checkpoint: start from `qwen-ft-format-only-mechbal`** (the best SFT checkpoint so
  far), not base `qwen2.5-coder:14b`. Less tool-syntax relearning needed, and KL-anchoring to a
  somewhat-broken reference is fine here — GRPO's whole job in this pilot is to move *away* from that
  checkpoint's specific biases (silence-after-first-success, raw-JSON reversions) via reward, not stay
  close to them.

### Open engineering risks (check before committing to the full build, not after)

1. **vLLM `--enable-lora` + hot-swap compatibility** with the PEFT adapter format `train_lora.py`
   already produces — likely fine (vLLM expects a standard HF PEFT adapter directory, which is exactly
   `train_lora.py --out`'s output) but unverified in this repo. Worth a 5-minute smoke test
   (load the existing `format_only_mechbal` adapter into a vLLM instance) before writing the full loop.
2. **GPU memory budget**: serving (vLLM) and training (HF/PEFT forward+backward) concurrently on one
   GPU may not fit. Likely resolvable by not running them simultaneously (generate a rollout batch,
   pause vLLM inference, run the backward pass, resume) rather than requiring co-residency, but adds
   per-step latency worth costing into the pilot estimate once measured.
3. **Reward signal variance at N=8/T=60** — needs a quick offline check (run the existing eval harness
   for a few seeds, look at regret spread) to confirm there's enough variance across G=4 samples for
   the group-relative advantage to carry real signal before investing in the full loop.
4. **Regex-gaming risk on the reversion penalty** — `count_raw_json_reversions` is a pattern match, not
   a semantic classifier; GRPO could learn to dodge the specific regex rather than the underlying
   failure mode. Mitigate by periodically eyeballing a rollout sample during the pilot, not just
   watching the aggregate reward curve.

### Cost / scope

RL needs orders of magnitude more rollouts than SFT needs demonstrations, and each rollout here is a
long, expensive session — this is a materially bigger and slower experiment than either SFT attempt
so far. Per [[no-auto-reps]]-style practice, don't launch a full run without first scoping a concrete
pilot: episode length, group size G, number of GRPO steps, and an estimated box-hour/cost budget,
proposed and agreed before spending GPU time.

## Open follow-ups — Phase 1 (immediate)

1. **Implement the GRPO loss/step + outer-step loop** — DONE (`rl_reward.py`, `rl_train.py`,
   `rl_rollout.py`, `rl_urn_pilot.py`), including checkpoint/resume support.
2. **Run one outer step for real and time it** — DONE, see "Time estimate" above.
3. **Run the 5-8 step pilot and read it as a direction check** — RUN FIVE TIMES (`rl_urn_smoke` void on
   the old pi*-based reward; v2/v3/v4 each root-caused and fixed a distinct real bug — exploration
   diversity, step-size, multi-epoch instability, a length confound; v5 ran clean to completion with
   all fixes applied). **Reward stayed flat across all of them (final v5: `[37.89, 35.62, 36.76, 36.52,
   36.46, 37.69, 35.5, 37.36]`).** With every tuning/stability explanation exhausted, this now points to
   a structural cause, not a bug: vanilla per-episode GRPO has no per-decision credit assignment, and
   this task's reward depends on a handful of pivotal decisions diluted among many easy ones (see
   "Pilot v5" above for the full reasoning). **Not resolved. Two live options, neither started:**
   (a) construct a per-decision advantage from this project's existing π*/wait2/`value_of_builds`
   scoring machinery instead of one flat episode-level scalar, or (b) accept vanilla GRPO here needs a
   training budget several orders of magnitude past pilot scale. GPU box work paused pending a decision
   between these (see "Pilot v5" above for what was/wasn't preserved before the box was released).
4. **Only if a future pilot shows a clear upward trend, propose and cost a separate convergence run**
   (rough order of magnitude 30-50+ outer steps, sharpened using the pilot's actual measured per-step
   cost and learning curve) — a new decision point, not something to commit to now. Not started.
5. **After Phase 1 training completes (pilot or convergence run), run the falsifiable legibility +
   transfer check** (see "Falsifiable check" above) using the existing tool eval harness unchanged, read
   in the same two-stage order (legibility vs. pre-FT baseline, then policy transfer vs. pre-FT
   baseline). Not started — blocked on (3).

## Open follow-ups — Phase 2 (tool framing, conditional on Phase 1's result)

1. **Scope a pilot** — DONE above ("Pilot scope"): N=8/T=60, G=4, ~20-30 capped GRPO steps, tool arm
   only, init from `qwen-ft-format-only-mechbal`. Only relevant if Phase 1 is inconclusive.
2. **Reward function** — DONE above ("Reward function (concrete)"), simplify the same way Phase 1's
   was: start with `reward = 1 - regret/regret_eager_reference` alone, add `n_malformed`/`n_unknown`/
   `n_raw_json_reversion` penalty terms only if a pilot run actually shows reward hacking, not
   pre-emptively.
3. **vLLM forced-tool-call path** — RESOLVED by design, not by fixing vLLM: Decision 1 above bypasses
   vLLM's tool-calling feature entirely (render via `apply_chat_template` + plain `/v1/completions` +
   the existing untagged-tool-call extractor), so the drop-silently bug never triggers. Not yet
   implemented as the `raw_chat.py` vLLM-branch change.
4. **Custom loop vs. library** — staying with a custom loop (consistent with the rest of this
   hand-rolled pipeline); library options noted but not evaluated further unless the custom loop hits a
   wall.
5. **Pre-pilot risk checks** (from "Open engineering risks" above) — cheap, do these before writing the
   full loop: (a) 5-minute vLLM `--enable-lora` hot-swap smoke test against the existing
   `format_only_mechbal` adapter, (b) reward-variance check at N=8/T=60 over a few seeds using the
   existing eval harness, no new code needed for either.
6. **Comparison to SFT.** If RL installs the policy and survives the tool-calling channel, compare
   against the SFT urn numbers (`qwen-finetune-transfer-plan.md` Result 2: 7% first-sight, 101% of
   π\* balls) — is RL's ceiling in the tool framing similar, better, or does it hit a different wall?
