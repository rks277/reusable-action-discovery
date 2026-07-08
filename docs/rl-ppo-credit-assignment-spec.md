# RL Phase 1 (urn framing): per-decision credit-assignment spec — standalone

Self-contained as of 2026-07-08: absorbs everything a reader needs from `rl-finetuning-plan.md` and
`qwen-finetune-transfer-plan.md` (both moved to `docs/old/`, superseded by this doc for RL/SFT purposes) to
understand and continue this work without opening either. `online-tool-investment-plan.md` (§1 below draws
on it for the headline claim/task definition) is NOT archived — it's the broader project doc covering more
than just this RL phase, still active. (`docs/box-setup.md`, the GPU-box runbook, is also NOT archived —
still the reference for provisioning/setup mechanics.)

## Status (2026-07-08)

**Per-decision credit-assignment machinery IMPLEMENTED, UNTESTED ON A REAL BOX.** The design questions in
§7 are resolved (privileged critic first; SAC-derived extensions deferred — see §7 for the decision
record). §4-§6 built and unit-tested locally (CPU-only torch installed in `.venv` for this) against
synthetic/heuristic-policy data, since the GPU box from the GRPO pilots (§3) was released and no rollout
data was persisted to replay (`rl_rollout.collect_batch` never wrote to disk) — substituted
`rl_reward.builds_to_transcript` + `pi_star`'s eager/wait2 heuristics to exercise the new code paths
end-to-end without an LLM. Specifically verified, all passing:

- `rl_reward.per_decision_rewards` sums to the exact scalar `episode_reward` on eager/wait2/none.
- `rl_critic`: feature extraction (privileged vs fair, shape/alignment/range), `returns_to_go`,
  `ReturnNormalizer` (matches `statistics.mean/stdev`), MLP trains (loss ↓ over 200 steps on synthetic
  data), critic checkpoint save/load round-trips to identical predictions.
- `rl_train._per_decision_logprobs`: shift-indexing verified against the old whole-sequence masked-sum
  approach on hand-built fake logits (token-count-weighted recombination of per-turn means reproduces the
  whole-sequence sum to 1e-3).
- `build_example`'s new `turn_spans` output is backward-compatible (SFT callers ignore the extra key).
- Full module import chain (`rl_urn_pilot.py` → `rl_train.py` → `rl_critic.py`/`train_lora.py`/
  `urn_session.py`) resolves without error.

**NOT yet verified — needs a real GPU box, nothing to test this against locally:**

- The actual `model(input_ids=...)` forward/backward path through `_per_decision_logprobs` (only the
  pure-tensor shift math was checked, not a real tokenizer/model's logits).
- Whether reward actually moves now that credit assignment is per-decision (the entire point).
- Wall-clock/VRAM impact of the critic's extra CPU-side forward passes each step (expected negligible
  given its size, unverified in practice).
- `N_EPOCHS` stays at 1 in this implementation (see `rl_train.py`'s `N_EPOCHS` comment) per §8's own
  suggested order — re-enabling multi-epoch reuse is deliberately deferred until a real run confirms this
  pipeline is stable at `n_epochs=1`, not bundled into this same untested change.

Design choices made autonomously while implementing (engineering-level, not framing decisions, so not
escalated for separate sign-off — flagging here for visibility): critic trains jointly from step 0, no
warm-up gate; critic stays on CPU throughout, never moved to the GPU the 14B policy occupies;
`CLIP_EPS=0.2` as the untested starting default; per-decision log-prob is a MEAN within each turn (not a
sum), for the same length-confound reason pilot v5 (§3) made the whole-episode version a mean.

**Pre-box-run review applied (2026-07-08, same day, before any box spend) — three changes, §8.5 has the
full rationale:** (1) the critic is now fitted to plateau on each outer step's batch
(`rl_critic.fit_critic`, early-stopped, ≤500 CPU-side iterations) instead of getting ONE gradient step
per outer step — the original wiring would have given it ~8 total updates across a whole pilot, i.e. a
near-constant baseline that silently degenerates the advantage to REINFORCE-with-a-mean-baseline and
never cashes in the privileged `rate` feature; (2) a **step-0 advantage-ordering diagnostic**
(`rl_train.advantage_diagnostic`) prints mean advantage bucketed by decision type × ground-truth role
every step — if the buckets don't already separate sensibly on the FIRST batch, before any policy
update, the run should be stopped and the credit signal debugged rather than burning outer steps;
(3) **behavioral metrics per step** (`rl_train.behavioral_metrics`: first-sight %, lateness — the
project's standing lead metrics) are printed each step and persisted in the checkpoint manifest, because
a disposition shift will resolve there several steps before it beats `mean_reward`'s noise at ~100
episodes/step. All three re-verified against the local self-tests (still green, new paths covered). A
fourth concern — the advantage is now in raw balls units (~5-15 early) instead of the unit-scale
z-scores every stable pilot ran, tilting the pg:KL balance ~10x at the same `KL_BETA` — is deliberately
NOT pre-tuned away; it's logged (`mean_abs_advantage`) with the response documented at the `KL_BETA`
constant (raise beta / normalize advantages, NOT another LR change) if `mean_kl` runs.

**Immediate next step: provision a fresh GPU box** (the one used for the v2-v5 pilots was released, no
persistent disk) and run one outer step of `rl_urn_pilot.py` to find out whether any of the above actually
works against a real model. Not done yet — costs money, flagging rather than doing automatically.

---

## 1. Orientation — the research question this phase serves

**The headline claim being investigated (not re-litigated here, just the load-bearing context):** LLMs
that can build a reusable script once (fixed cost, capped budget) and reuse it free on same-type problems
later **don't treat the budget as a resource-allocation problem** — they build eagerly on first sight
instead of waiting to see whether a problem type actually recurs, even though the *same model*, given the
*identical* underlying decision stripped of the coding framing (a bag of colored balls: keep a ball once
and its color's future draws all auto-collect for free), allocates far better. This is read as a
**framing/recognition failure**, not an inability to allocate: Haiku scores allocation regret ≈ 0 in the
urn (matches the optimal reference exactly) while showing regret 1633 in the tool framing on the *same
streams with the same information*. Full result, all the supporting numbers and caveats: see the archived
`online-tool-investment-plan.md` if that history matters; not needed to continue the work below.

**The urn/balls task, concretely** (what every number and every training run in this doc refers to):

- A stream of **T** items is drawn one at a time from **N** distinct color-classes (families). Some
  classes are **hot** (recur often), most are **trap** (rarely recur) — fixed at stream-generation time,
  randomized per seed which color gets which role.
- At each draw, the model may **KEEP** the current ball (locks in this color: every future draw of that
  color auto-collects, free, for the rest of the stream) or **PASS** (that ball scores nothing, and the
  color remains undecided — the same color can be kept later at a subsequent sighting). At most **B**
  colors can ever be kept (**B ≪ N**, a scarce write/keep budget).
- **Reward = total balls collected** by the end of the stream: kept colors contribute every occurrence
  from the keep position onward; never-kept colors contribute 0. This is the literal, stated objective —
  nothing else is optimized.
- **π\*** = the reference optimal policy, an exact finite-horizon belief-state DP (`exact_dp.ExactDP`)
  over a symmetric-Dirichlet(α) prior on the color-rate mix. Used only as an **evaluation** yardstick in
  this phase (see §2) — NOT part of the RL reward.
- **Lateness** = keep position minus first sighting (0 = kept on first sight = "eager"; ≥1 = waited for a
  repeat first). **First-sight %** = fraction of keeps at lateness 0. These are this project's standing
  lead metrics — more robust than regret at pilot-scale seed counts.
- Standard params for this phase: `N=8, T=60, B=3` (`urn_session.py`'s `UNIFORM/N/T/B`). **T=60 is
  load-bearing, not arbitrary** — at short horizons (e.g. T=20) even π\*'s own optimal policy degenerates
  toward 100%-eager (not enough remaining runway for waiting to pay off), which would make training
  against this reward pointless for the question being asked. T≥60 with B/N≈0.25-0.35 reliably reserves.

**Why this is being trained via RL now, not continuing the SFT approach:** an SFT arm (imitation learning
on π\*-optimal demonstrations, full detail archived in `qwen-finetune-transfer-plan.md`) already
**installs the urn policy near-optimally** — 7% first-sight, 101% of π\*'s balls collected, essentially
solved via imitation — but that installed competence **did not visibly transfer** to the tool-calling
framing in the evaluations run so far (the fine-tuned model still built eagerly there). That SFT
tool-eval history is genuinely messy — several real training/corpus/serving bugs were found and fixed
along the way, and at least one prior "clean transfer-fails" verdict was itself later retracted and
reinvestigated after a corpus bug surfaced — so the specific tool-side numbers shouldn't be treated as a
clean, final result without reading that doc's full revision history. What IS solid regardless of that
mess: (a) the urn-side install number above, and (b) the untouched base (pre-fine-tune) model's own tool
baseline is clean and eager (12 seeds: 0 malformed calls, 88% first-sight, 0.125 lateness, regret
2934±324) — i.e. the base model doesn't need fixing, it's eager, not broken.

**The sharper question RL asks that SFT couldn't:** SFT only ever tests whether an **imitated** policy
transfers across framings. It can't distinguish "the policy doesn't transfer" from "the tool-calling
channel itself is too fragile to carry any policy, imitated or not" (a real risk — SFT's tool-calling
eval independently hit multiple serving/format bugs, documented in the archived doc). RL sidesteps this
by testing whether a policy the model **discovers itself** under its own reward signal — never shown a
single demonstration — installs and transfers differently, independent of whatever the tool-calling
channel's own fragility turns out to be. **Phase 1 (this doc): RL entirely within the urn framing, from
the untouched base model** — cheap, and it avoids the entire tool-calling engineering surface (no JSON
tool-call schema, no `<tool_call>` wrapper, no malformed/unknown-tool/raw-JSON-reversion failure modes
exist in this framing at all: `urn_session.py` parses a plain `DECISION: KEEP/PASS` regex from ordinary
chat text). A Phase 2 (RL directly in the tool framing) exists as a fully specced contingency plan if
Phase 1's eventual zero-shot tool-transfer eval comes back ambiguous specifically because of channel
fragility — parked, full spec archived in `rl-finetuning-plan.md`, not reproduced here since it's not the
active work and depends on Phase 1's outcome first.

**Falsifiable check, once Phase 1 training ever succeeds** (not yet reached — training hasn't produced a
result yet): run the existing zero-shot tool eval and read it in two stages. (1) **Legibility first**
(necessary, not sufficient): are `n_malformed_tool_calls`/`n_unknown_tool_calls`/raw-JSON-reversion counts
still clean vs. the pre-FT baseline's 0/0? LoRA updates the same shared weight matrices both the urn and
tool generation paths flow through, so there's no architectural guarantee urn-only training leaves
tool-calling mechanics untouched — this must be checked, not assumed. (2) **Only then, policy transfer**:
first-sight/lateness vs. the pre-FT baseline's 88%/0.125 — did the reward-discovered urn policy transfer
better than the imitated one did?

---

## 2. Reward: raw balls collected, no reference policy (revised 2026-07-07)

The reward is **`reward = balls collected`** (`rl_reward.episode_reward`) — the urn's literal, stated
objective. No DP, no π\*, no normalization, zero dependency on `exact_dp`/`skirental_scorer`.

This wasn't the original design. The first version was `reward = 1 - regret/regret_eager` (both terms
π\*-relative), and a first pilot ran on it (`runs/rl_urn_smoke`, now VOID — do not resume it). Its reward
history never showed a clean trend, and investigating why surfaced a **design** problem, not a numerical
bug:

1. One episode hit `regret_eager ≈ 1e-13` (eager was already ~optimal on that stream) and
   `regret/regret_eager` blew up to ~1.57e15, dominating that step's batch mean. A patched epsilon guard
   turned out to be treating a symptom, not the root cause.
2. Per-episode data showed `regret`/`regret_eager` frequently large-magnitude *negative* — π\* was
   **losing to a trivial two-line heuristic** (`wait_k_builds(k=2)`: hand-solve a type's first sighting,
   build on its confirmed 2nd-sighting repeat), not just occasionally landing near zero.
3. **Root cause:** π\* (`exact_dp.ExactDP`, symmetric-Dirichlet(α) belief) is exactly Bayes-optimal for a
   world where the type-rate vector is drawn *once* from a smooth, continuous Dirichlet — but the actual
   stream generator (`stream_builder.build_stochastic_stream`) hard-assigns a **fixed discrete two-point
   split** every seed (a fixed hot rate, a fixed trap rate; only the color↔role assignment is randomized).
   Being Bayes-optimal for the wrong prior family gives no guarantee against the true distribution — this
   is a misspecification, not a bug, and no `alpha` reparametrizes a continuous Dirichlet into an exact
   discrete two-point mixture.
4. Measured the damage directly: under the pilot's own training distribution (every seed forced to have
   an early trap), π\* collects 39.8 balls/seed vs. wait2's 43.0 (π\* loses, z=−13.9, wins only 52% of
   seeds); under genuinely unconditioned i.i.d. sampling, 45.2 vs. 45.5 (still loses, though closer).
   **π\*, as built, is not a trustworthy "can't be beaten with the same info" reference against this
   benchmark's real generator** — using it as both the reward's value term and its normalizing
   denominator meant the signal was untrustworthy independent of the epsilon bug.
5. **Fix:** drop the reference policy from the reward entirely. GRPO's own group-relative advantage
   normalization (`compute_advantages`: z-score within each G-sized group of rollouts sharing one stream)
   already does the baseline-subtraction the `1 - regret/regret_eager` rescaling was for — so it was
   solving a problem GRPO's own machinery already solves, and removing it eliminates the entire class of
   reference-policy-miscalibration bugs, not just this one. π\*/wait2/eager/clairvoyant remain valid,
   useful **evaluation** yardsticks (printed alongside `mean_reward` each step for interpretability) —
   just not baked into the gradient.

---

## 3. Pilot history (GRPO, episode-scalar advantage) — five runs, five real bugs, still flat

All five pilots below ran **GRPO** (Group Relative Policy Optimization): reward is one scalar per episode,
advantage is `(r_i - mean(r_group)) / std(r_group)` within a G-sized group of rollouts sharing one
stream/seed, applied uniformly across every token of that episode. No critic, no per-decision structure —
this is the design this doc's §4-§8 ultimately replaces, once every other explanation for a flat reward
curve had been exhausted.

**Infra used by all five** (still current, unaffected by this doc's changes): rollout generation reuses
`urn_session.py`'s existing sampling-mode decoding against Ollama unchanged — no new serving layer.
Rollouts are batched per **policy iteration** ("outer step": ~20-30 seeds × G=4 rollouts ≈ 80-120
episodes under one fixed checkpoint), not per gradient step, to amortize Ollama's ~5min merge→GGUF
convert→`ollama create` resync cost. Every outer step: (1) serve the current checkpoint via Ollama, (2)
collect a batch of rollouts, (3) stop Ollama, load the HF/PEFT policy + resume optimizer state, run one
GRPO update, (4) save checkpoint (adapter + optimizer + a small manifest: step number, reward history,
next-seed cursor) — required so an interrupted/released box costs at most the current in-flight step, not
the whole run (these GPU boxes have no persistent disk). Measured per-outer-step cost: ~6-8 min (resync
~3.3-3.7min, rollout collection ~1.7-3min, GRPO update ~53s at 100 episodes/step).

**An OOM bug worth remembering if this pipeline is touched again:** `prepare_model_for_kbit_training`
upcasts non-4bit params (norms, LoRA A/B, biases) to fp32 for QLoRA training stability — but this promotes
the *whole* downstream computation including Q/K/V to fp32 via ordinary type promotion, which disqualifies
PyTorch's flash/mem-efficient SDPA kernels and silently falls back to a kernel that materializes a full
`(seq_len, seq_len)` fp32 score matrix — confirmed to blow past 79GB by ~3-4K tokens. Fixed in
`load_policy_model` by casting fp32 params back to bf16 after PEFT-wrapping. A second, related bug: a
freshly-constructed PEFT wrapper's decoder layers were stuck in eval mode (`.train()` was never called
recursively), silently disabling gradient checkpointing despite `model.is_gradient_checkpointing==True` —
fixed by calling `model.train()` explicitly after wrapping. Both fixes are already in `rl_train.py`'s
`load_policy_model` — don't reintroduce either regressing this.

### `rl_urn_smoke` — VOID (pi\*-based reward, §2's bug)

7 outer steps on the original `1 - regret/regret_eager` reward. Reward history
`[0.558, -1.937, 0.010, 0.634, <corrupted>, 0.097, 0.097]` — never trended. Superseded entirely by §2's
fix; do not resume.

### v2 — exploration insufficiency + step-size diagnosis

Fresh run on the fixed (raw-balls) reward, 8 outer steps planned, stopped after step 7 on request once the
curve showed no upward trend.

- **Reward history (7 steps):** `[36.42, 37.64, 36.61, 35.63, 36.51, 35.95, 34.62]` — flat, drifting *down*
  toward the eager baseline (34.56), not up toward wait2 (~42-44).
- **Root-caused via a direct rollout diagnostic** (8 rollouts, temperature=0.7, one fixed stream, final
  checkpoint): 6/8 rollouts were byte-identical (KEEP on the first three distinct colors — pure eager),
  0/8 hit a parse failure. **The model re-converged onto the base model's own pre-existing eager
  disposition** — not a new local optimum, just insufficient exploration diversity to escape the starting
  policy's already-low-entropy bias toward eager.
- **Compounding step-size problem:** each outer step is exactly ONE `optimizer.step()` (100 `backward()`
  calls accumulate into one update, not 100 updates) — a 7-8 step pilot is only 7-8 total weight updates,
  ever. AdamW's per-parameter step is ~bounded by `lr` regardless of gradient magnitude, so at the original
  `lr=1e-5` cumulative movement after 7 steps is ~7e-5/weight — plausibly too small to shift behavior at
  all, independent of the exploration problem.
- **Temperature sanity check** (same checkpoint, 8 rollouts each): raising sampling temperature increases
  behavioral diversity without breaking parsing —

  | temp | distinct patterns (of 8) | dominant-mode share | reward range | unparsed |
  |---|---|---|---|---|
  | 0.7 | 3 | 6/8 | 37-44 | 0 |
  | 1.0 | 6 | 3/8 | 21-37 | 0 |
  | 1.2 | 4 | 4/8 | 23-49 | 0 |

  1.2 gave the widest, most informative two-sided spread (two rollouts near-clairvoyant at 48-49 via
  wait-then-commit patterns, two clearly worse at 23) — exactly what GRPO's group-relative advantage
  needs. Parsing stayed clean (0 unparsed) at every temperature tested.
- **Fix attempted:** `RL_LR` raised `1e-5 → 3e-5`; `N_EPOCHS=3` added (reuse each collected batch for 3
  separate `optimizer.step()`s, multiplying weight updates without extra rollout collection — the
  expensive part); rollout temperature raised `0.7 → 1.2`.

### v3 — multi-epoch reuse without an importance-ratio clip caused KL blowup

Relaunched fresh from base model with v2's fix. Temperature=1.2 worked as intended
(`n_zero_advantage_groups` dropped to 0/25 at step 0, `group_std` rose, confirming real diversity). **But
`mean_kl` exploded — `-1.03, -5.49, -16.63, -30.81, -56.82` over 5 steps, accelerating, while `mean_reward`
stayed flat/noisy (37.58, 37.63, 36.05, 35.75, 38.39), not tracking the divergence at all.** Killed at
step 5.

**Root cause, not just "LR too high":** the loss `-(advantage * seq_logp)` is only a valid gradient
estimator when the policy being updated is the one that generated the sampled actions (the on-policy
assumption). Reusing one rollout batch across `N_EPOCHS=3` violates this with nothing to catch it: epoch
2+ recomputes `seq_logp` under the already-shifted post-epoch-1 weights but reapplies the SAME advantage
computed from the original policy, with no mechanism to reduce an example's gradient contribution once its
probability has already moved. Every epoch pushes further in the same direction on the identical examples
— exactly the failure mode PPO's clipped importance-ratio surrogate exists to prevent, which this
implementation didn't have.

**Fix:** reverted `N_EPOCHS` to 1 (kept `RL_LR=3e-5`, temperature=1.2), isolating whether the LR bump
alone was enough before building a real PPO clip. Next pilot restarts fresh from base model, not resumed
from v3's destabilized checkpoint.

### v4 — stable, but reward still flat — two more root causes found

`N_EPOCHS=1` fixed the instability (`mean_kl` bounded: `0.000, -0.143, -0.629, -1.457, ...`, ~20x smaller
than v3's runaway; `n_zero_advantage_groups` stayed low, 0-2/25). But `mean_reward` still didn't move
(`37.43, 36.47, 36.53, 35.18, 36.54`) — killed at step 5. Two further issues:

1. **Reverting `N_EPOCHS` gave up most of the intended step-size fix**, not just the unsafe part: the
   original plan was LR×3 *and* epochs×3 together (~10x cumulative movement); dropping epochs back to 1
   left only the LR×3 (~3x). A flat curve at this point was expected, not a new failure.
2. **The loss used a raw SUM of per-token log-probs, and episode length is confounded with the behavior
   being trained.** `urn_session.run_episode` stops generating decision turns the moment the keep budget
   is exhausted — an eager episode (commits all 3 keeps almost immediately) produces a short transcript; a
   reserve/wait episode produces a much longer one. A sum-based loss therefore gave longer (reserve-like)
   episodes systematically larger-magnitude gradients than shorter (eager-like) ones at the *same*
   advantage value — noise correlated with the thing being learned, not a length-neutral signal.

**Fixes:** `_seq_logprob` changed from a raw sum to a length-normalized **mean** over labeled tokens
(verified against an independent `log_softmax` reference computation, exact match to 1e-4, before
relaunching); `RL_LR` raised `3e-5 → 6e-5` (a second, deliberately incremental step, kept separate from
the `N_EPOCHS` fix so it's legible on its own — still ~40% below SFT's `1e-4`).

### v5 — both fixes applied, ran clean to completion, still flat — the structural diagnosis

Ran the full 8 steps to completion without intervention (first pilot to do so — single process, no OOM, no
crash). `mean_kl` stayed small and bounded throughout; `n_zero_advantage_groups` stayed low (0-1/25) — the
exploration and stability fixes all held.

**Final reward history (8 steps): `[37.89, 35.62, 36.76, 36.52, 36.46, 37.69, 35.5, 37.36]`** — flat,
oscillating in the same 35-38 range as every prior attempt, no trend toward wait2 (~42-44) anywhere. This
is the fourth consecutive pilot (after the pi\*-reward-fixed v2, then v3, v4, v5, each fixing one further
real bug) to show no improvement, despite independently fixing: the reward's reference-policy dependency,
insufficient exploration diversity, step size, multi-epoch instability, and a length confound. **Tuning is
ruled out as the remaining explanation.**

**Root cause: the loss applies ONE scalar advantage uniformly across every token in the episode, with no
per-decision or per-timestep credit propagation** — no critic/value function, which GRPO deliberately
omits (designed for single-shot completions graded as one homogeneous unit, e.g. one math proof, not a
multi-turn sequential-decision trajectory). Reward genuinely depends on only a handful of pivotal decisions
(a first-sighting KEEP/PASS on a color that turns out to matter) out of the ~3-13+ decisions in an episode;
most other decisions are easy/low-information and don't determine the outcome. A flat advantage applied to
all of them dilutes the pivotal decisions' gradient with noise from the rest, and from pure environmental
luck (which stream got drawn) unrelated to the model's choices. This is the standard weakness of vanilla
per-episode REINFORCE/GRPO relative to actor-critic methods (PPO with GAE, TD-learning — the family used
for genuinely sparse-reward tasks like checkpoint-based racing-game RL): those work with sparse reward
specifically because a learned value function propagates credit backward through every intervening
timestep via bootstrapping, not because sparse reward is inherently easy to learn from with a flat,
uncredited scalar.

This also reframes why SFT (§1) reached near-optimal urn performance from only 170 sessions: imitation
learning got dense, per-decision supervision (every one of ~60-100 KEEP/PASS decisions per session
directly labeled with π\*'s correct action) — deliberately not replicated for RL, so the policy would be
*discovered* under reward rather than imitated (§1's whole point) — and that choice is exactly what cost
the sample efficiency here. Fixing this for real means constructing a genuine per-decision advantage
instead of one flat episode-level scalar — which is what the rest of this document specs and implements.

**End-of-pilot-run state:** GPU box released, no persistent disk. Model checkpoints/GGUF artifacts from
all five pilots were NOT pulled (all runs are diagnostic dead ends, not validated results; the merged
model alone is ~28-29GB/run, pure regenerable intermediate; LoRA adapter+optimizer ~800MB/run). Preserved
locally: full `pilot.log`s + `manifest.json`s for all five runs (`runs/box_archive_2026-07-08/`, ~6MB) and
the rollout-diversity diagnostic script (`scripts/box/diag_rl_pilot_rollouts.py`).

---

## 4. The fix: per-decision MDP instead of episode-scalar

Treat each individual KEEP/PASS decision as its own timestep with its own reward, log-prob, and advantage
— instead of the whole episode as one undifferentiated unit with one reward, one log-prob, one advantage.

**The key fact that makes this cheap, not just correct:** "balls collected" is *already* exactly
decomposable per decision, with zero injected outside knowledge (no π\*, no oracle):

- A **KEEP** at `class_position` out of `class_size` total occurrences of that color contributes exactly
  `class_size - class_position + 1` balls (itself plus every future occurrence, auto-collected).
- A **PASS** contributes exactly `0` directly.
- Summing these across all decisions in an episode reproduces the scalar `episode_reward` exactly. This
  isn't reward shaping — it's decomposing the same objective into its already-additive terms
  (`rl_reward.per_decision_rewards`, verified by a sum-matches-scalar self-test).

## 5. Critic (value function) — the genuinely new component

Needed to give **PASS decisions** a meaningful advantage: their direct reward is always 0, but passing on
a color that turns out to be great should still register as a mistake *relative to what keeping would
have earned* — that requires a baseline estimate of expected value at that decision point.

### 5.1 Architecture: tiny separate MLP (not sharing the LLM backbone)

Rejected alternative: a shared-backbone value head (linear head on the LLM's last hidden state, trained
jointly). More expressive, but touches the model's forward pass, adds a second loss to balance against
the policy loss, and complicates the already-OOM-sensitive model-loading code (§3's OOM history). **Use a
separate, decoupled critic instead** (`rl_critic.py`) — simpler, cheaper, and this state space is
small/simple enough that a handful of engineered features should capture most of the signal.

**Input features**, computed per decision point from state already available in `run_episode`'s loop:

| feature | meaning |
|---|---|
| `t / T` | fraction of the episode elapsed |
| `budget_left / B` | fraction of keep-budget remaining |
| `class_position / T` | how many times this color has recurred so far (this draw included) |
| `n_seen_unkept / N` | how many other not-yet-kept colors are "in play," competing for the budget |

**Network:** 4-5 features → 2 hidden layers × 64 units, ReLU → 1 linear scalar output `V(s_t)`, in "balls"
units (same units as the reward). A few thousand parameters — trains in milliseconds, negligible memory.

### 5.2 Privileged critic — decided 2026-07-08: yes, privileged first

**The critic gets a 5th feature: `rate`** — the color's true underlying draw probability, from the
stream's hidden metadata (`stream_builder.py`'s `role`/`rate` fields, already computed, currently used
only for scoring/analysis, never shown to the model).

This is legitimate, not "cheating": the critic is a training-time-only variance-reduction tool, never
consulted by the deployed policy and never seen by the LLM — the policy still only ever sees the same
draw sequence it always has and still has to infer "hot or trap?" from context on its own. Giving the
*critic* privileged info is standard practice in actor-critic RL (asymmetric/centralized-critic designs),
specifically because it makes the baseline far more accurate, which makes the resulting advantage a
cleaner measure of "was this specific decision good or bad" — isolating the policy's actual decision
quality from noise about which random stream got drawn. Without `rate`, the critic has to reconstruct
"hot or trap?" purely from recurrence-so-far — the same noisy inference problem the policy faces, giving a
fair but much slower-converging, noisier baseline.

**Decision (explicit sign-off obtained 2026-07-08): use the privileged version first.** The immediate goal
is "does RL work here at all" — a maximally clean training signal is the fastest way to find out. If it
works, the natural follow-up is rerunning with the non-privileged (fair, 4-feature) critic to see how much
of the effect depended on the privileged information — a clean two-run comparison, not a confound baked
into one attempt. This was escalated for explicit sign-off (not just made autonomously) given how much
this project has historically cared about same-information framing (§2's π\* debate is exactly this kind
of issue playing out once already) — the concern was addressed the same way: the policy's own information
is unchanged either way, only a training-time-only baseline estimator's input differs.

### 5.3 Training

- **Separate optimizer** (its own Adam instance), higher LR than the policy (~1e-3) — a tiny, well-posed
  regression problem (predict Monte Carlo return-to-go; the target never depends on the critic's own
  weights, so there's no staleness/instability analogous to the policy's PPO-clip need).
- **Loss:** plain MSE between `V(s_t)` and the exact return-to-go `G_t` (sum of remaining per-decision
  rewards to the end of that episode — no bootstrapping needed, episodes are short: ~3-13 decisions).
- **Fit to plateau each outer step, not one gradient step** (`rl_critic.fit_critic`; revised 2026-07-08
  pre-box-run review, §8.5 item 1). The original implementation gave the critic ONE Adam step per outer
  step — so an 8-step pilot trains it with 8 total updates, while this module's own self-test needs
  ~200 iterations to fit even a single episode's decisions. A fresh MLP after a handful of updates is
  approximately a constant: `A_t = G_t - const` is just REINFORCE with a mean baseline, and the whole
  §5.2 privileged-feature machinery never engages. `fit_critic` runs full-batch gradient steps with
  early stopping (patience 25 on a 1e-4 min-delta, cap 500) and updates the `ReturnNormalizer` once per
  batch, not per iteration (re-updating on the same returns each iteration would inflate its sample
  count and overweight the current batch against the warm-started history). Cost is nil — CPU, a few
  hundred datapoints, milliseconds next to one 14B forward pass. Fitting the current batch closely is
  the intent, not a hazard: the baseline should track V under the *current* policy, and the warm start
  keeps it anchored across steps.
- **Warm-started across outer steps**, not refit from scratch each time — ~100 episodes/step gives only a
  few hundred decision-level datapoints, too little to fit a fresh model well each step. **Trains jointly
  from step 0, no separate warm-up gate** (decided autonomously, engineering-level: an undertrained early
  critic gives a noisier, not systematically biased, baseline — not judged worth an extra hyperparameter
  for this pipeline's first real test).
- **Return normalization:** a running mean/std of observed `G_t` (`ReturnNormalizer`, Welford's online
  algorithm), normalize the regression target, un-normalize before computing the advantage (keeps the
  advantage in interpretable "balls" units).
- **Persisted alongside the policy checkpoint** (`checkpoint/critic.pt` + `checkpoint/critic_optimizer.pt`),
  same save/resume cycle as the adapter/optimizer, so pilot resumability isn't broken.
- **Stays on CPU throughout** (decided autonomously, engineering-level) — never moved to the GPU the 14B
  policy occupies; trivially fast at this size, keeps VRAM free.

## 6. Advantage, PPO clip, and log-prob plumbing

**Advantage — Monte Carlo return minus baseline (not full GAE):** for each decision `t`,
`A_t = G_t - V(s_t)`, where `G_t` is the exact return-to-go (sum of per-decision rewards from `t` to the
end of the episode) and `V(s_t)` is the critic's baseline. This is GAE with `lambda=1`. Full GAE (the
`delta_t + gamma·lambda·A_{t+1}` backward recursion) is skipped — it exists to trade bias/variance over
*long* horizons via bootstrapping; these horizons are short enough (3-13 decisions) that the exact Monte
Carlo return is already low-variance. **No group-relative (GRPO-style) z-score standardization layered on
top** — decided autonomously, engineering-level: the critic-based baseline already targets the same
variance-reduction goal; can be added back later if the plain version proves too noisy.
`rl_train.compute_advantages` (the old group z-score utility) is kept, tested, and simply unused for now —
the specified fallback if the critic underperforms, not dead code from an abandoned design.

**PPO clipped surrogate — fixes v3's instability (§3) AND re-enables multi-epoch reuse.** Directly targets
v3's root cause: the loss `-(advantage * seq_logp)` has no correction for the policy having moved since
sampling, so multi-epoch reuse compounds unboundedly. Fix:

- Cache each decision's log-prob **at sampling time** (`old_logp_t`, epoch 0, before any update) —
  separate from both the live/current `new_logp_t` and the frozen-reference `ref_logp_t`. At epoch 0,
  `old_logp_t := new_logp_t.detach()` (the standard trick: ratio starts at exactly 1 but gradient still
  flows through `new_logp_t`, giving the same gradient as a plain policy-gradient loss on the first pass).
- Each subsequent epoch: `ratio_t = exp(new_logp_t - old_logp_t)`.
- `loss_t = -min(ratio_t * A_t, clip(ratio_t, 1-ε, 1+ε) * A_t)` (standard PPO-clip; `ε=0.2` as a starting
  default, genuinely untested on this task).
- Once an example's ratio drifts outside `[1-ε, 1+ε]`, its gradient contribution stops growing — the exact
  mechanism missing when v3's `N_EPOCHS=3` caused the runaway.
- **`N_EPOCHS` stays at 1 in this implementation regardless** — re-enabling multi-epoch reuse (the clip's
  actual payoff) is deliberately deferred until a real box run confirms this whole new pipeline is stable
  at `n_epochs=1` first (§8's build order).

**Per-decision log-probs — the tokenization/masking rework.** `build_example` (`train_lora.py`) now also
returns `turn_spans`: a `[(start, end), ...]` list, one half-open range into `input_ids`/`labels` per
ASSISTANT message, in order — `urn_session.run_episode` emits exactly one assistant message per KEEP/PASS
decision, so turn `i` here IS decision `i` in `per_decision_rewards`. Backward-compatible: SFT training
(this file's other callers) ignores the extra key. `rl_train._per_decision_logprobs` replaces the old
`_seq_logprob`: still **one forward pass** over the full concatenated transcript (efficient — no N
separate forward passes per episode), but instead of collapsing every assistant token into one scalar, it
segments the per-token log-probs by turn span and returns one **length-normalized mean** per decision
(mean, not sum — same length-confound reasoning as v4/v5's whole-episode fix, §3, generalized to the
within-episode case: some decisions elicit longer reasoning replies than others, and a sum would give
those a bigger-magnitude gradient at the same advantage). Reference log-probs get the identical
restructuring, still computed once via `disable_adapter()` and cached/reused across epochs.

**What stays the same:** G-way rollout collection per seed (kept for exploration diversity — the
temperature=1.2 fix, §3, proved this matters — no longer the *source* of the advantage, just still cheap
and useful); the KL-vs-frozen-reference term (kept, now computed and applied per-decision); checkpointing/
resume, the Ollama serve/resync cycle, temperature=1.2 (all unaffected); `MAX_GRAD_NORM` clipping (kept as
a floor-level safety net regardless of the PPO clip).

**Known scale caveat on the KL term (2026-07-08 review, §8.5 item 4):** every stable pilot ran
`KL_BETA=0.05` against unit-scale (z-scored) advantages; the per-decision advantage is now in raw
"balls" units (|A| plausibly ~5-15 while the critic is still converging), so the pg:KL balance inside
`pg_loss + beta·kl_t` is ~10x tilted toward the policy gradient relative to those pilots. AdamW's moment
normalization absorbs *overall* gradient scale but not the *relative* weighting inside the summed loss.
Deliberately not retuned pre-emptively (one change at a time; the balls-unit advantage is worth keeping
for interpretability if it holds) — `grpo_step` logs `mean_abs_advantage` each step so the imbalance is
visible, and if `mean_kl` runs on the first real run, the first lever is raising `KL_BETA` or normalizing
advantages by a running |A| scale, NOT another LR change (v3's "LR too high" first-guess already cost one
pilot's worth of misdiagnosis).

## 7. Design decisions and sign-off record

1. **Privileged (`rate`-aware) vs. fair critic (§5.2).** ESCALATED for explicit sign-off, given the
   project's same-information framing history (§2). **Decided 2026-07-08: privileged first**, fair
   4-feature version as a planned follow-up comparison once RL is confirmed to work at all.
2. **Group-relative (GRPO-style) standardization on top of the critic baseline?** Engineering-level, not
   escalated. **Decided: no, start without it** — see §6.
3. **Clip epsilon and re-enabled `N_EPOCHS`.** `ε=0.2` and `N_EPOCHS` staying at 1 for the first real run
   — genuinely untested guesses/defaults, not yet tried against a real model (§6, §8).
4. **Critic warm-up gate.** Engineering-level, not escalated. **Decided: no gate** — train jointly from
   step 0 (§5.3).
5. **Two SAC-derived extensions, discussed but explicitly deferred (2026-07-08), not implemented:**
   (a) a genuine cross-outer-step replay buffer to amortize the ~5min Ollama resync cost across more
   gradient steps than one batch currently buys (this pipeline already gets a *related* benefit for free
   once the PPO clip makes multi-epoch reuse of one batch safe, but a true replay buffer spanning multiple
   outer steps' rollouts is a further, separate idea); (b) an explicit entropy-bonus term in the loss, as
   a more principled alternative to decode-time temperature for maintaining exploration. Both were judged
   out of scope for this round — the credit-assignment gap (§3-§6) is the diagnosed root cause; these are
   unvalidated extensions layered on top of a not-yet-tested change. Revisit only after §4-§6 are
   confirmed to actually move reward on a real box.
   (SAC and DQN themselves were considered and rejected as wholesale alternative algorithms, not just as
   sources of borrowable ideas: SAC's core efficiency mechanism, off-policy replay reuse, targets expensive
   *environment interactions*, but this pipeline's measured bottleneck is the Ollama resync, not rollout
   generation — and its continuous-control machinery doesn't fit this discrete text-action space anyway.
   DQN's epsilon-greedy/value-based paradigm doesn't map onto autoregressive generation without abandoning
   training the LLM's own generative disposition, which would defeat the point of this investigation.
   Policy-gradient-family methods (GRPO → PPO+critic) remain the right fit regardless of the downstream
   decision being discrete.)

## 8. Implementation (files, status, and suggested build order)

**Files, all implemented as of 2026-07-08:**

- `rl_reward.py` — `per_decision_rewards` (§4) added; `episode_reward` (scalar) kept unchanged, still used
  for logging/evaluation. `builds_to_transcript` added to reconstruct a decision-turn list from a
  `pi_star.py` heuristic's `builds` dict, for testing without an LLM.
- `rl_critic.py` — new module: `extract_features` (privileged/fair, §5.1-5.2), `returns_to_go`,
  `ReturnNormalizer`, `build_critic`/`train_critic_step`/`critic_values` (the MLP, §5.1/5.3),
  `save_critic`/`load_critic`; `fit_critic` (fit-to-plateau, §5.3 / §8.5 item 1 — the entry point
  `grpo_step` actually calls; `train_critic_step` kept as the single-step primitive underneath).
- `rl_train.py` — `_per_decision_logprobs` replaces `_seq_logprob` (§6); `grpo_step` reworked to: compute
  per-decision rewards/returns/advantages via the critic, cache sampling-time log-probs, apply the
  PPO-clipped loss per decision, fit the critic alongside (`fit_critic`), log
  `critic_loss`/`mean_advantage`/`mean_abs_advantage`; `advantage_diagnostic` + `behavioral_metrics`
  (§8.5 items 2-3) printed every step and returned in the stats dict;
  `save_checkpoint`/new `load_critic_optimizer_state` persist the critic, and the manifest now also
  carries `behavior_history` (first-sight %/lateness per step) so the disposition trend survives a box
  interruption the same way the reward trend does.
- `train_lora.py` — `build_example` emits `turn_spans` (§6), backward-compatible.
- `rl_urn_pilot.py` — builds/resumes the critic + its optimizer + normalizer each outer step, alongside
  the existing policy adapter/optimizer resume; appends per-step behavior to the manifest's
  `behavior_history` (backward-compatible with pre-review manifests via `.get`) and prints
  first-sight %/lateness in the per-step summary line.

**Suggested build order** (why it was built in this sequence, useful if any of it needs to be redone):

1. Per-decision reward decomposition (§4) — cheap, exact, verifiable in isolation before touching any
   model code. DONE, self-tested.
2. Critic (§5) trained standalone against synthetic/heuristic-policy episodes (real replay data wasn't
   available — see Status at top for why) — validates the value estimates look sane before wiring into
   the loss. DONE, self-tested (loss decreases, save/load round-trips).
3. Per-decision log-prob plumbing (§6) — the tokenization rework, independently testable against hand-built
   fake logits (turn-span recombination should reproduce the whole-sequence masked sum). DONE, verified to
   1e-3.
4. PPO-clipped loss (§6) using the now-available per-decision advantages and log-probs. DONE, implemented
   inside the same `grpo_step` rewrite as step 3 — not separately re-verified beyond the shift-indexing
   check above, since it needs a real model forward pass to test meaningfully.
5. Re-enable `N_EPOCHS>1` only after 1-4 are confirmed stable at `N_EPOCHS=1` on a real box. NOT DONE —
   this is the next lever to pull, strictly after a real run, not before.

**What a real box run needs to check, in order:** (a) does the pipeline run at all without crashing/OOM
across a full outer step against the real 14B model; (b) **the step-0 mechanism gate** (§8.5 item 2):
does the very first batch's `advantage_diagnostic` already order decision quality correctly
(`keep_hot > keep_trap`, `pass_trap > pass_hot_first`) *before any policy update*? If not, STOP — the
remaining outer steps have nothing to train on; debug the critic/decomposition instead of burning box
time; (c) does `mean_kl` stay bounded (the v3 failure mode) at `n_epochs=1` — watching
`mean_abs_advantage` for the pg:KL scale caveat (§6); (d) do the behavioral metrics (first-sight %,
lateness) drift in the reserve direction across several outer steps — expected to resolve *before*
`mean_reward` beats its own noise at ~100 episodes/step; (e) does `mean_reward` actually trend upward —
the entire point of this doc's changes; (f) only then, try re-enabling `n_epochs>1` with the PPO clip
active.

## 8.5 Pre-box-run review (2026-07-08) — changes made before any box spend

An external review of §4-§8 as implemented, done the same day, before provisioning a box. Overall
verdict: the credit-assignment diagnosis (§3) is well-supported (the strongest evidence being the SFT
contrast — dense per-decision imitation installed the policy from 170 sessions while episode-scalar RL
got nothing from ~800 episodes; same model, task, and data scale, differing only in credit density),
and the §4-§6 design is the standard, appropriately-sized remedy. But as wired, the first run had a
real chance of coming back flat for reasons *other than* the hypothesis being wrong — which would be
ambiguous in exactly the way pilots v2-v5 were. Three changes plus one watch-item, all applied and
re-verified against the local self-tests:

1. **Critic was getting one gradient step per outer step — fixed with `fit_critic` (fit to plateau).**
   The catch: 8 outer steps = 8 total Adam updates for a fresh MLP whose own self-test needs ~200
   iterations on far less data. Consequence if unfixed: a near-constant baseline, advantage degenerates
   to `G_t - const` (REINFORCE with a mean baseline), the privileged `rate` feature (§5.2) never does
   anything, and the pilot silently doesn't test the thing it exists to test. Fix costs milliseconds
   (CPU, few hundred datapoints). Detail in §5.3.
2. **Step-0 advantage-ordering diagnostic added (`advantage_diagnostic`).** Buckets mean advantage by
   decision type × ground-truth role (keep_hot / keep_trap / pass_hot_first / pass_hot_later /
   pass_trap — role comes from the same hidden stream metadata the privileged critic uses, never shown
   to the policy). Purpose: decouple "is the credit signal right?" from "did 8 policy updates move a
   14B model?" — the two things a flat reward curve can't distinguish, and the exact ambiguity that
   made v2-v5 expensive. If the ordering is present at step 0, later flatness indicts step count, not
   credit assignment; if absent, stop immediately. This is the cheapest possible falsification point
   for the whole §4-§6 hypothesis and it runs before any meaningful spend.
3. **Behavioral metrics per step (`behavioral_metrics`), persisted in the manifest.** The project's own
   standing lead metrics (first-sight %, lateness — plan §0) were oddly absent from the RL logging,
   which judged pilots on `mean_reward` alone. At ~100 episodes/step, reward noise is large; a real
   eager→reserve disposition shift shows up in first-sight % several steps earlier. Also the reason a
   longer run (20-40 outer steps, not 8) should be planned: the "wait" signal is second-order — per-
   decision rewards make first-sight keeps on hot colors look locally *great* (`class_size -
   class_position + 1` is maximal at first sight), and waiting only pays through budget conservation
   entering via `G_t` minus an accurate baseline — so expect slow movement even if everything works.
4. **Watch-item, deliberately not pre-tuned: pg:KL scale imbalance** (~10x vs. every stable pilot, from
   the advantage's raw balls units). Logged via `mean_abs_advantage`; response documented at `KL_BETA`
   and in §6. Not changed now because it's a hypothesis about a *future* failure, and this project's
   pilot history says: one change at a time, and don't re-guess "LR" first.

One reviewer note kept for the `n_epochs>1` step later (§8 step 5): the per-turn MEAN log-prob makes
the PPO ratio `exp(mean_new - mean_old)` a per-token geometric-mean ratio, so the clip binds much later
for long turns than short ones. Irrelevant while `n_epochs=1` (the clip is inert on-policy), but when
multi-epoch reuse is enabled, remember the effective clip width varies with turn length.

## 8.6 Box sizing — the first real run targets a 40GB A100 (2026-07-08)

Single H100s were out of capacity; a 40GB A100 is available and is sufficient. The two memory phases
never overlap (Ollama is stopped before the HF/PEFT policy loads), so the card is sized against the
worse phase, not the sum:

- **Training phase — fits with margin.** QLoRA 4-bit 14B base ≈ 9-10GB; with the two §3 OOM fixes
  active (bf16 recast keeping SDPA fast kernels + gradient checkpointing actually on), measured
  scaling was ~1.7MB/token → ~15-20GB peak at this pilot's ≤3.7k-token episodes. (The 80GB OOM war
  stories in `box-setup.md` all came from the SFT corpus's 21k-token tool-bridge sessions, which this
  run never touches.) The merge subprocess loads the base in bf16 (~28GB) while Ollama is down — fits.
- **Serving phase — the binding constraint, two knobs changed in `rl_urn_pilot.py`:**
  1. **GGUF quant `q8_0` by default** (`--gguf-outtype`, was hardcoded f16). f16 (~28GB) + 8-slot KV
     (~5-7GB) is exactly borderline on 40GB, and the failure mode isn't a crash — it's the runbook's
     documented silent partial CPU offload (~8x rollout slowdown). q8_0 (~15GB) is effectively
     lossless and leaves real headroom. The residual quant mismatch between the sampled policy and the
     bf16 training weights is a mild off-policy wrinkle accepted knowingly: the pre-FT baselines ran
     on the stock Ollama tag (Q4), so q8_0 sampling is *closer* to the training weights than the
     baselines were to theirs. On an 80GB box, pass `--gguf-outtype f16`.
  2. **`num_ctx` pinned to 8192 explicitly** (base-model step 0 now serves a ctx-capped variant tag,
     `qwen-rl-base-ctx8k`; FT modelfiles get the same `PARAMETER num_ctx` line). Two reasons: KV
     sizing becomes planned (8 slots × 8192 × ~0.2MB/token ≈ 13GB) instead of whatever the box's
     Ollama version defaults to; and it gives 2x headroom over the longest measured episode (~3.7k
     tokens at temp 1.2) — Ollama truncates context *from the front* when `num_ctx` is exceeded,
     which would silently corrupt long episodes rather than error.
- **Speed expectation:** ~1.5-2x slower than H100 on generation/update, but per-step cost is
  resync-dominated (much of it CPU-bound merge/convert), so figure ~9-11 min/outer step vs. the
  measured ~7 — a 20-step run lands ~3-3.5h plus setup/debug.
- Runbook section for provisioning this box: `docs/box-setup.md` §B3.
