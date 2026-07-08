# PPO-style per-decision credit assignment — spec (2026-07-08)

Companion to `rl-finetuning-plan.md` (read "Pilot v5" there first — this doc exists because of what it
found).

**Status (2026-07-08, later same day): IMPLEMENTED, UNTESTED ON A REAL BOX.** §9 sign-off obtained
(privileged critic first; both SAC-derived ideas — cross-outer-step replay buffer, entropy bonus —
deferred, not folded in). §1-§8 built and unit-tested locally (CPU-only torch installed in `.venv` for
this purpose) against synthetic/heuristic-policy data, since the GPU box from the GRPO pilots was
released and no rollout data was persisted to replay (`rl_rollout.collect_batch` never wrote to disk) —
substituted `rl_reward.builds_to_transcript` + `pi_star`'s eager/wait2 heuristics to exercise the new code
paths end-to-end without an LLM. Specifically verified, all passing:
  - `rl_reward.per_decision_rewards` sums to the exact scalar `episode_reward` on eager/wait2/none.
  - `rl_critic`: feature extraction (privileged vs fair, shape/alignment/range), `returns_to_go`,
    `ReturnNormalizer` (matches `statistics.mean/stdev`), MLP trains (loss ↓ over 200 steps on synthetic
    data), critic checkpoint save/load round-trips to identical predictions.
  - `rl_train._per_decision_logprobs`: shift-indexing verified against the old whole-sequence masked-sum
    approach on hand-built fake logits (token-count-weighted recombination of per-turn means reproduces
    the whole-sequence sum to 1e-3).
  - `build_example`'s new `turn_spans` output is backward-compatible (SFT callers ignore the extra key).
  - Full module import chain (`rl_urn_pilot.py` → `rl_train.py` → `rl_critic.py`/`train_lora.py`/
    `urn_session.py`) resolves without error.

**NOT yet verified — needs a real GPU box, nothing to test this against locally:**
  - The actual `model(input_ids=...)` forward/backward path through `_per_decision_logprobs` (only the
    pure-tensor shift math was checked, not a real tokenizer/model's logits).
  - Whether reward actually moves now that credit assignment is per-decision (the entire point).
  - Wall-clock/VRAM impact of the critic's extra CPU-side forward passes each step (expected negligible
    given its size, unverified in practice).
  - `N_EPOCHS` stays at 1 in this implementation (see `rl_train.py`'s `N_EPOCHS` comment) per §10's own
    suggested order — re-enabling multi-epoch reuse is deliberately deferred until a real run confirms
    this pipeline is stable at `n_epochs=1`, not bundled into this same untested change.

Design choices made autonomously while implementing (engineering-level, not framing decisions, so not
escalated for separate sign-off — flagging here for visibility): critic trains jointly from step 0, no
warm-up gate (§9.4); critic stays on CPU throughout, never moved to the GPU the 14B policy occupies;
`CLIP_EPS=0.2` as the untested starting default (§9.3); per-decision log-prob is a MEAN within each turn
(not a sum), for the same length-confound reason pilot v5 made the whole-episode version a mean.

## 0. Why this doc exists

Five pilots (`rl_urn_smoke`, v2-v5 in `rl-finetuning-plan.md`) progressively fixed real bugs — the
reward's dependency on a miscalibrated π* reference, insufficient rollout exploration diversity,
too-small step size, a multi-epoch instability (KL blowup), and a sequence-length confound in the loss
— and reward still never moved. With every tuning explanation exhausted, the remaining diagnosis is
structural: **vanilla per-episode GRPO applies one scalar advantage uniformly to every token in an
episode, with no per-decision credit propagation.** This task's reward depends on a handful of pivotal
decisions (first-sighting KEEP/PASS calls on colors that turn out to matter) diluted among many easier
ones (most PASS decisions are close to obviously-correct and don't determine the outcome) — a flat
per-episode signal buries the pivotal decisions' gradient in noise from the rest, and from pure
environmental luck (which stream got drawn) unrelated to the model's choices. Standard sparse-reward RL
(checkpoint-based racing games, robotics) handles exactly this via a learned value function that
propagates credit backward through every intervening timestep via bootstrapping — GRPO deliberately
omits that (it's designed for single-shot-graded-as-a-whole completions, e.g. one math proof, not
multi-turn sequential decision trajectories like this one). This doc specs what adding that back looks
like, concretely, for this codebase.

## 1. Reframe the problem: episode-scalar → per-decision MDP

Currently the whole episode (however many KEEP/PASS turns it has) is treated as one undifferentiated
unit: one reward, one log-prob, one advantage. The fix: treat each individual decision as its own
timestep with its own reward, log-prob, and advantage.

**The key fact that makes this cheap, not just correct:** "balls collected" is *already* exactly
decomposable per decision, with zero injected outside knowledge (no π*, no oracle) —

- A **KEEP** at `class_position` out of `class_size` total occurrences of that color contributes
  exactly `class_size - class_position + 1` balls (itself plus every future occurrence, auto-collected).
  This is already what `_balls_collected` (`urn_session.py`) sums over kept classes.
- A **PASS** contributes exactly `0` directly.
- Summing these across all decisions in an episode reproduces the current scalar reward exactly. This
  isn't reward shaping — it's decomposing the same objective into its already-additive terms.

## 2. Reward: per-decision array, not a scalar (`rl_reward.py`)

Replace `episode_reward()`'s scalar output with a **per-decision reward array**, aligned 1:1 with the
transcript's decision turns: `[class_size - class_position + 1 if KEEP else 0, ...]`.

**Required self-test:** `sum(per_decision_rewards) == episode_reward(...)`'s old scalar value, on the
same episode — a cheap, exact correctness check for the decomposition, not just a smoke test.

## 3. Critic (value function) — the genuinely new component

Needed to give **PASS decisions** a meaningful advantage: their direct reward is always 0, but passing
on a color that turns out to be great should still register as a mistake *relative to what keeping
would have earned* — that requires a baseline estimate of expected value at that decision point.

### 3.1 Architecture: tiny separate MLP (not sharing the LLM backbone)

Rejected alternative: a shared-backbone value head (add a linear head on the LLM's last hidden state,
train jointly). More expressive, but touches the model's forward pass, adds a second loss to balance
against the policy loss, and complicates the OOM-sensitive model-loading code this pilot has already
fought all session. **Use a separate, decoupled critic instead** — simpler, cheaper, and this state
space is small/simple enough that a handful of engineered features should capture most of the signal.

**Input features** (computed per decision point from state already available in `run_episode`'s loop —
no new data collection needed):

| feature | meaning |
|---|---|
| `t / T` | fraction of the episode elapsed |
| `budget_left / B` | fraction of keep-budget remaining |
| `class_position / T` | how many times this color has recurred so far (this draw included) |
| `n_seen_unkept / N` | how many other not-yet-kept colors are "in play," competing for the budget |

**Network:** 4 features → 2 hidden layers × 64 units, ReLU → 1 linear scalar output `V(s_t)`, in "balls"
units (same units as the reward, directly comparable/subtractable). A few thousand parameters — trains
in milliseconds, negligible GPU memory impact.

### 3.2 Open framing decision: should the critic see privileged information the policy doesn't?

**Recommend adding a 5th feature: `rate`** — the color's true underlying draw probability, from the
stream's hidden metadata (`role`/`rate` fields already computed by `stream_builder.py`, currently used
only for scoring/analysis, never shown to the model).

This is legitimate, not "cheating" — the critic is a training-time-only variance-reduction tool, never
consulted by the deployed policy and never seen by the LLM; the LLM still only sees the same draw
sequence it always has and still has to learn to infer "is this probably hot or trap?" from context on
its own. Giving the *critic* privileged info is standard practice in actor-critic RL (asymmetric /
centralized-critic designs) specifically because it makes the baseline far more accurate, which makes
the resulting advantage a cleaner measure of "was this specific decision good or bad" — isolating the
policy's actual decision quality from noise about which random stream got drawn, rather than
conflating the two. Without `rate`, the critic has to reconstruct "hot or trap?" purely from
recurrence-so-far — exactly the same noisy inference problem the policy faces, giving a fair but much
slower-converging, noisier baseline.

**Recommendation: use the privileged version first.** The immediate goal is "does RL work here at
all" — a maximally clean training signal is the fastest way to find out. If it works, a natural,
well-motivated follow-up is rerunning with the non-privileged (fair, 4-feature) critic to see how much
of the effect depended on the privileged information — a clean two-run comparison, not a confound
baked into one attempt. **This choice needs explicit sign-off before implementing** — it's a framing
decision, not just an engineering detail, given how much this project has cared about
same-information framing elsewhere (the π* debate in `rl-finetuning-plan.md`).

### 3.3 Training

- **Separate optimizer** (its own Adam instance), higher LR than the policy (~1e-3) — a tiny,
  well-posed regression problem (predict Monte Carlo return-to-go; the target never depends on the
  critic's own weights, so there's no staleness/instability analogous to the policy's PPO-clip need).
- **Loss:** plain MSE between `V(s_t)` and the exact return-to-go `G_t` (sum of remaining per-decision
  rewards to the end of that episode — no bootstrapping needed, episodes are short: ~3-13 decisions).
- **Warm-started across outer steps**, not refit from scratch each time — ~100 episodes/step gives only
  a few hundred decision-level datapoints, too little to fit a fresh model well each step.
- **Return normalization:** track a running mean/std of observed `G_t`, normalize the regression
  target, un-normalize before computing the advantage (keep advantage in interpretable "balls" units).
- **Persisted alongside the policy checkpoint** (`checkpoint/critic.pt`), same save/resume cycle as the
  adapter/optimizer (`rl_train.save_checkpoint`/`load_manifest`), so pilot resumability isn't broken.

## 4. Advantage estimation: Monte Carlo return minus baseline (not full GAE)

For each decision `t`: `A_t = G_t - V(s_t)`, where `G_t` is the exact return-to-go (sum of per-decision
rewards from `t` to the end of the episode) and `V(s_t)` is the critic's baseline at that decision
point. This is GAE with `lambda=1`. **Skip full GAE** (the `delta_t + gamma·lambda·A_{t+1}` backward
recursion) — it exists to trade bias/variance over *long* horizons via bootstrapping; our horizons are
short enough (3-13 decisions) that the exact Monte Carlo return is already low-variance and needs no
extra machinery. Simpler to implement, simpler to debug.

## 5. PPO clipped surrogate — fixes v3's instability AND re-enables multi-epoch reuse

Directly targets the root cause diagnosed in pilot v3: the loss `-(advantage * seq_logp)` has no
correction for the policy having moved since sampling, so multi-epoch reuse compounds unboundedly
(mean_kl exploded `-1 → -57` over 5 steps with no reward gain). Fix:

- Cache each decision's log-prob **at sampling time** (`old_logp_t`, epoch 0, before any update) —
  separate from both the live/current `seq_logp_t` and the frozen-reference `ref_logp_t`.
- Each subsequent epoch: `ratio_t = exp(new_logp_t - old_logp_t)`.
- `loss_t = -min(ratio_t * A_t, clip(ratio_t, 1-ε, 1+ε) * A_t)` (standard PPO-clip; ε=0.2 as a starting
  default).
- Once an example's ratio drifts outside `[1-ε, 1+ε]`, its gradient contribution stops growing — the
  exact mechanism missing when `N_EPOCHS=3` caused the runaway. With this in place, **multi-epoch
  reuse becomes safe again**, so `N_EPOCHS` can go back up (2-4) to recover the sample-efficiency
  benefit originally wanted from it, this time without the instability.

## 6. Per-decision log-probs — the tokenization/masking rework

`_seq_logprob` (`rl_train.py`) currently collapses the *entire* transcript's assistant tokens into one
scalar (mean, as of the length-normalization fix). Needs to become: still **one forward pass** over the
full concatenated transcript (efficient, keep this — don't do N separate forward passes per episode),
but instead of collapsing everything, **segment the per-token log-probs by which decision-turn they
belong to** and return a list of per-turn scalars, aligned index-for-index with the per-decision
rewards/advantages from §2. Requires `build_example` (`train_lora.py`) or a variant to also emit
turn-boundary token indices, not just the flat `input_ids`/`labels` it produces now. Same restructuring
needed for the reference log-probs (still computed once via `disable_adapter()`, still cached and
reused across epochs — that part of v4's design was fine and carries over unchanged).

## 7. What stays the same

- **G-way rollout collection per seed** (`rl_rollout.py`) — keep it, still useful for exploration
  diversity (temperature=1.2 fix proved this matters); no longer the *source* of the advantage (that's
  now the critic/Monte-Carlo-baseline), but still cheap and worth keeping for the diversity it buys.
- **KL-vs-frozen-reference term** — keep it, now computed and applied per-decision like everything else.
- **Checkpointing/resume, Ollama serve/resync cycle, temperature=1.2** — all unaffected.
- **`MAX_GRAD_NORM` clipping** — keep as the floor-level safety net regardless of the PPO clip.

## 8. New / modified files (summary)

- `rl_reward.py` — `episode_reward()` returns a per-decision array; add the sum-matches-scalar
  self-test.
- New: a small critic module (e.g. `rl_critic.py`) — MLP definition, feature extraction from episode
  state, training loop (own optimizer, MSE loss, return normalization), save/load.
- `rl_train.py` — `_seq_logprob` reworked for per-decision segmentation; `grpo_step` reworked to: (a)
  compute per-decision rewards/returns/advantages via the critic, (b) cache sampling-time log-probs,
  (c) apply the PPO-clipped loss per decision, (d) train the critic alongside, (e) re-enable
  `N_EPOCHS>1` once the clip is in place.
- `train_lora.py`'s `build_example` (or a new variant) — emit turn-boundary indices alongside
  `input_ids`/`labels`.
- `rl_urn_pilot.py` — persist/resume the critic checkpoint alongside the adapter/optimizer.

## 9. Open design choices needing a decision before implementing

1. **§3.2 — privileged (`rate`-aware) vs. fair critic.** Recommended: privileged first, fair as a
   follow-up comparison. Needs explicit sign-off given the project's same-information framing history.
2. **Does group-relative (GRPO-style) standardization still play any role**, e.g. as an extra
   normalization layer on top of the Monte-Carlo-baseline advantage? Recommend: no, start without it —
   the critic-based baseline already targets the same variance-reduction goal; add back only if the
   plain version is still too noisy.
3. **Clip epsilon (ε) and re-enabled `N_EPOCHS` value** — ε=0.2 and `N_EPOCHS=3-4` are reasonable
   starting points, genuinely untested guesses until tried.
4. **Critic warm-up.** An untrained critic early on could give a *worse* baseline than no baseline
   (advantage ≈ raw return, higher variance but not systematically biased toward a bad estimate) —
   worth deciding whether to gate PPO-clip training behind a few steps of critic-only fitting first, or
   just let both train jointly from step 0 and accept some early noise.

## 10. Suggested implementation order (if not doing it all at once)

1. Per-decision reward decomposition (§2) — cheap, exact, easy to verify in isolation before touching
   any model code.
2. Critic (§3) trained standalone against logged rollouts from an existing pilot's data (e.g. replay
   v5's collected episodes) — validates the value estimates look sane before wiring it into the loss.
3. Per-decision log-prob plumbing (§6) — the tokenization rework, independently testable against a
   known transcript (turn boundaries should sum back to the whole-sequence log-prob as a sanity check).
4. PPO-clipped loss (§5) using the now-available per-decision advantages and log-probs.
5. Re-enable `N_EPOCHS>1` (§5's payoff) only after 1-4 are confirmed stable at `N_EPOCHS=1`.
