# RL fine-tuning plan (GRPO) — alternative to SFT bridge

Companion to `qwen-finetune-transfer-plan.md` (the SFT attempt this supersedes as the active
approach — read it first for the two SFT results and why both are stalled) and
`online-tool-investment-plan.md` (headline claim, §0 Orientation for notation). GPU box mechanics
live in `docs/box-setup.md`.

## Status (2026-07-06)

**PLANNED, not started.** No training run, no code beyond what's described here. This doc exists to
scope the approach before spending GPU time on it.

## Why RL instead of SFT

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

## Algorithm: GRPO

GRPO (Group Relative Policy Optimization), not PPO:

- Reward here is a clean scalar computed at episode end from the existing scorer (regret vs. π\*,
  balls collected) — no reward model needed, so PPO's extra machinery buys nothing.
- No critic/value network to train — smaller memory footprint on the single GPU that already needed
  a cuDNN backward-pass workaround (`train_lora.py`'s `enable_cudnn_sdp(False)`) to survive long
  sessions in plain SFT.
- Advantage is computed within a group of G rollouts sampled from the same starting condition (same
  seed/stream): `A_i = (r_i - mean(r_group)) / std(r_group)`. This is the standard credit-assignment
  substitute for a value function when reward is episode-level rather than per-token.

## Reward design

Reuse the existing scorer, not a new metric:

- **Tool arm:** `reward = -regret` (or normalized, `1 - regret/regret_eager`, to keep it roughly in
  [0,1] across seeds), from the same episode-end computation the eval harness already produces.
  Add a per-episode penalty for malformed/no-tool-call turns (e.g. `-0.1 * n_malformed`) so the
  model can't win by degenerating into something that happens to end early with few penalties —
  this also directly discourages the collapse pattern SFT hit.
- **Urn arm (comparison only, not required):** `reward = balls_collected / balls_pi_star` for that
  seed — the same "% of π\*" number already reported in the SFT results tables.

## Rollout generation

- Environment = the existing `run_session()` in `driver.py`, unchanged, run in **sampling mode**
  (temperature > 0) instead of eval's greedy/near-greedy decoding.
- Per training prompt (a seed/stream instance), sample a group of **G rollouts** (G=4–8) at the
  current checkpoint. This is the expensive part: each rollout is a full session — up to 60
  problems, up to a 200k-token cap — so a gradient step costs G× that. **Start with short episodes**
  (N=8/T=20, mirroring the SFT urn demos' length) for iteration speed and cost before scaling to
  full-length sessions.
- Serving for rollouts should be **vLLM, not Ollama** — RL needs high-throughput sampling of many
  rollouts, not the few careful greedy generations eval needs, and Ollama's tool-call parsing was
  already a source of friction in the SFT pipeline (`qwen-finetune-transfer-plan.md` §Serving: "vLLM
  +hermes is a dead end for Qwen-Coder tool-calling" — that parser problem applies equally to
  RL-time sampling, so it needs a working forced/parseable tool-call path in vLLM, not the GGUF→
  Ollama route used for SFT eval).

## Loss / masking

Reuse the SFT collator's masking logic (`train_lora.py: make_collator`, prefix-diffed against the
tokenizer's chat template) but apply it to **log-probs** instead of NLL labels: sum/average
log-prob only over assistant-generated spans (including tool_call spans — those are the decisions),
excluding injected user turns (`FORMAT_REMINDER`, next-problem prompts) and tool-result turns. The
existing template-diffing approach is directly reusable — no new masking logic needed.

## Guardrails

- **KL penalty against the SFT/base checkpoint** (standard in GRPO) — without it, reward hacking
  toward a degenerate-but-high-reward policy (e.g. spamming `submit_answer` on garbage, or
  exploiting an edge case in the regret computation) is a real risk with a coarse episode-level
  reward.
- **Keep the malformed-output penalty non-trivial.** Otherwise GRPO could "solve" the collapse by
  learning to avoid tool calls in ambiguous states entirely, rather than actually recovering from
  them — a shortcut that would look like success on the malformed-output metric while failing the
  actual task.

## Infra

Don't reach for TRL's `GRPOTrainer` as-is — it assumes single-turn prompt→completion, not this
multi-turn tool-loop-with-env-injected-turns shape. Two options:

1. **Custom GRPO loop**, reusing `driver.py` for rollouts (sampling mode) and the assistant-token
   masking logic for per-token log-probs, hand-rolling the group-relative advantage + PEFT gradient
   step. Consistent with the rest of this pipeline (SFT trainer and eval driver are both hand-rolled
   already).
2. **A multi-turn-agentic-RL library** (e.g. `verifiers`, OpenPipe's ART) that already handles
   interleaved env turns — worth a quick look before committing to (1), but not evaluated yet.

LoRA config (rank/alpha/target modules/dropout) can likely stay identical to the SFT run
(`train_lora.py`: `r=32, α=64, dropout=0.05`, all seven linear target modules) — GRPO updates the
same adapter weights, just with a policy-gradient loss instead of NLL.

## Cost / scope

RL needs orders of magnitude more rollouts than SFT needs demonstrations, and each rollout here is a
long, expensive session — this is a materially bigger and slower experiment than either SFT attempt
so far. Per [[no-auto-reps]]-style practice, don't launch a full run without first scoping a concrete
pilot: episode length, group size G, number of GRPO steps, and an estimated box-hour/cost budget,
proposed and agreed before spending GPU time.

## Open follow-ups

1. **Scope a pilot** — short episodes (N=8/T=20), small G (4), a capped number of GRPO steps, tool
   arm only, before considering a full-length run. Not started.
2. **Reward function implementation** — wire the existing regret/balls scorer into a per-episode
   scalar reward callable from a rollout loop; add and tune the malformed-turn penalty. Not started.
3. **vLLM forced-tool-call path for Qwen2.5-Coder** — the SFT plan's serving section documents that
   `vLLM+hermes` silently drops tool calls under `tool_choice="auto"` for this model family; RL
   rollout sampling needs a working forced/parseable path in vLLM specifically (not the GGUF→Ollama
   route, which isn't built for high-throughput sampling). Not started.
4. **Decide custom loop vs. library** (`verifiers`/ART) for the multi-turn GRPO rollout-and-update
   loop. Not evaluated.
5. **Comparison to SFT.** If RL installs the policy and survives the tool-calling channel, compare
   against the SFT urn numbers (`qwen-finetune-transfer-plan.md` Result 2: 7% first-sight, 101% of
   π\* balls) — is RL's ceiling in the tool framing similar, better, or does it hit a different wall?
