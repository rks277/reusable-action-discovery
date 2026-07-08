"""RL Phase 1 GRPO loss/step + checkpointing (docs/rl-finetuning-plan.md "Phase 1: RL on urn, from base
model"). The one genuinely new piece of code this phase needs -- rollout generation (`rl_rollout.py`)
and reward (`rl_reward.py`) both reuse existing harness code unchanged.

Design choices, all argued in the doc, not re-litigated here:
  - Reference policy = the SAME model with its LoRA adapter disabled (`peft`'s `disable_adapter()`
    context) -- valid specifically because Phase 1 starts from the untouched base model, so "reference"
    and "base" are the same weights. No second model copy in memory.
  - Per-example forward/backward (not batched across the group) -- simplest correct thing for a first
    pilot; episodes have different lengths (variable # of KEEP/PASS decisions), so batching would need
    padding/collation machinery this pilot doesn't need yet. Revisit if per-step wall-clock is a problem.
  - KL estimator = the plain log-ratio (seq_logp - ref_seq_logp), not a lower-variance estimator (e.g.
    k3) -- standard, unbiased in expectation; swap later only if KL estimates prove too noisy.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.rl_train --dry-run   # no GPU/model
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from scripts.creator.tool_disposition_benchmark.rl_critic import (
    ReturnNormalizer, critic_values, extract_features, load_critic, returns_to_go, save_critic,
    train_critic_step)
from scripts.creator.tool_disposition_benchmark.rl_reward import episode_reward, per_decision_rewards
from scripts.creator.tool_disposition_benchmark.rl_rollout import group_by_seed
from scripts.creator.tool_disposition_benchmark.train_lora import (
    BASE_MODEL, LORA_ALPHA, LORA_DROPOUT, LORA_R, TARGET_MODULES, build_example)
from scripts.creator.tool_disposition_benchmark.urn_session import B, N, T

RL_LR = 6e-5    # 1e-5 -> 3e-5 -> 6e-5 (2026-07-08). Each outer step is ONE optimizer.step() (100
                # backward() calls accumulate into a single update, not 100 updates), so an 8-step pilot
                # takes only 8 total weight updates ever; AdamW's per-parameter step is ~bounded by `lr`
                # regardless of gradient magnitude (moment-normalized), so 1e-5 gave a ceiling of only
                # ~8e-5 cumulative movement -- too small to see anything. 3e-5 (pilot v4, N_EPOCHS=1,
                # stable) still showed a flat reward curve after 5 steps, so raising further. NOTE this
                # is a DIFFERENT risk axis than the N_EPOCHS staleness bug (see below) -- that was a
                # specific compounding pathology now removed; this is the plain, still-live reason RL_LR
                # was originally kept well below SFT's 1e-4: single-episode reward is sparse and noisy,
                # so a bigger single step still means trusting a noisier gradient direction more. 6e-5
                # (6x the original, still ~40% below SFT's rate) is a deliberately incremental second
                # step up, not a jump straight to SFT's value, so this change is legible on its own
                # rather than conflated with a much larger one. MAX_GRAD_NORM still clips per-step
                # regardless of LR as a floor-level safety net.
N_EPOCHS = 1    # tried 3 (2026-07-08 pilot v3): mean_kl exploded (-1 -> -57 over 5 steps, still
                # accelerating) with NO reward improvement -- root cause, not just "LR too high": the
                # loss (-(advantage * seq_logp)) is only a valid gradient estimator when the policy being
                # updated is the one that generated the sampled actions. Reusing one rollout batch across
                # multiple epochs breaks that -- epoch 2+ reapplies the epoch-0 advantage to an
                # already-shifted policy with no correction, so each epoch pushes further in the same
                # direction with nothing to stop it once an example has already moved. `grpo_step` now
                # has the PPO clip (CLIP_EPS below) that fixes exactly this, per
                # docs/rl-ppo-credit-assignment-spec.md §5 -- but this constant STAYS at 1 until the new
                # per-decision+critic+clip pipeline has a confirmed-stable run at n_epochs=1 (the doc's
                # own suggested build order, §10 step 5): re-enabling multi-epoch reuse is the LAST lever
                # to pull, not bundled into the same untested step as everything else.
KL_BETA = 0.05
CLIP_EPS = 0.2  # PPO clip epsilon (spec §5) -- a standard default, genuinely untested on this task (spec
                # §9.3): no pilot has run with it yet. Once ratio_t = exp(new_logp_t - old_logp_t) drifts
                # outside [1-CLIP_EPS, 1+CLIP_EPS], that decision's gradient contribution stops growing --
                # the mechanism the N_EPOCHS=3 blowup above was missing.
MAX_GRAD_NORM = 1.0


# --------------------------------------------------------------------- model loading (resume-aware)
def load_policy_model(qlora: bool, resume_adapter: Path | None):
    """Base-loading mirrors `train_lora.load_hf` exactly (same torch/cuDNN workaround, same
    quantization config) so the RL policy starts from bit-identical footing to the SFT pipeline --
    only the PEFT-wrapping step differs (resume an existing adapter vs. fresh `get_peft_model`)."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    if qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kw)
    base.config.use_cache = False
    if qlora:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    else:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()

    if resume_adapter is not None and resume_adapter.exists():
        model = PeftModel.from_pretrained(base, str(resume_adapter), is_trainable=True)
        print(f"resumed adapter from {resume_adapter}")
    else:
        model = get_peft_model(base, LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
            task_type="CAUSAL_LM", target_modules=TARGET_MODULES))
        print("initialized fresh LoRA adapter from base (no resume checkpoint found)")

    # `prepare_model_for_kbit_training` upcasts every non-4bit param (norm weights, LoRA A/B, biases)
    # to fp32 for training stability -- standard QLoRA practice. But Qwen2RMSNorm's `self.weight *
    # hidden_states.to(input_dtype)` and LoRA's `base_out + lora_B(lora_A(x))` both then promote the
    # WHOLE downstream computation (including Q/K/V) to fp32 via normal type promotion. fp32 Q/K/V
    # disqualifies PyTorch's flash/mem-efficient SDPA kernels (bf16/fp16 only), forcing a silent
    # fallback to the "math" backend, which materializes a full (seq_len, seq_len) score matrix in
    # fp32 -- confirmed via direct reproduction to blow past 79GB on a single ~3-4k token episode
    # (measured OOM signature: reserved memory scaling ~quadratically with sequence length, e.g.
    # 1378 tokens -> 65.9GB). `train_lora.py`'s proven, default `--backend unsloth` path never hits
    # this because Unsloth's LoRA kernels manage compute dtype internally; this HF/PEFT-mirroring path
    # needs it done explicitly. Cast back to bf16 (matching `bnb_4bit_compute_dtype`/`torch_dtype`
    # above) -- confirmed via direct repro this restores the bf16 fast-kernel path with no OOM even at
    # 3671 tokens, longer than any sequence seen in this pilot's rollouts so far.
    n_cast = sum(1 for p in model.parameters() if p.dtype == torch.float32)
    for p in model.parameters():
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
    print(f"cast {n_cast} fp32 params (norms/LoRA/biases) back to bf16 to keep SDPA on the fast path")

    # `AutoModelForCausalLM.from_pretrained` on a quantized model leaves every decoder layer's own
    # `.training` flag False even though the freshly-constructed PEFT wrapper's top-level `.training`
    # defaults to True (nn.Module.__init__'s default) -- neither `get_peft_model` nor
    # `PeftModel.from_pretrained` calls `.train()` to recursively fix this. HF's
    # `GradientCheckpointingLayer.__call__` gates checkpointing on `self.gradient_checkpointing AND
    # self.training`, so every decoder layer was silently running WITHOUT checkpointing despite
    # `model.is_gradient_checkpointing == True` -- full (non-checkpointed) activations were retained,
    # confirmed via direct repro: peak backward memory grew ~14MB/token (5049 tokens -> 82GB, OOM) with
    # layers stuck in eval mode, dropping to ~1.7MB/token (7049 tokens -> 22.5GB) after this fix. This
    # was the real cause of the OOMs that resurfaced after the fp32/SDPA fix above -- that fix removed
    # the O(seq^2) math-kernel blowup, but checkpointing being inactive still left memory scaling
    # (linearly, but steeply) with sequence length, biting again on any batch with an unusually long
    # episode. Verified `model.train()` doesn't disturb gradient flow: the no-grad reference pass still
    # emits its expected (harmless) "no inputs require grad" warning once per checkpointed layer, while
    # the live pass's `seq_logp.requires_grad` stays True and LoRA params get real nonzero gradients
    # after `.backward()`.
    model.train()

    model.print_trainable_parameters()
    return model, tokenizer


# --------------------------------------------------------------------- log-probs
def _per_decision_logprobs(logits, input_ids, turn_spans: list[tuple[int, int]]):
    """Per-decision (per KEEP/PASS turn) MEAN log p(token), one scalar per entry in `turn_spans`
    (`build_example`'s per-assistant-turn `(start, end)` ranges, UNSHIFTED input_ids/labels space) --
    replaces the old whole-episode `_seq_logprob` (docs/rl-ppo-credit-assignment-spec.md §6), which
    collapsed every assistant token in the transcript into one scalar. Mean, not sum, WITHIN each turn
    for the same reason the old fix (2026-07-08) made the whole-episode version a mean: a sum would give
    a decision whose reasoning happens to run longer a bigger-magnitude gradient at the same advantage,
    which is exactly the length-confound pattern already found once at the episode level (see
    `rl_reward.per_decision_rewards`'s docstring for how this aligns with the reward's own decomposition).

    Standard causal-LM shift (logits[t] predicts input_ids[t+1]): shift-space index j predicts unshifted
    position j+1, so an unshifted span [start, end) maps to shift-space [start-1, end-1).

    Deliberately avoids materializing a full (seq_len, vocab≈152k) log-softmax tensor in fp32 -- that
    doubles the size of an already-large per-example tensor and is the EXACT failure mode
    `train_lora.py` already documents hitting during eval ("casting the padded batch's logits to fp32
    ... tried to allocate ~73GB"). Instead: `gather` the target token's raw logit and compute
    `logsumexp` directly on the (bf16) logits -- both reduce the vocab dimension away, so neither ever
    creates a second same-shape tensor; only the tiny (1, seq_len) result is upcast to fp32."""
    import torch
    shift_logits = logits[:, :-1, :]
    shift_ids = input_ids[:, 1:]
    gathered = shift_logits.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)   # (1, L-1)
    logsumexp = torch.logsumexp(shift_logits, dim=-1)                        # (1, L-1) -- no full
                                                                              # (L-1, vocab) copy made
    token_logp = (gathered - logsumexp).float().squeeze(0)                   # (L-1,)
    return [token_logp[start - 1:end - 1].mean() for start, end in turn_spans]


def compute_advantages(rewards: list[float]) -> list[float]:
    """Group-relative (GRPO-style) advantage within one G-sized group -- z-score within the group,
    zero if degenerate (all G rollouts scored identically). NO LONGER CALLED by `grpo_step` (superseded
    by the critic-based per-decision advantage, docs/rl-ppo-credit-assignment-spec.md §9.2: "start
    without [this] -- the critic-based baseline already targets the same variance-reduction goal; add
    back only if the plain version is still too noisy"). Kept, tested, and unused rather than deleted
    for exactly that reason -- it's the specified fallback if the critic underperforms, not dead code
    from a design that's gone."""
    if len(rewards) < 2:
        return [0.0 for _ in rewards]
    mean = st.mean(rewards)
    std = st.pstdev(rewards)
    if std < 1e-9:
        return [0.0 for _ in rewards]
    return [(r - mean) / std for r in rewards]


# --------------------------------------------------------------------- one PPO update over a whole batch
def grpo_step(model, tokenizer, optimizer, batch_results: list[dict], critic, critic_optimizer,
             normalizer: ReturnNormalizer, *, beta: float = KL_BETA, max_grad_norm: float = MAX_GRAD_NORM,
             clip_eps: float = CLIP_EPS, n_epochs: int = N_EPOCHS) -> dict:
    """Per-decision PPO-clipped credit assignment (docs/rl-ppo-credit-assignment-spec.md), replacing the
    original episode-scalar GRPO update -- see that doc's §0 for why: a flat per-episode advantage buried
    the gradient from a handful of pivotal decisions among many easy ones, and reward never moved across
    5 pilots even after fixing every other bug (reference misspecification, exploration diversity, step
    size, an N_EPOCHS staleness blowup, a length confound). `batch_results`: the flat list
    `rl_rollout.collect_batch` returns ({seed, slots, row}).

    Per rollout:
      1. `rl_reward.per_decision_rewards` decomposes the scalar reward into one term per KEEP/PASS turn.
      2. `rl_critic.returns_to_go` gives the exact Monte Carlo return G_t at each decision (no
         bootstrapping -- episodes are short, spec §4).
      3. `rl_critic.extract_features`/`critic_values` gives the baseline V(s_t) from a PRIVILEGED critic
         (spec §3.2, sign-off 2026-07-08: sees the color's true draw rate; the policy never does -- a
         training-time-only variance-reduction input, standard asymmetric-critic practice).
      4. Advantage A_t = G_t - V(s_t) directly -- NO group-relative z-score layered on top (spec §9.2;
         see `compute_advantages`'s docstring for why that's parked, not deleted).

    `batch_results` still comes from `collect_batch`'s G-way-per-seed sampling (kept for exploration
    diversity, spec §7) but `group_by_seed` is used here ONLY for an informational reward-spread log, not
    to compute the advantage.

    Critic trains ALONGSIDE the policy every step (own optimizer + MSE loss, spec §3.3), warm-started
    across outer steps (caller persists/reloads `critic`/`critic_optimizer`/`normalizer` via
    `rl_critic.save_critic`/`load_critic` -- see `rl_urn_pilot.py`), on the flattened per-decision
    (features, returns) pairs across the WHOLE batch -- decisions, not episodes, are its training
    examples. Trained BEFORE forming this step's advantages, so the advantage always uses the freshest
    baseline; no separate warm-up gate (spec §9.4 -- judged not worth an extra hyperparameter for this
    pipeline's first real test: an undertrained early critic gives a noisier, not systematically biased,
    baseline).

    PPO clip (spec §5) is what fixes pilot v3's root cause and makes multi-epoch reuse safe again --
    `n_epochs` defaults to `N_EPOCHS` (currently 1 regardless: see that constant's comment) rather than
    being bumped here, per the doc's own suggested build order (§10): confirm this new pipeline is
    stable at n_epochs=1 before compounding with multi-epoch reuse.

    Returns {mean_reward, critic_loss, mean_advantage, loss, mean_kl, mean_group_std} for logging."""
    import torch

    for r in batch_results:
        kept = {int(k): v for k, v in r["row"]["kept"].items()}
        r["reward_info"] = episode_reward(r["slots"], kept)
        transcript = r["row"]["transcript"]
        r["dec_rewards"] = per_decision_rewards(r["slots"], transcript)
        r["returns"] = returns_to_go(r["dec_rewards"])
        r["features"] = extract_features(r["slots"], transcript, T=T, B=B, N=N, privileged=True)
        assert abs(sum(r["dec_rewards"]) - r["reward_info"]["reward"]) < 1e-6, \
            "per-decision decomposition must reproduce the scalar reward exactly"

    # ---- critic: one MSE step on the whole batch's flattened decisions, then read fresh baselines ----
    all_features = [f for r in batch_results for f in r["features"]]
    all_returns = [g for r in batch_results for g in r["returns"]]
    critic_loss = train_critic_step(critic, critic_optimizer, normalizer, all_features, all_returns)
    flat_values = critic_values(critic, normalizer, all_features)
    cursor = 0
    for r in batch_results:
        k = len(r["features"])
        r["advantages"] = [g - v for g, v in zip(r["returns"], flat_values[cursor:cursor + k])]
        cursor += k

    # informational only (NOT used for the advantage -- see docstring)
    groups = group_by_seed(batch_results)
    group_stds = [st.pstdev([m["reward_info"]["reward"] for m in members]) for members in groups.values()
                  if len(members) >= 2]

    n = len(batch_results)
    device = next(model.parameters()).device

    examples = [build_example(r["row"]["messages"], tokenizer, tools=None) for r in batch_results]
    for r, ex in zip(batch_results, examples):
        assert len(ex["turn_spans"]) == len(r["dec_rewards"]), \
            (len(ex["turn_spans"]), len(r["dec_rewards"]))   # turn_spans must align 1:1 with decisions
    lens = sorted(len(ex["input_ids"]) for ex in examples)
    print(f"  sequence lengths: min={lens[0]} p50={lens[len(lens)//2]} p95={lens[int(0.95*len(lens))]} "
          f"max={lens[-1]}", flush=True)

    # ---- reference (frozen base) per-decision log-probs -- computed ONCE, reused every epoch ----
    ref_logps = []
    with torch.no_grad():
        for i, ex in enumerate(examples):
            ids = torch.tensor([ex["input_ids"]], device=device)
            labels = torch.tensor([ex["labels"]], device=device)
            with model.disable_adapter():
                ref_logits = model(input_ids=ids).logits
            ref_logps.append([lp.detach() for lp in
                              _per_decision_logprobs(ref_logits, ids, ex["turn_spans"])])
            del ref_logits, ids, labels
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

    # "sampling-time" log-probs for the PPO ratio -- populated from epoch 0's own forward pass (the
    # standard on-policy-epoch trick: at epoch 0 the policy being updated IS the one that produced the
    # rollouts, so old_logp := new_logp.detach() gives ratio=1 while still routing gradient through
    # new_logp; genuinely fixed/stale only from epoch 1 onward).
    old_logps = None
    n_decisions_total = sum(len(r["dec_rewards"]) for r in batch_results)

    total_loss = total_kl = 0.0
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        epoch_loss = epoch_kl = 0.0
        new_logps_this_epoch = []
        for i, (r, ex) in enumerate(zip(batch_results, examples)):
            ids = torch.tensor([ex["input_ids"]], device=device)
            labels = torch.tensor([ex["labels"]], device=device)

            logits = model(input_ids=ids).logits
            new_logps = _per_decision_logprobs(logits, ids, ex["turn_spans"])
            del logits
            new_logps_this_epoch.append([lp.detach() for lp in new_logps])

            cur_old = old_logps[i] if old_logps is not None else [lp.detach() for lp in new_logps]
            per_ex_loss = 0.0
            for new_lp, old_lp, ref_lp, A in zip(new_logps, cur_old, ref_logps[i], r["advantages"]):
                ratio = torch.exp(new_lp - old_lp)
                pg_loss = -torch.min(ratio * A, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A)
                kl_t = new_lp - ref_lp
                per_ex_loss = per_ex_loss + pg_loss + beta * kl_t
                epoch_kl += kl_t.item()
            per_ex_loss = per_ex_loss / max(len(new_logps), 1) / n   # equal weight per EPISODE
                                                                     # regardless of its decision count,
                                                                     # then equal weight per episode
                                                                     # across the batch (n = # rollouts)
            per_ex_loss.backward()
            epoch_loss += per_ex_loss.item()
            del new_logps, ids, labels
            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()   # per-example variable-length forward/backward fragments the
                                           # caching allocator badly enough (measured: OOM partway
                                           # through a 100-example loop despite expandable_segments)
                                           # that periodic reclaim is needed, not just del/refcounting.
                print(f"  epoch {epoch+1}/{n_epochs} [{i+1}/{n}] seqlen={len(ex['input_ids'])} "
                      f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
                      f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

        if old_logps is None:
            old_logps = new_logps_this_epoch

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += epoch_loss
        total_kl += epoch_kl
        print(f"  epoch {epoch+1}/{n_epochs} done: loss={epoch_loss:.4f} "
              f"mean_kl={epoch_kl/n_decisions_total:.3f}", flush=True)

    rewards = [r["reward_info"]["reward"] for r in batch_results]
    all_advantages = [a for r in batch_results for a in r["advantages"]]
    return {"mean_reward": st.mean(rewards), "reward_std": st.pstdev(rewards),
            "reward_min": min(rewards), "reward_max": max(rewards),
            "critic_loss": critic_loss, "mean_advantage": st.mean(all_advantages),
            "mean_group_std": st.mean(group_stds) if group_stds else 0.0,
            "max_group_std": max(group_stds) if group_stds else 0.0,
            "loss": total_loss,
            "mean_kl": total_kl / (n_decisions_total * n_epochs), "n_groups": len(groups)}


# --------------------------------------------------------------------- checkpoint / resume
def save_checkpoint(model, tokenizer, optimizer, critic, critic_optimizer, normalizer: ReturnNormalizer,
                    out_dir: Path, step: int, reward_history: list[float], next_seed: int) -> None:
    import torch
    ck = out_dir / "checkpoint"
    ck.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ck / "adapter"))
    tokenizer.save_pretrained(str(ck / "adapter"))    # merge_lora.py loads the tokenizer FROM the
                                                       # adapter dir (exact chat_template.jinja used in
                                                       # training) -- omitting this breaks the merge
                                                       # step's AutoTokenizer.from_pretrained call.
    torch.save(optimizer.state_dict(), ck / "optimizer.pt")
    save_critic(critic, normalizer, ck / "critic.pt")             # spec §3.3: warm-started across steps
    torch.save(critic_optimizer.state_dict(), ck / "critic_optimizer.pt")
    manifest = {"step": step, "reward_history": reward_history, "next_seed": next_seed}
    (ck / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_manifest(out_dir: Path) -> dict | None:
    manifest_path = out_dir / "checkpoint" / "manifest.json"
    return json.loads(manifest_path.read_text()) if manifest_path.exists() else None


def load_optimizer_state(optimizer, out_dir: Path, device) -> None:
    import torch
    optimizer.load_state_dict(torch.load(out_dir / "checkpoint" / "optimizer.pt", map_location=device))


def load_critic_optimizer_state(critic_optimizer, out_dir: Path, device) -> None:
    import torch
    critic_optimizer.load_state_dict(
        torch.load(out_dir / "checkpoint" / "critic_optimizer.pt", map_location=device))


def _dry_run():
    """No GPU/model -- exercises the reward/advantage/grouping logic against fake rollouts, the
    checkpoint manifest round-trip, and (if torch is installed) the per-decision log-prob shift-indexing
    math against hand-built fake logits (docs/rl-ppo-credit-assignment-spec.md §10 step 3: "independently
    testable against a known transcript" -- no real tokenizer/model needed for this part, since the shift
    math only depends on tensor shapes and indices, not real vocabulary)."""
    import tempfile
    from scripts.creator.tool_disposition_benchmark.stream_builder import (
        StochasticStreamSpec, build_stochastic_stream)
    from scripts.creator.tool_disposition_benchmark.urn_session import UNIFORM, N, T, B, MAG
    from scripts.creator.tool_disposition_benchmark.pi_star import eager_builds

    slots, _ = build_stochastic_stream(StochasticStreamSpec(
        families=UNIFORM, n_hot=B, T=T, budget=B, guarantee_trap_early=1.0, magnitude=MAG, seed=9000))
    fake_batch = []
    for i in range(4):
        kept = eager_builds(slots, B) if i % 2 == 0 else {}
        fake_batch.append({"seed": 9000, "slots": slots,
                           "row": {"kept": {k: v for k, v in kept.items() if v is not None},
                                   "messages": [{"role": "user", "content": "x"},
                                               {"role": "assistant", "content": "DECISION: PASS"}]}})
    for r in fake_batch:
        kept = {int(k): v for k, v in r["row"]["kept"].items()}
        r["reward_info"] = episode_reward(r["slots"], kept)
    groups = group_by_seed(fake_batch)
    advs = compute_advantages([m["reward_info"]["reward"] for m in groups[9000]])
    assert len(advs) == 4 and abs(sum(advs)) < 1e-6, advs   # z-scored -> sums to ~0
    print(f"advantages: {[round(a, 3) for a in advs]}")

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        (out_dir / "checkpoint").mkdir()
        manifest = {"step": 3, "reward_history": [0.1, 0.2, 0.3], "next_seed": 9012}
        (out_dir / "checkpoint" / "manifest.json").write_text(json.dumps(manifest))
        loaded = load_manifest(out_dir)
        assert loaded == manifest, loaded
    print("rl_train dry-run OK (reward/advantage/grouping/manifest logic; no model loaded)")

    try:
        import torch
    except ImportError:
        print("torch not installed -- skipping _per_decision_logprobs shift-indexing check "
              "(runs on the GPU box where this module is actually used)")
        return

    torch.manual_seed(0)
    vocab, seq_len = 7, 12
    logits = torch.randn(1, seq_len, vocab)
    input_ids = torch.randint(0, vocab, (1, seq_len))
    labels = torch.full((1, seq_len), -100)
    turn_spans = [(2, 5), (7, 9), (10, 12)]     # 3 fake "decisions" -- disjoint, in order, like
                                                # build_example's real assistant-turn spans
    for start, end in turn_spans:
        labels[0, start:end] = input_ids[0, start:end]

    # ground truth: the OLD whole-sequence masked approach (sum, not mean, so it's directly comparable
    # to a token-count-weighted recombination of the new per-turn means)
    shift_logits, shift_ids = logits[:, :-1, :], input_ids[:, 1:]
    shift_mask = (labels[:, 1:] != -100)
    gathered = shift_logits.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
    logsumexp = torch.logsumexp(shift_logits, dim=-1)
    whole_token_logp = (gathered - logsumexp).float().squeeze(0)
    whole_sum = (whole_token_logp * shift_mask.squeeze(0)).sum().item()

    per_turn_means = _per_decision_logprobs(logits, input_ids, turn_spans)
    assert len(per_turn_means) == len(turn_spans)
    recombined_sum = sum(mean.item() * (end - start) for mean, (start, end) in zip(per_turn_means, turn_spans))
    assert abs(recombined_sum - whole_sum) < 1e-3, (recombined_sum, whole_sum)
    print(f"_per_decision_logprobs shift-indexing OK (recombined sum {recombined_sum:.4f} "
          f"== whole-sequence masked sum {whole_sum:.4f})")

    # critic checkpoint round trip (rl_critic.save_critic/load_critic, called from save_checkpoint)
    from scripts.creator.tool_disposition_benchmark.rl_critic import build_critic
    with tempfile.TemporaryDirectory() as td:
        critic = build_critic(n_features=5)
        normalizer = ReturnNormalizer()
        normalizer.update([1.0, 2.0, 3.0])
        path = Path(td) / "critic.pt"
        save_critic(critic, normalizer, path)
        critic2, normalizer2 = build_critic(n_features=5), ReturnNormalizer()
        load_critic(critic2, normalizer2, path, device="cpu")
        assert normalizer2.mean == normalizer.mean and normalizer2.std == normalizer.std
    print("critic checkpoint round trip OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        _dry_run()
    else:
        raise SystemExit("no standalone train mode -- see scripts/creator/tool_disposition_benchmark/"
                         "rl_urn_pilot.py for the outer-step orchestration loop")
