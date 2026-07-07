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

from scripts.creator.tool_disposition_benchmark.rl_reward import episode_reward
from scripts.creator.tool_disposition_benchmark.rl_rollout import group_by_seed
from scripts.creator.tool_disposition_benchmark.train_lora import (
    BASE_MODEL, LORA_ALPHA, LORA_DROPOUT, LORA_R, TARGET_MODULES, build_example)

RL_LR = 1e-5    # deliberately much smaller than SFT's 1e-4 -- policy-gradient updates on a sparse,
                # episode-level reward are noisier per-step than dense token-level SFT supervision;
                # this is a starting point to sanity-check against the pilot's actual reward curve, not
                # a tuned value.
KL_BETA = 0.05
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
def _seq_logprob(logits, input_ids, labels):
    """Sum of log p(token) over positions where labels != -100, using the standard causal-LM shift
    (logits[t] predicts input_ids[t+1]). Same label convention `build_example` produces, consumed the
    same way HF's own labels-based loss would, just kept as a per-token log-prob instead of an NLL.

    Deliberately avoids materializing a full (seq_len, vocab≈152k) log-softmax tensor in fp32 -- that
    doubles the size of an already-large per-example tensor and is the EXACT failure mode
    `train_lora.py` already documents hitting during eval ("casting the padded batch's logits to fp32
    ... tried to allocate ~73GB"). Instead: `gather` the target token's raw logit and compute
    `logsumexp` directly on the (bf16) logits -- both reduce the vocab dimension away, so neither ever
    creates a second same-shape tensor; only the tiny (1, seq_len) result is upcast to fp32."""
    import torch
    shift_logits = logits[:, :-1, :]
    shift_ids = input_ids[:, 1:]
    shift_mask = (labels[:, 1:] != -100)
    gathered = shift_logits.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)   # (1, L-1)
    logsumexp = torch.logsumexp(shift_logits, dim=-1)                        # (1, L-1) -- no full
                                                                              # (L-1, vocab) copy made
    token_logp = (gathered - logsumexp).float()
    return (token_logp * shift_mask).sum(dim=-1).squeeze(0)   # scalar (batch size is always 1 here)


def compute_advantages(rewards: list[float]) -> list[float]:
    """Group-relative advantage within one G-sized group. A degenerate group (all G rollouts scored
    identically -- a real possibility early on, e.g. if temperature sampling hasn't diverged the
    policy's decisions yet) returns zero advantage for all members rather than dividing by zero: no
    learning signal from that group this step, not a crash."""
    if len(rewards) < 2:
        return [0.0 for _ in rewards]
    mean = st.mean(rewards)
    std = st.pstdev(rewards)
    if std < 1e-9:
        return [0.0 for _ in rewards]
    return [(r - mean) / std for r in rewards]


# --------------------------------------------------------------------- one GRPO update over a whole batch
def grpo_step(model, tokenizer, optimizer, batch_results: list[dict],
             beta: float = KL_BETA, max_grad_norm: float = MAX_GRAD_NORM) -> dict:
    """batch_results: the flat list `rl_rollout.collect_batch` returns ({seed, slots, row}). Scores
    reward per rollout, computes group-relative advantage within each seed's group, then does ONE
    gradient step averaged over the whole batch (per-example forward/backward, see module docstring).
    Returns {mean_reward, mean_balls, loss, mean_kl, n_zero_advantage_groups} for logging."""
    import torch

    for r in batch_results:
        kept = {int(k): v for k, v in r["row"]["kept"].items()}
        r["reward_info"] = episode_reward(r["slots"], kept)

    groups = group_by_seed(batch_results)
    n_zero_adv_groups = 0
    for seed, members in groups.items():
        advs = compute_advantages([m["reward_info"]["reward"] for m in members])
        if all(a == 0.0 for a in advs):
            n_zero_adv_groups += 1
        for m, a in zip(members, advs):
            m["advantage"] = a

    optimizer.zero_grad()
    n = len(batch_results)
    total_loss = total_kl = 0.0
    device = next(model.parameters()).device

    examples = [build_example(r["row"]["messages"], tokenizer, tools=None) for r in batch_results]
    lens = sorted(len(ex["input_ids"]) for ex in examples)
    print(f"  sequence lengths: min={lens[0]} p50={lens[len(lens)//2]} p95={lens[int(0.95*len(lens))]} "
          f"max={lens[-1]}", flush=True)

    for i, (r, ex) in enumerate(zip(batch_results, examples)):
        ids = torch.tensor([ex["input_ids"]], device=device)
        labels = torch.tensor([ex["labels"]], device=device)

        logits = model(input_ids=ids).logits
        seq_logp = _seq_logprob(logits, ids, labels)
        del logits

        with torch.no_grad():
            with model.disable_adapter():
                ref_logits = model(input_ids=ids).logits
            ref_seq_logp = _seq_logprob(ref_logits, ids, labels)
        del ref_logits
        kl = (seq_logp - ref_seq_logp).detach()
        total_kl += kl.item()

        loss = (-(r["advantage"] * seq_logp) + beta * (seq_logp - ref_seq_logp)) / n
        loss.backward()
        total_loss += loss.item()
        del seq_logp, ref_seq_logp, kl, loss, ids, labels
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()   # per-example variable-length forward/backward fragments the
                                       # caching allocator badly enough (measured: OOM partway through
                                       # a 100-example loop despite expandable_segments) that periodic
                                       # reclaim is needed, not just relying on refcounting + `del`.
            print(f"  [{i+1}/{n}] seqlen={len(ex['input_ids'])} "
                  f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
                  f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    rewards = [r["reward_info"]["reward"] for r in batch_results]
    # Per-group reward spread, not just the batch-wide mean -- distinguishes "this step's batch was
    # uniformly hard" (per-group std similar to the overall std) from "one catastrophic outlier
    # dominated a single group" (one group's std much larger than the rest, which also dilutes GRPO's
    # advantage signal for that group's other members, since they all share that inflated denominator).
    group_stds = [st.pstdev([m["reward_info"]["reward"] for m in members]) for members in groups.values()
                  if len(members) >= 2]
    return {"mean_reward": st.mean(rewards), "reward_std": st.pstdev(rewards),
            "reward_min": min(rewards), "reward_max": max(rewards),
            "mean_group_std": st.mean(group_stds) if group_stds else 0.0,
            "max_group_std": max(group_stds) if group_stds else 0.0,
            "loss": total_loss,
            "mean_kl": total_kl / n, "n_groups": len(groups),
            "n_zero_advantage_groups": n_zero_adv_groups}


# --------------------------------------------------------------------- checkpoint / resume
def save_checkpoint(model, tokenizer, optimizer, out_dir: Path, step: int,
                    reward_history: list[float], next_seed: int) -> None:
    import torch
    ck = out_dir / "checkpoint"
    ck.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ck / "adapter"))
    tokenizer.save_pretrained(str(ck / "adapter"))    # merge_lora.py loads the tokenizer FROM the
                                                       # adapter dir (exact chat_template.jinja used in
                                                       # training) -- omitting this breaks the merge
                                                       # step's AutoTokenizer.from_pretrained call.
    torch.save(optimizer.state_dict(), ck / "optimizer.pt")
    manifest = {"step": step, "reward_history": reward_history, "next_seed": next_seed}
    (ck / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_manifest(out_dir: Path) -> dict | None:
    manifest_path = out_dir / "checkpoint" / "manifest.json"
    return json.loads(manifest_path.read_text()) if manifest_path.exists() else None


def load_optimizer_state(optimizer, out_dir: Path, device) -> None:
    import torch
    optimizer.load_state_dict(torch.load(out_dir / "checkpoint" / "optimizer.pt", map_location=device))


def _dry_run():
    """No GPU/model -- just exercises the reward/advantage/grouping logic against fake rollouts, and
    the checkpoint manifest round-trip against a temp dir."""
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
