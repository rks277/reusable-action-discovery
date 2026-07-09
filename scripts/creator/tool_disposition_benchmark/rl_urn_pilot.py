"""RL Phase 1 outer-step orchestration loop (docs/rl-finetuning-plan.md "Phase 1: RL on urn, from base
model"). Ties together rl_rollout.py (Ollama-served sampling), rl_reward.py (regret-based reward),
and rl_train.py (GRPO update) into the batched-per-policy-iteration design the doc specs:

  repeat for `--steps` outer steps:
    1. serve the current checkpoint via Ollama (base model at step 0; merge/GGUF/`ollama create` from
       the last saved adapter otherwise -- this ~5min resync is why rollouts are batched per POLICY
       iteration, not per gradient step)
    2. collect a batch of rollouts (`--seeds-per-step` fresh seeds x `--G` each) via Ollama, conc=8
    3. stop Ollama (frees VRAM), load the HF/PEFT policy + resume optimizer state, run ONE GRPO update
    4. save checkpoint (adapter + optimizer + manifest with the running reward history + next seed
       cursor) -- REQUIRED for resumability (see doc's "Checkpointing / resumability"): every step
       leaves the run in a state that survives an interrupted box, not just script-restart.

Every outer step unloads the model to free VRAM (measured ~5-10s reload cost, not a bottleneck) --
optimizer state is therefore resumed from checkpoint every step, not only on a genuine process restart.

Launch (from the box, already ssh'd in, repo synced, venv active, Ollama already has qwen2.5-coder:14b
pulled):
  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.rl_urn_pilot --steps 1
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # must be set before
# torch initializes its CUDA allocator -- matches every SFT box script (run_format_only_mechbal_
# training.sh etc.), needed here for the same reason: 100 per-example forward passes over DIFFERENT
# sequence lengths fragments the allocator (this is what actually caused the first real OOM, not raw
# memory pressure -- see rl_train.py's _seq_logprob for the other half of that fix).

import argparse
import statistics as st
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.pi_star import eager_builds, wait_k_builds
from scripts.creator.tool_disposition_benchmark.rl_critic import ReturnNormalizer, build_critic, load_critic
from scripts.creator.tool_disposition_benchmark.rl_reward import episode_reward
from scripts.creator.tool_disposition_benchmark.rl_rollout import RL_SEED_START, collect_batch
from scripts.creator.tool_disposition_benchmark.rl_train import (
    grpo_step, load_critic_optimizer_state, load_manifest, load_optimizer_state, load_policy_model,
    save_checkpoint, RL_LR)
from scripts.creator.tool_disposition_benchmark.urn_session import B, MAG, N, T, UNIFORM

N_CRITIC_FEATURES = 5    # 4 fair + 1 privileged (`rate`) -- spec §3.2, sign-off 2026-07-08
CRITIC_LR = 1e-3          # spec §3.3: tiny well-posed regression problem, higher LR than the policy is
                          # fine since the target never depends on the critic's own weights


def _baseline_reward(slots: list[dict], builds: dict) -> float:
    return episode_reward(slots, {cid: v for cid, v in builds.items() if v is not None})["reward"]


def batch_baselines(batch: list[dict]) -> dict:
    """EVALUATION-only baselines (eager, wait2) over the batch's distinct streams -- NOT fed into the
    reward/gradient (see docs/rl-finetuning-plan.md "Reward, revised"), just printed alongside
    mean_reward each step so a wait2-like trend is visible without waiting for a separate eval pass.
    Cheap: both are O(T) heuristics, no DP involved."""
    streams = {r["seed"]: r["slots"] for r in batch}.values()
    eager_r = [_baseline_reward(s, eager_builds(s, B)) for s in streams]
    wait2_r = [_baseline_reward(s, wait_k_builds(s, B, 2)) for s in streams]
    return {"eager_mean": st.mean(eager_r), "wait2_mean": st.mean(wait2_r)}

BASE_MODEL_TAG = "qwen2.5-coder:14b"
BASE_CTX_TAG = "qwen-rl-base-ctx8k"
RL_MODEL_TAG = "qwen-rl-urn-pilot"

# 40GB-A100 sizing (2026-07-08; the H100s were out of capacity, spec §8.6). Two knobs keep the serving
# phase deterministic on the smaller card -- the failure mode to avoid is the runbook's documented
# gotcha (box-setup.md §A1): a GGUF + KV footprint that doesn't quite fit doesn't crash, it silently
# partial-offloads to CPU and slows rollouts ~8x.
GGUF_OUTTYPE = "q8_0"  # ~15GB for 14B vs f16's ~28GB. f16 + 8-slot KV (~34-36GB total) is exactly
                       # borderline on 40GB; q8_0 is effectively lossless and leaves real headroom.
                       # On an 80GB box, pass --gguf-outtype f16 to remove even that quant mismatch
                       # between the sampled policy and the bf16 training weights (a mild off-policy
                       # wrinkle either way; the pre-FT baselines ran on the stock Ollama tag, which
                       # is Q4, so q8_0 is already closer to the training weights than those were).
NUM_CTX = 8192         # pinned explicitly (stock modelfile leaves Ollama's default, which varies by
                       # version) for two reasons: (1) deterministic KV sizing -- 14B GQA is ~0.2MB/tok,
                       # so 8 slots x 8192 = ~13GB, planned rather than discovered; (2) headroom over
                       # the longest measured episode (~3.7k tokens at temp 1.2) -- Ollama TRUNCATES
                       # context from the front when num_ctx is exceeded, which would silently corrupt
                       # long episodes rather than erroring.


def sh(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _ctx_line() -> str:
    return f"PARAMETER num_ctx {NUM_CTX}\n"


def serve_checkpoint(out_dir: Path, gguf_outtype: str) -> str:
    """Returns the Ollama model tag to sample the NEXT batch of rollouts from. No checkpoint yet
    (step 0) -> the untouched base model (as a num_ctx-capped variant -- same weights/template, only
    the context parameter pinned; see NUM_CTX). Otherwise merge -> GGUF convert -> `ollama create`
    from the adapter the previous step just saved."""
    adapter = out_dir / "checkpoint" / "adapter"
    if not adapter.exists():
        sh(["sudo", "systemctl", "start", "ollama"]); time.sleep(3)
        Path("base_ctx.modelfile").write_text(f"FROM {BASE_MODEL_TAG}\n" + _ctx_line())
        sh(["ollama", "create", BASE_CTX_TAG, "-f", "base_ctx.modelfile"])
        return f"{BASE_CTX_TAG}:latest"

    merged = out_dir / "merged"
    sh(["python", "-m", "scripts.creator.tool_disposition_benchmark.merge_lora",
        "--adapter", str(adapter), "--out", str(merged)])
    gguf = Path.home() / f"rl-urn-pilot-{gguf_outtype}.gguf"
    sh(["python", "llama.cpp/convert_hf_to_gguf.py", str(merged), "--outfile", str(gguf),
        "--outtype", gguf_outtype])
    sh(["sudo", "systemctl", "start", "ollama"]); time.sleep(3)

    ref = subprocess.run(["ollama", "show", "--modelfile", BASE_MODEL_TAG],
                         check=True, capture_output=True, text=True).stdout
    modelfile = "FROM " + str(gguf) + "\n" + _ctx_line() + "\n".join(
        line for line in ref.splitlines() if not (line.startswith("FROM ") or line.startswith("#")))
    ft_path = Path("ft.modelfile")
    ft_path.write_text(modelfile)
    sh(["ollama", "create", RL_MODEL_TAG, "-f", str(ft_path)])
    return f"{RL_MODEL_TAG}:latest"


async def _collect(model_tag: str, seeds: list[int], G: int, temperature: float, conc: int) -> list[dict]:
    client = RawChat()
    return await collect_batch(client, model_tag, seeds, G, N=N, T=T, B=B, pool=UNIFORM,
                               magnitude=MAG, temperature=temperature, conc=conc)


def main() -> None:
    import asyncio
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True, help="run outer steps [resumed_step, steps)")
    ap.add_argument("--seeds-per-step", type=int, default=25)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.2)   # 1.2, not 0.7: load-bearing for
    # exploration diversity (pilot v2's rollout diagnostic: 0.7 collapsed to 6/8 byte-identical eager
    # rollouts; 1.2 gave the widest two-sided reward spread with 0 parse failures -- spec §3). Default
    # aligned with the spec 2026-07-08 so launching without the flag can't silently reintroduce v2's
    # exploration collapse.
    ap.add_argument("--qlora", action="store_true", default=True)
    ap.add_argument("--gguf-outtype", default=GGUF_OUTTYPE,
                    help="GGUF quant for the rollout-serving merge (see GGUF_OUTTYPE comment: q8_0 "
                         "default sized for a 40GB A100; f16 on an 80GB box)")
    ap.add_argument("--n-epochs", type=int, default=None,
                    help="override rl_train.N_EPOCHS for THIS invocation (spec §8 step 5: re-enable "
                         "multi-epoch reuse only after a stable n_epochs=1 run -- this flag exists so "
                         "that continuation is a relaunch, not an on-box edit of the constant; the PPO "
                         "clip is what makes >1 safe, and it's untested at >1, so back up the "
                         "checkpoint dir first and watch mean_kl on the first steps)")
    ap.add_argument("--out", type=Path, default=Path("runs/rl_urn_pilot"))
    args = ap.parse_args()

    load_dotenv()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.out)
    start_step = manifest["step"] if manifest else 0
    reward_history = list(manifest["reward_history"]) if manifest else []
    behavior_history = list(manifest.get("behavior_history", [])) if manifest else []
    next_seed = manifest["next_seed"] if manifest else RL_SEED_START
    print(f"{'resuming from' if manifest else 'starting fresh at'} step {start_step} "
          f"({len(reward_history)} rewards so far, next_seed={next_seed})", flush=True)

    if start_step >= args.steps:
        print(f"already at step {start_step} >= --steps {args.steps}; nothing to do.")
        return

    for step in range(start_step, args.steps):
        t0 = time.time()
        model_tag = serve_checkpoint(args.out, args.gguf_outtype)
        t_serve = time.time()
        seeds = list(range(next_seed, next_seed + args.seeds_per_step))
        print(f"\n=== outer step {step}: sampling {model_tag}, seeds {seeds[0]}-{seeds[-1]} "
              f"x G={args.G} ===", flush=True)
        batch = asyncio.run(_collect(model_tag, seeds, args.G, args.temperature, args.conc))
        t_rollout = time.time()
        print(f"  resync (merge/GGUF/ollama create): {t_serve - t0:.0f}s", flush=True)
        print(f"  rollout collection: {t_rollout - t_serve:.0f}s ({len(batch)} episodes)", flush=True)
        baselines = batch_baselines(batch)

        sh(["sudo", "systemctl", "stop", "ollama"]); time.sleep(2)

        resume_adapter = args.out / "checkpoint" / "adapter"
        has_checkpoint = resume_adapter.exists()
        model, tokenizer = load_policy_model(args.qlora, resume_adapter if has_checkpoint else None)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=RL_LR)
        if has_checkpoint:
            load_optimizer_state(optimizer, args.out, next(model.parameters()).device)

        # critic: same resume-every-step pattern as the policy (spec §3.3: warm-started, not refit from
        # scratch each outer step) -- rebuilt fresh here (a few thousand params, negligible cost) and
        # loaded from the same checkpoint dir the policy adapter/optimizer resume from.
        critic = build_critic(N_CRITIC_FEATURES)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
        normalizer = ReturnNormalizer()
        if has_checkpoint:
            load_critic(critic, normalizer, args.out / "checkpoint" / "critic.pt", "cpu")
            load_critic_optimizer_state(critic_optimizer, args.out, "cpu")

        epoch_kw = {"n_epochs": args.n_epochs} if args.n_epochs is not None else {}
        stats = grpo_step(model, tokenizer, optimizer, batch, critic, critic_optimizer, normalizer,
                          **epoch_kw)
        t_grpo = time.time()
        next_seed += args.seeds_per_step
        reward_history.append(stats["mean_reward"])
        behavior_history.append(stats["behavior"])
        save_checkpoint(model, tokenizer, optimizer, critic, critic_optimizer, normalizer, args.out,
                        step + 1, reward_history, next_seed, behavior_history)

        del model, optimizer
        torch.cuda.empty_cache()

        print(f"  GRPO update: {t_grpo - t_rollout:.0f}s  mean_reward(balls)={stats['mean_reward']:.2f} "
              f"(std={stats['reward_std']:.2f} min={stats['reward_min']:.2f} max={stats['reward_max']:.2f})  "
              f"[baselines: eager={baselines['eager_mean']:.2f} wait2={baselines['wait2_mean']:.2f}]  "
              f"loss={stats['loss']:.4f}  mean_kl={stats['mean_kl']:.3f}  "
              f"critic_loss={stats['critic_loss']:.4f} ({stats['critic_fit_iters']} iters)  "
              f"mean_advantage={stats['mean_advantage']:.3f}  mean|A|={stats['mean_abs_advantage']:.2f}  "
              f"first_sight={stats['behavior']['first_sight_pct']:.0f}%  "
              f"lateness={stats['behavior']['mean_lateness']:.3f}  "
              f"group_std(mean={stats['mean_group_std']:.3f} max={stats['max_group_std']:.3f})", flush=True)
        print(f"  step {step} total: {time.time() - t0:.0f}s", flush=True)

    print(f"\nreward history (step 0 = first completed step): "
          f"{[round(r, 3) for r in reward_history]}", flush=True)
    print("behavior history (first_sight% / lateness): "
          f"{[(round(b['first_sight_pct']), round(b['mean_lateness'], 2)) for b in behavior_history]}",
          flush=True)


if __name__ == "__main__":
    main()
