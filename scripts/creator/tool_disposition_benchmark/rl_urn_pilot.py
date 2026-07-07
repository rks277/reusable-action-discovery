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
from scripts.creator.tool_disposition_benchmark.rl_reward import episode_reward
from scripts.creator.tool_disposition_benchmark.rl_rollout import RL_SEED_START, collect_batch
from scripts.creator.tool_disposition_benchmark.rl_train import (
    grpo_step, load_manifest, load_optimizer_state, load_policy_model, save_checkpoint, RL_LR)
from scripts.creator.tool_disposition_benchmark.urn_session import B, MAG, N, T, UNIFORM


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
RL_MODEL_TAG = "qwen-rl-urn-pilot"
GGUF_NAME = "rl-urn-pilot-f16.gguf"


def sh(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def serve_checkpoint(out_dir: Path) -> str:
    """Returns the Ollama model tag to sample the NEXT batch of rollouts from. No checkpoint yet
    (step 0) -> the untouched base model. Otherwise merge -> GGUF convert -> `ollama create` from the
    adapter the previous step just saved."""
    adapter = out_dir / "checkpoint" / "adapter"
    if not adapter.exists():
        sh(["sudo", "systemctl", "start", "ollama"]); time.sleep(3)
        return BASE_MODEL_TAG

    merged = out_dir / "merged"
    sh(["python", "-m", "scripts.creator.tool_disposition_benchmark.merge_lora",
        "--adapter", str(adapter), "--out", str(merged)])
    gguf = Path.home() / GGUF_NAME
    sh(["python", "llama.cpp/convert_hf_to_gguf.py", str(merged), "--outfile", str(gguf), "--outtype", "f16"])
    sh(["sudo", "systemctl", "start", "ollama"]); time.sleep(3)

    ref = subprocess.run(["ollama", "show", "--modelfile", BASE_MODEL_TAG],
                         check=True, capture_output=True, text=True).stdout
    modelfile = "FROM " + str(gguf) + "\n" + "\n".join(
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
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--qlora", action="store_true", default=True)
    ap.add_argument("--out", type=Path, default=Path("runs/rl_urn_pilot"))
    args = ap.parse_args()

    load_dotenv()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.out)
    start_step = manifest["step"] if manifest else 0
    reward_history = list(manifest["reward_history"]) if manifest else []
    next_seed = manifest["next_seed"] if manifest else RL_SEED_START
    print(f"{'resuming from' if manifest else 'starting fresh at'} step {start_step} "
          f"({len(reward_history)} rewards so far, next_seed={next_seed})", flush=True)

    if start_step >= args.steps:
        print(f"already at step {start_step} >= --steps {args.steps}; nothing to do.")
        return

    for step in range(start_step, args.steps):
        t0 = time.time()
        model_tag = serve_checkpoint(args.out)
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

        stats = grpo_step(model, tokenizer, optimizer, batch)
        t_grpo = time.time()
        next_seed += args.seeds_per_step
        reward_history.append(stats["mean_reward"])
        save_checkpoint(model, tokenizer, optimizer, args.out, step + 1, reward_history, next_seed)

        del model, optimizer
        torch.cuda.empty_cache()

        print(f"  GRPO update: {t_grpo - t_rollout:.0f}s  mean_reward(balls)={stats['mean_reward']:.2f} "
              f"(std={stats['reward_std']:.2f} min={stats['reward_min']:.2f} max={stats['reward_max']:.2f})  "
              f"[baselines: eager={baselines['eager_mean']:.2f} wait2={baselines['wait2_mean']:.2f}]  "
              f"loss={stats['loss']:.4f}  mean_kl={stats['mean_kl']:.3f}  "
              f"group_std(mean={stats['mean_group_std']:.3f} max={stats['max_group_std']:.3f})  "
              f"zero_adv_groups={stats['n_zero_advantage_groups']}/{stats['n_groups']}", flush=True)
        print(f"  step {step} total: {time.time() - t0:.0f}s", flush=True)

    print(f"\nreward history (step 0 = first completed step): "
          f"{[round(r, 3) for r in reward_history]}", flush=True)


if __name__ == "__main__":
    main()
