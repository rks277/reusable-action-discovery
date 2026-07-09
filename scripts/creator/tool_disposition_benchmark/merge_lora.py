"""Merge a Phase-3 LoRA adapter into the BF16 base for vLLM serving (docs/box-setup.md §B1).

IMPORTANT: loads the base in BF16 -- NOT the 4-bit QLoRA training base -- and applies the adapter
there. This is the standard QLoRA merge; the small numeric mismatch (adapter optimized against a 4-bit
forward, merged into bf16) is absorbed by the a_script recalibration in the eval step. Do NOT reuse
train_lora.py's inline --merge-out for QLoRA runs (that would merge onto the 4-bit model).

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.merge_lora \
    --adapter runs/phase3_ft/pistar --out runs/phase3_merged/pistar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-Coder-14B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (runs/phase3_ft/<arm>)")
    ap.add_argument("--out", required=True, help="output dir for the merged bf16 checkpoint")
    ap.add_argument("--device", default="cpu", choices=["cpu", "auto"],
                    help="cpu (default) merges entirely in host RAM -- REQUIRED for the RL pilot's "
                         "per-step resync on a single card (rl_urn_pilot.py), where the parent process "
                         "still holds the GPU's CUDA context/cache from training and a device_map=auto "
                         "bf16 load (~28GB) OOMs the leftover ~14GB (observed 2026-07-08, step 8 on a "
                         "40GB A100). CPU merge needs ~28GB RAM (box has plenty) and the GGUF-convert "
                         "step downstream is CPU anyway, so nothing is lost but a little wall-clock. "
                         "Use auto only for a standalone merge when the GPU is known-empty.")
    a = ap.parse_args()

    device_map = None if a.device == "cpu" else "auto"
    print(f"[merge] loading base {BASE} in bf16 on {a.device} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map=device_map)
    print(f"[merge] applying adapter {a.adapter} ...", flush=True)
    model = PeftModel.from_pretrained(model, a.adapter)
    print("[merge] merge_and_unload ...", flush=True)
    model = model.merge_and_unload()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    # tokenizer + chat_template.jinja: take from the adapter dir (the exact template used in training,
    # which is what vLLM's hermes parser must round-trip against)
    AutoTokenizer.from_pretrained(a.adapter).save_pretrained(a.out)
    print(f"[merge] saved merged model -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
