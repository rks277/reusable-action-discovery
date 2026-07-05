"""Phase 3 LoRA SFT trainer (docs/qwen-finetune-transfer-plan.md § Training).

Fine-tunes Qwen2.5-Coder-14B-Instruct on the phase3_sft_data corpus to install the reserve-then-build
allocation policy. Two arms, identical config, only the labels differ:
  --arm pistar : treatment (exact-DP optimal decisions)
  --arm eager  : control   (build/keep on first sight) -- rules out "any SFT helps"

Data = the two matching jsonl files for the arm (urn_<arm>.jsonl + tool_bridge_<arm>.jsonl), chat format
{"messages":[...]}. SFT = next-token prediction with the loss masked to ASSISTANT tokens only (system /
user / tool tokens -> -100); assistant tool_calls ARE trained (they are the decision), tool RESULT turns
are masked. Masking is done here, backend-agnostically, by prefix-diffing the tokenizer's own chat
template -- so it is correct for the multi-turn / multiple-assistant-turns-per-problem tool sessions and
does not depend on the template carrying {% generation %} tags.

Backends (spec: Unsloth primary, HF+PEFT+TRL fallback if Unsloth has friction with this checkpoint):
  --backend unsloth : FastLanguageModel (fastest single-GPU path)
  --backend hf      : AutoModelForCausalLM + PEFT LoraConfig

CRITICAL CORRECTNESS CHECK (runs at startup, before any training): the chat template must render an
assistant tool_call as the SAME text vLLM's `hermes` parser reads back at eval -- an unescaped
<tool_call>{"name": "write_script", "arguments": {...}}</tool_call> block. Mismatch = the model learns a
format the eval harness can't parse (the Phase-2 bug class). `verify_template()` renders one tool session
and asserts this, printing the sample to eyeball.

Usage (see docs/box-setup.md §B for the box):
  # local / on-box, NO model load, NO GPU -- data prep + template check + length stats only:
  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.train_lora --arm pistar --dry-run

  # smoke train (20 steps, few sessions) -- confirm the pipeline end-to-end before a real run:
  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.train_lora --arm pistar --smoke

  # 🛑 real run (needs go-ahead per [[no-auto-reps]]):
  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.train_lora --arm pistar
  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.train_lora --arm eager
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Heavy deps (torch / transformers / unsloth / peft) are imported lazily inside the functions that need
# them, so --dry-run works with only `transformers` installed (just the tokenizer), and so this module
# imports cleanly for review / py_compile without a GPU stack present.

BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
DATA_DIR = Path("runs/phase3_sft_data")
OUT_ROOT = Path("runs/phase3_ft")

# LoRA + hyperparameters -- the spec's starting point (docs/qwen-finetune-transfer-plan.md § Training).
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LR = 1e-4
EPOCHS = 2
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 16
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
VAL_HOLDOUT = 10          # sessions held out per arm for a train/val loss curve (sanity only)
SEED = 42

# max_seq_len is MEASURED, not guessed (corpus token counts are a crude len//4 estimate). We tokenize
# every session, then round the observed max up to one of these caps. Sessions longer than the top cap
# are DROPPED (not truncated -- truncation would silently cut the decision structure).
SEQ_LEN_CAPS = [4096, 8192, 16384, 32768]


# --------------------------------------------------------------------- data loading
def load_sessions(arm: str, data_dir: Path) -> list[list[dict]]:
    """Load + concatenate the arm's urn and tool_bridge slices; normalize tool_call arguments."""
    sessions: list[list[dict]] = []
    for slice_name in ("urn", "tool_bridge"):
        path = data_dir / f"{slice_name}_{arm}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing corpus file {path} -- run phase3_demos.py first")
        with path.open() as f:
            for line in f:
                sessions.append(_normalize_tool_calls(json.loads(line)["messages"]))
    return sessions


def _normalize_tool_calls(messages: list[dict]) -> list[dict]:
    """The corpus stores tool_call arguments as a JSON *string* (json.dumps). The Qwen2.5 chat template
    renders `arguments` with `| tojson`, so a string would be double-encoded (escaped) -- wrong, and it
    would mismatch what the hermes parser expects at eval. Parse arguments back to a dict so the template
    emits an object literal. Idempotent (leaves dicts alone)."""
    for m in messages:
        for tc in m.get("tool_calls", []) or []:
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                tc["function"]["arguments"] = json.loads(args)
    return messages


# --------------------------------------------------------------------- render + assistant-only masking
def _ids(out) -> list[int]:
    """Normalize apply_chat_template(tokenize=True) to a flat list[int]. transformers 5.x returns a
    BatchEncoding (dict-like with 'input_ids'); older returns a plain list; guard the batched [[...]]
    shape too."""
    ids = out["input_ids"] if not isinstance(out, list) else out
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def build_example(messages: list[dict], tokenizer) -> dict:
    """Tokenize the whole conversation via the tokenizer's chat template and build a label vector that is
    the input ids on ASSISTANT turns and -100 everywhere else. Uses prefix-diffing (Qwen's template is
    prefix-additive: each message renders to <|im_start|>role\\n...<|im_end|>\\n appended to the prefix),
    which is robust to multi-turn / multiple-assistant-turns-per-problem and needs no {% generation %}
    template support. Returns unbounded ids (caller truncates/drops by measured max_seq_len)."""
    input_ids: list[int] = []
    labels: list[int] = []
    prev: list[int] = []        # empty prefix (transformers refuses apply_chat_template([])); the first
                                # message renders any template preamble and is masked as non-assistant
    for i, msg in enumerate(messages):
        cur = _ids(tokenizer.apply_chat_template(messages[: i + 1], tokenize=True,
                                                 add_generation_prompt=False))
        seg = cur[len(prev):]
        prev = cur
        input_ids.extend(seg)
        labels.extend(seg if msg["role"] == "assistant" else [-100] * len(seg))
    assert len(input_ids) == len(labels)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


# --------------------------------------------------------------------- template correctness check
def verify_template(tokenizer, sessions: list[dict]) -> None:
    """Assert an assistant write_script tool_call renders to the unescaped <tool_call> block the hermes
    parser reads at eval, and that assistant-only masking actually selects the decision tokens. Prints a
    rendered + masked sample to eyeball on the box."""
    tool_sess = next((s for s in sessions if any(
        tc["function"]["name"] == "write_script"
        for m in s for tc in m.get("tool_calls", []) or [])), None)
    assert tool_sess is not None, "no write_script call found in corpus -- cannot verify template"

    rendered = tokenizer.apply_chat_template(tool_sess, tokenize=False, add_generation_prompt=False)
    assert "<tool_call>" in rendered, "chat template did not emit <tool_call> blocks for tool_calls"
    assert '"name": "write_script"' in rendered or '"name":"write_script"' in rendered, \
        "write_script tool name missing/mangled in rendered template"
    assert '\\"code\\"' not in rendered and '{\\"' not in rendered, \
        "tool_call arguments look double-escaped -- _normalize_tool_calls should have parsed them to dicts"

    ex = build_example(tool_sess, tokenizer)
    n_tok, n_lbl = len(ex["input_ids"]), sum(1 for x in ex["labels"] if x != -100)
    print("=== template + masking check (one tool_bridge session) ===")
    idx = rendered.find("<tool_call>")
    print("  rendered tool_call snippet:", repr(rendered[idx:idx + 160]))
    print(f"  tokens={n_tok}  trained(assistant) tokens={n_lbl} ({n_lbl / n_tok:.0%})  "
          f"masked={n_tok - n_lbl}")
    # decode a short unmasked span to confirm it is assistant content, not user/tool
    span = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100][:40]
    print("  first trained tokens decode to:", repr(tokenizer.decode(span)))
    print("  template check OK\n")


# --------------------------------------------------------------------- length measurement
def build_and_measure(sessions: list[dict], tokenizer, max_seq_len: int | None) -> tuple[list[dict], int]:
    """Build every example, report the length distribution, choose max_seq_len (unless pinned), and DROP
    sessions longer than it (never truncate)."""
    examples = [build_example(s, tokenizer) for s in sessions]
    lens = sorted(len(e["input_ids"]) for e in examples)
    n = len(lens)
    p = lambda q: lens[min(n - 1, int(q * n))]
    print(f"=== tokenized length over {n} sessions ===")
    print(f"  min={lens[0]}  mean={sum(lens) // n}  p50={p(.5)}  p95={p(.95)}  max={lens[-1]}")
    if max_seq_len is None:
        max_seq_len = next((c for c in SEQ_LEN_CAPS if c >= lens[-1]), SEQ_LEN_CAPS[-1])
    kept = [e for e in examples if len(e["input_ids"]) <= max_seq_len]
    dropped = n - len(kept)
    print(f"  -> max_seq_len={max_seq_len}  kept={len(kept)}  dropped(>cap)={dropped}"
          + ("  ** DROPPED sessions exceed the top cap -- consider a shorter corpus **"
             if dropped and max_seq_len == SEQ_LEN_CAPS[-1] else ""))
    return kept, max_seq_len


# --------------------------------------------------------------------- collator
def make_collator(tokenizer):
    from transformers import DataCollatorForSeq2Seq        # pads input_ids w/ pad, labels w/ -100
    return DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, padding="longest")


# --------------------------------------------------------------------- model loading (two backends)
def load_unsloth(max_seq_len: int, qlora: bool):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=max_seq_len, dtype=None, load_in_4bit=qlora)
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, target_modules=TARGET_MODULES, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, bias="none", use_gradient_checkpointing="unsloth", random_state=SEED)
    return model, tokenizer


def load_hf(max_seq_len: int, qlora: bool):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kw)
    model.config.use_cache = False
    if qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        task_type="CAUSAL_LM", target_modules=TARGET_MODULES))
    model.print_trainable_parameters()
    return model, tokenizer


# --------------------------------------------------------------------- train
def train(args) -> None:
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    if args.backend == "unsloth":
        model, tokenizer = load_unsloth(args.max_seq_len or SEQ_LEN_CAPS[-1], args.qlora)
    else:
        model, tokenizer = load_hf(args.max_seq_len or SEQ_LEN_CAPS[-1], args.qlora)

    sessions = load_sessions(args.arm, args.data_dir)
    verify_template(tokenizer, sessions)
    examples, max_seq_len = build_and_measure(sessions, tokenizer, args.max_seq_len)

    rng = random.Random(SEED)
    rng.shuffle(examples)
    if args.smoke:
        examples = examples[:8]
    val_n = 0 if args.smoke else min(VAL_HOLDOUT, len(examples) // 5)
    val, tr = examples[:val_n], examples[val_n:]
    print(f"train sessions={len(tr)}  val sessions={len(val)}  (arm={args.arm})")

    ds_tr = Dataset.from_list(tr)
    ds_val = Dataset.from_list(val) if val else None

    out = args.out or (OUT_ROOT / args.arm)
    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=1 if args.smoke else EPOCHS,
        max_steps=20 if args.smoke else -1,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=1 if args.smoke else GRAD_ACCUM,
        learning_rate=LR, lr_scheduler_type="cosine", warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY, max_grad_norm=MAX_GRAD_NORM,
        bf16=True, logging_steps=1, save_strategy="no" if args.smoke else "epoch",
        eval_strategy="epoch" if ds_val is not None else "no",
        report_to="none", seed=SEED,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_val,
                      data_collator=make_collator(tokenizer))
    trainer.train()

    if args.smoke:
        print("\nSMOKE OK -- pipeline ran end-to-end. Not saving. 🛑 get go-ahead before the real run.")
        return
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"\nsaved LoRA adapter -> {out}")
    if args.merge_out:
        print(f"merging adapter into base -> {args.merge_out}")
        merged = model.merge_and_unload()
        merged.save_pretrained(args.merge_out)
        tokenizer.save_pretrained(args.merge_out)


# --------------------------------------------------------------------- dry run (no model, no GPU)
def dry_run(args) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    sessions = load_sessions(args.arm, args.data_dir)
    print(f"arm={args.arm}  sessions={len(sessions)}\n")
    verify_template(tokenizer, sessions)
    _, max_seq_len = build_and_measure(sessions, tokenizer, args.max_seq_len)
    print(f"\nDRY-RUN OK. Chosen max_seq_len={max_seq_len}. No model loaded, no GPU used.")


# --------------------------------------------------------------------- cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["pistar", "eager"], required=True)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=None, help="adapter output dir (default runs/phase3_ft/<arm>)")
    ap.add_argument("--merge-out", type=str, default=None, help="also merge adapter into base -> this dir")
    ap.add_argument("--backend", choices=["unsloth", "hf"], default="unsloth")
    ap.add_argument("--qlora", action="store_true", help="4-bit QLoRA (fallback if bf16 VRAM is tight)")
    ap.add_argument("--max-seq-len", type=int, default=None, help="pin; default = measured max rounded up")
    ap.add_argument("--smoke", action="store_true", help="20 steps on 8 sessions; no save")
    ap.add_argument("--dry-run", action="store_true", help="data prep + template check + stats only")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
