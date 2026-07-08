"""Phase 3 LoRA SFT trainer (docs/qwen-finetune-transfer-plan.md § Training).

Fine-tunes Qwen2.5-Coder-14B-Instruct on the phase3_sft_data corpus to install the reserve-then-build
allocation policy. Two arms, identical config, only the labels differ:
  --arm pistar : treatment (exact-DP optimal decisions)
  --arm eager  : control   (build/keep on first sight) -- rules out "any SFT helps"

Data = the two matching jsonl files for the arm (urn_<arm>.jsonl + tool_bridge_<arm>.jsonl) PLUS the
arm-independent tool-calling anchor (anchor_tool.jsonl, option (c) -- single-problem tool sessions that
preserve tool-calling modality without teaching build timing; --anchor none reproduces the pre-anchor
corpus), chat format {"messages":[...]}. SFT = next-token prediction with the loss masked to ASSISTANT
tokens only (system /
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
EPOCHS = 2     # REVERTED 4->2 (2026-07-06): 4 epochs overfit into verbatim phrase-memorization (the
               # model got stuck repeating a fixed anchor rationale string regardless of prompt/reminder
               # content -- not a format problem, a collapsed output distribution no prompt can fix).
               # Back to 2 to test the driver.py FORMAT_REMINDER fix against a non-collapsed checkpoint.
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
def load_sessions(arm: str, data_dir: Path, anchor: str = "tool", mechanics: str = "on",
                  recovery: str = "on", skip_urn: bool = False) -> list[list[dict]]:
    """Load + concatenate the arm's urn and tool_bridge slices (+ the arm-independent tool-calling
    anchor unless anchor='none', + the arm-independent mechanics bridge unless mechanics='off', + the
    arm-independent error-recovery bridge unless recovery='off'); normalize tool_call arguments. The
    anchor (option c), mechanics bridge (Design A fix, 2026-07-06), and error-recovery bridge
    (error-recovery ablation, folded into mechanics bridge training, 2026-07-06 --
    docs/qwen-finetune-transfer-plan.md) are all shared byte-identically across arms -- they preserve
    tool-calling modality / long-context mechanics / bad-turn recovery without teaching build timing.

    skip_urn (2026-07-07, "Adapter F" / format-only run -- docs/qwen-finetune-transfer-plan.md "Corpus
    regeneration" follow-up): skips the urn slice entirely (tool_bridge is loaded as usual -- it's empty
    under Design A regardless), so the adapter trains on ZERO urn exposure. Used to test whether tool-
    modality degradation is caused by gradient interference from co-training with urn's dominant,
    long, text-heavy signal, independent of the corpus-content fixes."""
    sessions: list[list[dict]] = []
    for slice_name in ("urn", "tool_bridge"):
        if skip_urn and slice_name == "urn":
            continue
        path = data_dir / f"{slice_name}_{arm}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing corpus file {path} -- run phase3_demos.py first")
        with path.open() as f:
            for line in f:
                sessions.append(_normalize_tool_calls(json.loads(line)["messages"]))
    if anchor == "tool":
        path = data_dir / "anchor_tool.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path} -- run phase3_demos.py to regenerate the corpus (it now includes the "
                "tool-calling anchor), or pass --anchor none")
        with path.open() as f:
            for line in f:
                sessions.append(_normalize_tool_calls(json.loads(line)["messages"]))
    if mechanics == "on":
        path = data_dir / "mechanics_bridge.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path} -- run phase3_demos.py to regenerate the corpus (it now includes the "
                "mechanics bridge), or pass --mechanics off")
        with path.open() as f:
            for line in f:
                sessions.append(_normalize_tool_calls(json.loads(line)["messages"]))
    if recovery == "on":
        path = data_dir / "error_recovery.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path} -- run phase3_demos.py to regenerate the corpus (it now includes the "
                "error-recovery bridge), or pass --recovery off")
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


def _session_tools(session: list[dict]):
    """A session is a TOOL session iff any assistant turn carries tool_calls (tool_bridge + anchor). At
    EVAL the harness passes tools=TOOL_SCHEMAS(), so Qwen's template injects a ~685-token <tools> system
    block; the urn demos are pure text and pass no tools. Rendering must MATCH eval per session-type, or
    the FT model is out-of-distribution at eval (the <tools>-block train/eval mismatch that made the FT
    model revert to CoT with tools= passed — 2026-07-05). Urn sessions -> None (no tools block)."""
    from scripts.creator.tool_disposition_benchmark.session_state import TOOL_SCHEMAS
    return TOOL_SCHEMAS() if any(m.get("tool_calls") for m in session) else None


def build_example(messages: list[dict], tokenizer, tools=None) -> dict:
    """Tokenize the whole conversation via the tokenizer's chat template and build a label vector that is
    the input ids on ASSISTANT turns and -100 everywhere else. Uses prefix-diffing (Qwen's template is
    prefix-additive: each message renders to <|im_start|>role\\n...<|im_end|>\\n appended to the prefix),
    which is robust to multi-turn / multiple-assistant-turns-per-problem and needs no {% generation %}
    template support. `tools` (when set) is passed to apply_chat_template so the rendered prompt carries
    the same <tools> block the eval harness produces — it lands in the (masked) system prefix, so it does
    not change which tokens are trained. Returns unbounded ids (caller truncates/drops by max_seq_len).

    For ASSISTANT turns, the trained span EXCLUDES the '<|im_start|>assistant\\n' role-declaration
    prefix -- only the actual content (+ '<|im_end|>\\n') is labeled, computed by diffing against
    apply_chat_template(messages[:i], add_generation_prompt=True) (the exact prefix a real inference
    call supplies BEFORE the model generates anything). Getting this wrong (labeling the whole
    per-message increment, prefix included, as earlier versions of this function did) trains the model
    to treat '<|im_start|>assistant\\n' as legitimate content it can emit mid-generation -- confirmed
    2026-07-06 as the likely root cause of the Design A tool-eval collapse: 3 of 5 saved failing
    checkpoints hallucinate the literal word "assistant" as content in 90%+ of their turns, and the one
    training slice containing two back-to-back assistant messages with no separating turn
    (mechanics_bridge's old 'reuse' branch, fixed in phase3_demos.py) is the only place that shape could
    have been reinforced from an in-context example rather than merely a masking-label artifact.

    Also returns `turn_spans`: a `[(start, end), ...]` list, one (start, end) half-open range into
    `input_ids`/`labels` per ASSISTANT message, in message order -- the trained-token span for that turn
    ALONE (same exclusion of the role-declaration prefix as the labels themselves). Added
    2026-07-08 for RL's per-decision credit assignment (docs/rl-ppo-credit-assignment-spec.md §6):
    `urn_session.run_episode` emits exactly one assistant message per KEEP/PASS decision, so turn i here
    IS decision i in `rl_reward.per_decision_rewards`. SFT training (this file's own callers) ignores
    the extra key; it changes no existing behavior."""
    input_ids: list[int] = []
    labels: list[int] = []
    turn_spans: list[tuple[int, int]] = []
    prev: list[int] = []        # empty prefix (transformers refuses apply_chat_template([])); the first
                                # message renders any template preamble and is masked as non-assistant
    for i, msg in enumerate(messages):
        cur = _ids(tokenizer.apply_chat_template(messages[: i + 1], tokenize=True,
                                                 add_generation_prompt=False, tools=tools))
        seg = cur[len(prev):]
        start_before = len(input_ids)
        if msg["role"] == "assistant":
            gen = _ids(tokenizer.apply_chat_template(messages[:i], tokenize=True,
                                                      add_generation_prompt=True, tools=tools))
            assert gen[:len(prev)] == prev and cur[:len(gen)] == gen, \
                "chat template not prefix-additive under add_generation_prompt -- masking assumption broken"
            n_prefix = len(gen) - len(prev)     # '<|im_start|>assistant\n' -- NOT trained
            labels.extend([-100] * n_prefix + seg[n_prefix:])
            turn_spans.append((start_before + n_prefix, start_before + len(seg)))
        else:
            labels.extend([-100] * len(seg))
        input_ids.extend(seg)
        prev = cur
    assert len(input_ids) == len(labels)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids),
            "turn_spans": turn_spans}


# --------------------------------------------------------------------- template correctness check
def verify_template(tokenizer, sessions: list[dict]) -> None:
    """Assert an assistant write_script tool_call renders to the unescaped <tool_call> block the hermes
    parser reads at eval, and that assistant-only masking actually selects the decision tokens. Prints a
    rendered + masked sample to eyeball on the box."""
    tool_sess = next((s for s in sessions if any(
        tc["function"]["name"] == "write_script"
        for m in s for tc in m.get("tool_calls", []) or [])), None)
    assert tool_sess is not None, "no write_script call found in corpus -- cannot verify template"

    tools = _session_tools(tool_sess)
    rendered = tokenizer.apply_chat_template(tool_sess, tokenize=False, add_generation_prompt=False,
                                             tools=tools)
    assert "<tool_call>" in rendered, "chat template did not emit <tool_call> blocks for tool_calls"
    assert "<tools>" in rendered, \
        "tool session rendered WITHOUT a <tools> block -- will mismatch the eval harness (tools= passed)"
    assert '"name": "write_script"' in rendered or '"name":"write_script"' in rendered, \
        "write_script tool name missing/mangled in rendered template"
    assert '\\"code\\"' not in rendered and '{\\"' not in rendered, \
        "tool_call arguments look double-escaped -- _normalize_tool_calls should have parsed them to dicts"

    ex = build_example(tool_sess, tokenizer, tools=tools)
    n_tok, n_lbl = len(ex["input_ids"]), sum(1 for x in ex["labels"] if x != -100)
    print("=== template + masking check (one tool_bridge session, tools= injected to match eval) ===")
    idx = rendered.find("<tool_call>")
    print("  rendered tool_call snippet:", repr(rendered[idx:idx + 160]))
    print(f"  tokens={n_tok}  trained(assistant) tokens={n_lbl} ({n_lbl / n_tok:.0%})  "
          f"masked={n_tok - n_lbl}")
    # decode a short unmasked span to confirm it is assistant content, not user/tool
    span = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100][:40]
    decoded = tokenizer.decode(span)
    print("  first trained tokens decode to:", repr(decoded))
    # REGRESSION GUARD (2026-07-06): the trained span must never start with the chat template's own
    # role-declaration text ('<|im_start|>assistant\n' renders 'assistant\n' as literal decoded text) --
    # this is the exact masking bug diagnosed as the likely root cause of the Design A tool-eval collapse
    # (the model hallucinating the literal word "assistant" as content). If this fires, build_example's
    # prefix-exclusion logic has regressed.
    assert not decoded.lstrip().startswith("assistant"), \
        ("trained span starts with the literal role-declaration text 'assistant' -- build_example is "
         "training the model to emit '<|im_start|>assistant\\n' as content again (the 2026-07-06 bug)")
    print("  template check OK\n")


# --------------------------------------------------------------------- length measurement
def build_and_measure(sessions: list[dict], tokenizer, max_seq_len: int | None) -> tuple[list[dict], int]:
    """Build every example, report the length distribution, choose max_seq_len (unless pinned), and DROP
    sessions longer than it (never truncate)."""
    examples = [build_example(s, tokenizer, tools=_session_tools(s)) for s in sessions]
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
    # torch 2.12 + cu13: the cuDNN fused-attention backend crashes in the BACKWARD pass on long
    # sequences ("mha_graph.execute ... got false") -- it survives the short smoke sessions but dies
    # on the ~21k-token tool_bridge session. Disable ONLY the cuDNN SDPA backend; flash + mem-efficient
    # stay on (both are memory-efficient, so no OOM -- eager would materialize a 21k x 21k score matrix
    # and blow past 80 GB). Requires attn_implementation="sdpa" (set below).
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

    sessions = load_sessions(args.arm, args.data_dir, args.anchor, args.mechanics, args.recovery,
                         args.skip_urn)
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
        # eval_strategy left "no" even when ds_val exists (2026-07-06 OOM fix): Trainer's automatic
        # eval never got its own per_device_eval_batch_size set, so it defaulted to 8 -- batching up to
        # 8 long mechanics_bridge sessions (~19k tokens) together and casting the padded batch's logits
        # to fp32 tried to allocate ~73GB. The val split is "sanity only" (see VAL_HOLDOUT above), not
        # needed for the adapter's actual downstream behavior, so skip it rather than risk re-tuning
        # eval batch size under a different OOM.
        eval_strategy="no",
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
    sessions = load_sessions(args.arm, args.data_dir, args.anchor, args.mechanics, args.recovery,
                         args.skip_urn)
    print(f"arm={args.arm}  anchor={args.anchor}  mechanics={args.mechanics}  sessions={len(sessions)}\n")
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
    ap.add_argument("--anchor", choices=["tool", "none"], default="tool",
                    help="include the arm-independent tool-calling anchor (option c) to preserve "
                         "tool-calling; 'none' reproduces the pre-anchor corpus")
    ap.add_argument("--mechanics", choices=["on", "off"], default="on",
                    help="include the arm-independent mechanics bridge (Design A fix, 2026-07-06) -- "
                         "policy-neutral long-context tool sessions that re-teach correct tool-call "
                         "syntax/naming under Design A (N_TOOL=0); 'off' reproduces the pre-fix corpus")
    ap.add_argument("--recovery", choices=["on", "off"], default="on",
                    help="include the arm-independent error-recovery bridge (error-recovery ablation, "
                         "folded into mechanics bridge training, 2026-07-06) -- bad-turn -> "
                         "harness-nudge -> recovery sessions, teaching the "
                         "conversational shape driver.py's retry logic creates on a malformed turn; "
                         "'off' reproduces the pre-ablation (Result 2) corpus")
    ap.add_argument("--skip-urn", action="store_true",
                    help="'Adapter F' / format-only run (2026-07-07): train on ZERO urn exposure -- "
                         "anchor + mechanics_bridge + error_recovery only -- to test whether tool-"
                         "modality degradation is caused by gradient interference from co-training "
                         "with urn, independent of the corpus-content fixes")
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
