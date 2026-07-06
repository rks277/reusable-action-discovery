# Adapter-merge transfer plan — alternative to joint-corpus SFT

Companion to `qwen-finetune-transfer-plan.md` (the joint-corpus SFT attempt this plan targets as a
fix — read it first for Design A's corpus, training config, and the Result 2 collapse table) and
`rl-finetuning-plan.md` (the other alternative under consideration, also PLANNED/not started). GPU
box mechanics live in `docs/box-setup.md`. Code lives in `scripts/creator/tool_disposition_benchmark/`
(`train_lora.py`, `merge_lora.py`, `phase3_demos.py`).

## Status (2026-07-06)

**PLANNED, not started, GATED — and the bar just got higher.** `qwen-finetune-transfer-plan.md`'s
"Mechanics bridge training — root-cause diagnosis and fix" found and fixed two concrete bugs behind
Result 2's collapse (a `train_lora.py` masking bug that trained the model to emit the chat template's
own `<|im_start|>assistant\n` role text as content, reinforced by a `phase3_demos.py` corpus bug in
`mechanics_bridge`'s "reuse" branch that was the only slice with the shape needed to teach it — hence
the name). Both are fixed; the corpus is regenerated locally and ready to retrain. Mechanics bridge
training (bug fixes + the already-built `error_recovery.jsonl`, no adapter split) is now the standing
next experiment — run it first. This plan's entanglement question only becomes live if that retrain
**still** collapses, which would now be much stronger evidence for entanglement than before (two known, concrete bugs would already
be ruled out). No training run, no code beyond the small changes scoped below.

## Why adapter merging instead of joint SFT

Design A's corpus blends three slices into one LoRA training run: `urn_pistar` (teaches the
reserve-then-build policy), `anchor_tool` (teaches tool-calling modality, policy-neutral), and
`mechanics_bridge` (teaches long-context tool-session mechanics, policy-neutral). The urn side
transfers cleanly (101% of π\* in the urn, replicated across two runs). The tool side is **blocked**:
every one of five checkpoints collapses into a different malformed-output pattern from problem 2
onward (`qwen-finetune-transfer-plan.md` Result 2). The working hypothesis there is a **coverage**
gap — none of the training slices ever show a bad turn followed by a correction, so the eval
harness's retry-nudge puts the model in a conversational state with zero training exposure.

That hypothesis is plausible but untested against an alternative explanation: that co-training one
LoRA adapter on all three slices at once **entangles** the policy signal (from urn) with the format
signal (from anchor/mechanics) in ways that produce the collapse — five different corpus mixtures
in Design A produced five different collapse patterns, which is at least consistent with
interference between objectives sharing one adapter's gradient updates, not just missing coverage.

Adapter merging removes that confound for free: train the policy skill and the format skill as
**separate LoRA adapters** on the same base model, with no shared gradient step between them, then
combine them in weight space. If the collapse was corpus-entanglement, separating training should
reduce it. If it isn't, this cleanly isolates that (see "Reading the outcome" below).

## The split

**Adapter P ("policy")** — urn only, no tool-calling exposure at all:
- Corpus: `urn_pistar.jsonl` (170 sessions) only. `tool_bridge_pistar.jsonl` is empty in Design A
  already (`N_TOOL=0`), so this is close to a no-op change.
- Command (existing flags, no code change): `train_lora.py --arm pistar --anchor none --mechanics off`.

**Adapter F ("format")** — tool-calling robustness only, zero reserve-timing signal:
- Corpus: `anchor_tool.jsonl` (150, single-problem, `ANCHOR_HAND_FRAC` balanced) +
  `mechanics_bridge.jsonl` (25, persistent-session, k=1/k≥2 balanced) + a **new**
  `error_recovery.jsonl` slice (the SFT plan's "not yet attempted" fix, folded in here instead of
  into the joint corpus).
- **New corpus piece — `error_recovery.jsonl`:** one `render_error_recovery_session()` generator
  (same conventions as `render_anchor_session` / `render_mechanics_bridge_session` in
  `phase3_demos.py`), covering the 5 collapse patterns actually observed in Result 2 — unregistered
  `submit_answers` (plural), pseudo-`<script>` tag wrapping real code, empty-`{}` repetition,
  verbatim anchor-phrase loop, empty markdown fence. Each session: a turn emits exactly one of
  these patterns, the next user turn is the real `driver.py: FORMAT_REMINDER` text (not a
  paraphrase — it must match what eval actually injects), and the assistant turn after that
  recovers with a correctly-parsed `<tool_call>`. Keep it policy-neutral the same way
  `mechanics_bridge` does: balance build-vs-hand and k=1/k≥2 across sessions so recovering from a
  bad turn never correlates with reserve-vs-eager timing. Needs its own `_selftest_error_recovery`
  (gold-correctness, FORMAT_REMINDER text matches driver.py verbatim, non-degenerate pattern/timing
  mix), same pattern as the existing `_selftest_anchor` / `_selftest_mechanics_bridge`.
- **New `train_lora.py` flag — `--skip-urn`:** today `load_sessions` hard-requires
  `urn_<arm>.jsonl` / `tool_bridge_<arm>.jsonl` to exist. Adapter F's run needs to skip both
  entirely (it has no policy signal to carry), so add `--skip-urn` to bypass that requirement.
  Everything else (anchor/mechanics loading, masking, template-match assertions) is reused as-is.

## The merge

PEFT already implements weighted multi-adapter combination — no new merge math needed. Load both
adapters onto one `PeftModel` and combine in weight space:

```python
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "runs/phase3_ft/policy", adapter_name="policy")
model.load_adapter("runs/phase3_ft/format", adapter_name="format")
model.add_weighted_adapter(
    adapters=["policy", "format"], weights=[1.0, 1.0],
    adapter_name="merged", combination_type="dare_ties",   # "linear" as a simpler first pass
)
model.set_adapter("merged")
model = model.merge_and_unload()
```

`dare_ties` (or plain `ties`) matters more than `linear` here specifically because Adapter P and
Adapter F were never trained jointly and may update the same `q_proj`/`down_proj` directions for
unrelated reasons — TIES trims low-magnitude noise and elects a sign per-parameter before summing,
which is the standard fix for merge-time interference between independently-trained adapters.

**`merge_lora.py` change:** currently takes one `--adapter` dir. Extend it to take two
(`--adapter-policy`, `--adapter-format`) plus `--weight-policy` / `--weight-format`
(default 1.0/1.0) and a `--combine` choice (`linear` / `ties` / `dare_ties`, default `dare_ties`).
Everything downstream (bf16 merge, GGUF convert, Ollama Modelfile, serving) is unchanged from the
single-adapter path documented in `box-setup.md §B2`.

## Why this is cheap to iterate on

The merge weights `(w_policy, w_format)` and `combination_type` are **merge-time** hyperparameters,
not training-time ones — re-merging with a different ratio is a linear-algebra op on already-trained
adapters (minutes, no backward pass), not a retrain. Design A's actual iteration cost five full
retrains to explore five corpus mixtures, each producing a different collapse pattern
(`qwen-finetune-transfer-plan.md` Result 2 table). Here, Adapter P and Adapter F are each trained
**once**; the merge ratio can be swept afterward at near-zero marginal cost per point.

## Eval plan

Same harness, same protocol as the existing runs (`driver.py`, A2 protocol, seeds 2000–2011,
GGUF→Ollama serving per `box-setup.md §B2` — vLLM+hermes remains a dead end for Qwen-Coder tool
calls, see `qwen-finetune-transfer-plan.md §Serving`):

1. **Urn eval on Adapter P alone** (sanity check, cheap, text-only, vLLM is fine here) — confirm the
   policy-only adapter reproduces the existing urn result (reserve near-optimal, ~101% of π\*) before
   spending time on any merge. If it doesn't, stop — something about removing anchor/mechanics from
   the urn run changed the urn result, which would be a surprise worth understanding on its own.
2. **Tool eval on Adapter F alone** (merged weight = format only, `w_policy=0`) — confirm the format
   adapter, by itself, produces a tool session that stays legible under the retry-nudge loop for the
   full 60 problems (the thing Design A never achieved on *any* checkpoint). This isolates whether
   `error_recovery.jsonl` actually fixes legibility independent of any policy signal at all.
3. **Tool eval on the merged model**, sweeping `(w_policy, w_format)` starting at `(1.0, 1.0)` —
   look for reserve-then-build behavior (first-sight %, lateness) surviving in the tool framing while
   the session stays parseable.
4. Only after (2) and (3) succeed at producing a legible full-length session does a regret-level
   comparison against the pre-FT baseline (tool A2 lateness 0.125, regret 2934±324, n=12) become
   meaningful.

## Reading the outcome

- **Merged model still collapses in tool eval, same as Design A** → the collapse isn't a
  corpus-entanglement artifact; combining *any* policy-bearing signal with *any* tool-modality
  signal breaks legibility at the weight level too. Points toward RL (`rl-finetuning-plan.md`)
  instead, since RL's on-policy rollouts get error→correction→recovery exposure without needing it
  hand-synthesized into a merge-time adapter.
- **Adapter F alone (step 2) is legible, but the merge stays eager (no reserve behavior)** → a
  genuinely clean negative-transfer result — cleaner than Design A's, because format-robustness and
  policy-transfer are now actually decoupled (Design A never got a legible enough tool eval to
  claim this). Adapter F is then reusable as a fixed ingredient for any future SFT or RL attempt.
- **Some `(w_policy, w_format)` shows reserve behavior in tool while staying legible** → transfer
  win. Compare against the SFT urn ceiling (`qwen-finetune-transfer-plan.md` Result 2: 7%
  first-sight, 101% of π\* balls) the same way the RL plan's open follow-ups propose.

## Open follow-ups

1. **Write `error_recovery.jsonl` generator + selftest** — the one genuinely new corpus piece;
   everything else reuses existing slices. Not started.
2. **`--skip-urn` flag on `train_lora.py`** — small, mechanical. Not started.
3. **Two-adapter `merge_lora.py`** — extend the existing single-adapter merge script per the sketch
   above. Not started.
4. **Run Adapter P and Adapter F training** — per [[no-auto-reps]], scope box-hours and get
   go-ahead before launching; each is a single-corpus run comparable in size to existing Design A
   attempts, so cost should be similar per-adapter, not additive with a joint corpus.
5. **Merge-weight sweep + eval**, per the eval plan above.
