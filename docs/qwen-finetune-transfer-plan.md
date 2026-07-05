# Qwen fine-tune-transfer experiment — Phase 3 plan (bridge SFT)

Companion to `online-tool-investment-plan.md` (headline claim; **read its §0 Orientation for notation**)
and `online-tool-investment-working-notes.md` (Phases 1-2 status). GPU box mechanics live in
`docs/box-setup.md` (§B is the vLLM serving path). This doc is the **complete, concrete spec** for
**Phase 3** (fine-tune Qwen2.5-Coder-14b to install the allocation policy) and **Phase 4** (the transfer
eval). Corpus is built; everything below is meant to be executable once approved. **Stop-points marked
🛑 need explicit go-ahead per [[no-auto-reps]] before any billable box time.**

## The question

Qwen-14b's tool failure is *genuine absence* of the allocation policy, not framing-suppression (plan §3:
N-disclosure barely moves it — urn A2 737, tool A2 2934, eager on all 12 seeds). So it's the right
subject: can we **install** the reserve-then-build policy by SFT on π\*-optimal demonstrations, and does
it **transfer** from the urn framing into the tool framing? A positive transfer is real learning; a
*negative* transfer (learns the urn, still eager in the tool) is itself a strong finding — the framing
gap would be a hard wall even for a model freshly taught to allocate.

## Status coming in

- **Phase 1 (done):** Qwen-Coder urn slope (0.5b→32b) is a noisy suboptimal plateau, not smooth scaling;
  competence only appears as a frontier jump at Opus. a_script scales smoothly (0.21→0.96). **FT target =
  Qwen2.5-Coder-14b** (a_script 0.83, a_hand 0.00 — codes fine, allocates badly).
- **Phase 2 (done):** Qwen-14b tool baseline is eager (A2: 88% first-sight, lateness 0.125, regret 2934 ±
  324, n=12 seeds 2000-2011; no-N: 96%/0.043/3268). This is the **pre-FT baseline** every Phase 3/4 result
  is compared against. urn A2 baseline = 737 (seeds 2000-2023).
- **Corpus (done 2026-07-04)** — see next section. Both pre-training tweaks applied + regenerated.

## The transfer-design decision: bridge SFT, not zero-shot urn-only

- **(a) Urn-only SFT → zero-shot tool eval.** Cleanest test of *pure* abstract transfer, but per this
  project's own thesis (coding framing suppresses recognition even in models that have the skill),
  plausibly returns a flat zero — informative but not constructive.
- **(b) Bridge SFT — CHOSEN.** Train on a large, surface-diverse urn corpus (learn the *policy*, not a
  lexical template) **plus a small tool-framed slice of the same policy**, so the abstraction has a
  foothold in the target vocabulary (`write_script`/`run_script`/`submit_answer`) without the SFT signal
  being dominated by tool-domain memorization.

## Corpus — as built (`runs/phase3_sft_data/`)

Generated deterministically by `scripts/creator/tool_disposition_benchmark/phase3_demos.py` from
`exact_dp.ExactDP.policy_builds` (treatment) / `pi_star.eager_builds` (control) + `family_kit` reference
code — **no LLM calls, no GPU, free, reproducible.** All four files are **A2** (disclose exact N),
matching the eval arm.

| file | sessions | est. tokens | notes |
|---|---|---|---|
| `urn_pistar.jsonl` | 170 | ~471k | treatment urn |
| `tool_bridge_pistar.jsonl` | 6 | ~95k | treatment tool bridge |
| `urn_eager.jsonl` | 170 | ~679k | control urn |
| `tool_bridge_eager.jsonl` | 6 | ~93k | control tool bridge |

- **Format:** chat JSONL, `{"messages": [...]}` per line; roles `system` / `user` / `assistant` / `tool`.
  Assistant turns carry either `content` ending in `DECISION: KEEP|PASS` (urn) or a `tool_calls` array
  (`write_script` / `run_script` / `submit_answer`). Tool results are `role:"tool"` messages. This is the
  **exact** protocol `driver.py` / `urn_session.py` present at eval time — the FT model sees no format it
  wasn't trained on.
- **Two arms, identical except labels.** Same prompts, streams, seeds, formatting; only the assistant
  *decisions* differ (π\* reserve-then-build vs eager first-sight). This is what makes the eager arm a
  clean control for "any SFT helps."
- **Urn diversity (tweak ii, done):** 7 vocabularies, each with its own attribute word — ball→color,
  ticket→material, coin→metal, tile→symbol, gem→kind, card→suit, token→shape — threaded through the
  system prompt + rationales (was hard-coded "color"). All 7 appear ~evenly across the 170 sessions.
  N∈{6,8}, T∈{60,80,100}, B/N≈0.25-0.35 (empirically the band where π\* genuinely reserves; short-T /
  generous-B collapses even the exact DP toward eager). urn seeds 5000+.
- **Tool bridge (tweak i, done):** cut 30→6 sessions so the corpus is **~85/15 urn:tool by TOKEN**
  (measured tool share 16.8% pistar / 12.0% eager, ≈15% averaged — tool sessions run ~6× longer than urn,
  so the old 30 was ~50/50 by token). Real eval config (N=8, T=60, MAG=100), B alternating 2/3, **fresh
  seeds 4000-4005** disjoint from all eval seeds.
- **Validation guards** (`phase3_demos.py --selftest`, green): FAMILY_CODE == `family_kit` reference
  (100 samples/family); every rendered decision == recomputed `policy_builds`; every `submit_answer` ==
  true gold; N disclosed in both prompts; **π\* reserve rate 93% k≥2 / eager 0%** (arms genuinely
  distinct). Re-run `--selftest` after any regeneration.
- **Known design limit:** on "wait" turns the tool bridge submits gold with no shown work — fine for
  teaching the build/wait *decision* (a_hand≈0 is already in the cost model) but it trains "emit the
  number" on those turns. Acceptable; noted.

## Training — concrete spec

> **STATUS: both arms TRAINED (2026-07-04, H100 209.20.159.76).** `--backend hf` + **`--qlora`** (bf16
> LoRA OOM'd at step 6 on the ~21k-token tool sessions → 4-bit + `expandable_segments`, see box-setup
> §B0). r=32/α=64, 2 epochs, max_seq_len=32768 (0 dropped), ~22 min/arm. Converged, eval loss still
> falling at epoch 2 (no overfit): **pistar** train 0.69→0.04, eval e1 0.122 → e2 0.062; **eager** train
> 0.48→0.03, eval e1 0.049 → e2 0.031 (eager's simpler policy fits tighter — expected). Adapters saved
> `runs/phase3_ft/{pistar,eager}/` (~550 MB each) + backed up locally. **NEXT = steps 5-6 (merge/serve/
> eval), awaiting go-ahead.** Note: QLoRA means merge dequantizes to bf16 for serving; a_script recalib
> (eval step 3) absorbs any 4-bit numeric drift.

**Base checkpoint:** `Qwen/Qwen2.5-Coder-14B-Instruct` (HF, not the Ollama GGUF — GGUF isn't trainable).
Download to the box (~28 GB).

**Method: LoRA** (not full FT — cheaper, one H100, fast iteration), **bf16** base (14B×2B ≈ 28 GB fits an
80 GB H100 comfortably; QLoRA 4-bit is the fallback only if VRAM is tight). Gradient checkpointing ON.

**Framework:** **Unsloth** primary (`FastLanguageModel`, fastest single-GPU path, has
`train_on_responses_only`); **HF Transformers + TRL `SFTTrainer` + PEFT** fallback if Unsloth has friction
with this checkpoint. Both consume the `{"messages":[...]}` files directly.

**LoRA config (starting point — tune only if a run looks obviously wrong):**

| param | value |
|---|---|
| rank `r` | 32 |
| `lora_alpha` | 64 |
| `lora_dropout` | 0.05 |
| target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` (all linear) |
| bias | none |

**Training hyperparameters:**

| param | value | note |
|---|---|---|
| epochs | 2 | try 3 only if underfit on the val-loss curve |
| learning rate | 1e-4 | cosine decay, warmup_ratio 0.03 |
| per-device batch | 1 | sequences are long |
| grad accumulation | 16 | → effective batch ~16 |
| `max_seq_len` | **set from measured max** (see below); default 16384, bump to 32768 if any session exceeds | do NOT truncate tool sessions — that would cut the decision structure |
| packing | **OFF** | each session is one example; preserves within-session context + masking boundaries |
| optimizer | adamw_8bit (Unsloth) / adamw_torch | |
| weight_decay / max_grad_norm | 0.0 / 1.0 | |
| seed | fixed | |

> **max_seq_len must be measured, not guessed.** The corpus token counts above are a crude
> `len(json.dumps)//4`; the real Qwen tokenizer count differs. **Pre-train step:** tokenize every session
> with `apply_chat_template` and report max / p95 / mean; set `max_seq_len` to cover the max (round up to
> 16384 or 32768). If the true max blows past 32768, drop the longest tool sessions rather than truncate.

**Loss masking (the core mechanic):** SFT is next-token prediction over the rendered transcript with the
loss computed **only on assistant tokens** — system / user / tool tokens set to label `-100`. Use
Unsloth's `train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part=
"<|im_start|>assistant\n")` (TRL fallback: `DataCollatorForCompletionOnlyLM` or the chat template's
assistant-mask). **Assistant `tool_calls` ARE trained** (they're the decision); `role:"tool"` result
turns are masked. Net effect: the model is optimized to *produce the right decision given the context*,
not to predict problems or tool outputs.

**Chat template:** use the tokenizer's built-in Qwen2.5 template (`apply_chat_template`), which serializes
`tool_calls` into Qwen's `<tool_call>{...}</tool_call>` form. This **must** match what vLLM's `hermes`
parser reads back at eval (§ serving) — mismatch = the model learns a format the harness can't parse,
exactly the bug class that silently zeroed builds in Phase 2. Add a round-trip assertion (below).

**Two runs, identical config:** treatment = `{urn_pistar + tool_bridge_pistar}`; control =
`{urn_eager + tool_bridge_eager}`. Hold out ~10 sessions per arm as a val split for a train/val-loss curve
(sanity only — the real test is Phase 4, not val loss). Output: one LoRA adapter per arm (~tens of MB).

**Training script (`train_lora.py`, to be written in step 4, on the box):** CLI
`--arm {pistar,eager} --data-dir runs/phase3_sft_data --out runs/phase3_ft/<arm> [--smoke]`. Loads the
two matching jsonl files, concatenates + shuffles (fixed seed), tokenizes via chat template, applies
response-only masking, trains LoRA with the config above, saves the adapter. `--smoke` = 20 steps on a
handful of sessions to confirm the pipeline end-to-end before either real run.

**VRAM / time (H100 80GB):** bf16 LoRA + grad-checkpointing + seq-16k + batch-1 ≈ 40-55 GB. ~1M tokens ×
2 epochs is tiny → **~30-60 min per arm**. Data gen is already done and free.

## Serving (Phase 4 prep) — merge + vLLM

> **STATUS (2026-07-04): merge DONE, vLLM serving in progress (step 5).** Both arms merged via the
> dedicated **`merge_lora.py`** (loads base in **bf16**, applies adapter, `merge_and_unload`, saves) →
> `runs/phase3_merged/{pistar,eager}` (28 GB each). vLLM 0.24.0 installed (it pinned torch 2.12→2.11,
> CUDA still fine — training was already done). pistar being served; next = smoke-test the tool-call
> round-trip before eval. **Do NOT use `train_lora.py --merge-out` for QLoRA runs** (merges onto the
> 4-bit model); `merge_lora.py` is the correct path.

1. Merge the adapter into the base — **use `merge_lora.py`** (`--adapter runs/phase3_ft/<arm> --out
   runs/phase3_merged/<arm>`), which loads the base in bf16 (NOT the 4-bit QLoRA base) and merges there.
   Gives clean bf16 weights for serving + `a0` recalib.
2. Serve via vLLM (box-setup.md §B):
   ```bash
   vllm serve <merged-path> --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.92 \
     --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes
   ```
   `hermes` = the Qwen2.5 tool-call template. Route the harness with
   `LOCAL_BACKEND=vllm VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=EMPTY`.
3. **🛑 vLLM path is UNTESTED in this project** — smoke-test the `LOCAL_BACKEND=vllm` branch of
   `raw_chat.py` with a single seed and assert one real `write_script` tool call round-trips (template →
   generation → hermes parse → harness sees the build) **before** any eval traffic. Budget debugging time
   here, same as the Ollama tool-call bugs in Phase 2.

## Eval plan (Phase 4)

> ### RESULTS IN PROGRESS (2026-07-04) — read this first
>
> **Held-out urn A2, n=24 seeds 2000-2023, temp≈0.7 sampled (same as pre-FT). Both arms done + control:**
>
> | urn A2 | first-sight | lateness | regret | traps/seed | balls |
> |---|---|---|---|---|---|
> | pre-FT Qwen-14b | 76% | 0.36 | 737 | ~0.96 | — |
> | **eager-FT (control)** | 100% | 0.000 | 864±378 | 1.08 | 88% |
> | **pistar-FT (treatment)** | 54% | 0.615 | 795±524 | 0.62 | 88% |
> | π\* target (Haiku A2) | 28% | 1.19 | 0 | ≈0.67 | 100% |
>
> **Clean BEHAVIORAL win, attributable to demo content (the lead metric):** the two arms moved in
> *opposite* directions — π\*-demos → **reserves** (lateness 0.36→0.615, first-sight 76→54%, traps
> 0.62 ≈ π\*'s 0.67), eager-demos → **fully eager** (100% / 0.000, traps 1.08). Opposite directions
> **rules out "any-SFT-helps"**; the reserve policy (and better type-selection) is installable and the
> demo content drives it. **BUT outcome (regret/balls) is ~equal across arms** (795 vs 864, overlapping
> ±hundreds; both 88% of π\*): the treatment reserves + avoids traps, but *waiting* forfeits early
> reuses that offset the trap-avoidance gain (the lateness⊥regret tradeoff, at this g=1/N=8/T=60). So:
> **partial install** — behaviorally π\*-like (reserves, low traps) but short of π\*'s first-sight ~7-28%
> and with no significant regret gain at n=24. **temp=0 (greedy) readout DONE — byte-identical to temp≈0.7
> for BOTH arms** (pistar 54%/0.615/795, eager 100%/0/864). So the policies are **temperature-robust /
> high-confidence**: the partial reserve is the model's genuine learned policy, not a sampling artifact,
> and greedy does not sharpen it further. (`urn_session --temp 0` via a new `temperature` kwarg in
> `raw_chat.chat`; default unchanged.)
>
> **⚠️ TOOL-TRANSFER EVAL (step 6b) IS BLOCKED — training-design issue found.** The FT model **does not
> emit tool calls** in the tool framing — it reverts to base CoT hand-solving (verified: no `<tool_call>`
> with OR without a `tools=` block). Cause: **170 of 176 training sessions are the urn, which responds in
> free *text* (`DECISION: KEEP/PASS`), not tool calls** — training ~97% on text suppressed the base
> model's tool-calling; the 6 tool-bridge sessions couldn't preserve it. So the tool eval can't run on
> this model as trained. Fix options (need a re-train, ~44 min; a design call for the user): (a) give the
> urn demos a tool-call modality (e.g. a `decide(KEEP/PASS)` tool) so all sessions are tool-calling and
> the modality is consistent; (b) raise the tool-bridge fraction (risks the "collapses into tool-domain
> SFT" confound the design wanted to avoid); (c) mix in generic tool-calling data to preserve the skill.
> **Note the urn eval is unaffected** (it's text) — so the primary "did it learn to allocate" question is
> answerable now; only the transfer question waits on a re-train.
>
> **DECISION (2026-07-04): transfer eval PAUSED, urn result BANKED (deliberate, not abandoned).** The urn
> behavioral finding (policy installable + content-driven, clean vs control, temp-robust) stands on its
> own. Box released. To RESUME the transfer eval later, pick a fix for the tool-calling suppression:
> (a) unify everything to tool-calling (convert urn harness+demos to a keep/pass tool; re-baseline pre-FT
> — cleanest, ~half-day); (b) bump tool-bridge share so tool-calling survives + add a bridge-only control
> to isolate transfer (fast, ~1h, larger tool foothold); (c) mix in generic tool-calling data to preserve
> the skill (cleanest for pure abstract-transfer; needs a generic tool-use data source). Adapters backed
> up at `runs/phase3_ft/{pistar,eager}`; merged checkpoints regenerable via `merge_lora.py`.

Run each fine-tune (treatment, control) + the pre-FT baseline — three-way. **Lead with lateness /
first-sight, not the regret level** (standing convention; regret is noisier at n=12 and now has an extra
moving part — the recalibrated a_script).

1. **Held-out urn A2** — `urn_session.py --model <served> --announce-n --seeds 2000..2023`. Seeds 2000-2023
   are disjoint from the training urn (5000+) and tool (4000+) seeds, and match the pre-FT urn A2 baseline
   (737). *Did the FT even move it off the plateau toward π\* in its own training domain?*
2. **Tool A2 (the real test)** — `arm_a1_announce.py --model <served> --announce-n` on seeds 2000-2011.
   Three-way vs pre-FT (lateness 0.125 / first-sight 88% / regret 2934). *Does urn competence transfer to
   the tool framing?*
3. **`a0_oracle_gap.py --model <served>`** post-FT — a_script may drift from training on correct
   `write_script` calls; regret needs recalibrated costs (don't reuse 0.83 blindly for the FT model).
4. **Catastrophic-forgetting sanity check** — a few held-out coding problems to confirm general ability
   isn't wrecked, only the allocation policy changed.

Interpretation grid:

| urn A2 | tool A2 | reading |
|---|---|---|
| π\*-arm improves, eager-arm flat | π\*-arm improves | **transfer** — policy installed + crosses the framing gap |
| π\*-arm improves | π\*-arm flat/eager | **framing wall** — learned to allocate, coding framing still suppresses it (strong finding) |
| both arms improve equally | — | confound: "any SFT helps" → re-examine (masking? tool-JSON cleanup?) |
| π\*-arm doesn't improve even on urn | — | training didn't take → revisit LR/epochs/rank before any tool claim |

## Implementation steps (order of work + stop-points)

1. ~~Urn + tool-bridge demo generators (local, free)~~ **DONE.**
2. ~~Control-condition variant (eager labels)~~ **DONE.**
3. ~~Chat-JSONL + validate (assertions + eyeball + size/token report + 85/15 confirmed)~~ **DONE**
   (`--selftest`; token shares reported above).
4. ~~Box setup + `train_lora.py` + measure `max_seq_len` (32768) + `--smoke`~~ **DONE (2026-07-04).**
   HF backend (not Unsloth — transformers-5.13 friction); two 5.13 fixes in the script.
5. ~~🛑 Real training runs (treatment + control)~~ **DONE (2026-07-04).** `--qlora` + `expandable_segments`
   (bf16 OOM'd on the ~21k-token sessions). Both converged; adapters `runs/phase3_ft/{pistar,eager}` +
   backed up locally. Curves in the Training STATUS block.
6. **Merge + serve via vLLM (IN PROGRESS, step 5).** ~~Merge~~ **DONE** (`merge_lora.py` → `runs/phase3_merged/`).
   Serving pistar; 🛑 smoke-test the `LOCAL_BACKEND=vllm` tool-call round-trip next, before eval.
7. Eval (step 6): held-out urn A2 (2000-2023), tool A2 (2000-2011), `a0_oracle_gap` recalibration,
   forgetting check — vs pre-FT (urn 737 / tool 2934). Per arm: serve, eval, swap.
8. Write results into working-notes / plan / memory (same pattern as Phases 1-2).

## Cost / logistics

- One 80 GB H100, ephemeral (box-setup.md — fresh IP, no persistent disk, full re-setup each time).
- Rough time: setup+download ~30-45 min; 2 LoRA runs ~1-2 h; merge+serve+smoke ~30 min (+ vLLM debugging
  buffer); eval ~30 min. **~4-6 box-hours → ~$15-30** at typical H100 rates, most of the variance in the
  untested vLLM path. Data gen already done + free.
- **Always `rsync runs/` back before releasing the box** (box-setup.md §A.6 — the Qwen A2 raw dirs were
  lost this way once). Includes adapters, merged checkpoints (large — decide whether to keep), and eval
  run dirs.

## Open risks

- **Bridge ratio (15%)** is a knob, not a constant — revisit only if the first result is ambiguous (too
  little tool anchoring, or so much it collapses into plain tool-domain SFT).
- **a_script drift post-FT** — the recalibration step exists to keep this out of the regret comparison.
- **vLLM path untested** — budget debugging (step 6 smoke test gates it).
- **max_seq_len vs long tool sessions** — measure first; truncation would silently corrupt the decision
  structure. Drop over-long sessions rather than truncate.
- **Catastrophic forgetting** — step 7 sanity check.
- **Merged-checkpoint size** — two ~28 GB bf16 checkpoints; decide up front whether to rsync them back or
  regenerate from adapters (adapters are tiny; keeping only adapters + base is the cheap option).
