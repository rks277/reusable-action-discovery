# Qwen fine-tune-transfer experiment — Phase 3 (bridge SFT)

Companion to `online-tool-investment-plan.md` (headline claim; **read its §0 Orientation for notation**)
and `online-tool-investment-working-notes.md` (Phases 1-2 status). GPU box mechanics live in
`docs/box-setup.md` (§B0 training, §B2 GGUF→Ollama serving — the only working serving path, see below).

## Status (2026-07-06)

Two SFT attempts, then a **root-cause diagnosis that found and fixed two concrete bugs** — a training
masking bug and a `mechanics_bridge` corpus-construction bug — that together are a much
better-supported explanation for Result 2's collapse than either of the two working hypotheses that
came before it (data-coverage gap; urn/tool objective entanglement). The retrain that applies both
fixes is called **mechanics bridge training** below (the bug lived in that slice) — this supersedes the
original error-recovery ablation's rationale, though the error-recovery corpus built for that ablation
is folded into the same retrain as a smaller, complementary piece.

1. **Original run (2026-07-05) — "framing wall," now judged CONFOUNDED.** SFT installed the reserve
   policy in the urn and the tool eval came back 100% eager. But the tool-vocabulary training data was
   ~25:1 eager-flavored (150 build-only anchor sessions vs. 6 policy-bearing `tool_bridge` demos), so
   "100% eager" may just reflect what the anchor taught, not a real inability to transfer. See **Result 1**.

2. **Design A re-run (2026-07-06) — pure-transfer redesign, tool eval BLOCKED.** Rebuilt the corpus so
   the tool vocabulary carries **zero** reserve-timing signal (`N_TOOL=0`, balanced anchor), so any
   reserve behavior at tool eval could only have transferred from the urn. The urn side replicated clean
   (reserve intact, even strengthened). But getting a *legible tool-call channel* out of the fine-tuned
   model in a long (60-problem) session turned out to be its own unsolved problem: five different training
   configs each produced a **different** malformed-output collapse from problem 2 onward, and an eval-time
   format reminder didn't fix any of them. See **Result 2** — this is the open thread.

3. **Mechanics bridge training (2026-07-06) — two concrete bugs found and fixed, not yet retrained.**
   Diffing the saved failing transcripts against the training corpus found: (a) a `train_lora.py`
   masking bug that trained the model to treat the chat template's own `<|im_start|>assistant\n`
   role-declaration text as legitimate content, and (b) a `mechanics_bridge` corpus bug (its "reuse"
   branch) that is the only place in the entire corpus with the exact shape needed to reinforce that bug
   from an in-context example. Both are now fixed; see "**Mechanics bridge training — root-cause
   diagnosis and fix**" below for the full evidence chain. This is a materially stronger candidate
   explanation than the coverage-gap hypothesis that motivated the original error-recovery ablation.

**Original working hypothesis for the collapse (superseded, kept for the record):** every training
session (urn, anchor, mechanics bridge) is well-formed by construction — none of them contain a "turn
goes wrong → correction → recovery" example. The eval harness's retry-nudge puts the model into exactly
that never-seen conversational shape the instant one turn fails, and it has no learned behavior there.
This hypothesis motivated the original error-recovery ablation (below), whose corpus is still included
going forward, but it's no longer the leading explanation — see the mechanics bridge training section
for why. This was tested directly: restating the
exact `<tool_call>` syntax at the point of failure did not help, on two different checkpoints. That rules
out "the model forgot the format" — it's a distributional gap in training, not missing information.

## The question

Qwen-14b's tool failure is *genuine absence* of the allocation policy, not framing-suppression
(N-disclosure barely moves it — urn A2 regret 737, tool A2 regret 2934, eager on all 12 seeds — see
`online-tool-investment-plan.md` §3). So: can we **install** the reserve-then-build policy by SFT on
π\*-optimal demonstrations, and does it **transfer** from the urn framing into the tool framing? Positive
transfer is real learning; negative transfer (learns the urn, still eager in tool) is itself a strong
finding — a framing gap that survives a model freshly taught to allocate.

**Pre-FT baseline (Qwen2.5-Coder-14B-Instruct, the number every result below is measured against):**
tool A2 lateness 0.125, first-sight 88%, regret 2934±324 (n=12, seeds 2000-2011); urn A2 regret 737
(seeds 2000-2023).

## Corpus (current: Design A)

Generated deterministically by `scripts/creator/tool_disposition_benchmark/phase3_demos.py` — no LLM
calls, no GPU, free, reproducible (`--selftest` guards every invariant below). Current composition for
the `pistar` arm (345 sessions total):

| slice | file | sessions | purpose |
|---|---|---|---|
| urn | `urn_pistar.jsonl` | 170 | teaches the reserve-then-build policy (text `DECISION: KEEP/PASS`, 7 vocabularies, A2 protocol — discloses N) |
| tool_bridge | `tool_bridge_pistar.jsonl` | **0** | intentionally empty — see "why zero" below |
| anchor | `anchor_tool.jsonl` | 150 | arm-independent, policy-neutral tool-calling modality (single-problem, `ANCHOR_HAND_FRAC=0.5` balanced build/hand so it can't teach an eager reflex either) |
| mechanics bridge | `mechanics_bridge.jsonl` | 25 | arm-independent, policy-neutral long-context sessions (real N=8/T=60 streams, build timing forced half-k=1/half-k≥2 via `random_builds` so it can't teach reserve or eager) |

**Why `N_TOOL=0` (no policy-bearing tool_bridge):** the original run's `tool_bridge` (6 sessions,
π\*-labeled) put a real reserve-timing signal into tool vocabulary — a positive tool-eval result then
wouldn't distinguish "transferred from the urn" from "directly taught in tool_bridge." Removing it makes
transfer attribution airtight: nothing in the tool vocabulary demonstrates reserve-vs-eager timing, so
reserve-in-tool (if it appeared) could only come from the urn.

**Why the anchor and mechanics bridge exist:** removing `tool_bridge` also removed the model's only
exposure to tool-calling at all (anchor) and to a *persistent 60-problem* tool session (mechanics bridge)
— both needed just to keep the model *capable* of emitting tool calls, without smuggling back a
reserve-vs-eager signal. Both are **arm-independent** (generated once, shared by both SFT arms) and
**policy-neutral by construction**: the anchor is single-problem (no recurrence exists to demonstrate
timing); the mechanics bridge forces a fixed 50/50 split between committing at first sighting and
committing after a random later sighting, so build timing carries no correlation with recurrence either
way. `_selftest_anchor` / `_selftest_mechanics_bridge` assert this (gold-correctness, N disclosure,
non-degenerate k=1/k≥2 split) on every regeneration.

**Eager control:** not trained in Design A. The "any-SFT-helps" question was already answered by the
original run's urn 3-way (pistar/eager arms moved in opposite directions), and since the tool vocabulary
now contains zero reserve signal, transfer attribution doesn't need a control to stay clean — reserve-in-
tool can only be urn-sourced regardless. Available as a belt-and-suspenders follow-up, not run.

## Training

LoRA on `Qwen/Qwen2.5-Coder-14B-Instruct` (HF checkpoint, not the Ollama GGUF — GGUF isn't trainable).
`r=32, α=64, dropout=0.05`, all linear target modules, `lr=1e-4` cosine, `EPOCHS=2` (see below), QLoRA
4-bit (`--qlora`, required — bf16 OOMs on long sessions) + `expandable_segments` for the cuDNN allocator.
Backend is `--backend hf` (Unsloth had transformers-5.13 friction). Exact config: `train_lora.py`.

**`load_hf` disables the cuDNN SDPA backend** (`torch.backends.cuda.enable_cudnn_sdp(False)`, keeps
flash/mem-efficient) — torch 2.12/cu13's cuDNN attention backward crashes on long (~20k-token) sessions;
this is now a standing fix, not a per-run workaround.

**`build_example` renders tool sessions WITH `tools=TOOL_SCHEMAS()`** (`_session_tools`), matching what
the eval harness injects — training without this produced a `<tools>`-block train/eval mismatch that made
an earlier model reverts to CoT at eval. Urn sessions stay tools-free (they're text, not tool-calling).

**EPOCHS=2, not 4:** 4 epochs was tried and made things worse — the model overfit into verbatim
phrase-memorization (repeating a fixed anchor rationale string regardless of prompt content, ignoring
even an explicit format correction). 2 epochs is the current default; see Result 2 for why more wasn't
the fix.

## Serving

**vLLM+hermes is a dead end for Qwen-Coder tool-calling** (confirmed on both the FT model and the base
model): Qwen2.5-Coder spontaneously emits tool calls as markdown-fenced JSON, not the `<tool_call>` XML
`hermes` requires, so `tool_choice="auto"` silently returns no tool call. Forced `tool_choice` proves the
capability is there — it's a parser mismatch, not incapacity.

**Working path: merge → GGUF → Ollama**, reusing the stock `qwen2.5-coder:14b` Modelfile template (which
has the same `<tools>`/`<tool_call>` rendering the base model was trained on):
```bash
python -m scripts.creator.tool_disposition_benchmark.merge_lora --adapter <adapter_dir> --out <merged_dir>
python llama.cpp/convert_hf_to_gguf.py <merged_dir> --outfile <name>.gguf --outtype f16
ollama show --modelfile qwen2.5-coder:14b > ref.modelfile
{ echo "FROM <name>.gguf"; grep -vE "^FROM |^# " ref.modelfile; } > ft.modelfile
ollama create <model-name> -f ft.modelfile
```
Full recipe: `box-setup.md §B2`. vLLM remains fine for the text-only urn eval.

## Result 1 — original run (2026-07-05), SUPERSEDED / CONFOUNDED

| qwen-ft-pistar (build-heavy anchor) | n | first-sight | lateness | regret | balls |
|---|---|---|---|---|---|
| urn A2 | 24 | 7% | 1.51 | −92±379 (≈0) | 101% of π\* |
| tool A2 | 10 (2 errored) | 100% | 0.000 | 5043±669 | — |

Reserve installed near-optimally in the urn; tool eval came back fully eager. Read at the time as a
"framing wall." **Retrospective problem:** the tool vocabulary in this corpus was ~25:1 eager-flavored
(150-session build-only anchor + only 6 π\*-labeled `tool_bridge` sessions), so the eager tool result
could have been *taught by the anchor* rather than reflecting a real transfer failure. Also had two
training bugs (tools-block train/eval mismatch, a cuDNN crash on long sessions) that are now fixed
standing defaults in `train_lora.py` (see Training above) — not re-litigated per run below.

## Result 2 — Design A pure-transfer redesign (2026-07-06), tool eval BLOCKED

**Urn dilution check (unaffected by the tool-vocabulary redesign) — clean, even stronger:**

| urn A2, Design A | n | first-sight | lateness | regret | balls |
|---|---|---|---|---|---|
| qwen-ft-pistar-a | 24 | 7% | 1.51 | −92±379 (≈0) | 101% of π\* |

Identical to Result 1's urn numbers — the balanced anchor didn't dilute the policy at all.

**Tool eval: every configuration collapses into a malformed-output loop from problem 2 onward.** All
five checkpoints solve problem 1 correctly (matches training: short context, matches the anchor's own
framing), then degrade into a *different* unparseable pattern the harness can't read as a real tool call.
`hit_cap=True` on every seed — the session burns its full token budget on retry loops before submitting
more than 1-3 of 60 problems.

| checkpoint | corpus change | problem 1 | problem 2+ failure |
|---|---|---|---|
| `qwen-ft-pistar-a` | `N_MECH=0` (no mechanics bridge) | ✅ | hallucinates `submit_answers` (plural, unregistered tool name) as plain text |
| `qwen-ft-pistar-a2` | `N_MECH=10` | ✅ | invents a `<script>...</script>` pseudo-tag wrapping real Python code |
| `qwen-ft-pistar-a3` | `N_MECH=25` | ✅ | degenerates into `{}` / empty-object repetition |
| `qwen-ft-pistar-a4` | `N_MECH=25`, `EPOCHS=4` | ✅ | **worse**: verbatim repetition of a fixed anchor rationale phrase, ignoring all subsequent turns including an explicit format correction |
| `qwen-ft-pistar-mech25-e2` | `N_MECH=25`, `EPOCHS=2`, + `driver.py` format-reminder fix | ✅ (2/60 before stalling) | collapses into an empty markdown fence (`` 'assistant\n```\n\n```' ``), ignoring the reminder shown on the immediately preceding turn |

**The format-reminder fix (`driver.py: FORMAT_REMINDER`)** restates the exact `<tool_call>{...}</tool_call>`
syntax on any turn with zero parsed tool calls (previously the retry nudge only listed tool *names*, not
the wrapper syntax). It's a real, motivated fix — the syntax genuinely does drop out of a 300k-token
context — but it was **tested against two checkpoints and didn't recover either one**, including one turn
where the correct syntax appears verbatim in the message immediately before the model repeats its own
broken pattern anyway. That rules out "missing information" as the cause. **The fix is left in place**
(it's harmless and may help future checkpoints) but is not sufficient on its own.

**Working hypothesis:** all three training slices (urn, anchor, mechanics bridge) are synthesized to be
well-formed on every turn — none of them ever show the model a turn that fails and needs correcting. The
eval harness's own retry logic (inject a nudge after a bad turn) puts the model in a conversational state
with **zero representation in training**, and its behavior there is undefined rather than "confused about
format." A prompt-level reminder can't fix an undefined region of behavior; only training on it can.

**Not yet attempted (until mechanics bridge training, below):** synthesizing training sessions that include a deliberate
malformed turn followed by a correction and a correct recovery, so the model has *seen* how to recover
once, in training, not just been told the syntax at eval time.

## Mechanics bridge training — root-cause diagnosis and fix (2026-07-06), not yet retrained

Prompted by a challenge to the original error-recovery ablation's premise (would 45 short,
single-problem error-recovery sessions — ~3% of corpus tokens — actually change trained behavior at
all?), diffed the
saved failing eval transcripts (`runs/arm_a1_announce_qwen-ft-pistar-{a,a2,a3,a4,mech25-e2}_latest_n-announced/`)
against the training corpus directly, instead of reasoning about the collapse from the outside. Found a
concrete, well-evidenced mechanism — not the coverage gap, not (necessarily) urn/tool entanglement.

**Evidence, step by step:**

1. **The real transcripts show the model hallucinating the literal chat-template role marker as
   content.** E.g. `qwen-ft-pistar-a`, one turn: `'{"name": "linear_congruence", ...}}\nassistant\n
   {"name": "linear_congruence", "arguments": {"inputs": {"SE'` — the word **"assistant"** appears
   mid-generation, as if the model is starting a second fake turn inside its own output. Checked across
   all 5 saved checkpoints (seed 2000, counting assistant turns whose content contains the literal
   string "assistant"):

   | checkpoint | turns with literal "assistant" text |
   |---|---|
   | `qwen-ft-pistar-a` | 50/55 |
   | `qwen-ft-pistar-a2` | 0/46 |
   | `qwen-ft-pistar-a3` | 52/55 |
   | `qwen-ft-pistar-a4` | 0/60 |
   | `qwen-ft-pistar-mech25-e2` | 60/61 |

   3 of 5 checkpoints show this on 90%+ of turns. `box-setup.md` had already logged this exact string
   (`content: "assistant\n\""`) as "known noise" from the GGUF serving path (§B2) — it isn't serving
   noise, it's trained behavior.

2. **The training masking labels the chat template's own role-declaration text as content to predict.**
   `train_lora.py`'s saved `verify_template` output (`runs/phase3_ft_logs_mech25/train_pistar_designA.log`):
   `first trained tokens decode to: '<|im_start|>assistant\nThis needs an exact, large computation...'`
   — the literal `<|im_start|>assistant\n` prefix (in a Qwen-style template the role name is plain text
   right after the special start token, not a single special token) was part of the LABELED span for
   every assistant turn in the corpus, because `build_example` masked at whole-message granularity. At
   real inference, the model is never asked to generate that prefix — the serving harness's prompt
   construction (`add_generation_prompt=True`) already supplies it before generation starts. Normally
   this is harmless (the model can't "re-emit" a prefix it's never asked to predict from scratch), but:

3. **One corpus slice contains the exact shape needed to reinforce it from an in-context example.**
   `mechanics_bridge`'s "reuse" branch appended a content-only rationale message immediately followed by
   a *separate* tool_calls-only assistant message — two consecutive assistant-role dicts with no
   intervening tool/user turn. Checked every slice for this shape:

   | slice | sessions with back-to-back assistant turns | total occurrences |
   |---|---|---|
   | `urn_pistar` | 0 | 0 |
   | `anchor_tool` | 0 | 0 |
   | `mechanics_bridge` | 23/25 | 390 |
   | `error_recovery` | 0 | 0 |

   Real eval (`driver.run_session`) never produces this shape — every turn is exactly one assistant
   dict, whether or not it carries tool_calls. On those 390 occurrences, the tokenized training sequence
   contains, mid-turn, a *second* `<|im_start|>assistant\n` directly following the first assistant
   message's own trained content — and per bug (2), that second role-prefix was *also* labeled as
   something to predict. The model was trained, hundreds of times, that "after finishing a response,
   emitting `<|im_start|>assistant\n` and continuing" is sometimes the correct next-token continuation.
   That is exactly the collapse pattern in the real transcripts.

**This is a better-supported explanation than either prior hypothesis:** it isn't "needs more
error-recovery examples" (that data wouldn't touch this mechanism at all — the model isn't messing up
for lack of a rescue path, it's being taught a specific bad continuation habit), and it doesn't require
positing entanglement between the urn and tool objectives (though that question remains separately
interesting for `docs/adapter-merge-transfer-plan.md` if this fix turns out insufficient).

**Fixes applied (both in this repo now, not yet retrained):**
1. `phase3_demos.py` — `render_tool_bridge_session` / `render_mechanics_bridge_session`'s "reuse"
   branch now emits ONE assistant message (rationale + `run_script` tool_calls together), matching
   every other branch and matching eval's actual per-turn shape. New regression guard
   `_selftest_no_consecutive_assistant` asserts no generated session, in any slice, ever contains two
   back-to-back assistant-role messages.
2. `train_lora.py` — `build_example` now excludes the `<|im_start|>assistant\n` role-declaration prefix
   from the trained span on every assistant turn (computed by diffing against
   `apply_chat_template(messages[:i], add_generation_prompt=True)`, the exact prefix a real inference
   call supplies before generating). `verify_template` now asserts the first trained span never starts
   with the literal role-declaration text, as a standing regression guard.

Both changes are corpus/training-pipeline fixes, independent of the original error-recovery ablation
below — the `error_recovery` slice built for that ablation is unaffected by (and doesn't overlap with)
either bug, and is still included in mechanics bridge training's corpus as a smaller, complementary
addition.

**Next step (not yet run):** run mechanics bridge training — retrain the `pistar` arm with both fixes
applied (plus `error_recovery`, already merged in) and rerun the tool A2 eval. This is now the
highest-confidence next experiment, ahead of any further corpus tuning or the adapter-merge plan.
**Attribution caveat:** because this retrain bundles the bug fixes with the already-built
`error_recovery` addition, a clean result won't by itself say which one mattered — if that distinction
matters later, an optional follow-up (`--recovery off`, same bug fixes) would isolate whether the bug
fixes alone were sufficient.

## Error-recovery ablation — data folded into mechanics bridge training (2026-07-06)

**Superseded as the leading hypothesis by mechanics bridge training's root-cause diagnosis above** —
kept as-is below since the corpus it produced (`error_recovery.jsonl`) is still a reasonable
complementary addition (general
resilience to occasional stochastic formatting slips is still worth having, even once the systematic
collapse bug is fixed), just no longer the explanation for Result 2's *systematic* collapse.

Before reaching for anything architectural (e.g. splitting the corpus into separately-trained adapters
so the policy and format objectives never share a gradient step — scoped in
`docs/adapter-merge-transfer-plan.md`), the direct, minimal test of the "missing coverage" hypothesis is
just to add the missing coverage to the **existing joint corpus** and retrain once. If that alone fixes
legibility, the adapter-merge complexity is unnecessary; if it doesn't, that's evidence the fix needs
isolation from the urn objective to survive training, and the adapter split becomes warranted instead of
speculative.

**What's built:** a new arm-independent slice, `error_recovery.jsonl` (`phase3_demos.py`,
`render_error_recovery_session` / `generate_error_recovery`, `N_RECOVERY=45`), covering the three actual
error branches in `driver.run_session` — not paraphrased, byte-matched to the harness's own strings:

| pattern | bad turn | harness response (verbatim-matched) | recovery |
|---|---|---|---|
| `reminder` (15 sessions) | one of the 5 malformed-content patterns actually observed in Result 2 (unregistered plural tool name as plain text, pseudo-`<script>` tag, empty-`{}` repetition, verbatim anchor-phrase loop, empty markdown fence) — no `tool_calls` key at all | `driver.FORMAT_REMINDER`, imported directly (not retyped) so it can never drift from what eval injects | a real `<tool_call>` submitting the correct answer |
| `wrong_name` (15 sessions) | a well-formed `tool_calls` entry naming an unregistered tool (`submit_answers`, plural) | the exact unknown-tool tool-role error `driver.py` constructs (`"no such tool '{name}'. Available tools: {sorted(known_tools)}."`) | retry with the correct tool name |
| `unknown_script` (15 sessions) | `run_script` on a name never `write_script`'d | the exact `session_state.op_run_script` "no script named" error | `write_script` → `run_script` → `submit_answer` |

All sessions are single-problem (like the anchor): no cross-problem state exists, so none of them can
carry a build-timing signal — they can only teach recovery, never reserve-vs-eager. Guarded by
`_selftest_error_recovery` (determinism/arm-independence, policy-neutral prompts, every final submit
correct, and every harness-side string byte-matched against `driver.py`/`session_state.py` — not
hand-retyped, to avoid exactly the near-miss-format bug class this project keeps hitting).

**Wiring:** `train_lora.py --recovery {on,off}` (default `on`), same pattern as `--anchor`/`--mechanics`.
Corpus regenerated locally (`phase3_demos.py`, no LLM calls, no GPU, free) — `runs/phase3_sft_data/`
now includes `error_recovery.jsonl` (45 sessions, ~29k est. tokens) alongside the existing slices, ready
to sync to a fresh box.

**Next step:** superseded by the root-cause section above — the retrain now happens with the two bugs
fixed AND `error_recovery` included, in one pass (see that section's "Next step" and attribution
caveat), rather than as a standalone test of the coverage-gap hypothesis.

## What's preserved locally (2026-07-06)

- **Adapters** (LoRA weights only — Trainer's per-epoch `checkpoint-N` subdirs, which duplicate the
  weights plus optimizer state, were deleted; keep this pattern for any future pull):
  `runs/phase3_ft/pistar_mech25-e2` (539 MB, the `EPOCHS=2` checkpoint used in Result 2's last row),
  `runs/phase3_ft/pistar_mech25-e4` (539 MB, the memorization-collapsed `EPOCHS=4` checkpoint).
  `runs/phase3_ft/{pistar,eager}` are the **original** (2026-07-05) Result-1 adapters, pre-Design-A.
- **Eval run dirs** (full transcripts): `runs/urn_qwen-ft-pistar-a_latest_n-announced` and
  `runs/arm_a1_announce_qwen-ft-pistar-{a,a2,a3,a4,mech25-e2}_latest_n-announced` — the source for every
  row in the Result 2 table above.
- **Logs**: `runs/phase3_ft_logs_mech25/` (training curves, merge/GGUF-convert logs, smoke-gate outputs
  for every Design A attempt); `runs/phase3_ft_logs/` (Result 1's logs).
- **Not kept** (regenerable, not pulled): merged bf16 checkpoints and GGUF files for every Design A
  attempt (28 GB / 29.5 GB each) — regenerate from an adapter via `merge_lora.py` + `convert_hf_to_gguf.py`
  in ~5 min if a specific checkpoint needs to be served again.
- **Corpus** is not stored as an artifact — it's deterministic from `phase3_demos.py`, always regenerate
  fresh on a new box rather than syncing the jsonl files.

## Open follow-ups

1. **Run mechanics bridge training** (both bug fixes + error_recovery, rerun the tool A2 eval) — corpus
   is built and regenerated locally, code fixes are in `phase3_demos.py`/`train_lora.py`, all selftests
   pass. This is now the highest-confidence next experiment. Not yet launched (per [[no-auto-reps]], needs go-ahead
   before spending box time).
2. **Attribution follow-up, only if (1) is clean and the distinction matters:** rerun with `--recovery
   off` (bug fixes only, no error_recovery data) to isolate whether the bug fixes alone were sufficient,
   separate from the error-recovery data's contribution.
3. **If (1) still collapses: adapter-merge split** — train the policy (urn-only) and format
   (anchor+mechanics+error_recovery-only) objectives as separate LoRA adapters with no shared gradient
   step, combine via PEFT's weighted-adapter merge. Fully scoped in `docs/adapter-merge-transfer-plan.md`,
   not started. A collapse surviving both bug fixes would be much stronger evidence for entanglement
   than anything available before this diagnosis.
4. **Publication-grade tool eval** — once tool-calling reliability is solved, rerun with more seeds and
   a clean regret-level comparison (a_script recalibration, `a0_oracle_gap.py`, not yet run against any
   Design A checkpoint) and a catastrophic-forgetting sanity check.
5. **Alternative fix, if neither (1) nor (3) works: unify modality.** Give the urn a `decide(KEEP/PASS)`
   tool call so training is 100% tool-calling from the start — the urn's own 170 long sessions would then
   double as long-context tool-calling practice, sidestepping the need for a separate mechanics bridge
   entirely. Bigger lift (urn harness rewrite + pre-FT re-baseline), not started.
