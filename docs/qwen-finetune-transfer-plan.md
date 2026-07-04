# Qwen fine-tune-transfer experiment — Phase 3 plan (bridge SFT)

Companion to `online-tool-investment-plan.md` (the headline claim) and `online-tool-investment-working-notes.md`
(Phases 1-2 status). This doc specs **Phase 3**: fine-tune Qwen2.5-Coder-14b so that whatever allocation
competence it picks up actually transfers into the tool game, and **Phase 4** (the transfer eval itself).

## Status coming in

- **Phase 1 (done):** Qwen-Coder urn slope (0.5b→32b) is a noisy suboptimal plateau, not a smooth
  scaling curve — competence only appears as a frontier jump at Opus. a_script calibration: coding
  ability scales smoothly with size (0.21→0.96); **FT target = Qwen2.5-Coder-14b** (a_script 0.83,
  a_hand 0.00 — codes fine, allocates badly).
- **Phase 2 (done):** Qwen-14b tool baseline is eager (96% first-sight, lateness 0.043, n=12 seeds
  2000-2011), matching every other model tested. Regret vs exact π\* (measured a_script=0.83) = 3268 ±
  659. This is the **pre-FT baseline** every Phase 3/4 result gets compared against.
- Full detail, tables, and the two Ollama tool-calling bugs fixed along the way: working-notes
  § Qwen-Coder urn slope / a_script calibration / tool baseline.

## The transfer-design decision: bridge SFT, not zero-shot urn-only

Two designs were considered:

- **(a) Urn-only SFT → zero-shot tool-game eval.** The scientifically cleanest test of pure abstract
  transfer, but tangential to the immediate goal here (make the model better at the tool game) and, per
  this project's own thesis (coding framing suppresses recognition even in models that have the skill),
  plausibly returns a flat zero — informative but not constructive.
- **(b) Bridge SFT — chosen.** Train on a large diverse corpus of urn demonstrations (so the model
  learns the abstract policy, not a lexical pattern tied to one prompt template) **plus a small slice of
  tool-framed demonstrations of the same optimal policy**, so the abstraction has a foothold in the
  target vocabulary (`write_script`/`run_script`/`submit_answer`) without the SFT signal being
  dominated by tool-domain memorization.

## Demonstration generation (local, free — no LLM calls, no GPU)

Both slices are synthesized directly from `exact_dp.ExactDP.policy_builds` (the exact same-info optimum
already built for scoring) and `family_kit`'s reference implementations — no model calls needed to
generate training data, so this step costs nothing.

### Urn slice (bulk of the corpus)
For many `(vocabulary, N, T, B, seed)` combinations: build a stochastic stream, run π\*'s policy forward,
render a full synthetic conversation (system prompt + one user turn per draw + assistant turn = short
CoT rationale distilled from the belief state + KEEP/PASS). **Vocabulary is varied per session**
(balls/colors, tickets/prizes, coins/mints, ...) plus N/T/B — the point is to force the model to abstract
the policy ("recurrence demonstrated + budget available → reserve for it") rather than pattern-match a
fixed prompt template, which is exactly the failure mode this whole project keeps finding.

### Tool-bridge slice (small fraction of the corpus)
Short (T≈15-20, not the full 60) tool-game sessions on the same uniform-hard 8-family pool (MAG=100),
on **fresh seeds (4000+)**, disjoint from the 2000-2011 eval seeds. Same π\* decisions, rendered as real
tool calls:
- **wait** (π\* says don't build): skip `write_script`, go straight to `submit_answer(gold)` — gold read
  directly off `family_kit`'s reference function, so the allocation signal isn't contaminated by hand-solve
  noise (Qwen's a_hand is ~0 for this pool; that's a separate, irrelevant capability gap).
- **build** (π\* says build now): `write_script(name, code)` using the family's canonical reference code
  (or a verified-correct script pulled from the real Phase 2 transcripts) → `run_script` → `submit_answer`.
- **reuse** (type already built): `run_script(existing)` → `submit_answer`.
Each tool call is preceded by a short rationale in the tool game's own vocabulary ("this type has
recurred — worth a script now"), so the reasoning pattern is anchored in both surface forms.

### Mixing ratio
Target **~85/15 urn/tool-bridge by session count**. Bridge is meant to be a nudge, not the dominant
signal — if it dominates this collapses into plain tool-domain SFT and the transfer question becomes
moot. Worth revisiting only if the first result is ambiguous; not sweeping it up front.

## Control condition (rules out the "any SFT helps" confound)

Same generators, same data volume/format/diversity, but the label source is swapped from π\*'s decisions
to the **model's own current eager policy** (build/keep on first sight — reusing `pi_star.eager_builds`
for the tool side, analogous first-sight logic for the urn side). If eager-SFT leaves tool-game
lateness/regret unchanged while π\*-SFT moves it, the gain is attributable to the reserve-policy content
specifically, not to generic exposure to clean, well-formatted decision transcripts (e.g. incidentally
better tool-call JSON formatting, which was already a real bug source in Phase 2).

## Data validation (before any GPU time)

- Format both slices as chat-format JSONL with assistant-turn spans marked for loss masking.
- Assert every rendered decision matches `policy_builds`'s actual output and every `submit_answer`
  matches the family's true gold (`grading.correct_to_sigfigs`).
- Eyeball a handful of rendered transcripts for sanity (rationale reads sensibly, matches the real
  `TOOL_SCHEMAS()`/message-role structure `driver.py` uses, so the fine-tuned model sees the exact same
  protocol at eval time).
- Report final dataset sizes / token counts, confirm the ~85/15 ratio in practice.

## Training infra

- Ollama serves GGUF, which isn't directly fine-tunable — need the original HF checkpoint
  (`Qwen/Qwen2.5-Coder-14B-Instruct`), not the Ollama-pulled tag.
- **LoRA**, not full fine-tune (cheaper, faster to iterate on a single H100). Leaning toward Unsloth for
  iteration speed; falls back to plain HF + PEFT if Unsloth has friction with this model.
- Conservative starting hyperparameters: rank ~16-32, 2-3 epochs, small LR — tune only if the smoke
  train or first real run looks obviously wrong.
- Smoke train (a handful of steps) before either real run, to confirm the pipeline runs end-to-end.
- **Serving:** merge the LoRA adapter into the base checkpoint, serve via vLLM. `lomekwi/raw_chat.py`
  already has a `LOCAL_BACKEND=vllm` routing branch, but it hasn't been exercised yet this project — gets
  its own smoke test before any real eval traffic.

## Eval plan

1. **Held-out urn re-eval** (fresh seeds, disjoint from training) for both treatment and control — do the
   fine-tunes even move the model off the plateau in its own training domain?
2. **The real test — tool-game re-eval on the same seeds 2000-2011** already used for the pre-FT
   baseline (96% first-sight, lateness 0.043, regret 3268) — direct three-way comparison (pre-FT /
   control / treatment) via `arm_a1_announce.py --model <served-tag>`, unmodified.
3. **Re-run `a0_oracle_gap`** on the fine-tuned checkpoint(s) — a_script may shift from fine-tuning, so
   regret needs recalibrated costs before it's trustworthy; don't reuse the pre-FT 0.83 blindly for the
   fine-tuned model.
4. **Lead with lateness, not regret** — consistent with this project's standing convention (regret is
   noisier at this sample size, and now has an extra moving part in the recalibrated a_script).

## Implementation steps (order of work)

1. Urn-diverse + tool-bridge demonstration generators (local, free).
2. Control-condition variant (swap π\* labels for eager-policy labels).
3. Format to chat-JSONL + validate (assertions + eyeball + size/token report). **Checkpoint: confirm
   concrete dataset sizes before touching training infra.**
4. Training infra on the box: pull HF checkpoint, install LoRA stack, write training script, smoke train.
5. **Real training runs (treatment + control) — stop and confirm before this step.**
6. Merge + serve via vLLM (with its own smoke test).
7. Eval: held-out urn, tool-game on matched seeds, `a0_oracle_gap` recalibration.
8. Write results into working-notes/plan + memory, same pattern as Phases 1-2.

## Open risks / things that could go sideways

- **Bridge ratio might need retuning** if 15% tool-framed data isn't enough to anchor the policy in the
  target vocabulary (or, conversely, is already enough to make this indistinguishable from plain
  tool-domain SFT) — see it as a knob, not a fixed constant, if the first result is ambiguous.
- **a_script drift post-FT** — fine-tuning on a lot of correct `write_script` calls could change script-writing
  reliability as a side effect, independent of the allocation policy; the recalibration step (eval step 3)
  exists specifically to keep this from contaminating the regret comparison.
- **vLLM serving path is untested in this project** — first real exercise of the `LOCAL_BACKEND=vllm`
  branch in `raw_chat.py`; budget time for this to need debugging, similar to the Ollama tool-calling
  bugs hit in Phase 2.
- **Catastrophic forgetting** — worth a quick sanity check that the fine-tuned model hasn't lost general
  coding ability, not just that its allocation policy changed.
