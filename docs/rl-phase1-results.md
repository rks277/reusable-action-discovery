# RL Phase 1 (urn framing) — RESULTS

Standalone, candid results writeup for the Phase 1 RL work. **Method/design/pilot-history live in
`docs/rl-ppo-credit-assignment-spec.md`** (per-decision PPO + privileged critic, the five flat GRPO
pilots, the build order, the pre-box-run review); this doc is only *what happened and what it means*,
including the caveats we found by reading the raw transcripts. Broader project framing:
`docs/online-tool-investment-plan.md`. All numbers here are from runs on 2026-07-08 (A100-40GB,
`ubuntu@150.136.64.191`), artifacts listed in §7.

---

## TL;DR

1. **RL works in the urn (in-framing).** From the *untouched base* `qwen2.5-coder:14b` (QLoRA,
   per-decision PPO + privileged critic, no demonstrations), a 20-step run **discovered the reserve
   policy from the balls reward alone** and reached π\* parity: paired held-out eval 75%→**32%**
   first-sight, 87%→**101%** of π\* balls. This matches the SFT install but is *reward-discovered*, not
   imitated — the distinction Phase 1 existed to make.
2. **It does NOT transfer to the tool framing — the framing wall holds for RL too.** Zero-shot tool eval
   of the same checkpoint: it stays essentially eager in the tool task (95% first-sight, lateness 0.238)
   despite reserving in the urn (32% / 0.903), over a **verifiably legible** tool channel (0 malformed /
   0 unknown calls). Its tool-frame behavior is nearly **indistinguishable from base**.
3. **Interpretation (the payoff):** SFT already showed an *imitated* reserve policy doesn't transfer; RL
   now shows a *self-discovered* one doesn't either, over a channel we verified is legible. That rules
   out both "it failed only because it was imitated" and "the tool channel can't carry any policy." The
   framing gap suppresses the disposition **regardless of how it was acquired**.

---

## 1. Urn result (in-framing): reward-discovered reserve reaches π\*

**Training:** 20 outer steps, untouched base `qwen2.5-coder:14b`, QLoRA, per-decision PPO + privileged
critic, `n_epochs=1`, `lr=6e-5`, temp 1.2, 25 seeds/step × G=4, training seeds 9000+, **no
demonstrations**. Step-0 mechanism gate passed cleanly (advantage ordering correct *before* any policy
update: `keep_hot +2.23 > keep_trap −2.10`, `pass_trap +2.80 > pass_hot_first −1.50`). In-training:
first-sight 77%→42%, reward 36→40, `mean_kl` grew monotonically to −0.568 and was **still coupled with
reward (productive drift), not converged** at stop — more steps / `n_epochs>1` could push further, but
π\* parity is already reached so that is optional.

**Paired held-out eval** (seeds 2000–2023, disjoint from training; **no-announce** = training condition;
q8_0 GGUF via Ollama; same-seed so the delta is the policy, not stream luck):

| metric | base | RL-final |
|---|---|---|
| first-sight % (eager) | 75% | **32%** |
| mean lateness | 0.375 | **0.903** |
| balls/seed | 33.7 | **39.3** |
| % of π\* balls | 87% | **101%** |
| balls-regret vs π\* | +5.2 | **−0.4** |
| unparsed | 0 | 1 |

RL is less extreme than SFT on first-sight (32% vs SFT's ~7%) yet hits the same balls result — an equally
good urn policy via a slightly less eager-suppressing route.

---

## 2. Tool-transfer result (the headline experiment): no transfer

**Setup.** Paired tool eval, `arm_a1_announce.py`, **A2 / `--announce-n`** (N disclosed, to match the
historical reference baseline), seeds 2000–2011 (the same 12 as that baseline). Both tags served at
**`num_ctx 8192`** with the stock qwen tool template, so the paired delta is weights, not context or
template: base control `qwen-rl-base-ctx8k` (Q4) vs `qwen-rl-urn-final` (q8_0) — the same Q4-vs-q8_0
quant wrinkle knowingly accepted in the urn eval.

| metric (tool A2) | base (Q4) | RL-final (q8_0) | *RL-final in the urn, for contrast* |
|---|---|---|---|
| first-sight % | 100% (23/23) | **95% (20/21)** | *32%* |
| mean lateness | 0.000 | **0.238** (one build, occ. #6) | *0.903* |
| n_malformed / n_unknown | 0 / 0 | **0 / 0** | — |
| valid tool calls / correct answers | 95 / 15 | 75 / 16 | — |
| builds/seed | 1.92 | 1.75 | — |
| regret vs π\* (a_script=1.0) | 3377±717 | 3729±534 | *−0.4 balls-regret* |

**Stage 1 — legibility: PASS.** Both models emitted **0 malformed / 0 unknown / 0 refused** tool calls.
Urn-only QLoRA did **not** damage the shared tool-calling weights. Verified, not assumed (§1 of the spec
flagged that LoRA touches weights both paths flow through).

**Stage 2 — policy transfer: FAILS.** RL-final stays eager in the tool framing (95% first-sight) while
reserving in the urn (32%) — a ~63-point first-sight gap *within the same model, same information*
(N disclosed on both sides). RL moved tool-frame behavior essentially not at all vs base.

---

## 3. Why the transfer failure is REAL and not a tool-channel artifact

We read the raw transcripts (this was the crux — the aggregate numbers alone are misleadingly clean).

**Tool-calling competence is intact — this is not a channel-failure result.** Across the 12 base seeds:
95 valid tool calls, 0 malformed/unknown/refused, 24 scripts written, 30 run calls, 22 answers submitted,
**15 correct**, including debugging a broken script and rewriting it, and reusing saved scripts. RL is
comparable (75 calls, 16 correct). **The base model plainly understands the tools** (this corrects an
in-session overstatement that briefly read the "no-tool turns" rate as a tool-calling deficiency — it is
not; see §4). What is absent from *both* models is any *allocation reasoning*.

**Two behavioral regimes, identical in base and RL, neither of them reserve:**
- **Eager-burst seeds** (RL 2000/2001/2002/2006): a clean write→run→submit loop that spends all B=3
  writes on the first 3 distinct types immediately (first-sight, budget exhausted by problem 2–3).
  Reactive "I need to solve X → write a script," never "should I spend a write here?" Assistant turns
  carry ~no deliberation. This is the cleanest measurement of disposition — working tools, budget in
  hand, no derailment — and it is unambiguously eager.
- **Idle-tail seeds** (RL 2003/2005/2007/2008/2010/…): after ~1–2 early first-sight builds the model
  gives up on the long tail — emitting empty ```json``` fences that yield no tool call, taking the
  harness's "respond with a REAL tool call" reminder each problem, leaving problems unanswered. It even
  fails to *reuse* (base seed 2000 problem 7 is the same 32-bit-xorshift type it built a script for at
  problem 1, yet it emits an empty fence instead of `run_script`). This is an idle/give-up tail, **not
  reserve** — zero waiting/budget/recurrence reasoning anywhere.

**Why "both eager" is robust to the tail (the specific worry: is the tail hiding reserve?).** Per-build
breakdown across all seeds:

| | builds counted | first-sight (occ. #1) | built on a recurrence |
|---|---|---|---|
| BASE | 23 | **23/23 = 100%** | **0** |
| RL-final | 21 | 20/21 = 95% | **1** (seed 2004, class 7, occ. #6) |

1. The metric is computed **only over builds that actually happened, at the moment of first sighting.**
   Base built 23 scripts and *every one* was on occurrence #1 — a literal 100%, nothing for the tail to
   erode. A skipped tail problem is not a "pass"; the model never returns to build it later, so it never
   enters the lateness metric.
2. **Reserve is the opposite of what the tail does.** Reserve = deliberately pass an early sighting,
   keep acting, build on a later recurrence. The tail is the model *ceasing to act*. It cannot
   manufacture a pass-then-build.
3. **Direction of bias runs against the finding, not for it.** The only thing a derailed tail can inject
   is a *stray late build* (a session that happens to emit a build on a later occurrence), which pushes
   lateness *up* — making the model look *less* eager. So the eager finding is conservative. RL's single
   late build is exactly such a stray: seed 2004, class 7, occurrence #6, but from the derailed tail with
   empty-fence context and no reasoning. Remove it and RL is also ~100% first-sight.
4. **It is a comparison, and the tail is common-mode** (base 84% ≈ RL 86% no-tool turns; ~2 scripts/seed
   both). Whatever the tail does, it does equally to both, so it cannot produce a base-vs-RL difference —
   and there is none.

**Mechanism reading:** in the urn the reserve policy lives in the `DECISION: KEEP/PASS` plain-text
decision turns where the model deliberates; RL sharpened *that*. The tool framing puts the model in
reactive "coding assistant, solve the problem in front of me" mode — a context RL-on-urn never saw a
gradient for. The disposition is context-bound, exactly as the recognition/framing hypothesis predicts.

---

## 4. The idle tail, characterized (not reserve, and probably not a model-size wall)

This matters for interpreting builds/seed and for deciding whether "we need a bigger Qwen."

- **It is real and it degrades the eval:** ~85% of assistant turns produce no tool call (base 84%,
  RL 86%), dominated by the long tail after the first handful of problems; many problems go unanswered.
  This is a genuine failure to *sustain* agentic tool use over a 60-problem session — distinct from, and
  not contradicted by, the fact that individual tool-call *mechanics* are clean.
- **It is NOT primarily a context-window wall, and NOT late-session decay:** the tail begins at
  **problem ~3, when the context is only ~1–2k tokens** — nowhere near the 8192 ceiling. So it is an
  *early-onset generation pathology* (empty-```json```-fence collapse), not "degrades after 30 turns of
  accumulated context."
- **`num_ctx 8192` compounds it, does not trigger it:** per-turn input tokens climb and pin at the
  ceiling (max 8184–8189 across long seeds) because the empty fences + reminders accumulate; once the
  window saturates, Ollama front-truncates the system prompt (the tool schema), so the model can never
  recover. The window makes the tail *permanent*; something else *starts* it.
- **It is a pre-existing property of qwen-coder-14b in this long multi-problem harness, common to base
  and RL** — not an RL artifact and not a tool-competence gap.

**Implication for "bigger Qwen?":** plausibly worth it, but for the *research* reason, not as "the
tool-calling fix." 32b is the only rung with **nascent allocation competence in the urn** (regret 216,
CI brushing 0, vs 14b's 737) and higher `a_script` (0.96 vs 0.83), and may be more robust to the tail as
a side effect. **But 14b was deliberately chosen** (plan §3/§7) because its urn failure is *genuine
absence* of the competence, not framing-suppression — the clean subject for "teach it, then test
transfer." 32b already has some competence, so RL/SFT on 32b tests a weaker claim (filed as the optional
stronger subject). Cheaper levers were **tested first (§4.1, 2026-07-09)**: raising `num_ctx` and
lowering temperature do **not** help (temperature *worsens* it); a harness empty-fence hard-retry
**partially** mitigates (breaks the runaway loop, ~halves regret, lifts realized builds) but does **not**
restore sustained engagement. So the tail is a robust 14b generation pathology, which *strengthens* the
"bigger Qwen" case — and, critically, none of these levers moved first-sight off ~91–100%, so the
transfer verdict never depended on the tail.

### 4.1 Idle-tail diagnostic (2026-07-09): cheap levers tested, config ruled out, harness retry partial

Paired re-eval on seeds 2000–2003 (n=4, same-seed vs the 8k/default-temp baseline; noisy — treat as
directional). Two arms:

**(a) Config levers — `num_ctx` 8192→16384 + `temperature` ~0.7→0.2** (both moved at once, via
`PARAMETER` on diagnostic Ollama tags; no harness change):

| arm | first-sight | lateness | submitted/seed | correct/seed | last-answered Q | hit token cap |
|---|---|---|---|---|---|---|
| base 8k / def | 100% | 0.000 | 2.25 | 1.75 | ~2 | 3/4 |
| base 16k / t0.2 | **100%** (6/6) | 0.000 | **0.50** | **0.00** | **~0.5** | 4/4 |
| urn 8k / def | ~95–100% | ~0 | 1.50 | 1.25 | ~1.5 | 1/4 |
| urn 16k / t0.2 | **100%** (8/8) | 0.000 | 1.25 | 1.00 | ~1.5 | 2/4 |

The empty-` ```json``` `-fence loop **persists** at 16k + low temp (confirmed in transcripts; both arms
run to the 300k token cap emitting empty fences; the urn model even began hallucinating tool results in
prose). Lower temperature made base *worse* (0 correct) — consistent with the collapse being a
low-entropy attractor. **Verdict: not a context-window wall, not a sampling-temperature artifact.**

**(b) Harness empty-fence hard-retry — `--empty-fence-retry 4`** (new lever, `driver.py`): a no-tool turn
is *pruned* from context and the same problem is re-prompted with an escalating reminder up to 4 attempts
before force-advancing (so empty fences never accumulate and self-reinforce). At the original 8k/default
config vs the same 8k baseline:

| arm | first-sight | builds/seed | regret (a_script=1.0) | scripts/seed | last-answered Q | hit cap | turns/seed |
|---|---|---|---|---|---|---|---|
| base 8k baseline | 100% | 1.92 | 3377 | 2.00 | ~2 | 3/4 | 46.8 |
| base 8k **+EFR4** | **91%** (10/11) | **2.75** | **1199** | 2.75 | ~4 | **1/4** | 29.8 |
| urn 8k baseline | 95% | 1.75 | 3729 | 2.50 | ~1.5 | 1/4 | 22.5 |
| urn 8k **+EFR4** | **92%** (11/12) | **3.00** | **1832** | 3.00 | ~2 | **0/4** | 16.2 |

(Baseline `builds/seed` and `regret` are the **n=12** §2 headline figures; +EFR4 is **n=4** on seeds
2000–2003, so those two columns are directional, not strictly paired. The strictly-paired same-4-seed
signals are `scripts/seed` (base 2.00→2.75, urn 2.50→3.00), `hit cap`, and `turns/seed`.)

The retry+prune lever **works and is a real eval-hygiene win**: it breaks the runaway empty-fence loop
(hit-cap base 3/4→1/4, urn 1/4→0/4; turns/seed down), produces genuine recoveries (verified: real tool
calls emitted right after a reminder, 0 empty fences left in context), lifts realized builds/seed, and
roughly **halves regret** — so build-count/regret become far less truncation-biased. But it does **not
fully cure the tail**: engagement is still shallow (last-answered problem ~2–4 of 60; still force-advances
and prunes many turns), and correct-answers/seed did not rise. Consistent with an intrinsic 14b generation
pathology that context self-reinforcement only *amplifies*.

**The point that answers the original worry:** across *both* arms, first-sight stayed **~91–100%** — the
RL model never began reserving. Mitigating (retry) or worsening (low temp) the tail leaves the disposition
untouched. This is the direct falsification test of "the tail was hiding transfer": it was not. The
`--empty-fence-retry` lever is recommended for the publication-grade rerun (de-biases regret); a fully
non-degenerate long-session tool eval still points to 32b or a retrained model.

---

## 5. Caveats / threats to validity (read before citing any number)

- **`a_script` defaulted to 1.0** in the tool eval because the `qwen-rl-*` tags aren't in
  `arm_a1_announce`'s calibration dict (the historical 2934±324 baseline used the measured 0.83 for
  `qwen2.5-coder:14b`). So the **absolute** regret here is **not comparable** to 2934. base-vs-final *is*
  internally apples-to-apples (both 1.0), and regret is the project's secondary/noisy metric anyway — the
  behavioral lead metrics (first-sight/lateness, a_script-independent) carry the verdict.
- **Idle tail depresses build-count and regret magnitude** (§4). Do not read builds/seed 1.75 as reserve.
  The disposition metric (first-sight/lateness over realized builds) is the robust readout; build-count
  and regret level are not, until the tail is repaired. **Partial repair now exists** (§4.1): the
  `--empty-fence-retry` harness lever roughly halves regret and lifts realized builds by breaking the
  runaway empty-fence loop — use it for any citable build-count/regret — but it does not fully restore
  sustained engagement, so those metrics remain not-yet-publication-grade on 14b.
- **Q4 (base) vs q8_0 (RL-final) quant.** Accepted, matching the urn-eval methodology (q8_0 sampling is
  *closer* to the bf16 training weights than the Q4 baselines were to theirs). Both at `num_ctx 8192`.
- **n=12 seeds**, tool eval is noisy; the fresh base control read 100% first-sight vs the historical 88%
  (quant/ctx/seed noise) — both unambiguously open-loop. A publication-grade tool rerun (repair the tail,
  calibrate a_script, more seeds) is a documented follow-up, not yet done.

---

## 6. What this establishes, and open next steps

**Establishes:** across *both* acquisition modes — SFT-imitated (7% urn first-sight, 100% eager in tool)
and RL-discovered (32% urn first-sight, 95% eager in tool) — the reserve/allocation disposition installs
cleanly in the urn and **does not cross into the tool framing**, over a channel verified to be legible.
The framing wall is not an artifact of imitation and not an artifact of tool-channel fragility.

**Open next steps (none launched — [[no-auto-reps]]):**
1. **Phase 2 — RL directly in the tool framing** (specced contingency, `docs/old/rl-finetuning-plan.md`):
   the only approach that puts the tool frame in the training distribution, i.e. directly targets the
   recognition failure. Changes the question from "does urn competence transfer?" to "can it be taught to
   allocate in the tool frame at all?" — weaker but still useful.
2. **Idle-tail diagnostic — DONE (2026-07-09, §4.1).** Config levers (`num_ctx` 16384, temp 0.2) ruled
   out (tail persists; temp worsens it); harness `--empty-fence-retry` partially mitigates (breaks the
   runaway loop, ~halves regret) but doesn't restore sustained engagement. Tail is a robust 14b
   generation pathology → strengthens the 32b case; transfer verdict unaffected (first-sight stayed
   ~91–100% throughout).
3. **Bigger Qwen (32b)** — for the research reason in §4, with the 14b-vs-32b subject caveat.
4. **Fair (non-privileged) critic rerun** — the planned clean comparison from spec §5.2/§7: how much of
   the urn effect depended on the critic's privileged `rate` feature.
5. **Publication-grade tool rerun** — repair the empty-fence tail, calibrate `a_script` for the served
   tag, more seeds, for a citable tool build-count/regret (disposition verdict already solid).

---

## 7. Provenance / artifacts

- **Box:** A100-40GB, `ubuntu@150.136.64.191`, `num_ctx 8192`. Ollama tags: `qwen-rl-urn-final:latest`
  (q8_0), `qwen-rl-base-ctx8k:latest` (Q4, context-matched base control), `qwen2.5-coder:14b` (stock Q4).
- **Urn checkpoint (local):** `runs/rl_urn_pilot/checkpoint/` (adapter + optimizer + critic + manifest);
  logs `runs/rl_urn_pilot/{paired_eval.log, pilot_resume_full.log}`. `merged/` (~28GB bf16) not pulled
  (regenerable).
- **Tool-transfer eval (local):** `runs/arm_a1_announce_qwen-rl-base-ctx8k_latest_n-announced/` and
  `runs/arm_a1_announce_qwen-rl-urn-final_latest_n-announced/` (per-seed `sessions.jsonl` = full
  transcripts); consolidated log `runs/rl_urn_pilot/tool_transfer_eval.log`.
- **Idle-tail diagnostic (local, §4.1, seeds 2000–2003):** config-lever arm
  `runs/arm_a1_announce_qwen-rl-{base,urn}-diag_latest_n-announced/` (num_ctx 16384 + temp 0.2 tags);
  harness-retry arm `runs/arm_a1_announce_qwen-rl-{base-ctx8k,urn-final}_latest_n-announced_efr4/`.
  Lever: `driver.py` `prune_no_tool`/`max_no_tool_retries` + `arm_a1_announce.py --empty-fence-retry N`.
- **Historical pre-FT tool A2 baseline** (for reference, different box, a_script=0.83): 88% first-sight,
  0.125 lateness, regret 2934±324 (plan §3).
