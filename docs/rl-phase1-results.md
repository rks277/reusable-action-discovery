# RL Phase 1 (urn framing) — RESULTS

Standalone, candid results writeup for the Phase 1 RL work. **Method/design/pilot-history live in
`docs/rl-ppo-credit-assignment-spec.md`** (per-decision PPO + privileged critic, the five flat GRPO
pilots, the build order, the pre-box-run review); this doc is only *what happened and what it means*,
including the caveats we found by reading the raw transcripts. Broader project framing:
`docs/online-tool-investment-plan.md`. §1–4 are from runs on 2026-07-08 (A100-40GB,
`ubuntu@150.136.64.191`); §5 (framing-generalization follow-ups) is from a fresh box on 2026-07-09
(A100-40GB, `ubuntu@129.213.82.160`); artifacts listed in §8.

---

## TL;DR

1. **RL works in the urn (in-framing).** From the *untouched base* `qwen2.5-coder:14b` (QLoRA,
   per-decision PPO + privileged critic, no demonstrations), a 20-step run **discovered the reserve
   policy from the balls reward alone** and reached π\* parity: paired held-out eval 75%→**32%**
   first-sight, 87%→**101%** of π\* balls.
2. **It does NOT transfer to the tool framing — the framing wall holds for RL too.** Zero-shot tool eval
   of the same checkpoint: it stays essentially eager in the tool task (95% first-sight, lateness 0.131)
   despite reserving in the urn (32% / 0.903), over a **verifiably legible** tool channel (0 malformed /
   0 unknown calls). Its tool-frame behavior is **indistinguishable from base** (95% vs 95% first-sight).
   *Confirmed publication-grade (2026-07-10): matched q8_0, n=24 paired seeds, EFR4, calibrated `a_script`
   — see §2.*
3. **Interpretation (the payoff):** a policy discovered from reward in the abstract task remains
   context-bound: it does not activate in reusable script creation, even though the tool channel is
   legible and the model retains ordinary script-writing and tool-use competence.
4. **Follow-up (2026-07-09, §5): the disposition survives a lexical re-skin but only partly survives a
   decision-modality change.** Same RL-final checkpoint, three novel surface vocabularies (coins/
   treasure-chest, arrows/quiver, potions/cauldron) that swap every content word while holding the
   decision structure fixed: RL-final stays clearly more reserved than base (pooled first-sight 19% vs
   89%) — the reserve policy is bound to the abstract KEEP/PASS structure, not the literal ball/bag/color
   words. Switching the SAME task from free-text decisions to actual tool calls (`keep`/`pass`) shrinks
   the gap by ~3x (RL-final 62% vs base 99%) and makes it wildly vocab-dependent (RL-final ranges
   44%–97% across the three vocabs, vs a tight 10–33% band in free text) — including one vocab
   (potions/cauldron) where RL-final collapses into an unconditional "keep the first 3 things shown, then
   stop looking" policy in 16/24 seeds, nearly indistinguishable from base.

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

RL reaches the urn reference through a policy that still keeps some colors on first sight: first-sight
falls to 32% while balls collected reaches 101% of π\*.

---

## 2. Tool-transfer result (the headline experiment): no transfer

**Setup (publication-grade rerun, 2026-07-10 — `docs/qwen-tool-transfer-rerun-spec.md`).** Paired tool
eval, `arm_a1_announce.py`, **A2 / `--announce-n`**, **24 paired seeds 2000–2023** (up from the original
12; disjoint from the 9000+ training seeds). Both tags served at **`num_ctx 8192`** and now at **matched
q8_0** through the identical merge→GGUF path, so the paired delta is weights alone, not quantization:
base control `qwen-rl-base-q8` (q8_0, no adapter, same `Qwen/Qwen2.5-Coder-14B-Instruct` base) vs
`qwen-rl-urn-final` (q8_0). Run with **`--empty-fence-retry 4`** (de-biases build-count/regret by breaking
the idle-tail loop) and **per-tag calibrated `a_script`** (base 0.89, RL-final 0.94; measured q8_0,
`a0_oracle_gap` MAG=100 k=8, a_hand=0). This supersedes the original confounded cell (base Q4 vs RL q8_0,
n=12, `a_script=1.0`, no EFR) whose numbers are archived in the run logs.

| metric (tool A2, n=24, matched q8_0, EFR4) | base-q8 (a_script 0.89) | RL-final (a_script 0.94) | *RL-final in the urn, for contrast* |
|---|---|---|---|
| first-sight % | 95% (59/62) | **95% (58/61)** | *32%* |
| mean lateness | 0.048 | **0.131** (max 5; one derailed-tail late build) | *0.903* |
| n_malformed / n_unknown / n_refused | 0 / 0 / 0 | **0 / 0 / 0** | — |
| builds/seed | 2.58 | 2.54 | — |
| hit token cap | 0/24 | **0/24** | — |
| regret vs π\* (calibrated a_script) | 1909±417 | 1821±396 | *−0.4 balls-regret* |

The verdict is unchanged and now confounder-clean: under matched q8_0, n=24, EFR4, and calibrated
`a_script`, RL-final stays **95% first-sight in the tool frame** while reserving at **32% in the urn** — a
~63-point within-model, same-quant, same-information dissociation — and is **indistinguishable from base**
(95% vs 95%) in the tool frame. EFR4 also cured the truncation pathology (0/24 token-cap hits vs the
original run's cap-bound tails), so builds/seed (2.5–2.6, up from ~1.8) and regret (~1.8–1.9k, roughly
half the old ~3.7k) are now citable, not truncation-biased.

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
transfer." 32b already has some competence, so training 32b tests a weaker claim (filed as the optional
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

## 5. Framing-generalization follow-ups (2026-07-09): vocab reskins and tool-call modality

Two follow-up probes on the SAME RL-final checkpoint (`qwen-rl-urn-final:latest`, q8_0) vs the SAME
base control (`qwen2.5-coder:14b`, Q4), on a fresh box (`ubuntu@129.213.82.160`), extending
`urn_session.py` (now backed by shared `urn_common.py`) with a `--vocab` flag and adding a sibling
harness `urn_tool_session.py`. Same N=8/T=60/B=3 streams, same π\* reference, same 24 seeds
(2000–2023), same `--temp 0`, in all cases — only the vocabulary and/or decision modality changes.

### 5.1 Vocab reskin (still free-text `DECISION: KEEP/PASS`): disposition transfers cleanly

Three HELD-OUT surface vocabularies that RL's ball-only training never saw:
**treasure_chest** (coins, attr="crest"), **quiver** (arrows,
attr="feather"), **cauldron** (potions, attr="label"). Same KEEP/PASS structure, only the words differ.

| | base, pooled (range) | RL-final, pooled (range) |
|---|---|---|
| first-sight % | 89% (71–100%) | **19%** (10–33%) |
| mean lateness | 0.134 | **0.968** |
| balls-regret vs π\* | 5.9 (84–86% of π\*'s) | **−2.4 (105–108% of π\*'s)** |

RL-final is **even more reserved here than in the original ball-framing eval** (19% vs the §1 headline
32%) while base stays comparably eager (89% vs 75%) — and RL-final again collects *more* balls than
π\* on average (105–108%), consistent with §1's balls-regret −0.4. The two distributions don't overlap
on any single vocab (base's low end 71% still exceeds RL-final's high end 33%). **Verdict: the reserve
policy is bound to the abstract KEEP/PASS decision, not the literal ball/bag/color wording** — a clean
positive result, in contrast to the tool-framing transfer failure in §2.

### 5.2 Tool-call modality: partial transfer, with a real methodological trap along the way

Same three vocabs, decision now elicited via an actual tool call (`keep`/`pass`, zero-argument) instead
of a free-text line — isolating decision MODALITY as an axis orthogonal to vocab. This is NOT the §2
tool-framing benchmark (different decision structure entirely, build-once-reuse with a hand-solve
alternative) — only its `chat_tools` plumbing is reused.

**The trap:** under the initial `tool_choice="auto"` + the vocab-reskinned system prompt reused
verbatim from §5.1, BOTH models made **zero real tool calls** in a smoke test. RL-final emitted the
literal trained `"DECISION: PASS"` text as plain message content; base emitted that text plus an empty
` ```json``` ` fence — the SAME empty-fence collapse §4 documents for the tool-framing benchmark's idle
tail, but here present from turn 1, not onsetting after a few problems. `tool_choice="required"` did
**not** fix it (verified: Ollama's OpenAI-compat shim accepts the parameter but doesn't enforce it as a
hard grammar constraint — confirmed by feeding the real system prompt through both `"auto"` and
`"required"` and getting identical zero-tool-call behavior). **The actual cause and fix:** the system
prompt's own closing paragraph explicitly instructed "end your reply with EXACTLY one line: DECISION:
KEEP/PASS" — a direct competing instruction. Rewriting that one paragraph to instruct tool use instead
(byte-identical everywhere else, verified) immediately produced clean tool-calling behavior (0 unparsed
in a follow-up smoke test) for both models. Left as the default (`--tool-choice required`, now largely
moot given the prompt fix, but kept as a second line of defense).

**Result, once the channel actually works:**

| | base, pooled (range) | RL-final, pooled (range) |
|---|---|---|
| first-sight % | 99% (99–100%) | **62%** (44–97%) |
| mean lateness | 0.009 | **0.412** |
| unparsed (typ. per 72 decisions) | 0–2 | **4–19** |

RL-final is still measurably more reserved than base (62% vs 99%) — the disposition has **not**
vanished under tool-calling — but the gap shrank roughly 3x versus free text (62% vs 19%) and the
between-vocab spread exploded (44–97%, stdev 30%, vs free text's tight 10–33%). Elevated `unparsed`
counts for RL-final (vs base's near-zero) suggest a milder, residual version of the same
channel-engagement friction that fully blocked the pre-fix run.

**The cauldron collapse, read from the raw transcripts:** in 16/24 cauldron seeds, RL-final calls
`keep` on the first three distinct potions shown with zero passes, exhausting its entire 3-keep budget
by decision turn 3 and never observing draws 4–60 of the 60-draw stream. Confirmed NOT a stream
artifact — cauldron and treasure_chest see byte-identical class-id sequences per seed (checked
directly). In treasure_chest/quiver, the same model runs 5–11 decision turns per seed, passing
repeatedly before committing — the genuine reserve pattern from §5.1. **No transcript explains why**:
every tool call in every vocab carries empty message content (the model never uses the "you may think
briefly first" allowance), so there's no chain-of-thought to inspect. Speculative, unverified reading:
"potion" may carry a stronger immediate-use association from pretraining than "coin"/"arrow" (naturally
hoardable resources) — but this is a guess, not something these transcripts confirm. **Open diagnostic**
(not run): sample cauldron at `temp > 0`, several samples/seed, to check whether the collapse is a hard
greedy-decoding attractor or persists under sampling variance.

---

## 6. Caveats / threats to validity (read before citing any number)

**Publication-grade rerun (2026-07-10) resolves the tool-cell caveats below.** The §2 headline is now the
matched-q8_0, n=24, EFR4, per-tag-calibrated-`a_script` run (`docs/qwen-tool-transfer-rerun-spec.md`):
the quant-mismatch, `a_script=1.0`, and n=12 caveats are **retired**, and the idle-tail caveat is
**downgraded** — EFR4 gave 0/24 token-cap hits and 0 malformed/unknown/refused across all 48 sessions, so
builds/seed and regret are no longer truncation-biased (though engagement on 14B is still shallow; a fully
non-degenerate long-session tool eval still points to 32B). The bullets below describe the **original
n=12** run and are kept for provenance.

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

## 7. What this establishes, and open next steps

**Establishes:** a reserve/allocation disposition discovered from reward installs cleanly in the urn and
**does not cross into the reusable-script framing**, over a channel verified to be legible. The learned
policy generalizes across free-text vocabularies but remains strongly context- and modality-dependent.

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
6. **Cauldron collapse diagnostic (from §5.2, not yet run)** — sample the tool-modality cauldron arm at
   `temp > 0`, several samples/seed: is the "keep first 3, stop looking" policy a hard greedy-decoding
   attractor, or does it persist under sampling variance? No chain-of-thought exists in the temp=0
   transcripts to explain the vocab-specific collapse, so this is the cheapest next lever.
7. **Tool-modality vocab-dependence, unexplained** — §5.2's RL-final range (44–97% first-sight across
   3 vocabs, vs free text's 10–33%) is a real, measured effect with no mechanistic explanation yet (empty
   tool-call content in every transcript, both vocabs). Worth a lens other than transcript-reading if
   pursued further (e.g. logprob/entropy comparison at the decision token across vocabs).

---

## 8. Provenance / artifacts

- **Box (§1–4):** A100-40GB, `ubuntu@150.136.64.191`, `num_ctx 8192`. Ollama tags:
  `qwen-rl-urn-final:latest` (q8_0), `qwen-rl-base-ctx8k:latest` (Q4, context-matched base control),
  `qwen2.5-coder:14b` (stock Q4).
- **Urn checkpoint (local):** `runs/rl_urn_pilot/checkpoint/` (adapter + optimizer + critic + manifest);
  logs `runs/rl_urn_pilot/{paired_eval.log, pilot_resume_full.log}`. `merged/` (~28GB bf16) not pulled
  (regenerable).
- **Tool-transfer eval — original n=12 (local):** `runs/arm_a1_announce_qwen-rl-base-ctx8k_latest_n-announced/`
  and `runs/arm_a1_announce_qwen-rl-urn-final_latest_n-announced/` (per-seed `sessions.jsonl` = full
  transcripts); consolidated log `runs/rl_urn_pilot/tool_transfer_eval.log`.
- **Tool-transfer eval — publication-grade rerun n=24 (local, 2026-07-10):**
  `runs/arm_a1_announce_qwen-rl-base-q8_latest_n-announced_efr4/` and
  `runs/arm_a1_announce_qwen-rl-urn-final_latest_n-announced_efr4/` (24 seeds each, matched q8_0, EFR4).
  Box: A100-40GB `ubuntu@158.101.121.179`. Both tags built via `scripts/box_make_q8_tags{,_resume}.sh`
  (adapter merge → q8_0 GGUF at num_ctx 8192; the resume script drops the redundant `extra_special_tokens`
  list that transformers 4.57.6 rejects). Per-tag `a_script` (base 0.89 / RL-final 0.94) from
  `runs/a0_oracle_gap_20260710_15*/`, registered in `arm_a1_announce._A_SCRIPT`.
- **Idle-tail diagnostic (local, §4.1, seeds 2000–2003):** config-lever arm
  `runs/arm_a1_announce_qwen-rl-{base,urn}-diag_latest_n-announced/` (num_ctx 16384 + temp 0.2 tags);
  harness-retry arm `runs/arm_a1_announce_qwen-rl-{base-ctx8k,urn-final}_latest_n-announced_efr4/`.
  Lever: `driver.py` `prune_no_tool`/`max_no_tool_retries` + `arm_a1_announce.py --empty-fence-retry N`.
- **Historical pre-FT tool A2 baseline** (for reference, different box, a_script=0.83): 88% first-sight,
  0.125 lateness, regret 2934±324 (plan §3).
- **Box (§5, 2026-07-09):** fresh A100-40GB, `ubuntu@129.213.82.160` (no persistent disk; adapter
  re-pulled from local `runs/rl_urn_pilot/checkpoint/adapter/` (274MB, LoRA weights only) and re-merged/
  re-quantized to `qwen-rl-urn-final:latest` q8_0, same recipe as §1's box; base `qwen2.5-coder:14b`
  pulled fresh, both served at Ollama defaults, no `num_ctx` override needed (urn turns are short)).
- **Harness changes (§5):** `scripts/creator/tool_disposition_benchmark/urn_common.py` (new — shared
  `VOCAB`/`_art`/`render_system`/scoring/`report_summary`, extracted from `urn_session.py` so both
  harnesses share logic byte-for-byte; `render_system` gained an optional `response_instruction` param,
  default reproduces the original DECISION-line text byte-for-byte); `urn_session.py`'s `--vocab` flag
  (`ball`/`treasure_chest`/`quiver`/`cauldron`, `nargs="+"` for pooled multi-vocab runs);
  `urn_tool_session.py` (new — tool-call modality variant, `runs/urn_tool_*` prefix, `--tool-choice`).
  `rl_rollout.py`/RL training code untouched (re-verified via its own `_selftest()` after the refactor).
- **§5 run artifacts (local):** `runs/urn_{qwen2.5-coder_14b,qwen-rl-urn-final_latest}_vocab-{treasure_chest,quiver,cauldron}/`
  (§5.1, free text) and `runs/urn_tool_{qwen2.5-coder_14b,qwen-rl-urn-final_latest}_vocab-{treasure_chest,quiver,cauldron}/`
  (§5.2, tool call) — per-seed `stream.json`/`session.json` (full transcripts, `"how"`-tagged decisions).
  Consolidated logs: `runs/vocab3_eval.log` (§5.1), `runs/vocab3_tool_eval.log` (§5.2).
