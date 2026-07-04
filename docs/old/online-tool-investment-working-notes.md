# Online-tool-investment — working notes

Companion to `online-tool-investment-plan.md` (the polished narrative + headline claim + related-work
verdict) and `same-info-optimal-dp.md` (the DP construction). This doc is the technical status board:
what's built, what's calibrated, what's running, and the immediate next steps. Renamed 2026-07-03 from
`online-tool-investment-stochastic-design.md` (was written before the exact-DP result matured; folded
forward here).

## Status snapshot (2026-07-03)

- Reference policy = exact belief-state DP (`exact_dp.py`), cap=3 certified lossless. **Done.**
- Haiku: eager-first-B fidelity (100%, lateness 0) on g=1; g=0 control done (regret is
  distribution-dependent, NOT the headline — the open-loop *policy* is); A1 disclosure-immune. **Done.**
- Mechanism reframe: it's a **recognition/framing failure** (H2), not value-of-information. **Done**
  (transcript analysis), then **proven by a paired control** (the urn/balls abstraction-gap task,
  below) rather than just argued.
- Abstraction-gap urn: Haiku paired result done (partial gap); Opus urn done (aces it). **Opus
  tool-task cell ASSUMED eager + shelved** (continued_frac exploded at MAG=1000; buys rigor not a
  finding). See §Abstraction-gap control + §Opus tool-task confound below.
- 5-agent adversarial novelty check done — verdict + phrases-to-avoid in `plan.md` §5b. **Done.**
- **Qwen-Coder urn slope (0.5b→32b, 24 seeds) DONE** (H100 box): competence is a noisy suboptimal
  *plateau* (Qwen-32b ≈ Haiku), NOT a smooth slope — the near-optimal behavior is a **frontier jump at
  Opus**. Lateness ⊥ regret. See §Qwen-Coder urn slope below.
- **Qwen-Coder a_script calibration DONE** — coding ability scales smoothly with size (0.21→0.96,
  0.5b→32b), unlike the flat urn plateau. **FT target = 14b** (a_script 0.83; 32b higher at 0.96 but
  too slow to iterate on). Phase 1 of the fine-tune-transfer experiment (Qwen chosen for open weights)
  is now complete; see §Qwen-Coder a_script calibration below.
- **Qwen-14b tool baseline (Phase 2) DONE** — eager, same as every other model tested (96% first-sight,
  lateness 0.043). Found + fixed two real Ollama tool-calling JSON bugs along the way (not just a
  speed optimization); early-stop-on-budget-exhaustion + analytic tail-scoring cut the 12-seed run to
  ~5.5 min total. See §Qwen-Coder tool baseline (Phase 2) below.

## Reference policy: π\* — the exact same-information optimum (DP)

**π\* is the exact Bayes-optimal policy against the model's own information** — knows only {N, T, B} +
"a distribution over N types", via an **exchangeable Dirichlet(α) prior** (no idea about the
frequent/rare split; infers everything from the stream, like the model). Computed by an exact
finite-horizon belief-state DP — **not** a relaxation. Implemented in `exact_dp.py` (`ExactDP`); wired
into the scorer as `skirental_scorer.exact_pistar_report`. Full construction in `same-info-optimal-dp.md`.

**Structure:** π\* reserves the first sighting of a type and builds on the **2nd sighting** (demonstrated
recurrence) — never builds on sight. Because π\* is the exact same-info optimum, (π\* − model) is the
honest same-information regret, not a bound.

> **Tractability (the count-cap).** The uncapped joint DP is intractable at N=12,T=60 (~10¹³ states).
> `ExactDP` caps *unbuilt* counts at `cap` (force-build a type on its (cap+1)-th sighting) — a valid
> same-info policy; a K-sweep certified it's **lossless** here (`V(3)=V(4)=V(5)=V(6)` exactly). So
> `cap=3` is the exact optimum at ~540k states / ~3s, pure Python. Re-run the sweep to re-certify if the
> cost model or α changes.

> **Retired — the old Whittle π\*.** `pi_star.py`'s Whittle-index construction is superseded: at N≈12
> the constant-price relaxation over-builds (~44% first-sight) and can't match the context-dependent
> exact optimum. Kept in the codebase only for `value_of_builds`/`clairvoyant_builds`/`eager_builds`
> helpers, which the exact-DP pipeline still reuses.

## Two-layer metric (separates policy quality from draw luck)

Haiku runs a **fixed eager policy regardless of the stream**, so realized per-seed value swings with
the draw (on ~30% of g=1 streams the first-B arrivals happen to be hot, so eager "looks optimal" while
executing the identical mindless rule). Two layers:

1. **Behavioral fidelity (luck-free, qualitative).** Read off the action log: lateness (build position
   relative to first sighting) + does the built-set == first-B-distinct-arrivals. No per-model cost
   calibration needed — this is the layer every new arm/model only needs a modest sample size to pin.
2. **Expected policy regret (luck-free headline number).** Once fidelity pins the policy (e.g. Haiku ≡
   eager-first-B), price that fixed policy against π\* **analytically over many synthetic streams**,
   removing draw-luck from the mean (tiny SE vs a noisy realized mean).

**Haiku g=1 (30 sessions, 88 builds):** 100% first-sight, lateness 0.000, 30/30 built-set==first-B.
Analytic E[regret] over 1500 g=1 streams = 1417±40 (eager traps 1.21 vs π\* 0.50). Realized 30-seed
regret 1107±275 (consistent, ~1.1 SE).

## g=0 control (done) — regret is distribution-dependent, NOT the headline

Natural-rate draws (no forced early trap): Haiku's traps/seed collapse **1.10 → 0.10** (= π\*'s own
0.10) — i.e. under natural arrival Haiku rarely traps anyway (traps are naturally rare), so most of the
g=1 regret is a **trap-early-conditioning artifact**, not evidence of a different underlying policy.
On the uniform-hard N=8 pool: analytic E[regret] g=1 983 vs g=0 71. **Conclusion: regret is
distribution-dependent; the distribution-*independent* claim is the open-loop *policy itself*
(fidelity: builds first-B on sight, unconditionally) — that's what should headline, not the regret
magnitude.**

## A1 disclosure (done) — open-loop whether told or not

System prompt discloses (non-prescriptively) that some types recur and others don't. Result: **no
change** — 35/35 builds at first sight, lateness 0.000 (uniform-hard, n=12, seeds 2000–2011). Haiku is
open-loop whether it must infer recurrence (A0) or is handed it (A1). This is also the disclosure
condition used for the abstraction-gap urn (see below) — matched deliberately, not by accident (see
next section).

## Mechanism: H2 recognition/framing failure (not value-of-information)

Transcript analysis (A1 tool sessions): Haiku's build rationale is *always* per-problem — "I need to
solve X, let me write a script." It never reasons about types recurring, reserving budget, or waiting:
"recur" (148×) is entirely "recurrence **relation**" (algorithm-speak, not the phenomenon); "budget"
(47×) is entirely *retrospective* ("since I've used up my budget…"); "reserve/conserve" = 0×. So Haiku
treats `write_script` as "how I solve this hard computation, one problem at a time" and only notices
the budget once it's gone. It reactively reuses a saved script when a type repeats but never
proactively allocates. **This reframes the whole project: the contribution is that LLMs don't recognize
tool-creation as resource allocation under recurrence — a recognition failure, not the crowded VoI /
premature-commitment slice** (see `plan.md` §5b for the novelty positioning this forces).

## Abstraction-gap control — the urn/balls isomorph (proves H2 by paired control, not argument)

`urn_session.py`, `runs/urn_{haiku,opus}/`. Strips the tool cover story: colored balls drawn one at a
time from the *same* stochastic streams; KEEP collects current + all future same-color (= build +
reuse); PASS/never-kept scores 0 (= hand-solve at a_hand 0). Keeps map onto `model_builds`; π\* is the
same `exact_dp`. **Info condition must match the tool arm being compared against — this took two
corrections to get right:**

1. **Observation-parity bug (caught and fixed).** v1 silently skipped kept-color recurrences instead of
   showing them — the urn player observed a gappy, shorter-than-T stream, an information asymmetry vs
   the tool player. Fixed: auto-collect in place, tell the model ("another green ball → auto-collected,
   +1"), system prompt states the bag is unchanged and all T draws happen. Model now observes exactly T
   draws, matching the tool player.
2. **Info-condition mismatch (caught by the user).** The urn prompt ("bag of colored balls, unknown
   counts") *discloses that colors recur* — it's **A1-equivalent**, not A0. So the paired tool baseline
   must be A1 (disclosed), not A0 (hidden) — matching to A1 is also the *stronger* comparison (both
   sides told recurrence exists, so the gap can't be "didn't know recurrence was possible").

**Paired result (Haiku, seeds 2000–2011, uniform-hard N=8, g=1, A1 both sides):**

| | A1 tool (coding framing) | urn (allocation framing) |
|---|---|---|
| first-sight | 100% (35/35) | 53% (19/36) |
| lateness | 0.000 | **0.86** (max 6) |
| regret vs π\* | 1308 ± 520 | **640 ± 285** |
| traps/seed | — | 0.92 (π\* 0.75) |

Removing the coding framing roughly **halves misallocation** and makes Haiku reserve budget +
genuinely track frequencies (transcripts unambiguous, uses the auto-collect feedback to confirm
choices). **But not a clean dissociation** — urn regret ≠ 0 and Haiku builds *more* traps than π\*
(0.92 > 0.75), so the tool failure is partly framing, partly a real allocation weakness even in the
abstract. **Lead with lateness, not regret** — regret is noisy/seed-dependent (urn was −30±213 on a
different seed set, 3000–3011, vs +640±285 here; both n=12).

**Opus urn (same seeds, same streams; $0.54) — capability-graded competence:**

| | Haiku urn | Opus urn |
|---|---|---|
| first-sight | 53% | **8%** (3/36) |
| lateness | 0.86 | 1.14 |
| regret vs π\* | 640 | **−655 ± 389** (beats π\*) |
| traps/seed | 0.92 | **0.33** (π\* 0.75) |

Opus essentially aces the urn (waits for a repeat, near-zero traps). **Ceiling check strongly passes
for Opus** (vs Haiku's partial pass) — so for Opus the abstraction-gap should be a *clean* dissociation
once the tool-side is measured. This reframes the (crowded, per the novelty check) capability axis: if
Opus is also eager in the tool task, the recognition gap *widens* with capability — the stronger model
has *more* allocation competence for the tool framing to suppress.

### Opus tool-task confound (calibration work, 2026-07-03) — IN PROGRESS

Before running Opus on the tool task, calibrated whether the uniform-hard pool is actually hand-hard
for Opus (it wasn't at MAG=100 — pooled a_hand ≈ 0.25–0.38, `josephus` fully hand-solvable at 1.00).
Fix:
- **`crt_solve`, `modpow`: fixed by raising MAG to 1000.** Their old samplers ignored the magnitude
  parameter (or barely used it — modpow's exponent was fixed 12-40 regardless of `m`); patched both
  samplers in `family_kit.py` to scale their difficulty-driving parameter (CRT's congruence count `n`,
  modpow's exponent range) with `m`, calibrated to reduce to the old fixed behavior at m=100 (so
  Haiku's existing MAG=100 data/calibration is untouched). At m=1000 both hit a_hand=0.00 for Opus.
- **`josephus`: unfixable by magnitude** — plateaus at a_hand ≈ 0.5–0.6 for Opus regardless of N (no
  closed form exists for general K, but Opus tracks the O(N) recurrence reliably at any length; scaled
  its sampler's N range too, no effect). Instead of chasing difficulty, **structurally isolated** it:
  added `pinned_last_trap` to `StochasticStreamSpec`/`build_stochastic_stream` — draws the other 7
  families i.i.d. over T−1 slots (recursing into the normal path, which auto-renormalizes the trap
  rate), then appends josephus as a single forced trap at the *final* slot (T−1). By the last slot
  there are 0 remaining draws, so building never pays off regardless of hand-difficulty, and it cannot
  have influenced any earlier build/reserve decision. No changes needed to `exact_dp`/scorer — π\*
  naturally never recommends building with 0 remaining draws. Local invariant checks pass (60 slots,
  josephus only at slot 59, Σcounts=T, no class_id collision) — no model calls needed to verify this.
- Haiku is **untouched** (MAG=100, no pin) — confirmed clean already (`a0_newfams_haiku`: crt_solve
  0.00, josephus 0.00 at MAG=100).
- **Opus A1 tool run: attempted, then SHELVED.** On the hardened MAG=1000 pool, `continued_frac`
  (modulus-free numerator) exploded to ~55-digit answers → Opus perseverated dozens of turns per
  problem → sessions never finished (killed at 0 completions). Remaining fix = bound `continued_frac`'s
  term-scaling independent of the global MAG dial, and drop `matrix_power_mod` (a_script≈0.33 for Opus
  — can't reliably *write* the tool → use N=7). **Decision: Opus tool cell is ASSUMED eager** (prior
  constructed-design bait 20/20, lateness 0) and shelved to pre-publication — re-running buys rigor,
  not a new finding. See plan.md §4.

## Qwen-Coder urn slope (2026-07-03) — competence is a plateau + frontier jump, not a smooth slope

Ran the urn (identical game/streams to Claude, **24 seeds** 2000–2023, CONC=8) across the full
Qwen2.5-Coder ladder on the H100 box (`ubuntu@68.209.75.3`, Ollama, `OLLAMA_NUM_PARALLEL=8`; whole
6-size sweep ~2.5 min). Harness: `urn_session.py --model qwen2.5-coder:<size>` (model-string
pass-through routes to Ollama, pricing zeroed, robust `parse_decision` handles small-model rambling).
π\* is the same model-independent exact-DP optimum.

| model | keep lateness | regret vs π\* | traps/seed | unparsed |
|---|---|---|---|---|
| Qwen 0.5b | 0.31 | 745 ± 476 | 1.00 | 7 |
| Qwen 1.5b | 0.30 | 1731 ± 505 | 1.00 | 27 |
| Qwen 3b | 0.20 | 660 ± 379 | 0.96 | 0 |
| Qwen 7b | 0.02 | 1545 ± 545 | 0.96 | 2 |
| Qwen 14b | 0.17 | 491 ± 336 | 0.96 | 0 |
| Qwen 32b | 0.13 | 901 ± 363 | 1.08 | 0 |
| **Haiku** (12 seeds) | 0.86 | 640 ± 285 | 0.92 | — |
| **Opus** (12 seeds) | 1.14 | **−655 ± 389** | 0.33 | — |

(π\* traps: 0.67 on the 24-seed Qwen set, 0.75 on the 12-seed Claude set — different seed sets; regret is vs π\* on each model's own seeds. For a strictly paired comparison, re-score Qwen on seeds 2000–2011 only — instant, session data on the box.)

Findings:
- **No smooth scaling across Qwen.** Regret is a noisy, suboptimal plateau (~500–1700, CIs overlap heavily); 0.5b→32b shows no clean trend. Qwen-32b (901) ≈ Haiku (640).
- **Competence is a frontier JUMP, not a slope.** Only Opus breaks away (beats π\*, −655); matches [[creator-frontier-inversion-cross-family]] (the interesting behavior is a frontier effect the Qwen ladder tops out below). So the two-point "Haiku < Opus" that looked like a slope is really **flat plateau → frontier jump**.
- **Lateness trends with capability** (Qwen ~0.02–0.31 eager < Haiku 0.86 < Opus 1.14) **but regret decouples from it.**
- **Format-adherence floor ≈ 3b** (1.5b unparsed 27; 3b/14b/32b clean 0).

**Lateness ⊥ regret (methodological point).** Qwen's far-lower lateness (eager) does *not* give it lower regret than Haiku, because regret is dominated by *type-selection* (trap vs hot — a ~2,600-utility swing per misallocated keep: a hot type kept ≈ +1,170 vs the same type un-kept and hand-solved ≈ −1,480), and Qwen and Haiku make the *same* ~0.3-excess-trap error over π\*. Keep-lateness only moves the small "missed early occurrences" term (≈ u_reuse − u_hand ≈ 179/occurrence → ~300 total for Haiku's extra waiting), inside the seed noise. So a model can wait-but-pick-wrong (Haiku) or not-wait-but-pick-similarly-wrong (Qwen) → similar regret. **Lateness = behavioral (did it *reserve*); regret = outcome (did it *allocate right*); report both, don't infer one from the other.** Opus is the only model that both waits *and* converts it into trap-avoidance (0.33) — its negative regret is the trap-avoidance, not the waiting.

Implication for the transfer experiment: the FT-target choice is about *tool-writing ability* + clean protocol, not "best urn score" (the Qwen plateau is flat). Box state: repo rsynced + `.venv`; `.env` OLLAMA_BASE_URL→localhost; runs in `runs/urn_qwen2.5-coder_<size>/`.

## Qwen-Coder a_script calibration (2026-07-03) — picking the fine-tune target

`a0_oracle_gap.py` (uniform-hard pool, MAG=100, k=6) across the ladder — measures whether each size can *write a working tool* (a_script), independent of the urn result. a_hand = 0.00 for every size (pool is hand-hard throughout, as intended — MAG=100 is fine here since a_script tests algorithm-writing correctness, not hand-arithmetic difficulty, and is not magnitude-sensitive at this problem scale).

| model | a_hand | **a_script (pooled)** |
|---|---|---|
| 0.5b | 0.00 | 0.21 |
| 1.5b | 0.00 | 0.35 |
| 3b | 0.00 | 0.50 |
| 7b | 0.00 | 0.75 |
| **14b** | **0.02** | **0.83** |
| 32b | 0.00 | 0.96 |

Smooth, monotonic climb with size (unlike the flat urn-competence plateau) — coding ability scales cleanly; allocation competence doesn't. Both clear the ≥0.8 usability bar (14b: 0.83; 32b: 0.96, near-perfect, only `crt_solve`/`matrix_power_mod` at 0.83).

**Infra note (H100-specific gotcha, worth remembering for any future large-model run on this box):** the initial 32b a_script run stalled badly (~8.6 tok/s, 8%/92% CPU/GPU split) — cause: `OLLAMA_NUM_PARALLEL=8` (set for the cheap urn sweep) × Ollama's default ~32768-token context meant 32b's KV cache no longer fit purely on the H100, forcing partial CPU offload. Fix: created `qwen2.5-coder:32b-ctx8k` via an Ollama Modelfile (`PARAMETER num_ctx 8192` — ample for these short single-turn problems) → 100% GPU placement, ~68 tok/s single-stream (~8× faster), confirmed before committing the real run. Context window only matters for long transcripts; irrelevant to a_script correctness at this scale, but will matter again for Phase 2's persistent 60-problem tool session (long transcript) — size `num_ctx` to the real session length then, not reused blindly from this fix.

**FT-target decision: 14b, not 32b — on wall-clock, not correctness.** Even after the context fix, the 32b a_script run still took several minutes; a real Phase 3 SFT + repeated eval loop on 32b would multiply that many times over, and 32b needs the ctx8k variant threaded through everywhere to stay fast. **14b (a_script 0.83, a_hand 0.00, clean urn protocol, 0 unparsed) is fast, comfortably clears the usability bar, and is on the same suboptimal allocation plateau (regret 491) as everything else below Opus** — the profile Phase 2 needs, at a size cheap enough to iterate on. 32b remains a documented fallback/stronger-alternative if 14b turns out too weak once real tool-writing (not just single isolated a_script instances) is tested.

## Qwen-Coder tool baseline (Phase 2, 2026-07-03) — eager, same as every model tested

Ran the actual tool task (60-problem persistent session, A1 announce, uniform-hard N=8, MAG=100,
seeds 2000–2011 — same design/seeds as Haiku's A1 tool arm) on **qwen2.5-coder:14b** via Ollama on the
H100 box. Harness: extended `arm_a1_announce.py` to accept raw Ollama model tags (mirrors
`urn_session.py`'s `IS_LOCAL`/zero-pricing pattern) and to pass `stop_on_budget_exhausted=True` for
local models.

**Two real bugs found and fixed** (not just an optimization — these were silently zeroing out every
build before the fix):
1. **Triple-quoted code breaks JSON.** Qwen-14b sometimes writes a tool-call argument as a Python
   triple-quoted string (`"code": """\ndef solve...\n"""`) instead of a properly escaped JSON string —
   invalid JSON (reads as an empty string immediately followed by a stray quote), so every
   `write_script` call carrying real multi-line code failed to parse and the session never registered
   a single build. Fixed in `lomekwi/raw_chat.py`: `_repair_triple_quoted_strings` JSON-escapes any
   `"""..."""` span before the decode attempt (safe by construction — three consecutive double-quotes
   are never valid JSON, so this can only rescue an already-guaranteed parse failure, never break a
   working one).
2. **Tool-call envelope/parameter conflation.** The model sometimes puts the *script's* name in the
   tool-call's top-level `"name"` field (where the tool name belongs) and drops only `"code"` into
   `"arguments"`, omitting the wrapper entirely — `{"name": "bitwise_operations", "arguments":
   {"code": ...}}` instead of `{"name": "write_script", "arguments": {"name": "bitwise_operations",
   "code": ...}}`. Added a third recognized shape to `_coerce_tool_call_obj`: `"code"` is a
   `write_script`-only argument in this schema (run_script takes `"inputs"`), so its presence
   unambiguously identifies the intended tool even when the envelope is malformed.

**Early-stop infra (reused, not new):** `driver.run_session`'s existing `stop_on_budget_exhausted` flag
(built earlier for a Haiku pilot) truncates the session the instant the write budget (B=3) is spent —
no further BUILD decisions are possible, so all decision signal is final. `skirental_scorer.
model_builds_from_actions`/`value_of_builds` were already designed to value a truncated session
analytically (reuse the untouched tail if built, hand-solve it if not) — "killing the truncation
confound" per their own docstring. Net effect: **most sessions finished within 2-3 problems / ~13-20
turns instead of running the full 60-problem session** — all 12 seeds completed in **~5.5 minutes
total** (vs. an estimated 15-25 min for full T=60 sessions), at zero cost (local model).

**Result:** 23 builds across 12 seeds, **22/23 (96%) at first sight**, mean lateness 0.043 — Qwen-14b
is eager in the tool task, matching Haiku (100%) and Opus's assumed-eager tool behavior. Built-set ==
first-B-distinct-arrivals on 9/12 seeds (75%). Regret vs exact π\* (Costs built with the **measured**
a_script=0.83, not the ~1 default, so the model's realized script-writing unreliability is priced into
both its own value and π\*'s): mean 3268 ± 659, positive on 11/12 seeds; model_traps/seed=1.00 vs
π\*_traps/seed=0.75.

**Paired against Qwen-14b's own urn result — a small/no framing gap, unlike Haiku's:** urn lateness
0.17 vs tool lateness 0.043 — both already mostly eager. This is the opposite of Haiku's large
within-model gap (urn 0.86 → tool 0.00) and is exactly what the "flat suboptimal plateau, not a smooth
slope" reading (§ Qwen-Coder urn slope, above) predicts: Qwen never showed much reserving behavior in
the *bare* urn either, so there's little latent allocation competence left for the coding framing to
suppress. The urn→tool suppression effect so far only shows up in models that have real competence to
suppress (Haiku partially, Opus — assumed — fully); a model sitting on the plateau is eager in both
framings. **Caveat:** don't directly compare the regret *magnitudes* across urn (491, computed with the
default a_script=1.0) and tool (3268, computed with the measured a_script=0.83) here — different cost
models, not a paired apples-to-apples number. Lateness is the valid cross-framing comparison.

**Confirms the FT-target profile:** Qwen-14b is eager in the tool task (like every model tested) *and*
sits on the suboptimal plateau in the bare urn — the "codes fine, allocates badly" combination Phase 3
needs to test whether SFT on π\*-optimal urn demonstrations can move it off the plateau, and whether
that transfers to the tool game.

## Novelty / related work

Full 5-agent adversarial check + verdict + phrases-to-avoid is in `plan.md` §5b (not duplicated here).
Headline: core investment/amortization slice is CLEAR; abstraction-gap *method* is not novel (borrow,
domain is); capability axis is CROWDED (make it secondary); reference policy is a safe textbook
assembly, correctly attributed.

## Immediate next steps

Phase 1 (Qwen urn baseline + a_script calibration) and **Phase 2 (Qwen-14b tool baseline) are both
DONE.** **Phase 3 is fully specced** in `docs/qwen-finetune-transfer-plan.md` (bridge SFT: diverse urn
demonstrations + a small tool-framed slice, plus an eager-policy control fine-tune) — see that doc for
the full design, implementation steps, and open risks. **FT target = Qwen2.5-Coder-14b** (a_script 0.83, a_hand 0.00, clean urn protocol, eager tool
baseline confirmed; 32b scores higher (0.96) but too slow to iterate on for Phase 3's repeated SFT+eval
loop). Continuing the transfer arc:
1. **Phase 3 (next) — SFT Qwen-14b on π\* urn demonstrations** (generate perfect traces from
   `exact_dp`) + a control FT; first re-eval on held-out urn seeds (does it even *learn* the allocation
   policy off the plateau toward π\*?). The early-stop-on-budget-exhaustion technique from Phase 2
   (§ Qwen-Coder tool baseline) applies directly to speeding up Phase 4's re-eval loop too.
2. **Phase 4 — eval the fine-tuned Qwen-14b on the tool game** (does urn competence *transfer*?) —
   re-run the same `arm_a1_announce.py --model <finetuned-tag>` harness used for Phase 2.
3. (Deferred) Opus tool cell for the paired 2×2 — assumed eager, pre-publication tightening (needs the
   `continued_frac` bound + drop `matrix_power_mod`). R1 rung (mechanism isolation). More seeds for a
   stable regret *level* (lateness already tells the story).

## Platform note (from the earlier design, still relevant if the Qwen ladder resumes)

- **Qwen-Coder** was chosen as the main open-weights platform for a future fine-tuning/mitigation arm
  (needs open weights; Claude can't be fine-tuned). Not yet started — see `plan.md` §5 deferred list.
- Haiku serves as the frontier proof-of-concept so the finding isn't dismissed as a small-model
  artifact; Opus (via the abstraction-gap 2×2) is now doing double duty as the frontier anchor instead
  of a separate capability-ladder run.

## Open items

- Confirm π\*'s exchangeable-Dirichlet prior is not sensitive to α in a way that matters (spot-checked,
  not swept).
- `matrix_power_mod`/`continued_frac`/`linrec_mod` covers only matter if a cross-model uniform pool
  beyond Haiku/Opus is revisited (e.g. Sonnet or the Qwen ladder) — not urgent while the 2×2 is the
  active thread.
- Keep R/λ set so a single build never pays off standalone (one-shot wasteful) but recurrence (count
  ≥~2-3) does — already true of the current Costs constants; re-verify if magnitudes are bumped further
  for a new family/model pairing.
