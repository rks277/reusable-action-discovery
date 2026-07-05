# LLMs Don't Recognize Reusable-Tool Creation as Resource Allocation

**Status (2026-07-04): core result CONFIRMED on a clean same-information footing.** A same-information audit added an N-disclosed arm; the headline survived it and got *stronger*. Pre-audit docs preserved in `docs/old/`. Next open thread = Phase 3 fine-tune transfer (§7; demo corpus already built).

---

## 0. Orientation (read first) — notation, files, conventions

**The task.** A model answers a stream of **T** numeric problems one at a time, drawn i.i.d. from **N** distinct problem *types* (a.k.a. classes). It may `write_script` (build a reusable solver — a fixed one-time cost, capped at **B** writes for the whole session) and later `run_script` for free on same-type problems. Some types recur often (**hot**), most appear rarely (**trap**). Optimal play spends the B scarce writes on hot types once they've shown they recur.

**Key quantities.**
- **π\*** — the reference optimal policy: an exact finite-horizon belief-state DP (`exact_dp.ExactDP`), symmetric Dirichlet(α=1) prior over the type mix. "Same-information optimum" *only if* the model also knows N (see §2). Reserves budget, builds on the ~2nd sighting.
- **lateness** — build/keep position minus first sighting (0 = built on first sight = "eager"; ≥1 = waited for recurrence). **first-sight %** — fraction of builds at lateness 0. These are the **lead metrics** (behavioral, robust).
- **regret** — value(π\*) − value(model), analytic over full realized streams. **Secondary/noisy** at n=12–24 seeds; lead with lateness, not regret level.
- **traps/seed** — how many of the model's builds were on trap (rarely-recurring) types (wasted budget). π\* itself builds ~0.75 traps/seed here.
- **a_hand / a_script** — P(model solves a problem correctly *by hand* / *with a correct script*), measured per-model by `a0_oracle_gap.py`. The pool is "uniform-hard" = a_hand≈0 (no closed form; must iterate). a_script feeds the regret cost model.
- **g** (`guarantee_trap_early`) — fraction of seeds forced to have a trap in the first B slots (g=1 = every seed; stresses eager-vs-wait early). **MAG** — number-magnitude difficulty dial (100 for Haiku/Qwen, 1000 needed to make Opus's pool hand-hard). **m\*** — break-even reuse horizon (build pays iff a type recurs ≥ m\* times).
- The 8-family uniform-hard pool + all params live in `scripts/creator/tool_disposition_benchmark/` (`stream_builder.py`, `family_kit.py`). Models: **Haiku** = claude-haiku-4-5, **Opus** = claude-opus-4-8, **Qwen** = qwen2.5-coder:{0.5b…32b} (open-weights, for fine-tuning).

**Conventions.** `[[name]]` = a cross-reference to the author's memory notes (not in-repo); the key one, **[[no-auto-reps]]**, means: do NOT launch expensive/multi-seed runs without explicit user go-ahead — propose plan + cost first. All code is under `scripts/creator/tool_disposition_benchmark/`; run with `PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.<module>`.

**Data / compute location (2026-07-04).** Claude A2 runs are LOCAL (`runs/urn_haiku_n-announced/`, `runs/urn_opus_n-announced/`, `runs/arm_a1_announce_n-announced/`). **The Qwen A2 raw run dirs are NOT local — they were produced on an ephemeral H100 box that is now DOWN; only the aggregate numbers in §3 survive.** GPU boxes here are transient (fresh IP each time, no persistent disk); each needs full re-setup — **the complete, verified runbook is `docs/box-setup.md`** (Ollama + concurrency override, model pulls, repo/venv/.env sync, smoke test, launch commands, vLLM for Phase 3, and the "always rsync `runs/` back before releasing the box" rule that we learned the hard way). Re-run Qwen work only if the raw transcripts are needed again.

---

## 1. The claim

When a stream of numeric problems lets you **build a reusable script once** (fixed cost) and **reuse it free** on later same-type problems, under a **scarce write budget**, the right behavior is to treat the *B* writes as a budget and spend them on types that recur. **LLMs don't frame it that way — they build eagerly on first sight and never reserve.** This is a **recognition/framing failure**, not an inability to allocate: the *same model* allocates competently when the identical decision is posed as a bare urn/balls game, and fails the moment it's dressed as tool-writing.

## 2. Two information conditions (why there are two, and both matter)

The reference policy π\* (`exact_dp.ExactDP`) is constructed knowing the number of distinct types **N** (its Dirichlet-multinomial predictive `(α+k)/(N·α+t)` needs N). The model's information relative to π\* defines two conditions:

- **no-N (ecologically realistic).** The model is *not* told N — the real-deployment setting, since an agent never knows the size of the type-space it will face. Regret vs π\*(known-N) is then an **upper bound** on same-information regret (π\* is strictly stronger), and the question is "how close to an N-informed optimum can you get *without* knowing N."
- **A2 (same-information, N disclosed).** The model *is* told the exact N (`--announce-n`; urn system prompt states "exactly N distinct colors," tool prompt adds `prompts.n_types_note`). Now the model and π\* have identical information, so regret vs π\* is the honest same-information regret, and any residual gap **isolates framing from N-ignorance**.

The A2 arm was added after we noticed π\* had been given N while the model wasn't (the "audit," 2026-07-03). It did not overturn anything — it sharpened it.

## 3. The core result — the abstraction gap (recognition, not competence)

Same allocation decision, two framings, identical streams (seeds 2000–2011, uniform-hard N=8, g=1, T=60, B=3), reference = the exact belief-state DP π\*. **Lead metric = lateness / first-sight** (build position vs first sighting); regret is secondary.

### Haiku 2×2 — no-N and A2 side by side

| | urn — no-N | urn — A2 | tool — no-N | tool — A2 |
|---|---|---|---|---|
| first-sight | 53% | 28% | 100% | **100%** |
| lateness | 0.86 | 1.19 | 0.000 | **0.000** |
| regret vs π\* | 640±285 | **0±344** | 1308±520 | **1633±553** |
| traps/seed | 0.92 | 0.58 | — | 1.25 (π\* 0.75) |

**Reading:**
1. **Told N, Haiku allocates *perfectly* in the urn** — regret 0, matches π\* exactly. It provably *has* the competence.
2. **Told the *same* N, Haiku ignores it entirely in the tool task** — 100% first-sight, lateness 0, regret 1633 (N disclosure gave *zero* benefit here — even slightly worse, more traps). The coding framing suppresses allocation reasoning the model demonstrably has.
3. The old "partial gap" reading (urn regret 640, nonzero) was an artifact of *urn-side* N-ignorance. Remove it (A2) and the dissociation is **clean**: optimal in one framing, fully open-loop in the other, same information.

### Opus urn — near-optimal, robust to the audit

| | no-N | A2 |
|---|---|---|
| first-sight | 8% | 0% |
| lateness | 1.14 | 1.28 |
| regret vs π\* | −655±389 | **−685±390** (beats π\*) |
| traps/seed | 0.33 | 0.33 (π\* 0.75) |

Opus essentially aces the urn under both conditions (waits for a repeat, near-zero traps). **Do NOT headline "beats π\*."** The negative regret is (a) only ~1.75 SE below 0 at n=12, and (b) possible at all only because π\* is optimal against its *mis-specified* symmetric-Dirichlet prior — the real generator has fixed hot-count + trap-early (g=1) structure the prior doesn't model, so a more trap-averse policy can edge it. **Report as: "reaches the same-information optimum (regret ≈ 0 within noise) while being systematically more trap-averse (0.33 vs π\* 0.75)."** Lead with the behavioral signals (waits, avoids traps), not the regret sign. **Opus's paired tool cell is ASSUMED eager** (prior constructed-design: bait 20/20, lateness 0; shelved pre-publication — see §5). Combined reading: real allocation competence exists (Opus, and Haiku-in-the-urn), and the tool framing squanders it.

### Cross-model urn: competence is a plateau + frontier jump (holds under A2)
Qwen-Coder 0.5b→32b is a noisy *suboptimal plateau* under **both** conditions — no-N regret ~500–1700; A2 (N disclosed) regret 844/1743/1197/1494/737/216 for 0.5b→32b. Only Opus breaks fully away (beats π\*). **Do NOT claim "widens smoothly with capability"** — flat-then-frontier-jump (matches [[creator-frontier-inversion-cross-family]]).

**The decisive A2 finding: N-disclosure does NOT rescue the Qwen ladder** the way it rescued Haiku (640→0). Told N, Qwen 0.5b–14b stay suboptimal (737–1743); only 32b moves toward optimal (216, lowest, CI brushing 0 — the frontier beginning to show). So **"has latent allocation competence when told N" tracks the frontier: Haiku and Opus have it, the Qwen ladder up to 14b does not, 32b is starting to.** This distinguishes two failure modes — *framing suppression of an existing competence* (Haiku) vs *genuine absence of the competence* (Qwen ≤14b) — see the Qwen-14b 2×2 below.

### Qwen-14b 2×2 (the fine-tune target) — genuine incompetence, not suppression

| Qwen-14b | urn no-N | urn A2 | tool no-N | tool A2 |
|---|---|---|---|---|
| lateness | 0.17 | 0.36 | 0.043 | 0.125 |
| regret vs π\* | 491 | 737 | 3268 | 2934 |
| first-sight | — | 76% | 96% | 88% |

Unlike Haiku, N-disclosure barely moves 14b anywhere: urn stays suboptimal (737), tool stays eager (2934, positive regret on all 12 seeds). **14b lacks the allocation policy in both framings** — the clean "codes fine (a_script 0.83), allocates badly, and it's not N-ignorance" profile the fine-tuning experiment (§6) targets: teach a competence it demonstrably doesn't have, then test urn learning + tool transfer.

## 4. Mechanism — recognition/framing, not value-of-information
Transcript analysis (tool sessions): Haiku's build rationale is *always* per-problem ("I need to solve X, let me write a script"). "recur" = "recurrence *relation*" (algorithm-speak); "budget" appears only *retrospectively* ("since I've used up my budget"); "reserve/conserve" = 0×. It reactively reuses a saved script when a type repeats but never *proactively* allocates — even when told N. The A2 tool result (N known, still 100% eager) is the quantitative counterpart: it isn't deliberating about *when* to build at all.

## 5. What's solid vs. pending
**Solid (measured, clean):** Haiku urn+tool under both no-N and A2; Opus urn under both. The A2 Haiku 2×2 is the airtight same-information proof. a_script calibration (Qwen 0.21→0.96). Instrument (exact-DP cap=3 lossless; harness; scorer; urn isomorph). Novelty verdict (`docs/old/online-tool-investment-related-work.md`).

**Pending (do NOT run without go-ahead — [[no-auto-reps]]):**
- ~~Qwen A2 reruns~~ **DONE (2026-07-03)** — urn ladder + 14b tool, folded into §3.
- **Fine-tune transfer (Phase 3)** — the main open thread; demo corpus already built (§7 step 1 DONE); see §7.
- **Paired Opus tool cell** — replace the assumed cell with a measured one on the hardened pool (MAG=1000, josephus pinned-last, continued_frac bounded, drop matrix_power_mod). ~$30; buys rigor not a new finding.
- R1 rung (declarative lock-in), cost-regime/m\* sweep, more seeds for a stable regret *level*.

## 6. Positioning (from the adversarial novelty check — `docs/old/online-tool-investment-related-work.md`)
Lead with the amortization/investment mechanism + the abstraction-gap 2×2 (recognition-vs-competence in the tool-creation domain). Capability axis is CROWDED → secondary. Abstraction-gap is a *borrowed, validated method* in a novel domain; differentiator = **hold computation constant**. Reference is an explicit textbook oracle. **Phrases owned by others — do not headline:** "premature commitment," "budget-aware tool use," "knowing-doing gap," "content effects." Always qualify "regret" as amortization/ski-rental regret.

## 7. Next steps — Phase 3 fine-tune transfer (the main open thread)

**Full executable spec: `docs/qwen-finetune-transfer-plan.md`** (as-built corpus, LoRA/hyperparameter config, training-script design, loss-masking + chat-template mechanics, merge+serve, eval seeds, cost, stop-points). Summary below.

**Question:** can we *install* the allocation policy Qwen-14b lacks (via SFT on π\*-optimal urn demonstrations) and have it **transfer to the tool framing**? 14b is the right subject because its urn failure is genuine absence, not framing-suppression (§3) — so a gain would be real learning, and a *negative* transfer result (learns the urn, still eager in the tool) is itself a strong finding: it would show the framing gap is a hard wall even for a model freshly taught to allocate.

**All new runs are on the A2 (N-disclosed) protocol** — demonstrations must disclose N, matching the corrected same-information footing.

1. **Demo generator (`phase3_demos.py`) — DONE (2026-07-04), both tweaks applied + regenerated.** Redone under A2: both slices disclose N; stream params (N∈{6,8}, T∈{60,80,100}, B/N≈0.25–0.35) fixed so π\* genuinely reserves. Self-test (`--selftest`) confirms **93% of π\* urn commits at k≥2** and the eager control at 0% (clean contrast). Corpus at **`runs/phase3_sft_data/`**: `{urn,tool_bridge}_{pistar,eager}.jsonl` — **170 urn + 6 tool-bridge** sessions per policy. **The two pre-training tweaks are now applied** (2026-07-04): (i) tool-bridge cut 30→6 sessions so it is ~85/15 urn:tool *by token* — measured tool share 16.8% (pistar) / 12.0% (eager), ≈15% averaged over the two arms [tool sessions run ~6× longer than urn, so the old 30 was ~50/50 by token — the doc's earlier "~8 sessions" estimate was too high]; (ii) each URN_VOCAB entry now carries a per-vocab attribute word (ball→color, ticket→material, coin→metal, tile→symbol, gem→kind, card→suit, token→shape) threaded through the prompt + rationales, instead of hard-coding "color" — all 7 words appear ~evenly across the 170 urn sessions. Self-test still green.
   - *Design note:* on "wait" turns the tool-bridge demo submits the gold answer with no shown work — fine for teaching the build/wait *decision* (we score build-timing, and a_hand≈0 is already in the cost model), but it trains "emit the number" on those turns.
2. **Two SFT arms + control (LoRA, on a fresh GPU box — see §0 for setup):** treatment = π\* demonstrations; control = eager-policy demonstrations (same volume/format) to rule out "any SFT helps." Serve the fine-tuned model via vLLM (`LOCAL_BACKEND=vllm` branch in `lomekwi/raw_chat.py`, untested this project → smoke-test first). Need the HF checkpoint `Qwen/Qwen2.5-Coder-14B-Instruct` (Ollama GGUF isn't fine-tunable).
3. **Eval in order:** (a) held-out urn A2 (fresh seeds, disjoint from training seeds 4000/5000-range) — did it learn to allocate off the plateau toward π\*? (b) tool A2 on the paired seeds 2000–2011 — does it transfer, vs the 14b pre-FT baseline (urn A2 737, tool A2 2934)? (c) re-run `a0_oracle_gap` post-FT — a_script may drift, so regret needs recalibrated costs. Lead with lateness/first-sight, not the regret level.
4. **Optional stronger subject:** 32b (urn A2 216 — already near-frontier) would test transfer in a model that *has* nascent competence; slower to iterate, hold as a follow-up.

**Next actionable step:** the corpus is finalized (both tweaks applied + regenerated 2026-07-04) — next is to request a GPU box (setup in `docs/box-setup.md`) for the LoRA training + eval in (2)–(3). Propose a plan + cost first per [[no-auto-reps]].
