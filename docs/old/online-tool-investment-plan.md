# LLMs Don't Recognize Reusable-Tool Creation as Resource Allocation

**Status:** Active. Last rewrite 2026-07-03 (full consolidation — supersedes the incremental logs below). The headline **evolved** twice: `capability-graded recurrence-recognition` (falsified) → `fail the optimal online VoI policy` (partly right, wrong mechanism) → **the current framing below**, which the transcript analysis forced.

---

## 1. The claim (current)

A model that must solve a stream of numeric problems, where a **reusable script can be built once** (fixed write cost) and then **reused for free** on later problems of the same type, under a **scarce write budget**, is facing a **resource-allocation problem**: some types recur (a script pays off on every occurrence), some appear once (a script is wasted), and only *B* of them can get a script. The right behavior is to treat the *B* writes as a scarce budget and spend them on types that will recur.

**LLMs don't do this. They treat `write_script` as "how I solve this one hard problem," build eagerly on first sight, and never reserve budget — even when told some types recur.** The failure is not that they gather too little information before committing (that was our previous, wrong framing); it is that **they never frame the task as allocation at all.** This is a **recognition/framing failure**, not a value-of-information failure.

**What resolves it (the core result, §4):** an *abstraction-gap* control shows the same model allocates competently when the identical decision is posed as a bare urn/balls game but builds eagerly the moment it's dressed as tool-writing — so the failure is **recognition (framing), not an inability to allocate**. That latent allocation competence is a **plateau + frontier jump**: the Qwen-Coder ladder (0.5b→32b) is a noisy suboptimal plateau (≈ Haiku), and only Opus reaches near/above-optimal — so real competence, and the recognition gap that squanders it, is a **frontier phenomenon** (not a smooth capability slope).

Positioning: the lab's **recognition axis** (does the model recognize the structure of the task?), not the crowded VoI / premature-commitment / ski-rental slice. Per [[online-investment-novelty-verdict]]: do **not** headline "uniform across capability" (unproven) or "premature commitment" (Mehta owns the phrase).

---

## 2. Setup (stochastic design + exact reference)

Full detail in `docs/online-tool-investment-working-notes.md` and `docs/same-info-optimal-dp.md`.

- **Streams:** *T* i.i.d. draws from a distribution over *N* types (*B* hot + *N−B* trap). Family↔rate assignment randomized per seed. A consequential-seed knob **g** = P(a trap is forced into the first *B* slots).
- **Cost model (per Haiku A0):** R=100, λ=0.1; u_hand = −98.7, u_build = 49.2, u_reuse = 80. Building beats hand even single-use, so the real scarcity is the **write budget**, not per-token cost.
- **Reference π\* = the EXACT finite-horizon belief-state DP** (`exact_dp.py`) — the same-information Bayes-optimal policy. Symmetric Dirichlet(α=1) predictive; canonical state = (built-count-multiset, unbuilt-count-multiset); closed-form budget=0 base case. Made tractable at N=12,T=60 by a **count-cap K**: a K-sweep certified `V(3)=V(4)=V(5)=V(6)` exactly → **cap=3 is exact** (~540k states, ~3s, pure Python). Replaces the retired Whittle relaxation (over-built at N≈12). Wired into the scorer as `exact_pistar_report`. π\* **reserves the first sighting** and builds on recurrence — never builds a type on first sight.
- **Two-layer metric** (separates policy quality from draw luck — see [[exact-dp-reference-and-poc]]):
  1. **Behavioral fidelity (luck-free, model-independent):** which policy is the model running? Read straight off the action log (build lateness; is built-set == first-*B* distinct arrivals?). No per-model calibration needed.
  2. **Expected policy regret (luck-free headline number):** once fidelity pins the policy, price it vs π\* analytically over thousands of streams (tiny SE). Expensive model runs only need enough sessions to pin the policy.

---

## 3. What we've established (Haiku)

All on the stochastic design with the exact-DP reference.

- **Eager open-loop, 100%.** Across 30 g=1 sessions / 88 builds: **every build at first sight** (lateness 0.000), and **30/30** built-set == first-*B* distinct arrivals. Haiku ≡ "build the first *B* distinct types on sight." Zero exceptions, never reserves. Within-class reuse is **perfect** (1 tool/class, 0 rebuilds) — so the failure is cleanly isolated to **build allocation**, not tool quality.
- **Regret is distribution-dependent, so it is NOT the headline.** g=1: analytic E[regret] = 1417 ± 40 over 1500 streams (eager traps 1.21 vs π\* 0.50; realized 30-seed 1107 ± 275, consistent). **g=0 control DONE:** under natural draws model traps/seed collapse 1.10 → 0.10 = π\* 0.10; on the uniform-hard pool analytic regret falls g=1 983 → g=0 71. So the big regret number is a **trap-early-conditioning artifact**. The headline is the **open-loop policy itself** (distribution-independent), not the regret magnitude.
- **Disclosure-immune (A1 announce, n=12).** Told non-prescriptively that some types recur several times and others appear once, Haiku *still* builds 35/35 at first sight (lateness 0.000). Open-loop whether recurrence must be **inferred** (A0) or is **disclosed** (A1).
- **MECHANISM — recognition/framing, not VoI (transcript analysis).** Haiku's build rationale is *always* per-problem — *"I need to solve X, let me write a script."* It **never** reasons about types recurring, reserving budget, or waiting: "recur" appears 148× but is entirely "recurrence **relation**" (algorithm-speak); "budget" 47× is entirely **retrospective** (*"since I've used up my budget…"* after exhaustion); "reserve/conserve" 0×. It spends its *B* writes greedily on the first hard problems and only *notices* the budget once it's gone. It **reactively** reuses a saved script when a type repeats but **never proactively allocates**. So "doesn't wait for recurrence evidence" is the wrong description — it isn't deliberating about *when* to build at all.

### Prior (constructed-design) evidence — consistent, superseded metric
The earlier hand-constructed recurring+one-off design showed the same behavior across a full Claude ladder (Haiku/Sonnet/Opus): bait rate ~100%, mean lateness 0.00, disclosure-immune. That work established the phenomenon and built the instrument, but used bait/lateness on constructed streams rather than the stochastic design + exact-DP regret; treat it as corroborating, not as the capability result. (Its "uniform across capability" conclusion is **not** carried forward — see §4.)

---

## 4. The core result — the abstraction gap (recognition, not competence)

The central experiment runs the **same allocation decision in two framings** on identical streams: the **tool task** (build a reusable script once, vs hand-solve each time) and an **isomorphic urn/balls task** (KEEP the current ball → collect it + every future same-color ball; PASS/never-keep → 0). If a model allocates well in the bare urn but builds eagerly the moment it's dressed as tool-writing, the failure is **recognition (framing), not an inability to allocate**. Harness `urn_session.py`; the matched tool arm is **A1** because the urn prompt ("a bag of colored balls; you don't know the counts"; KEEP locks in all future of that color) *discloses that colors recur* — so it is A1-equivalent, if anything more disclosing. **Lead metric = lateness / first-sight** (stable; regret is noisy + seed-dependent — see caveats). All cells: Haiku + Opus, seeds 2000–2011, uniform-hard, g=1, T=60, B=3, A1 both sides, identical streams; reference π\* is the same `exact_dp`.

Design details fully in `docs/online-tool-investment-working-notes.md` (§ abstraction-gap), incl. the two bugs caught and fixed (observation-parity: kept colors auto-collect in place and are reported to the model so it sees exactly T draws; and the info-condition = A1 correction).

### The 2×2

| | urn (allocation framing) | tool (coding framing) |
|---|---|---|
| **Haiku** | lateness 0.86, first-sight 53% — *partial reserve* | lateness 0.00, first-sight 100% — *eager* |
| **Opus** | lateness 1.14, first-sight 8% — *near-optimal (aces it; beats π\*)* | **[assumed] eager** — lateness ~0, first-sight ~100% |

Supporting regret (secondary — noisy at n=12, do not headline the level): Haiku urn 640±285 vs A1 tool 1308±520; Opus urn −655±389 (beats π\*); Haiku urn traps 0.92 vs Opus urn 0.33 (π\* 0.75).

Three readings, one story:
1. **Within model (urn → tool): the coding framing suppresses allocation.** Haiku: partial (halves misallocation, 1308→640; first-sight 100%→53%). Opus: large (assumed) — from acing the urn to eager building.
2. **Across models (urn): competence is a plateau + frontier jump, NOT a smooth slope.** The Qwen-Coder ladder (0.5b→32b, 24 seeds; §5) is a noisy *suboptimal plateau* — regret ~500–1700, no clean scaling, Qwen-32b (901) ≈ Haiku (640) — and only **Opus breaks away to beat π\*** (−655, traps 0.33). So it is flat-then-jump at the frontier (matches [[creator-frontier-inversion-cross-family]]), *not* "competence rises with scale." Lateness *does* trend (Qwen ~0.02–0.31 eager < Haiku 0.86 < Opus 1.14), but **regret decouples from lateness**: regret is dominated by *type-selection* (trap vs hot, a ~2,600-utility swing per misallocated keep), and Qwen and Haiku make the same ~0.3-excess-trap error, so a 4× lateness gap barely moves regret. Opus's win is trap-avoidance, not waiting per se.
3. **Combined — the recognition gap is a frontier phenomenon.** Real allocation competence appears only at the frontier (Opus), and the tool framing then squanders even that (Opus assumed eager). Reframes the crowded capability-axis literature (§5b): not "does capability fix tool disposition?" but "competence appears only at the frontier, and the tool framing squanders it." **Do NOT claim "widens smoothly with capability"** — the Qwen ladder is a flat plateau; state it as plateau→frontier-jump.

### The Opus tool cell is ASSUMED eager, not measured (shelved to pre-publication)

We take Opus as eager in the tool task **on the strength of the prior constructed-design run** (`stream_sweep_opus_n20_announce`: bait 20/20, lateness 0 — Opus built one-offs on first sight in every seed). Re-running it on the *paired* stochastic design (seeds 2000–2011) buys **rigor/comparability, not a new qualitative finding** — we fully expect eager again (see prediction reasoning in the working notes). Given ~$30 + a calibration cycle + interpretation risk (below), the paired Opus tool cell is **shelved**; tighten before publication if a reviewer demands it or the story leans harder on it.

**If/when it is run, the pool needs the following** (calibrated 2026-07-03; details in working notes):
- **MAG=1000 for Opus** — `crt_solve`/`modpow` are only hand-hard (a_hand→0) there, not at MAG=100 (their samplers were patched to scale difficulty with magnitude; Haiku stays at MAG=100, unaffected).
- **`continued_frac` must be bounded** — at MAG=1000 its (modulus-free) numerator explodes to ~55 digits → Opus perseverates for dozens of turns → sessions never finish. Cap its term-scaling independent of the global MAG dial (it's already hand-hard for Opus at small terms).
- **Drop `matrix_power_mod`** — a_script≈0.33 for Opus (can't reliably *write* the tool), so building doesn't buy accuracy → unusable. Gives N=7. (Verified-clean for Opus at MAG=1000: lcg, modpow, crt_solve, quadratic_map_mod, xorshift_steps.)
- **`josephus` pinned last** — hand-solvable for Opus at *any* magnitude (no closed form for general K, but Opus tracks the O(N) recurrence reliably), so it can't be made hard by scaling. Pinned as a single forced trap at the final slot (`pinned_last_trap`) so its hand-easiness cannot influence any build/reserve decision (0 draws remain after it).

### Caveats (carry into the writeup)
- **Haiku's gap is partial, not a switch.** Urn regret ≠ 0 and Haiku builds *more* traps than π\* (0.92 vs 0.75) — so "framing fully explains the failure" is too strong. Honest claim: the coding framing **substantially suppresses** allocation Haiku can *partly* do.
- **Opus tool cell is assumed**, not measured on the paired design (see above).
- **Hand-solve asymmetry** — the tool task permits hand-solving (a fallback); the urn forbids it (PASS = 0). So an eventual real Opus-tool *lateness* could reflect *reserving* (recognition-positive) OR *hoping to hand-solve first* (mechanical). The **R1 rung** (declarative lock-in over real problems, deferred §5) isolates this, as does transcript reading.
- **Lead with lateness, not regret** — urn regret swung −30 (seeds 3000–3011) vs +640 (2000–2011) at n=12; lateness is stable across both seed sets.
- **R0 (the pure urn) removes three things at once** — surface, act-conflation (`write_script` *is* both the solve and the investment), computation load. R1 isolates which drives the tool-task suppression.

See [[abstraction-gap-urn-haiku]].

## 5. Deferred experiments (do NOT run without go-ahead — [[no-auto-reps]])

- **Paired Opus tool cell (pre-publication tightening).** Run Opus A1 tool on the hardened, pinned-josephus pool (seeds 2000–2011) to replace the *assumed* cell in §4 with a measured one on the paired design. ~$30 + a calibration cycle; buys rigor not a new finding (we expect eager). Harness ready (`arm_a1_announce.py --model opus`, pinned_last_trap wired); only `continued_frac` bounding + dropping `matrix_power_mod` remain. Do this only if a reviewer demands the airtight cell.
- **R1 rung — declarative lock-in over real problems.** Isolates *which* of {surface, act-conflation, computation-load} drives the tool-task suppression (R0/the urn removes all three at once). Also resolves the hand-solve-asymmetry interpretation risk in §4. The mechanism-sharpener; needs a new harness variant (real problems + a separate lock-in action + grading).
- **Qwen-Coder urn slope + a_script calibration — DONE (2026-07-03, H100 box, 0.5b→32b).** Urn (24 seeds): a noisy **suboptimal plateau**, no smooth scaling (regret ~500–1700; Qwen-32b 901 ≈ Haiku 640; lateness ~0.02–0.31, all eager) — the near-optimal behavior is a **frontier jump at Opus** only. Reinforces §4 reading 2 and matches [[creator-frontier-inversion-cross-family]]. Also the methodological point that **lateness ⊥ regret** (regret is type-selection-dominated). a_script calibration: coding ability scales *smoothly* with size (0.21→0.35→0.50→0.75→0.83→0.96, 0.5b→32b) — a clean dissociation from the flat urn plateau (Qwen gets better at writing tools but not at deciding when to build one). Full tables in working-notes §§ Qwen-Coder urn slope / a_script calibration. **FT target = Qwen2.5-Coder-14b** (a_script 0.83, a_hand 0.00, clean protocol; 32b scores higher (0.96) but too slow to iterate on for Phase 3's repeated SFT+eval loop — documented fallback). This completes **Phase 1 of the fine-tune-transfer experiment** (Qwen chosen for open weights). **Phase 2 — Qwen-14b tool baseline — also DONE (2026-07-03):** eager, same as every model tested (96% first-sight, lateness 0.043, n=12 seeds), confirming the "codes fine, allocates badly" FT-target profile. Along the way, found and fixed two real Ollama tool-calling JSON bugs (`lomekwi/raw_chat.py`: Python-triple-quoted code inside tool-call JSON, and a tool-name/script-name envelope conflation) that were silently zeroing out every build before the fix — not just a speed optimization. Paired against Qwen-14b's own urn result, the urn→tool lateness gap is small (0.17→0.043, both already eager) unlike Haiku's large gap (0.86→0.00) — consistent with Qwen sitting on the flat suboptimal plateau in *both* framings, i.e. having little latent allocation competence for the coding framing to suppress in the first place (only models with real urn competence — Haiku partially, Opus assumed fully — show a big suppression gap). Full detail + the early-stop-on-budget-exhaustion technique (cut the 12-seed run to ~5.5 min) in working-notes § Qwen-Coder tool baseline. **Next: Phase 3** (SFT on π\* urn demos, re-eval urn) → **Phase 4** (does urn competence transfer to the tool game?).
- **Cost-regime / m\* sweep.** Vary R/λ so building is *not* a per-instance win (m\*>1); show the eager failure persists where "build-everything" is genuinely wrong — kills the degenerate-cost critique. Mostly analytic, ~$10.
- **Framing / allocation intervention — DEMOTED.** Previously slated as central; now optional depth-probe. Rationale: the current prompt is **already fair and method-neutral** and supplies all the information (scarce persistent scripts), and the model fails it — so the recognition failure stands on the neutral prompt with no intervention. **F2 (prescriptive "wait until a type recurs") is cheating** (tests instruction-following, not disposition) — drop it. **F1 (non-prescriptive allocation framing)** has one narrow use: distinguishing "recognition gap, fixable by framing" from "robust inability" — but its "fixes it" branch *softens* the claim, so it's a later depth-probe, not a priority.
- **Demonstration / few-shot.** Does showing the wait-then-build policy transfer? ~$10.
- **Fine-tuning Qwen + held-out generalization eval.** Strongest mitigation and the reason Qwen is the platform, but contingent on the above; value is entirely in generalizing to held-out families/configs (else circular). Expensive; defer.
- **Robustness batch (cheap):** prompt-framing sensitivity (neutral vs tool-encouraging — framing dominated small models in CREATOR); design-param sensitivity (α, N, T, B), mostly analytic.
- **Realism bridge (stretch):** demonstrate the failure in a naturalistic tool-use setting (real coding tools / MCP). Best external-validity payoff; new harness, likely argued in prose instead.

---

## 5b. Related work & novelty (adversarial lit check, 2026-07-03)

Five parallel adversarial searches. **Verdict: CLEAR on the core; the defensible contribution is the *conjunction* — three of four ingredients are individually crowded.** Position accordingly.

- **Core (tool-building as budget-constrained investment under recurrence) — CLEAR.** No result collision. Neighbors to cite + distinguish: **TroVE** (per-problem Create/Skip/Import, no budget, retrospective), **"Library Learning Doesn't"** (2410.20274 — single-use libraries, but a post-hoc audit, *not* our claim), **LATM** (2305.17126 — owns the amortization *framing*, assumes the decision away), **Calibrate-Then-Act** (2602.16699 — cost-aware when-to-commit, but per-*action* cost not irreversible fixed-cost creation). White space = **irreversible fixed-cost creation × unknown recurrence × scarce build budget × measured as a recognition/allocation failure.**
- **Recognition mechanism + abstraction-gap control — METHOD NOT NOVEL, domain is.** The paired abstract-vs-embedded design is well-trodden: **2601.23048 "From Abstract to Contextual"** (near-exact, math domain), **Dasgupta content effects** (2207.07051), **GSM-Symbolic** (2410.05229), **LogiQAte** (2602.01132). Frame the control as a *borrowed, validated method* applied to a new domain; our differentiator = **hold computation constant** to isolate framing-as-allocation. Tensions: Dasgupta shows content usually *helps* (we claim it hurts — must engage); **2604.02910** shows abstraction doesn't always dominate → the clean dissociation may not appear. **Pre-register, and verify the abstract urn task is at ceiling before calling the gap recognition.**
- **Capability axis — CROWDED (most exposed flank).** Cannot claim first-to-show capability-graded tool disposition: **Model-Adaptive Tool Necessity / "knowing-doing gap"** (2605.14038, closest), **SMART** (2502.11435, large models under-use), **Tool-Use Tax** (2605.00136), **BAGEN** (2606.00198), **"When the Tool Decides"** (2606.14476, *opposite* direction — stronger defer more). Direction is contested. Make the capability axis **secondary**, positioned against this disagreement — never "first." (Consistent with our own [[build-proportion-inverse-in-capability]] / [[tool-disposition-inverse-capability]], now with external company.)
- **Decision-theory reference — no over-claim risk; cite as standard oracle.** An assembly of textbook primitives, not a named problem. Method: exact finite-horizon **belief-state DP** (Bellman 1957; DeGroot 1970; Bertsekas; Kaelbling–Littman–Cassandra 1998). Primitives: rent-or-buy (Karlin–Manasse–McGeoch–Owicki 1994), k-secretary (Kleinberg 2005), Dirichlet–multinomial (DeGroot 1970; Blackwell–MacQueen 1973). Flag Gittins (1979)/Whittle (1988; Weber–Weiss 1990) as related-but-heuristic to justify exact DP. Do **not** name the "claim → collect all future same-type" accrual as a problem or present the DP as a result; note tractability is a property of our small instance. Misattribution guard: 1988 Snoopy Caching = Karlin/Manasse/**Rudolph/Sleator**.

**Phrases owned by others — do NOT headline:** "premature commitment" (Mehta 2606.22936), "budget-aware tool use" (BATS 2511.17006), "knowing-doing gap" (2605.14038), "content effects" (Dasgupta), "tool overuse"/"self-aware agent" (SMART), "single-use library" (2410.20274), "tool-use tax" (2605.00136); always qualify "regret" as **amortization/ski-rental regret**.

**Strategic upshot:** lead with the amortization/investment mechanism (the open conjunction) + the abstraction-gap result (§4); the capability slope is secondary vs a contested field; the abstraction-gap is a borrowed method in a novel domain (hold computation constant); reference is an explicit oracle. Nothing we've *done* collides — the exposure is in *framing* (position the 2×2 as recognition-vs-competence in the tool-creation domain, not as a novel paired-design method or a novel capability-scaling claim).

## 6. Scope / honest limits

- The claim is about a **specific decision structure** — allocating a scarce, reusable, irreversible build budget under recurrence — not "LLMs are bad at all decision-making."
- It rests on the reference being well-defined and computed. **Done:** the exact belief-state DP is the same-information optimum, certified lossless at cap=3.
- Regret is **distribution-dependent** (g=0 defused it). The distribution-independent claim is the **open-loop policy** (fidelity: lateness / first-sight), which is what the abstraction-gap 2×2 (§4) measures.

## 7. Surviving capability signals (secondary)

- **Hand-solvability threshold:** stronger models hand-solve more → fewer builds triggered (same disposition, shifted threshold). Per-model A0: Haiku band m≈10, Sonnet m≈1000.
- **Tool generality:** stronger models write broader, more reusable tools (Sonnet's `poly_eval_exact`, built for a one-off, reused on 14 problems; Haiku builds narrow single-purpose tools).

## 8. Instrument (built, validated)

`family_kit.py` (exact-integer families, A0-calibrated per model), `stream_builder.py` (`StochasticStreamSpec`), `exact_dp.py` (the reference π\*), `run_stream_session.py` (persistent session, write budget, non-binding token cap), `skirental_scorer.py` (cost model + decision classification + `exact_pistar_report`), `a0_oracle_gap.py` (per-model hand-vs-build calibration), `poc_haiku.py` / `arm_a1_announce.py` (spend-guarded, resumable harnesses; reuse=1.00 confirmed).

**Status of the core result:** the abstraction-gap 2×2 (§4) is complete on paper — Haiku urn+tool and Opus urn are measured; the Opus tool cell is assumed-eager per prior constructed-design and shelved. No run is needed to state the headline (recognition failure, capability-graded competence, gap widens with capability). Remaining runs (§5) are pre-publication tightening or secondary; nothing runs without explicit go-ahead ([[no-auto-reps]]).
