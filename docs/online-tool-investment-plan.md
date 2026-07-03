# LLMs Don't Recognize Reusable-Tool Creation as Resource Allocation

**Status:** Active. Last rewrite 2026-07-03 (full consolidation — supersedes the incremental logs below). The headline **evolved** twice: `capability-graded recurrence-recognition` (falsified) → `fail the optimal online VoI policy` (partly right, wrong mechanism) → **the current framing below**, which the transcript analysis forced.

---

## 1. The claim (current)

A model that must solve a stream of numeric problems, where a **reusable script can be built once** (fixed write cost) and then **reused for free** on later problems of the same type, under a **scarce write budget**, is facing a **resource-allocation problem**: some types recur (a script pays off on every occurrence), some appear once (a script is wasted), and only *B* of them can get a script. The right behavior is to treat the *B* writes as a scarce budget and spend them on types that will recur.

**LLMs don't do this. They treat `write_script` as "how I solve this one hard problem," build eagerly on first sight, and never reserve budget — even when told some types recur.** The failure is not that they gather too little information before committing (that was our previous, wrong framing); it is that **they never frame the task as allocation at all.** This is a **recognition/framing failure**, not a value-of-information failure.

**Open question (the next experiment):** is this universal across the capability ladder, or does some model spontaneously recognize the allocation structure?

Positioning: the lab's **recognition axis** (does the model recognize the structure of the task?), not the crowded VoI / premature-commitment / ski-rental slice. Per [[online-investment-novelty-verdict]]: do **not** headline "uniform across capability" (unproven) or "premature commitment" (Mehta owns the phrase).

---

## 2. Setup (stochastic design + exact reference)

Full detail in `docs/online-tool-investment-stochastic-design.md` and `docs/same-info-optimal-dp.md`.

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

## 4. What's next — the capability axis (THE experiment)

**Question:** does **any** model spontaneously recognize the allocation structure, or is eager open-loop building universal? Operationally: **does build lateness stay 0 as capability rises?**

**Why this is #1:** it turns "Haiku does X" into a phenomenon, and it's the one question the neutral setup answers directly with no intervention. Two capability ladders, doing different jobs — **they don't substitute**:
- **Qwen-Coder ladder = the slope.** Many cheap points on the GPU box ([[oss-gpu-box-access]]), one controlled family, no per-point cost pressure. At small sizes it's *cleaner* than Claude — models can't hand-solve uniform-hard, so building is forced and the only DOF is timing.
- **Opus = the frontier anchor (non-substitutable).** The universality claim lives or dies at the frontier — a reviewer will say "a frontier model would recognize it," and Qwen-72B can't answer that; Opus can. And our own line says the interesting recognition effects are **frontier phenomena** that the Qwen ladder tops out below ([[creator-frontier-inversion-cross-family]], [[creator-qwen3b-floor-is-curiosity-gate]]). It pays off either way: Qwen flat → Opus upgrades "small models fail" to "everyone fails"; Qwen rising → Opus is the payoff point (does the trend continue or plateau?).
- **Sonnet dropped** — a redundant mid-ladder point the Qwen sizes already cover.

**Design:**
- **Arm:** A0 (neutral prompt). The claim is *spontaneous* recognition, so no disclosure. (A1 already known not to move Haiku; becomes a follow-up only for a model that shows nonzero lateness under A0.)
- **Pool:** uniform-hard N=8 (`lcg, modpow, continued_frac, crt_solve, josephus, quadratic_map_mod, xorshift_steps, matrix_power_mod`), MAG=100. All a_hand=0 → removes the hand-solvability confound so the *only* DOF is when/which to build.
- **Params:** T=60, B=3, cap=3, g=1, token_cap=300k, max_tokens=4096.
- **Seeds:** **shared across all models** (paired design — identical streams, so the only variable is the model). Directories keyed on model (`runs/arm_capability/{model}/seed_X`), seed range shared (e.g. 3000–3011). 12 seeds/model (~36 builds) — fidelity is near-deterministic, so this catches even a ~10% reserve rate; expand only on a signal.

**Primary metric — behavioral fidelity, with a 3-way outcome taxonomy** (needed so we don't misread capability as recognition):
1. **Eager (lateness 0):** built on first sight → the failure.
2. **Reserved (lateness > 0):** saw the type, chose hand/skip, built on a *later* sighting → the recognition signal.
3. **Hand-solved throughout (never builds):** capability, *not* recognition — reported separately. On uniform-hard this should be ~0; if a strong model cracks e.g. `josephus` by hand it lands here, and if this creeps up we bump MAG to keep building forced.

**Predictions:** flat lateness 0 across all → **universal recognition failure** (clean, strong headline). Lateness rises with scale → recognition **is** capability-graded (different, also-good framing). Higher models hand-solve instead of reserve → capability substitutes for allocation (bump MAG).

**Scope — fidelity only.** The regret/π\* layer needs per-model A0 constants (R,λ,C,r,h); we only have Haiku's. Defer per-model A0 + regret to a small Phase 2 (~$5/model), run only if fidelity shows something worth pricing.

**Cost:** Haiku fresh A0/uniform-hard ~$6 (existing Haiku runs are A1 or the N=12 pool — not strictly comparable, so re-run for a clean shared-seed anchor); Opus 12× ~$2.5 ≈ $30; Qwen ladder ~free on the GPU box (wall-clock only). Spend-guarded + resumable (cached seeds free), hard cap ~$55.

**Harness:** generalize `arm_a1_announce.py` → `arm_capability.py`: `--model` arg, `set_profile(model)`, per-model price multipliers in `cost_of` (Haiku 1/5/0.1/1.25 → Sonnet 3/15/0.3/3.75 → Opus 5/25/0.5/6.25), `announce_recurrence=False`, shared seeds, per-model dirs, fidelity report per model + a comparison table. Qwen runs via the Ollama/OSS path on the A10 box.

---

## 4b. Abstraction-gap control — IN PROGRESS (result of first run INVALIDATED)

The "urn/balls" isomorph (`urn_session.py`, `runs/urn_haiku/`): the byte-identical decision to the tool task, stripped of the tool cover story. Balls of clear colors are drawn from the SAME stochastic streams (paired by seed 3000–3011, uniform-hard N=8, g=1, T=60, B=3); KEEP collects the current ball + all future same-color (= build + reuse), PASS/never-keep scores 0 (= hand-solve at a_hand 0). Keeps map onto `model_builds`; reference π\* is the same `exact_dp`. A0 information (told T, B; not N or which colors are hot). Interactive turn-by-turn (matches the tool session's online mode).

The first run had an **observation-parity bug** (kept-color recurrences *silently skipped* → gappy < T stream, lost horizon), now **fixed**: kept colors are auto-collected IN PLACE and reported to the model ("another green ball → auto-collected, +1"), and the system prompt states the bag is unchanged and all T draws happen. The model observes exactly the same T-length stream as the tool player — never effectively >T or <T.

**Info condition = A1, not A0.** The urn prompt ("a bag of colored balls; you don't know the counts"; KEEP collects all future same-color) *discloses that colors recur* — so it is A1-equivalent (if anything more disclosing). The matched tool arm is therefore **A1**, and we already have it: the A1 announce run is Haiku, uniform-hard N=8, g=1, T=60, B=3 on seeds 2000–2011 — identical stream params. So the paired comparison runs the urn on the SAME seeds 2000–2011 and scores the existing A1 tool sessions for regret.

**PAIRED result (Haiku, seeds 2000–2011, uniform-hard, A1 disclosure both sides, identical streams; urn $0.12):**

| metric | A1 tool (coding framing) | urn (allocation framing) |
|---|---|---|
| first-sight | 100% (35/35) | 53% (19/36) |
| mean lateness | **0.000** | **0.86** (max 6) |
| regret vs π\* | **1308 ± 520** | **640 ± 285** |
| traps/seed | — | 0.92 (π\* 0.75) |

**Read — real but PARTIAL (earlier "H2 confirmed / competence exists" was too strong):** removing the coding framing, on identical streams, roughly HALVES the misallocation (regret 1308→640) and makes Haiku reserve budget (lateness 0→0.86, first-sight 100%→53%); transcripts show genuine frequency-tracking + use of the auto-collect feedback. So framing matters a lot — but it is **not a clean dissociation**: (1) urn regret is 640, not ≈0, and Haiku builds MORE traps than π\* (0.92 vs 0.75), so the tool failure is **partly framing, partly a genuine allocation weakness even in the abstract** (ceiling only partially met). (2) **Regret is noisy + seed-dependent** — urn regret was −30 on seeds 3000–3011 vs +640 on 2000–2011 (both n=12); **lead with LATENESS** (stable: 0.75/0.86 vs tool's 0.000), not the regret level. (3) Even in the urn Haiku sometimes keeps on first sight.

**Defensible claim:** the tool/coding framing SUBSTANTIALLY suppresses the allocation reasoning Haiku demonstrably can do — a large partial effect, not a switch.

**Opus urn (seeds 2000–2011, same streams; $0.54)** — urn competence is CAPABILITY-GRADED:

| metric | Haiku urn | Opus urn |
|---|---|---|
| first-sight | 53% | **8%** (3/36) |
| lateness | 0.86 | **1.14** |
| regret vs π\* | 640 | **−655 ± 389** (beats π\*) |
| traps/seed | 0.92 | **0.33** (π\* 0.75) |

Opus essentially aces the urn — waits for a repeat before keeping, keeps almost no traps (0.33 ≪ π\*'s 0.75), and beats π\* on realized g=1 streams by being more trap-averse than the Bayes-optimum. **Ceiling check strongly passes for Opus** (unlike Haiku's partial pass), so for Opus the abstraction-gap is a *clean* dissociation-in-waiting. **This reframes/subsumes the (crowded) capability axis:** if Opus is eager in the *tool* task (prior constructed-design: bait 20/20, lateness 0), the recognition gap *widens* with capability — the stronger model has MORE allocation competence for the tool framing to suppress. The claim becomes "tool framing suppresses allocation reasoning even in a model that demonstrably has it," which the urn supplies the competence baseline for. Caveats: regret noisy (don't headline −655; lead with first-sight 8% / traps 0.33); **no paired Opus tool baseline yet** (need Opus A1 tool on seeds 2000–2011, ~$6, to complete the Haiku/Opus × urn/tool 2×2).

Design caveats / next: R0 removes three things at once (surface, act-conflation, computation load) — R1 (declarative-lock-in) isolates which, deferred; Opus A1 tool run completes the paired 2×2; more seeds for a stable regret level. See [[abstraction-gap-urn-haiku]].

## 5. Deferred experiments (do NOT run without go-ahead — [[no-auto-reps]])

- **Cost-regime / m\* sweep.** Vary R/λ so building is *not* a per-instance win (m\*>1); show the eager failure persists where "build-everything" is genuinely wrong — kills the degenerate-cost critique. Mostly analytic, ~$10. *(The other original "existential check"; run alongside or after the capability axis.)*
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

**Strategic upshot:** lead with the amortization/investment mechanism (the open conjunction); capability axis is secondary vs a contested field; abstraction-gap is a borrowed method in a novel domain; reference is an explicit oracle. Nothing we've *done* collides — the exposure is entirely in the *framing* of the two planned experiments.

## 6. Scope / honest limits

- The claim is about a **specific decision structure** — allocating a scarce, reusable, irreversible build budget under recurrence — not "LLMs are bad at all decision-making."
- It rests on the reference being well-defined and computed. **Done:** the exact belief-state DP is the same-information optimum, certified lossless at cap=3.
- Regret is **distribution-dependent** (g=0 defused it). The distribution-independent claim is the **open-loop policy** (fidelity), which is what the capability axis measures.

## 7. Surviving capability signals (secondary)

- **Hand-solvability threshold:** stronger models hand-solve more → fewer builds triggered (same disposition, shifted threshold). Per-model A0: Haiku band m≈10, Sonnet m≈1000.
- **Tool generality:** stronger models write broader, more reusable tools (Sonnet's `poly_eval_exact`, built for a one-off, reused on 14 problems; Haiku builds narrow single-purpose tools).

## 8. Instrument (built, validated)

`family_kit.py` (exact-integer families, A0-calibrated per model), `stream_builder.py` (`StochasticStreamSpec`), `exact_dp.py` (the reference π\*), `run_stream_session.py` (persistent session, write budget, non-binding token cap), `skirental_scorer.py` (cost model + decision classification + `exact_pistar_report`), `a0_oracle_gap.py` (per-model hand-vs-build calibration), `poc_haiku.py` / `arm_a1_announce.py` (spend-guarded, resumable harnesses; reuse=1.00 confirmed).

**Next step:** build `arm_capability.py` and run the capability axis (Haiku + Opus first for the frontier verdict; Qwen ladder for the slope). Nothing runs without explicit go-ahead.
