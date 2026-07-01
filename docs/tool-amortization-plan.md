# Tool-Building as Amortized Investment: When Should an Agent Pay to Build a Reusable Script?

**Status:** Plan / direction proposal. Written 2026-06-29 after two literature scans (see *Novelty & Related Work* below). No experiments run yet — all sweeps are approval-gated per the standing no-auto-reps rule.

**Supersedes** the net-negative / capability-graded-spiral direction (`net-negative-plan.md`), which we abandoned because the per-instance "when to call / when to stop" axis is saturated (To-Call 2605.00737, OTC-PO 2504.14870, When2Tool 2605.09252, Re-FORC 2511.02130, EDS 2602.03485). This plan returns to the repo's native thread — *reusable* tool creation (CREATOR, ToolWorld/WoodWorld build studies) — on an axis the per-instance literature structurally ignores.

---

## 1. The one-sentence idea

An agent solving a **stream** of problems can write a reusable Python script **once** and amortize it over future problems. Deciding *whether to pay the fixed build cost now*, without seeing the future problems it would serve, is an **online rent-vs-buy (ski-rental) decision under uncertain reuse**. Models solve it badly, and — our prior data suggests — **capability-gradedly**: weak models over-build, strong models under-build (hand-grind). We characterize this miscalibration against the ski-rental optimum and test what corrects it.

## 2. Why this is the right axis (and why the field missed it)

Every paper in the crowded neighborhood optimizes tool use **per instance** — "for *this* query, call or not, how many, when to stop." They all assume tools are **throwaway** (every call fresh), cost is **homogeneous** (one call = one unit, paid and gone), and there is **no amortization**. That last assumption is the white space.

The amortized question is a *different optimization*:
- **Heterogeneous cost.** Writing a script is expensive once; calling it is ~free. Per-instance call-counting (OTC) cannot represent this asymmetry — which is exactly why it's a different problem.
- **Irreversible investment under uncertainty.** You commit a scarce *write* before observing the future demand it would serve. This is the canonical **rent-vs-buy** structure (rent = re-solve each instance by hand; buy = pay the script's fixed cost once, then reuse cheaply).
- **Online, not offline.** To-Call allocates a budget *offline* (top-K over a visible set). Here the stream is unseen; the decision is sequential and irrevocable.

This is also the repo's original setting: Python scripts reused throughout a CREATOR-style exam, and the ToolWorld/WoodWorld build-vs-grind findings. We already have the harness (persistent session over N distinct problems + a write budget) and prior results (build propensity inverse in capability: Haiku 0.97 > Sonnet 0.81 > Opus 0.73). What those results lacked is a **normative yardstick** and an **economic model** of building — which ski-rental supplies.

## 3. The normative core: ski-rental as the build oracle

For a problem class `c` with per-instance hand cost `h`, per-use script cost `r` (`r << h`), and fixed build cost `C`:

- **Full-information optimum (known horizon `m` = future applicable problems):** build `c` iff `m·(h − r) > C`, i.e. iff `m > C / (h − r)`. Define the **break-even horizon** `m* = C / (h − r)`.
- **Online optimum (unknown horizon):** the classic ski-rental policy — *rent (hand-solve) until cumulative rental spend equals the buy cost, then buy (build)* — is 2-competitive; the randomized/ML-augmented variants do better given a horizon predictor. This gives a principled **"build after you've seen ≈ `C/(h−r)` repeats"** rule to score models against, even when they cannot see the future.

This is the yardstick the tool-making literature never had: it lets us label any build decision as **correct / wrongly-built / wrongly-skipped**, and compute **regret** vs the optimal policy.

## 4. Claims, scoped by ambition (same three-tier discipline as before)

1. **MECHANISM (general claim).** LLM tool-*building* is miscalibrated against the rent-vs-buy optimum, and the miscalibration is **capability-graded** — weak models over-build (spend writes on horizons that don't amortize), strong models under-build (hand-grind past the break-even horizon). Explanatory, expected to transfer.
2. **PREDICTOR / threshold (within-regime claim).** A model's *effective* build threshold `m̂` (the horizon at which it actually starts building) is measurable and predictable within a contest-math regime across a capability ladder; the gap `m̂ − m*` is the calibration error. Not a general oracle.
3. **BENCHMARK (the bridge).** A **streaming-reuse benchmark** with controllable class density, reuse horizon, build/hand cost ratio, and announce-vs-infer horizon — the instrument that lets others extend the mechanism to new domains.

If the predictor (tier 2) fails to generalize across held-out classes/models, we fall back to the descriptive mechanism + benchmark.

## 5. Metrics

1. **Build-calibration error:** signed gap `m̂ − m*` between a model's effective build threshold and the break-even horizon; and **regret** = excess total cost vs the online ski-rental optimum (token-denominated, compute-matched).
2. **Decomposition:** wrongly-builds (built when `m < m*`) vs wrongly-fails-to-build (hand-ground when `m > m*`). This is the part the field has never measured.
3. **Reuse-vs-rebuild:** of problems where a previously-built script applies, the fraction the model actually **reuses** vs **redundantly rebuilds / re-derives by hand**. A pure efficiency leak every prior work assumes away (they dedup offline or store-everything).
4. **Capability grading:** all of the above across Haiku / Sonnet / Opus, plus the OSS ladder (Qwen 3B–72B, Llama-70B) on the Oracle A10 box — to test whether the *sign* of the miscalibration is a frontier effect or holds down the ladder.

## 6. Benchmark design

A persistent session presents a **stream** of distinct problems under a hard **write budget** (scripts the agent may author). Structure we control:

- **Class membership.** Some problems belong to a recurring class solvable by one parameterized script (e.g. "evaluate this finite product mod p", "count lattice paths with these constraints"); others are one-offs. The script for a class generalizes across its members by argument, not copy-paste.
- **Reuse horizon `m`.** Members per class, and their **arrival pattern** (front-loaded: repeats come early, building pays; back-loaded: repeats come late, online policy must gamble; interleaved).
- **Cost ratio `C/(h−r)`.** Tune so `m*` lands at an interesting, controllable value (e.g. `m* ≈ 3`), measured empirically per model from hand-cost and build-cost calibration runs.
- **Awareness.** Announce the class structure / horizon ("problems of this kind recur ~`m` times") vs require the model to **infer** it from the stream. This is the awareness lever from the E0 result (announced budget moved Haiku +2.6 solve) imported into the build decision.
- **Grading.** Exact-answer (sig-fig / integer) as in the existing disposition harness; cost in spent tokens (uncached input + output) and writes consumed.

**Contamination guard.** Per the capability-graded-contamination finding, problem *content* must be fresh for every model (strong models memorize old contest answers, which would collapse hand-cost and fake "code-unnecessary"). Use synthetic class-structured problems with randomized parameters, not historical AIME.

## 7. Experiments (all gated — propose, cost, wait for go-ahead)

- **A0 — Instrument + sanity (no capability claim yet).** Build the streaming-reuse benchmark; calibrate `h`, `r`, `C` per model; verify a **meaningful, compute-matched** gap exists between hand-only and build-optimal policies. *This is the rebuttal to TroVE-compute-matched (2507.22069): if building doesn't pay even under the oracle once compute is matched, the whole premise dies — so test it first.*
- **A1 — Capability ladder (headline).** Build-calibration error and its wrongly-built/wrongly-skipped decomposition across Haiku/Sonnet/Opus (+ OSS ladder). Does miscalibration **flip sign** with capability (weak over-build / strong under-build)?
- **A2 — Reuse-horizon dose-response.** Sweep `m` around `m*`; does each model's build rate track a threshold, and where does its `m̂` sit vs `m*`?
- **A3 — Awareness lever.** Announce horizon/class structure vs infer. Does telling a strong model "this recurs `m` times" correct under-building? (Ties the E0 awareness result to the build decision.)
- **A4 — Reuse-vs-rebuild.** Measure redundant rebuilds across the ladder; relate to capability.
- **A5 — Intervention (optional).** Can a build-and-reuse demo / scaffold move the decision toward the optimum, à la the playground demo that recovered Opus's build rate?

## 8. Novelty & related work (verified against two scans, 2026-06-29)

**The joint claim — tool creation as *online investment under uncertain reuse with a write budget*, scored against ski-rental, decomposed into wrongly-built/wrongly-skipped, and graded by capability — is unclaimed.** Each ingredient alone is owned; the combination is open.

| Work | What it owns | Why it is NOT this |
|---|---|---|
| **LATM** (Cai 2023, 2305.17126) | "Build once, reuse to cut average cost" via maker/user split | Amortization as a *fixed pipeline rationale*; build always happens; no decision, no uncertainty, no budget |
| **CREATOR** (Qian 2023, 2305.14318) | Disentangle tool creation from use; transfer lifts accuracy | Always creates; about accuracy, not build economics |
| **TroVE** (Wang 2024, 2401.12869) | Runtime import/create/skip branch; trim toolbox | Heuristic skip, no investment threshold, effectively offline over a fixed set |
| **TroVE compute-matched re-eval** (2507.22069) | Skeptic: toolbox gains ~vanish under matched compute | The threat we must out-argue — A0 compute-matches by construction; regret is compute-denominated |
| **UCT** (2602.01983) | Online build-on-retrieval-miss + offline consolidation | Build trigger is a heuristic miss, not calibrated; no capability analysis, no budget |
| **Voyager** (Wang 2023, 2305.16291) | Lifelong online skill library | Stores *every* verified skill; no budget, no when-to-build decision |
| **CRAFT** (Yuan 2024, 2309.17428) | Offline task-specialized toolset + retrieval | Built in a setup phase; no online decision |
| **Agentic Plan Caching** (2506.14852) | Online pay-once-reuse-cheap caching | Caches automatically on every success; no budgeted decision |
| **PETS-Online** (2602.16745) | Genuine online streaming compute allocation | Per-problem **independent**; nothing built/amortized across problems |
| **When2Tool** (2605.09252), **To-Call** (2605.00737), **OTC-PO** (2504.14870) | Per-instance call-or-not (mis)calibration; offline top-K budget | Per-instance, throwaway calls — the axis we are explicitly *not* on |
| **Ski-rental / rent-vs-buy** (SOAC NeurIPS'25; ML-augmented 1903.00092, 2002.05808) | The canonical online amortization-under-uncertainty primitive | Never applied to LLM tool/skill creation — **the hook we adopt and cite** |

**Do NOT claim:** "first to amortize tool cost" (LATM), "first build-or-not branch" (TroVE), "online skill library" (Voyager/UCT), or any per-instance calibration result. **Lead with:** capability-graded build *disposition* + reuse-vs-rebuild behavior (empirical headline), framed through the ski-rental investment lens (the analytical novelty), with explicit compute-matching (the skeptic rebuttal).

**Watch (fresh, could narrow the gap):** UCT (2602.01983), When2Tool (2605.09252), Library Drift (2605.19576, governs *retirement* not creation), Agent-Skills surveys (2602.20867, 2605.07358). Re-check v2s before freezing language.

## 9. Open risks

1. **Building genuinely may not pay (the 2507.22069 risk).** Mitigated by gating everything behind A0's compute-matched oracle gap. If A0 is null, pivot to the *disposition characterization* (do models build when they shouldn't / refuse when they should) as a pure descriptive result, which stands regardless of net benefit.
2. **Threshold may not generalize across classes (tier-2 risk).** Fall back to mechanism + benchmark.
3. **Sign of the capability effect.** Our build-inverse data is from game worlds (ToolWorld/WoodWorld) and may not transfer to math streams; A1 tests it directly rather than assuming it.

## 10. Relationship to existing repo assets

- **Harness:** extend `scripts/creator/tool_disposition_benchmark/` (persistent session over N distinct problems + write budget already exist) with class-structured streams, arrival-pattern control, and the ski-rental scorer.
- **Prior findings to unify:** build-proportion inverse in capability; playground-build-demo recovers Opus; recognition-vs-acquisition bottleneck (ToolWorld vs WoodWorld). This plan gives those the missing normative yardstick and economic model.
- **Related docs:** `lit-review-tool-creation-disposition.md`, `why-opus-builds-less-experiments.md`, `disposition-bench.md`.

**Recommended first move:** A0 only (instrument + compute-matched oracle gap). Single small calibration pass, no capability sweep, no reps — propose cost and wait for go-ahead before anything multi-rep.
