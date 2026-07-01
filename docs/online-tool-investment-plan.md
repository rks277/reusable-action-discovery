# LLMs Fail the Optimal Online Tool-Investment Policy

**Status:** Chosen reframe, 2026-07-01. **Supersedes** the *capability-graded recurrence-recognition* framing in `tool-amortization-plan.md` (and the earlier `net-negative-plan.md`), which the Haiku-vs-Sonnet runs **falsified** — the miscalibration is *not* capability-graded; it is uniform. All experiment infrastructure (family_kit, stream_builder, run_stream_session, skirental_scorer, per-model A0) carries over unchanged.

## Thesis

Solving a stream of problems where a reusable tool can be **built once** (fixed write cost) and then **amortized** over future problems, under a **scarce write budget**, is an **online investment problem under an unknown reuse distribution**. There is a well-defined *optimal online policy* for this setting — one that **gathers evidence** (waits for a type to recur) before committing an irreversible write.

**LLMs systematically deviate from it: they build eagerly on first encounter with a hand-hard problem, without waiting for evidence of recurrence.** This appears **uniform across capability** — Haiku and Sonnet both do it — so the contribution is not a capability law but a **general failure of online decision-making / value-of-information** in agentic tool building.

## Evidence so far (Claude, this benchmark)

- **Haiku** — eager builder: tools the first ~4 distinct types it sees, on first sighting; with a one-off guaranteed early (oneoff_per_block) + budget 4 it wastes writes on one-offs and never tools all recurring classes → solve loss (easy 18/24). (`haiku-eager-builder-result`)
- **Sonnet** — in the **easy** condition looked judicious (built **0** one-off tools, 24/24) — but that was an **artifact**: easy one-offs are hand-solvable for Sonnet, so its build-when-I-can't-hand-solve policy never fired on them. In the **hard** condition (one-offs *also* hand-hard) Sonnet builds a tool for the **first one-off it sees, on first sight, in all 3 seeds**. (`sonnet-builds-oneoffs-when-hard`)
- **Conclusion:** both models "build when they can't hand-solve," on first sight, **without waiting for recurrence evidence**. Neither does online learning. Capability changes only (a) the **hand-solvability threshold** (which problems trigger a build) and (b) **tool generality** (Sonnet's tools are broader and reused more) — *not* the build-timing disposition.

## Why "the model didn't know the distribution" is NOT a valid defense

The setting *is* online learning under an unknown reuse distribution — the standard online-learning regime. Optimal policies for it (ski-rental competitive ratios; randomized / ML-augmented ski-rental; no-regret online learning) achieve strong guarantees **without** knowing the distribution a priori — they adapt to the observed sequence. So the optimal-online benchmark operates under the **same missing information** as the model. If the model does worse than the online optimum, that is a **genuine deficiency**, not an artifact of missing information — because the benchmark shares that handicap and still wins by waiting/adapting.

(Consequently an "announce the recurrence structure" control is **not required** to defend the claim — the online-optimal benchmark already answers it. Announce remains an interesting *secondary* probe: does telling the model the structure fix the over-eagerness?)

The precise, defensible claim: models are not doing *nothing* — they run *some* reactive heuristic — but that heuristic **deviates from the optimal online policy in a specific direction: over-eager irreversible commitment instead of value-of-information gathering.**

## What's needed to make it solid

1. **The optimal online policy (the benchmark) — now central, previously deferred.** Characterize/bound the best achievable online policy under unknown reuse: ski-rental / rent-vs-buy with an unknown horizon distribution (competitive-ratio and/or no-regret). Plug it into `skirental_scorer` (the pluggable optimal slot) and report each model's **regret vs the online optimum** — the headline metric. If the true optimum is intractable, use a strong reference policy (e.g. deterministic 2-competitive ski-rental) and report regret vs that as a **lower bound** on the true gap.
2. **A larger test.** The current stream (N=24, 5 members/class) gives almost no room for online learning — the evidence window is tiny. Scale up: **longer streams (≈50–150 problems), more classes, larger and varied reuse horizons, more one-offs**, so gathering evidence genuinely pays and the model-vs-optimal gap is measurable. Vary **P(recurs)** and horizon to map *when* waiting matters (and confirm the optimum does wait in those regimes).
3. **More models.** Opus + an OSS ladder (Qwen / Llama on the Oracle A10 box). Because the claim is **uniform failure**, every model failing the same way *strengthens* it. If some model *does* wait, that itself is a finding (the frontier of online-decision competence).

## Surviving capability signals (secondary, not the headline)

- **Hand-solvability threshold:** stronger models hand-solve more → fewer builds triggered (same disposition, shifted threshold). Requires per-model A0 to locate (done: Haiku band m≈10, Sonnet band m≈1000).
- **Tool generality:** stronger models write broader, more reusable tools (Sonnet's `poly_eval_exact`, built for a one-off, was reused on 14 problems across families; Haiku builds narrow single-purpose tools).

## Scope / honest limits

- The claim is about a **specific decision structure** — online irreversible investment under unknown reuse — not "LLMs are bad at all decision-making."
- It stands or falls on the **optimal-online benchmark being well-defined and computed** (or tightly bounded). This is the analytical crux.
- Irreversibility matters: waiting costs the early instances you pass (can't go back). The value of waiting scales with the reuse horizon; the larger test must use horizons where waiting demonstrably pays.

## Related work / novelty (confirm via focused scan before committing)

- LLM decision-making under uncertainty — bandits/exploration, optimal stopping (secretary problem), test-time-compute allocation. Some prior work exists; **confirm the "fail to gather information before an irreversible tool-building investment" angle is fresh.**
- Ski-rental / rent-vs-buy applied to **LLM tool creation** — unclaimed per the two prior scans.
- Positioning: *LLMs fail online value-of-information in agentic tool building, uniformly across capability* — with model-vs-online-optimal regret as the quantitative result.

## Instrument (built, validated)

`family_kit.py` (owned exact-integer procedures; recurring families + easy/hard one-off pools, A0-calibrated per model), `stream_builder.py` (arrival incl. `oneoff_per_block`; controllable horizons / one-offs / difficulty), `run_stream_session.py` (persistent session, write budget, non-binding token cap), `skirental_scorer.py` (cost model + decision classification + regret; **needs the online-optimal policy plugged into its pluggable slot**), `a0_oracle_gap.py` (per-model hand-vs-build calibration).

**Recommended next step:** characterize the optimal online policy (item 1) — it is simultaneously the benchmark *and* the rebuttal to the "missing information" critique — then scale the test (item 2), then the model ladder (item 3).
