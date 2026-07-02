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

## Design log — what we've tried (2026-07-01)

The instrument is built and validated; the open work is choosing a **regime** where building is unambiguously right for recurring types and unambiguously wrong for one-offs, so an eager build is faultable.

1. **Instrument validated.** Family kit expanded to ~30 exact-integer families with a `GEN_CLUSTER` constraint (no script reuses across classes); exact arbitrary-precision int grading (float sig-figs gave false positives on >16-digit golds); **budget-constrained knapsack** optimum in the scorer (per-class m*<1 means "build everything" is per-class optimal → real scarcity is the write budget, so the optimum builds the top-B classes by gain); `chat_tools` caching fix (see `chat-tools-caching-fix`) — N=80 dropped 2.5M→60k tokens (~40×).
2. **Clean Haiku demonstrations.** eager build-on-first-sight, **mean_lateness = 0.00** (zero evidence-gathering — the thesis), wastes 1–2 of its 6 builds on one-off traps, starves 1–2 recurring classes. Reuse is **perfect** (1 tool/class, 0 rebuilds) so the failure is cleanly isolated to **build-timing/allocation**. (`tool-investment-pilot-haiku`)
3. **a_hand≈1 regime pivot.** Made every problem hand-solvable (no "couldn't do otherwise" excuse) but token-tedious. **Complexity-confound discovered + fixed:** the first a_hand≈1 probe looked optimal but was an artifact — recurring were all complex, one-offs mostly simple, so Haiku's real heuristic ("build multi-step, hand-solve simple") *aligned* with recurrence. Fix = **complexity-match** the pools (equal-complexity recurring + one-offs → complexity gives no signal; recurrence detectable only by waiting).
4. **The token channel is a DEAD END (key negative result).** Under caching, building never pays on tokens: reuse ≈ hand (both ~330–490 tok/problem, within noise), and the one-time **build cost dominates** (~5000 tok/build-problem, inflated by early-session *uncached* input — system+tools = 945 tokens is below the cache minimum, so it can't be pre-warmed). Net: build-a-recurring-class ≈ 8700 vs hand-it ≈ 4440 → building costs ~2×. So a token budget / "minimize tokens" objective would say **never build** — it cannot incentivize building. `run_and_submit` (collapse reuse to 1 turn) was added to try to rescue this, but **the model ignored it (0 uses)** and it wouldn't have fixed the build cost anyway → **removed**.
5. **Conclusion → building must be justified by ACCURACY, not tokens.** That requires **a_hand < 1**. The clean, non-degenerate design is **moderate a_hand (~0.6)**: building buys real accuracy on recurring types, hand-solving is a weak (quantified) fallback. Headline = **build-decision regret** vs the budget-constrained online optimum (NOT solve — solve is noisy because hand-solving partially recovers starved classes).
6. **The COMPOSABILITY pivot (key structural finding).** The `GEN_CLUSTER` "distinct direct operation" standard is **too weak** — verified empirically: in a moderate 4+5 run, Haiku built one `dot_product` tool and **reused it across weighted_sum + five arithmetic one-offs** (diff_of_products, pairwise, matvec, quad, sum_of_squares). So the entire **multiply-add "arithmetic web" (~19 families: products, dots, polynomials, matvec, pairwise, sum-of-powers) collapses into a SINGLE reusable skill** — those families are not independent classes. Real requirement: classes must be **genuinely non-composable** — operations no single tool can be stretched across. Those are the *iterative / number-theoretic / digit / order* primitives. Haiku's error profile is narrow (only big-mult, gcd, iterated-mod-product, big-division/modulo, and long digit procedures are hard), so moderate-**and**-non-composable families are scarce and must be tuned individually.
7. **Arrival fix.** Early runs degenerated because one-offs arrived *after* the budget was spent on recurring classes (forced "correct-skip" for the wrong reason). Fixed with **`random_oneoff_early`**: fully random order, but ≥1 one-off guaranteed in the first 6 slots — the eager-vs-wait decision is forced early, while budget is still free.

## Result so far — clean single-seed Haiku demonstration (2026-07-01)

`runs/stream_disposition_20260701_171639`. **5 recurring + 5 one-off, all non-composable, difficulty-matched** (a_hand 0.30–0.90, pool means 0.63 vs 0.65), budget = 5, `random_oneoff_early`, N=80. Recurring: euclid_gcd_chain, factorial_mod, lcg, int_div_sum, count_inversions. One-offs: kaprekar_routine, look_and_say, luhn_sum, continued_frac, mod_pair_sum. Calibrated in `runs/a0_moderate_tuned`.

- **Cross-family reuse = ZERO** (verified: each of 5 built tools maps to exactly one family) — the non-composable redesign holds, so the failure is genuine, not a reuse artifact.
- **mean_lateness = 0.00** — every build on first sighting; no evidence-gathering. *The thesis.*
- Haiku **built a tool for `continued_frac` at slot 0** — the very first problem, a **one-off** — an irreversible commitment with zero recurrence evidence. That wasted 1 of 5 budget slots.
- When **lcg** (a recurring class) first appeared at slot 10, budget was exhausted → **lcg wrongly-skipped** → all 15 members hand-solved at a_hand 0.60. The entire regret (**1870** vs budget-optimum) traces to that one eager commitment.
- Decision tally: **4 correct-build, 1 wrongly-built, 1 wrongly-skipped, 4 correct-skip.** Within-class reuse perfect (15/15), 0 rebuilds.
- **No excuse:** continued_frac has a_hand 0.85 (trivially hand-solvable) and is non-composable — building it was pure waste the model couldn't reuse its way out of.

### 20-seed Haiku sweep (`runs/stream_sweep_haiku_n20`, all 20 OK)

The single-seed pattern holds across seeds with striking consistency (seeds vary arrival order + instance sampling; runner = `run_stream_sweep.py`):

| metric | value |
|---|---|
| **bait rate** (built ≥1 one-off) | **20/20 = 100%** (rule-of-three 95% lower bound ≈ 0.86) |
| **mean lateness** | **0.00, sd 0.00** — every build on first sight, every seed |
| one-offs built / seed | 1.0 (sd 0) — wastes exactly 1 of 5 budget slots |
| recurring classes starved / seed | 1.05 (sd 0.22) |
| solve / 80 | 74.2 (sd 6.1) |
| regret vs budget-optimum | 1152 (sd 735) |

Not a family artifact: the wasted build lands on **whichever one-off arrives early** (continued_frac, look_and_say, luhn_sum, kaprekar all appear across seeds) and the starved recurring class varies (lcg, count_inversions, factorial_mod, euclid, int_div). It is a **general disposition** — Haiku *always* commits an irreversible build to a freshly-seen one-off rather than waiting for recurrence evidence.

### Awareness arm — disclosure does NOT fix it (`runs/stream_sweep_haiku_n20_announce`)

The system prompt was given a factual note that some problem types recur (several times) and others appear once, unlabeled — *without* prescribing "wait" (so we can't be accused of leading the model). Result across 20 seeds: **bait 20/20 = 100%, mean lateness 0.00 — identical to the hidden arm.** Even told recurrence exists, Haiku still commits builds on first sight. This rules out the "it didn't know recurrence was possible" objection: the failure is a genuine **online-decision** deficiency (over-eager irreversible commitment), not a recognition gap. (Run truncated at budget-exhaustion; solve/regret are not comparable across truncation, but the decision metrics — bait, lateness — are, and they match.)

**Cross-model note (Sonnet):** hidden-arm bait was also 100% (lateness 0). Sonnet additionally *flails* — it runs existing tools on the wrong family before writing the correct one — which is why build decisions are now attributed by the **authoring event** (which class was on screen when `write_script` fired), robust to that and to truncation.

### Full capability ladder — uniform, and disclosure-immune

Each model uses its **own** difficulty-calibrated non-composable set (moderate a_hand is model-specific: Haiku 5+5, Sonnet 4+4, Opus 3+3 — raw-arithmetic families like int_div/mod_pair fall out as capability rises since long division becomes perfect; continued_frac drops for Opus as its value explodes). Arrival guarantees the first one-off within the first **B** (=budget) slots, so the bait decision is always faced *with budget free* (this fixed a false-restraint artifact where an eager model spent its whole budget on early recurring first-sightings before any one-off appeared).

**Announced arm, 20 seeds each, bait rate / mean lateness:**

| model | set | bait rate | mean lateness |
|---|---|---|---|
| Haiku | 5+5, budget 5 | 20/20 = 100% | 0.00 |
| Sonnet | 4+4, budget 4 | ~100%¹ | 0.00 |
| Opus | 3+3, budget 3 | 20/20 = 100% | 0.00 |

¹ Sonnet's announced arm predates the arrival fix; raw 18/20 with two false-restraint seeds (budget exhausted before the one-off appeared) → true rate ~100%. To be re-run under the final setup.

**The failure is uniform across the capability ladder and immune to disclosure of the recurrence structure.** Even the strongest model, told some types recur, commits an irreversible build to a one-off on first sight in every seed, with zero evidence-gathering. (Opus run cost $1.15 total — truncation at budget-exhaustion + the first-B one-off guarantee make eager seeds terminate within a handful of problems.)

## Where to go next (steps to the end goal)

**End goal:** a defensible, quantitative demonstration that LLMs deviate from the optimal online tool-investment policy — building eagerly on first sight without gathering recurrence evidence — reported as **decision-regret vs the online optimum**, ideally uniform across the capability ladder.

1. ~~Commit to moderate-a_hand regime~~ **DONE** — regime chosen, non-composable family set built + calibrated (Haiku 5+5, Sonnet 4+4 via per-model difficulty profiles), arrival fixed (`random_oneoff_early`).
2. ~~Multi-seed Haiku sweep~~ **DONE, both arms** — hidden (100% bait, lateness 0) and announced (100% bait, lateness 0). Sonnet hidden also 100%; Sonnet announced running.
3. **Definitive re-runs under the final prompt** — the sweeps above that predate the final prompt (Haiku hidden full, Sonnet hidden) should be re-run under it for a clean dataset; report realized solve too (needs full, non-truncated runs). *(gated)*
4. **Opus** (+ optional OSS ladder). Needs its **own A0 recalibration** (moderate is model-specific; int_div/mod_pair already fell out for Sonnet — expect more raw-arithmetic families to drop for Opus). *(gated)*
5. **Confirm the online-optimal reference** in `skirental_scorer` (budget-constrained knapsack is in; verify against a 2-competitive ski-rental lower bound).

Infra now in place: authoring-based build attribution, `--stop-on-budget-exhausted` (cheap truncated arms), per-seed `--session-timeout` guard, per-model difficulty profiles, `--announce` awareness arm.

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
