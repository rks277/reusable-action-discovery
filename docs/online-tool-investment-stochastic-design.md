# Stochastic online-tool-investment design — working notes

Companion to `online-tool-investment-plan.md` and `online-tool-investment-related-work.md`.
Last updated 2026-07-02.

## Design

Problems are i.i.d. draws from a distribution over **N** types; session length **T**; write budget
**B** scripts. **B "hot" types** (`p_i > 0.2`) + **N−B "trap" types** (`p_i < ~0.05`), rates
well-separated. Per set, the **rate multiset is fixed but which family gets which rate is
randomized** (no memorization; forces online identification). `#hot = B` by design → budget exactly
covers the hot types. Building a tool buys **accuracy** (a_script≈1 ≫ a_hand); the token channel is
degenerate under caching, so **the write budget is the scarcity**.

**Model knows:** only {N, T, B} + "problems come from a distribution over N types." NOT the p_i,
which types are hot, or that #hot = B.

## Reference policy: π\* — the exact same-information optimum (DP)

**π\* is the EXACT Bayes-optimal policy against the model's own information** — knows only {N, T, B} +
"a distribution over N types", via an **exchangeable Dirichlet(α) prior** (no idea about the
frequent/rare split; infers everything from the stream, like the model). It is computed by an exact
finite-horizon belief-state DP — **not** a relaxation. **Implemented** in `exact_dp.py` (`ExactDP`);
wired into the scorer as `skirental_scorer.exact_pistar_report` (called from `score_run`). Full
construction, invariant, and optimal-policy structure in **`docs/same-info-optimal-dp.md`**.

**The claim** (headline): a policy with *identical information* to the model extracts far more value
by **gathering evidence before spending an irreversible build** — it reserves the first sighting of a
type and builds on the **2nd sighting** (demonstrated recurrence), spending its scarce builds on the
demonstrated winners; the model builds the first B distinct types on sight (lateness 0). Because π\*
is the exact same-info optimum, **(π\* − model) is the honest same-information regret** (not a bound).
On the 3-seed Haiku dry-run: **exact regret ≈ 2293/seed** (positive on all seeds), π\* builds **0
traps/seed** (never baited — it waits) vs the model's **1 trap/seed** built on sight.

> **Tractability (the count-cap).** The uncapped joint DP is intractable at N=12,T=60 (~10¹³ states).
> `ExactDP` caps *unbuilt* counts at `cap` (force-build a type on its (cap+1)-th sighting): this is a
> valid same-info policy, and a K-sweep certified it is **lossless** here — `V(K=3)=V(4)=V(5)=V(6)`
> exactly (and matches the uncapped exact at T≤40). So `cap=3` is the exact optimum at ~540k states /
> ~3s, pure Python. The cap is a *state-space bound*, NOT the build rule (π\* builds on the 2nd
> sighting, well inside it; the cap-forced 4th-sighting build essentially never fires). Re-run the
> sweep to re-certify if the cost model or α changes.

> **Retired — the old Whittle π\*.** `pi_star.py`'s Whittle-index construction (per-type DP under a
> constant price λ\*) is superseded. At N≈12 the constant-price relaxation *over-builds* (builds at
> first sight ≈44% of the time) and is structurally unable to match the exact optimum, whose decision
> is context-dependent (on remaining budget and competing types). Its old numbers (π\* 1285 vs model
> 209 ≈ "6×"; lateness ~1.5; 0.67 traps/seed) **understated** the regret — the exact DP roughly
> doubles it (≈2293 vs ≈1263) and drives π\* traps to 0.

**wait-one-repeat is ≈ the same-info optimum, NOT a cheat.** We earlier dismissed wait-one-repeat
(build first B to reach the 2nd sighting) as using extra info ("rare = one-off"). The exact DP shows
building on the 2nd sighting *is* the same-information optimal rule — so wait-one-repeat is a good
cheap proxy for π\*, not an over-informed comparison. (This also retires the earlier
mixture/hierarchical-prior idea — unnecessary.)

**Per-set clairvoyant** (knows realized counts, builds top-B at first sight; `pi_star.clairvoyant_builds`)
is the loose upper bound. (clairvoyant − π\*) = intrinsic price of online uncertainty (small: ~150–350
on the dry-run seeds); (π\* − model) = the honest reported regret.

## Platform (decided 2026-07-02)

- **Main platform: Qwen-Coder** (a few notable sizes, NOT a full ladder — phenomenon looks uniform).
  Rationale: the **fine-tuning / mitigation arm requires open weights** (can't train Claude), and
  weak models make difficulty trivial to hit. Coder variants clear the a_script (tool-writing) bar.
- **Haiku = frontier proof-of-concept** so the finding isn't dismissed as a small-model artifact.
  Framing: characterize + fix on open models, confirm the phenomenon persists at the frontier.
- Established already (OLD constructed design): Haiku/Sonnet/Opus all **100% bait, lateness 0, in
  hidden AND announced arms** — disclosure-immune. The stochastic redesign re-measures this cleanly
  with a proper regret metric.

## Family pool status (A0-validated, `runs/a0_*`)

Bars: a_hand low (building buys accuracy) + a_script ≈ 1 (model writes a working tool).
- **Haiku: ~18 usable families** — the original 13 (9 strong + 4 hardened) + all 6 new
  (matrix_power_mod, crt_solve, josephus, xorshift_steps, linrec_mod, quadratic_map_mod, all
  a_hand≈0). No shortage; Haiku PoC is ready.
- **Cross-model uniform pool** (only needed if we want a hard frontier set): **7 solid** (lcg,
  quadratic_map_mod, xorshift_steps, factorial_mod, modpow, crt_solve, look_and_say) → ~9–11 with
  small fixes. **Key rule:** uniform hardness needs *individually hard per-step ops* (big
  multiply/mod/bit-ops); many-small-steps families (josephus, small-coeff linrec) get hand-grinded by
  Opus.
- Difficulty is per-model only for the frontier; on Qwen almost anything is hard-by-hand, so the pool
  is unconstrained there.

## Next steps

1. ~~Stochastic arrival mode in `stream_builder`~~ **DONE** (`build_stochastic_stream`, trap-early knob).
2. ~~Reference policy π\*~~ **DONE, and upgraded from Whittle to the EXACT belief-state DP**
   (`exact_dp.py`): canonical (exchangeability-reduced) DP, validated lossless against the brute-force
   vector DP, with the count-cap certified converged (cap=3 exact, ~3s). Supersedes the retired
   Whittle `pi_star.py`. See `docs/same-info-optimal-dp.md`.
3. ~~Wire π\* + clairvoyant into the scorer~~ **DONE** (`skirental_scorer.exact_pistar_report` +
   `model_builds_from_actions`, called from `score_run`, keyed off `meta.json`). Values BOTH model and
   π\* analytically over full realized sizes (truncation confound gone); bait keyed off `role=="trap"`;
   reports `regret = value(π\*) − value(model)` + `clairvoyant_gap`. `score_run(pistar_dp=...)` lets a
   sweep build the DP value table ONCE (design constant of {N,T,B,cap,costs}) and reuse it. Re-scored
   the 3-seed dry-run: exact regret ≈ 2293/seed (pos 3/3), model 1 trap/seed & lateness 0 vs π\* 0
   traps & lateness ~0.67.
4. **Haiku PoC run** on the new design (the frontier anchor). ← NEXT
5. **Qwen-Coder main runs** (few sizes) + the **fine-tuning mitigation** arm.

## Open items

- Pin π\*'s exchangeable prior; confirm prior-insensitivity.
- `matrix_power_mod` cover fixed (was "×K times" ambiguous, tanked Opus a_script); re-verify if we
  use the frontier uniform pool. `continued_frac`/`linrec_mod`/`josephus` only matter for that pool.
- Keep R/λ set so a single build never pays (one-shot wasteful) but count ≥ ~3 does.
