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

## Reference policy: π\* (built — `pi_star.py`)

**π\* is a SAME-INFORMATION reference, NOT claimed optimal.** It has *exactly* the model's info
(knows only {N, T, B} + "a distribution over N types"), via an **exchangeable Dirichlet(α) prior** —
it has NO idea about the frequent/rare split and must infer everything from the stream, like the
model. Construction: per-type Whittle-index DP (predictive `q(k,t)=(α+k)/(Nα+t)`) → build region in
(k,t); the build price λ\* is tuned so expected builds = B (complementary slackness, uses only
{N,T,B}). See `docs/whittle-asymptotic-optimality.md`.

**The claim** (this is the headline, and it's clean): a policy with *identical information* to the
model extracts **~6× the value** (self-test: π\* 1285 vs eager/model 209, g=1.0). So the model is bad
at using the information it has. Since π\* ≤ the true optimum, **(π\* − model) is a conservative
LOWER BOUND on the model's regret.** π\* WAITS (build-lateness ~1.5); the model builds on sight (0).

**Do NOT hold π\* to beating wait-one-repeat.** wait-one-repeat (build first B to reach 2nd sighting)
*cheats* — its k=2 threshold bakes in structural knowledge (rare = one-off), which the model/π\* do
not have. It scores higher (2030) precisely because it has more info; that only reinforces "the model
is bad." Report it (and the clairvoyant upper bound, 3077) as **extra-info comparisons**, not the
reference. (This retires the earlier idea of a mixture/hierarchical prior — unnecessary: we don't
need π\* to be optimal, only same-info and much better than the model.)

**Per-set clairvoyant** (knows realized counts, builds top-B at first sight; `optimal_build_set` /
`fullinfo_value`) is the loose upper bound. (clairvoyant − π\*) = intrinsic price of online
uncertainty; (π\* − model) = the reported lower bound on the model's excess regret.

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
2. ~~π\* (Whittle-index DP + λ-tuning)~~ **DONE** (`pi_star.py`, self-test certifies 6× vs model).
3. ~~Wire π\* + clairvoyant into the scorer~~ **DONE** (`skirental_scorer.pistar_report` +
   `model_builds_from_actions`; wired into `score_run`, keyed off `meta.json`). Values BOTH model and
   π\* analytically over full realized sizes (truncation confound gone — built hot types get credited
   downstream reuse even in a budget-truncated transcript); bait keyed off `role=="trap"`; reports
   `regret_lb = value(π\*) − value(model)` + `clairvoyant_gap`. `score_run(pistar_price=...)` lets a
   sweep tune the Whittle price ONCE (design constant) and reuse it. Verified on the cached 5-seed
   smoke: regret_lb ≈ +1400/seed (pos 4/5), model 1.2 traps/seed & lateness 0 vs π\* 0.8 & ~0.5.
4. **Haiku PoC run** on the new design (the frontier anchor). ← NEXT
5. **Qwen-Coder main runs** (few sizes) + the **fine-tuning mitigation** arm.

## Open items

- Pin π\*'s exchangeable prior; confirm prior-insensitivity.
- `matrix_power_mod` cover fixed (was "×K times" ambiguous, tanked Opus a_script); re-verify if we
  use the frontier uniform pool. `continued_frac`/`linrec_mod`/`josephus` only matter for that pool.
- Keep R/λ set so a single build never pays (one-shot wasteful) but count ≥ ~3 does.
