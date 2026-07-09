# Online-tool-investment — working notes

Companion to `online-tool-investment-plan.md` (headline + positioning; **read its §0 Orientation for notation/files/conventions/data-location**) and `docs/old/same-info-optimal-dp.md` (the DP). Technical status board. Pre-audit docs in `docs/old/`. Jargon (N/T/B, hot/trap, lateness, a_hand/a_script, g, MAG, m\*, π\*, `[[no-auto-reps]]`) is defined in plan §0.

## The two information conditions (read first)

π\* (`exact_dp.ExactDP`) is constructed knowing N (predictive `(α+k)/(N·α+t)`). Two arms by what the *model* knows:
- **no-N** — model not told N (realistic; regret vs π\* is an *upper bound* on same-info regret).
- **A2** — model told exact N (`--announce-n`); same-information with π\*, isolates framing from N-ignorance.

**A2 arm (added 2026-07-03, run):**
- `urn_session.py --announce-n`: `_N_NOTE` → "There are exactly N distinct colors…"; dir `runs/urn_<model>_n-announced/`.
- `arm_a1_announce.py --announce-n`: `SessionState.announce_n_types=N` → `driver.py` appends `prompts.n_types_note(N)`; dir `runs/arm_a1_announce*_n-announced/`.
- Default (no flag) unchanged.

## Results (measured)

### Haiku 2×2 — the core result
| | urn no-N | urn A2 | tool no-N | tool A2 |
|---|---|---|---|---|
| first-sight | 53% | 28% | 100% | 100% |
| lateness | 0.86 | 1.19 | 0.000 | 0.000 |
| regret vs π\* | 640±285 | **0±344** | 1308±520 | **1633±553** |
| traps/seed | 0.92 | 0.58 | — | 1.25 |

Told N, Haiku is **optimal in the urn** (regret 0) and **fully eager in the tool task** (100% first-sight, regret 1633, N gave zero benefit). Clean same-information dissociation → framing failure, not N-ignorance. The old nonzero urn regret (640) was urn-side N-ignorance; A2 removes it.

### Opus urn — robust
| | no-N | A2 |
|---|---|---|
| first-sight | 8% | 0% |
| lateness | 1.14 | 1.28 |
| regret vs π\* | −655±389 | −685±390 (beats π\*) |
| traps/seed | 0.33 | 0.33 |

Aces the urn under both. **Do NOT headline "beats π\*"** — only ~1.75 SE at n=12, and π\* is optimal only vs its mis-specified symmetric-Dirichlet prior (real generator has fixed hot-count + trap-early structure); report as "reaches the optimum within noise + more trap-averse (0.33 vs 0.75)." Paired Opus *tool* cell ASSUMED eager (prior constructed-design bait 20/20; shelved pre-pub).

**Interpretable urn metric (added 2026-07-04):** `urn_session.report()` now also prints **balls collected** (the urn's literal objective) — model / π\* / clairvoyant per seed, plus balls-regret vs π\* and % of π\*'s balls. Balls-regret = the reuse-deficit ΔM and equals the utility regret / (100·a_script+78.7) = **/178.7** exactly (a_script=1 in the urn, budget fully used) — verified per-seed on Opus A2 (all 12 seeds match to the decimal). More readable than the utility number and it's what the model was told to maximize; still NOT magnitude-comparable to the tool game (different currency), so keep leading with lateness cross-framing. Haiku A2: model 38.4 = π\* 38.4 balls/seed (100%, balls-regret 0.0±1.9); clairvoyant 49.5 (the hindsight gap is irreducible uncertainty — why we regret vs π\*, not clairvoyant).

### Cross-model urn — plateau holds under A2; N-disclosure doesn't rescue Qwen
Qwen-Coder 0.5b→32b urn regret (24 seeds 2000–2023):

| size | no-N | A2 | A2 lateness | A2 first-sight |
|---|---|---|---|---|
| 0.5b | 745 | 844±498 | 0.65 | 71% |
| 1.5b | 1731 | 1743±474 | 0.60 | 81% |
| 3b | 660 | 1197±330 | 0.11 | 94% |
| 7b | 1545 | 1494±372 | 0.23 | 80% |
| 14b | 491 | 737±423 | 0.36 | 76% |
| 32b | 901 | 216±263 | 0.42 | 67% |

Noisy suboptimal plateau under both conditions; only Opus beats π\* → flat-then-frontier-jump. **N-disclosure does NOT rescue the Qwen ladder** (contrast Haiku 640→0): 0.5b–14b stay suboptimal told N; only 32b moves toward optimal (216). → latent allocation competence when told N tracks the frontier (Haiku/Opus have it, Qwen ≤14b doesn't, 32b starting). Lateness ⊥ regret (type-selection-dominated, ~2600-utility swing per misallocated keep). Spend ~$21 API + free local.

### Qwen-14b 2×2 (FT target) — genuine incompetence, not suppression
| | urn no-N | urn A2 | tool no-N | tool A2 |
|---|---|---|---|---|
| lateness | 0.17 | 0.36 | 0.043 | 0.125 |
| regret | 491 | 737 | 3268 | 2934±324 |
| first-sight | — | 76% | 96% | 88% |

N barely moves 14b: urn stays suboptimal (737), tool stays eager (2934, positive on all 12). Lacks the policy in both framings — unlike Haiku (urn A2 → 0). Clean FT target: teach the competence, test urn learning + tool transfer.

### Cost model (Haiku A0)
R=100, λ=0.1; u_hand=−98.7, u_build=49.2, u_reuse=80. Building beats hand even single-use → scarcity is the write budget, not per-token cost.

## Clean assets (no N dependence)

**Instrument (all reusable):** `exact_dp.py` (cap=3 certified lossless, ~540k states/~3s; Whittle `pi_star.py` retired, kept for `value_of_builds`/`clairvoyant_builds`/`eager_builds`); `driver.py`/`session_state.py` (persistent session, write budget, `stop_on_budget_exhausted`, `progress_cb`); `stream_builder.py` (`StochasticStreamSpec`, `pinned_last_trap`); `skirental_scorer.py` (`exact_pistar_report`, analytic truncation-safe `value_of_builds`); `urn_session.py`.

**a_script calibration (Qwen, MAG=100):** 0.5b 0.21, 1.5b 0.35, 3b 0.50, 7b 0.75, 14b 0.83, 32b 0.96 (a_hand=0). Smooth climb → **FT target = 14b**. Infra: 32b needs a `num_ctx`-capped Ollama Modelfile (else CPU offload, ~8× slowdown).

**Ollama tool-call fixes** (`lomekwi/raw_chat.py`): `_repair_triple_quoted_strings` (Python `"""…"""` inside tool-call JSON) + shape-C in `_coerce_tool_call_obj` (script-name in the `name` slot; keyed on write_script-only `code`).

**Early-stop technique** (method): budget spent → truncate + value tail analytically. Local 12-seed run → ~5.5 min.

**Novelty verdict** (`docs/old/online-tool-investment-related-work.md`): amortization slice CLEAR; abstraction-gap method borrowed (novel domain); capability axis crowded (secondary). Avoid: "premature commitment," "budget-aware tool use," "knowing-doing gap," "content effects."

## Mechanism (transcripts)
Haiku's tool rationale is per-problem; never reserves; "budget" only retrospective; "reserve/conserve" 0× — even under A2 (N known, still 100% eager). Not deliberating about *when* to build.

## Immediate next steps
1. ~~Qwen A2 reruns~~ **DONE (2026-07-03)** — urn ladder + 14b tool (raw dirs were on the now-down ephemeral box; only aggregates above survive — see plan §0).
2. **RL intervention — DONE (2026-07-08).** From the untouched Qwen-14b base, per-decision PPO with a privileged critic learned to reserve from balls reward alone. Held-out urn eval (n=24): first-sight 75%→32%, lateness 0.375→0.903, balls 87%→101% of π\*. Paired tool A2 eval (n=12): RL-final remained eager (95% first-sight) over a clean tool channel (0 malformed / 0 unknown). Full results and caveats: `docs/rl-phase1-results.md`.
3. **Framing-generalization probes — DONE (2026-07-09).** The learned policy transfers across three held-out free-text vocabularies (pooled 19% vs base 89% first-sight) but only partly through isomorphic `keep`/`pass` tool calls (62% vs 99%), with strong vocabulary dependence.
4. **Priority follow-ups:** publication-grade tool rerun; RL directly in the tool framing; fair-critic ablation; paired Opus tool cell; 32b subject; mechanism probe for vocabulary-sensitive tool-call transfer.

## Open items
- Re-verify π\*'s α-insensitivity under A2.
- Urn "exactly N colors" vs tool "N distinct types" disclosures — confirm close enough to call the arms same-information (both now state N).
- Qwen A2 raw transcripts live only on a destroyed box; if reviewers need them, re-run (box setup in plan §0).
