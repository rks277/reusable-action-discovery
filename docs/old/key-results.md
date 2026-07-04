# Tool-Disposition — Key Results

*Updated 2026-06-28.* AIME-2026, one session, 200k token cap, tools optional, exact-match. Detail:
[aime-disposition](aime-disposition.md), [plan](net-negative-plan.md).

## 2×2×3 grid — AIME I/II × Haiku/Sonnet/Opus × all-code/mix/by-hand (n=1/cell)

| | all-code | mix | by-hand |
|---|---|---|---|
| Haiku · I/II | 2 / 2 | 8 / 6 | 6 / 6 |
| Sonnet · I/II | 9 / 6 | 11 / 8 | 10 / 8 |
| Opus · I/II | 11 / 6 | 11 / 8 | 11 / 12 |

- All-code is the worst arm in all 6 cells; constraining code never loses.
- `mix > by-hand` only with a code-necessary problem: Paper I edge = entirely P9 (6⁶ sim); Paper II has none → mix ties by-hand (Haiku 6=6, Sonnet 8=8).
- Tool reliance inverse with capability: all-code catastrophic for Haiku (2/15 floor), deadweight for Opus (Paper II by-hand 12 ≫ all-code 6).

## Transcript analysis — why all-code fails (Haiku)

- The harm is a verification/rewrite spiral, not code itself: one palindrome problem → 5 scripts, 82k tok.
- Code-available Haiku ≈50k tok/problem vs ≈13k by-hand (which covered all 15 in 195k).

## E0 — announce vs enforce (Haiku, Paper I, 5 reps/cell)

| announce / enforce | solve | scripts |
|---|---|---|
| 5 / 15 (told tight, not forced) | 7.4 | 5.0 |
| 15 / 15 (all-code) | 4.8 | 7.4 |

- Awareness, not tightness: announce (unenforced) +2.6 solve, t=2.74, p≈0.03; announce main effect +1.1, enforce ≈0.
- Soft ≥ hard: a5/e15 7.4 > a5/e5 5.8, a15/e5 6.2.
- Method bug that hid it: `writes_remaining` showed the *enforced* count, contradicting the announced cap; fixed via `announce_budget` in `session_state.py`.

## Per-problem value-of-budget curves (Haiku, both papers, each problem in isolation, k∈{0,1,2,3}, 3 reps)

Each problem solved alone (no shared token pool, no allocation), script budget k, silent token ceiling.

| per-problem budget k | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| mean solve (30 problems) | 0.41 | 0.60 | 0.72 | 0.66 |

n\* (smallest k reaching the problem's max solve) distribution: **0:10, 1:6, 2:10, 3:2, unsolvable:2**.

Per-problem solve rates (3 reps each; n\* = smallest k at the problem's max):

| problem | k0 | k1 | k2 | k3 | n\* | | problem | k0 | k1 | k2 | k3 | n\* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PI-1  | 1.00 | 1.00 | 1.00 | 1.00 | 0 | | PII-1  | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
| PI-2  | 0.33 | 0.00 | 0.33 | 0.33 | 0 | | PII-2  | 0.00 | 0.00 | 1.00 | 0.67 | 2 |
| PI-3  | 1.00 | 1.00 | 1.00 | 1.00 | 0 | | PII-3  | 0.00 | 0.00 | 0.00 | 0.00 | — |
| PI-4  | 0.33 | 1.00 | 1.00 | 1.00 | 1 | | PII-4  | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
| PI-5  | 1.00 | 1.00 | 1.00 | 1.00 | 0 | | PII-5  | 0.67 | 0.67 | 1.00 | 0.67 | 2 |
| PI-6  | 1.00 | 1.00 | 1.00 | 0.67 | 0 | | PII-6  | 0.67 | 1.00 | 1.00 | 0.00 | 1 |
| PI-7  | 0.67 | 1.00 | 0.67 | 1.00 | 1 | | PII-7  | 0.33 | 0.33 | 0.33 | 0.33 | 0 |
| PI-8  | 1.00 | 0.67 | 0.67 | 1.00 | 0 | | PII-8  | 0.00 | 1.00 | 0.67 | 1.00 | 1 |
| PI-9  | 0.00 | 0.67 | 1.00 | 0.33 | 2 | | PII-9  | 0.67 | 0.67 | 1.00 | 0.67 | 2 |
| PI-10 | 0.00 | 0.00 | 0.33 | 0.33 | 2 | | PII-10 | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
| PI-11 | 0.00 | 0.00 | 0.33 | 0.00 | 2 | | PII-11 | 0.33 | 0.33 | 0.67 | 0.67 | 2 |
| PI-12 | 0.33 | 0.33 | 0.33 | 1.00 | 3 | | PII-12 | 0.00 | 0.33 | 0.67 | 0.67 | 2 |
| PI-13 | 0.00 | 0.67 | 1.00 | 0.67 | 2 | | PII-13 | 0.00 | 0.33 | 0.67 | 0.33 | 2 |
| PI-14 | 0.00 | 0.00 | 0.00 | 0.33 | 3 | | PII-14 | 0.00 | 1.00 | 1.00 | 1.00 | 1 |
| PI-15 | 0.00 | 0.00 | 0.00 | 0.00 | — | | PII-15 | 0.00 | 1.00 | 1.00 | 1.00 | 1 |

- Code helps per-problem in aggregate (0.41→0.72, k0→k2); the optimum is small — 10/30 need no code (n\*=0), 16/30 max out at k=1–2.
- Curve is **non-monotonic**: aggregate dips k2→k3 and several problems regress — over-verification can talk Haiku out of a correct answer *even in isolation* (no coverage to lose), so the spiral harms via two channels: session-level coverage starvation **and** per-problem over-verification. (n=3 reps; individual dips noisy, aggregate dip cleaner.)
- These `n*` values are the clean per-problem benefit ground truth for the budget predictor.
