# AIME 2026 — Script-Budget Disposition

*Last updated 2026-06-26.* Code:
[`run_aime_session.py`](../scripts/creator/tool_disposition_benchmark/run_aime_session.py).
Companion to [Tool-Disposition Benchmark — Findings](disposition-bench.md); this is a
contest-math probe of the **same** disposition question on a different problem distribution.
For how these results sit against the literature (and which prior papers' future-work directions
we and others have since addressed), see
[Budget / Tool-Disposition — Related Work & Direction Map](budget-disposition-related-work.md).

## What this is

A spin-off of the [disposition benchmark](disposition-bench.md) that swaps the synthetic
CREATOR-arithmetic problems for **real contest problems** — [MathArena AIME
2026](https://huggingface.co/datasets/MathArena/aime_2026) — and turns the **script budget into
the manipulated variable**. The question is no longer "does the model build a reusable kit"
(AIME problems share no reusable tool — each is a one-off), but: *given a fixed token cap, how
does constraining the model's freedom to write code change how many problems it gets right?*

Why AIME: integer answers 0–999 graded by **exact match** (no sig-fig machinery, no gold
recomputation), and it is the **freshest** contest set available, so contamination is minimal.
We run both **Paper I (the first 15 problems)** and **Paper II (the last 15)** — Paper I is the
primary write-up below; [Paper II](#paper-ii--replication-extension-and-the-script-budget-micro-study)
replicates the grid, overturns one Paper-I generalization, and adds a script-budget micro-study.
The harness is reused verbatim — same `driver`
loop, `SessionState`, `write_script`/`run_script`/`submit_answer` tools, and persistent-session
structure as `run_sweep.py`; only the prompt (AIME framing, tools optional) and the dataset
loader are overridden locally (the shared files are untouched). See the
[pivot rationale](#why-aime-and-not-gsm-hard-or-math) for why earlier candidates were rejected.

## Design

- **One persistent session, 15 problems, one at a time, no going back.** Each answer is graded
  by exact integer match.
- **Token cap = 200,000** (announced in the prompt, `tokens_remaining` on every tool result),
  `max_tokens` = 2048/turn. The cap is the binding constraint — see below.
- **Three arms vary how much the model may code:**

  | Arm | How | Disposition it induces |
  |---|---|---|
  | **all-code** (budget 15) | up to 15 scripts (= N; effectively unlimited) | free to write a solver per problem |
  | **mix** (budget 5) | up to 5 scripts | must hand-solve most, reserve code for the few that need it |
  | **by-hand** (no-scripts) | only `submit_answer` exposed; scripting tools stripped | reason every answer out |

- **n = 1 per cell.** This is an exploratory grid, not a powered measurement — read directions,
  not decimals.

Unlike the headline benchmark, **persistence/reusability are not meaningful here** (no shared
tool to reuse across AIME problems), so we report **solve count, coverage, and where the token
cap bites** instead.

## The grid (AIME Paper I, solves / 15)

| | all-code (b15) | mix (b5) | by-hand | row spread |
|---|---|---|---|---|
| **Haiku 4.5** | 2 | **8** | 6 | 6 |
| **Sonnet 4.6** | 9 | **11** | 10 | 2 |
| **Opus 4.8** | 11 | 11 | 11 | **0** |

(A Sonnet **b7** spot-check landed at 9/15, between b15 and b5 — consistent with the mix arm
being the peak.)

## Three findings

### 1. The all-code column is a *normal* capability ladder (2 → 9 → 11)

No inverse scaling here: with code freely available, stronger model → more solves. The
mechanism is *script quality*, visible in the transcripts:

- **Haiku thrashes.** In all-code it wrote **10 scripts but the cap bit at problem 3** — it
  burned the entire token budget iterating buggy code on the first two problems (solved P1–P2,
  reached nothing else). Code freedom is a *trap* for the weakest model.
- **Opus's scripts are first-try-correct.** Every problem on which Opus ran a script was solved
  in **all three** arms (`eff_solve_given_script = 1.0`), including P9 (a 6⁶ die-sticker
  enumeration that genuinely needs compute).

### 2. A scarce budget is a disposition lever — but only for the weaker two models

Both Haiku and Sonnet peak at the **mix** arm: **mix > by-hand > all-code**.

| | all-code | mix | by-hand |
|---|---|---|---|
| Haiku | 2 | **8** | 6 |
| Sonnet | 9 | **11** | 10 |

A tight budget forces the model off token-expensive code-iteration onto token-cheap
hand-solving, so it **covers more problems** under the fixed cap *and* still spends its few
scripts on the genuinely-computational ones (Haiku's single used script in the mix arm cracked
P9, which it fails by hand). The effect shrinks with capability — Haiku swings 6 points across
arms, Sonnet 2. **This is the AIME analogue of the disposition-bench [token-cap
ablation](disposition-bench.md#ablation-does-the-token-cap-shape-behavior--and-is-it-the-announcement-or-the-tightness)
finding that a *binding-but-not-suffocating* budget is the real behavioral lever** — here the
lever moves the solve rate, not just the tooling rate.

### 3. Opus is reasoning-bound, not script-bound — flat 11/11/11

The lever does nothing for Opus because its **own deliberation**, not its scripting, is the
token sink. It hit the 200k cap in **all three arms** — even by-hand (17 turns, 204k) — capping
coverage at ~12–13 problems regardless of tooling. Per arm:

| Opus arm | solve | submitted | scripts | correct | error(s) | unreached (cap) |
|---|---|---|---|---|---|---|
| all-code (b15) | 11 | 12 | 10 (used 7) | P1–10, 12 | P11 (798) | P13–15 |
| mix (b5) | 11 | 13 | 5 (used 5) | P1–9, 11, 12 | P10 (462), P13 (251) | P14–15 |
| by-hand | 11 | 12 | 0 | P1–9, 11, 12 | P13 (19) | P10, P14–15 |

The arms swap *which* problems they nail by hand vs. by code (e.g. P9 — the simulation — was
scripted in b15 but solved correctly *by hand* in b5/by-hand; P11 was wrong in b15 but right by
hand otherwise), but it nets out flat — exactly the n=1 variance you'd expect with the ceiling
fixed by reasoning length.

## The coverage / token-efficiency angle

Under a fixed token budget, coverage *reverses* the capability ladder. The by-hand arm makes it
starkest:

| by-hand arm | problems reached | hit cap? | accuracy-on-reached |
|---|---|---|---|
| Haiku | **all 15** (195k) | no | 6/15 = 0.40 |
| Sonnet | 11 (213k) | yes | 10/11 = 0.91 |
| Opus | 12 (204k) | yes | 11/12 = 0.92 |

Weaker models are **token-cheap per problem** and blanket the whole set at low accuracy; Opus is
**token-expensive per problem** (verbose reasoning), reaches the fewest, but is near-perfect on
what it attempts. So **Opus's verbosity is self-limiting** under a budget. This is the
capability-cost trade the headline benchmark frames as *verbosity is capability-inverse* (mean
response length Haiku ≫ Sonnet ≫ Opus) — here that same verbosity converts directly into a
*coverage ceiling* when tokens are scarce.

## Paper II — replication, extension, and the script-budget micro-study

We re-ran the full three-arm grid on **Paper II** (problems 16–30), then dug into the script
budget for Sonnet. Paper II both confirms the robust Paper-I effects and corrects one
generalization that Paper I's specific contents had masked.

### The completed 2×2×3 grid

| | all-code (b15) | mix (b5) | by-hand |
|---|---|---|---|
| Haiku · P I | 2 | 8 | 6 |
| Haiku · P II | 2 | 6 | 6 |
| Sonnet · P I | 9 | 11 | 10 |
| Sonnet · P II | 6 | 8 | 8 |
| Opus · P I | 11 | 11 | 11 |
| Opus · P II | 6 | **8** | **12** |

**What replicates:**

- **all-code is the worst arm in all six cells.** Free scripting never wins — constraining code
  is weakly-to-strongly beneficial everywhere.
- **Haiku's all-code thrash is identical across papers** (2/15 both): it writes 10–12 scripts,
  burns the token cap re-debugging the first 2–3 problems, and never escapes them.
- **mix ≫ all-code everywhere** (the disposition lever).

**What Paper II reveals that Paper I hid:**

- **The `mix > by-hand` edge is conditional on a "code-necessary" problem.** Paper I's edge came
  *entirely* from **P9** — a 6⁶ die-sticker simulation the models fail by hand but solve with one
  reserved script. Paper II contains no such problem (every reachable problem is hand-tractable
  for these models), so **mix ties by-hand**: Haiku 6 = 6, Sonnet 8 = 8.
- **Opus is *not* budget-indifferent — that was a Paper-I artifact.** On Paper II, Opus runs
  **by-hand 12 ≫ mix 8 ≫ all-code 6** — strictly monotone in *less code*. With no code-necessary
  problem, scripting is pure token deadweight for the strongest hand-solver, so removing it
  entirely is best: **Opus-by-hand's 12/15 is the single best cell on the Paper II grid** (P1–12
  correct, only P13 wrong at 20 vs 107, P14–15 unreached at the cap). The Paper-I flat 11/11/11
  was the special case where P9 made scripting pay for itself.

**Unifying principle (now clean across all six cells):** *under a binding token cap, code only
pays if it captures a problem you genuinely cannot do by hand.* When such a problem exists
(Paper I's P9) a small budget edges out by-hand; when none does (Paper II) code is deadweight,
and **the more capable the hand-solver, the more it gains from dropping code** — culminating in
Opus-by-hand at 12/15.

### Natural scripting on Paper II (all-code arm)

When free to write as many scripts as it likes, script *count* is U-shaped — the middle model
writes the fewest — but the real signal is **iteration depth per problem**:

| model | scripts written | distinct problems coded | scripts / coded-problem | reached |
|---|---|---|---|---|
| Haiku | 12 | 3 | **4.0** | 2 |
| Sonnet | 4 | 3 | 1.3 | 8 |
| Opus | 10 | 5 | 2.0 | 6 |

Haiku thrashes — all 12 scripts pile onto 3 problems in 5-deep debug chains
(`bug_paths → _v2 → _v3 → _memoized → _final`; `pentagon_analysis → … → _verify`). Sonnet writes
one clean script per problem (`grid_paths`, `urn_marbles`, …) and never iterates. Opus is in
between (`pent → pentcount → pentvalid → pentmin`). Coverage (reached) runs **inverse** to script
volume.

### The budget binds Haiku and Opus, but not Sonnet (mix arm, budget 5)

| model | natural (free) | wrote @ b5 | hit the budget? |
|---|---|---|---|
| Haiku | 12 | 5 / 5 | yes (exhausted) |
| Sonnet | 4 | 4 / 5 | no |
| Opus | 10 | 5 / 5 | **yes — tried a 6th, refused** |

The budget caps Haiku's thrash (coverage 2 → 9) and genuinely constrains Opus (it wants more);
Sonnet self-limits below the cap, so the lever doesn't touch it — which is why its grid row moves
least.

### Sonnet script-budget micro-study

Probing whether the budget *number* matters for Sonnet (who self-limits to ~2 scripts on
Paper II):

| budget | reps | solves | mean ± SD | scripts written | refusals |
|---|---|---|---|---|---|
| 3 | 5 | 6, 8, 9, 9, 10 | 8.40 ± 1.5 | 1–3 (mean 2) | 0 |
| 4 | 4 | 9, 10, 11, 11 | 10.25 ± 1.0 | 2–3 | 0 |
| 5 | 1 | 8 | — | 4 | 0 |
| 15 | 1 | 6 | — | 4 | 0 |

**Budgets 3 through 15 are all non-binding for Sonnet** — it writes ~2 scripts and is *never*
refused. Measured per-session SD ≈ **1.5** (lower than the binomial guess of 2.0; this is the
first real variance estimate for these runs).

The budget-4 mean (10.25) sits above budget-3 (8.40): two-sample **t = 2.11, p ≈ 0.07** —
borderline, not significant at 0.05. Critically, **no mechanism supports a real budget-3-vs-4
difference**: both arms write the same ~2 scripts with 0 refusals, so the manipulated variable
produced *no behavioral change*. The likely driver is sampling (budget-3 drew one unlucky 6).
Pooled over all non-binding runs, **Sonnet Paper II ≈ 9.2 ± 1.5**. With σ ≈ 1.5, separating a
real 1.85-problem gap at 80% power would need ~8 reps/condition; a 1-problem gap, ~36.

> **Open hypothesis — the announcement/anchor effect (untested).** A competing reading: that
> *announcing* a budget ≈ the natural count primes more efficient behavior **even when it never
> binds** (an anchor effect, distinct from binding). More reps at budget 4 cannot test this — both
> the anchor and the noise stories predict "budget-4 scores well." The decisive test decouples
> *what the model is told* from *what is enforced*: **announce-4 / enforce-15** vs
> **announce-15 / enforce-15** (identical enforcement; only the told number differs). The anchor
> hypothesis predicts the first wins; the noise hypothesis predicts a tie. The harness now supports
> this via `--announce-budget` (verified to change only the prompt's "up to *K* scripts" line), but
> the comparison is **not yet run**. It connects to the
> [announcement-vs-tightness ablation](disposition-bench.md#ablation-does-the-token-cap-shape-behavior--and-is-it-the-announcement-or-the-tightness)
> in the main benchmark, where the *token-cap* announcement effect was small — but the *script-budget*
> announcement is a different lever.

### Paper II golds

Within-paper 1-indexed (problems 16–30): P1=178, P2=243, P3=503, P4=279, P5=190, P6=50, P7=754,
P8=245, P9=669, P10=850, P11=132, P12=223, P13=107, P14=157, P15=393.

## Relation to the disposition benchmark

| | [disposition-bench](disposition-bench.md) (headline) | this (AIME) |
|---|---|---|
| Problems | synthetic CREATOR arithmetic, parameterized | real AIME 2026 contest, integer |
| Grading | exact match @ D sig figs | exact integer match |
| Reusable tool exists? | yes (the point — a kit of primitives) | no (each problem one-off) |
| Manipulated variable | difficulty (magnitude × D), token cap | **script budget** (all-code / mix / by-hand) |
| Headline disposition result | tool reliance **inverse** with capability | all-code column is a **normal** ladder; budget is a lever only sub-frontier |
| Shared mechanism | binding-but-not-suffocating budget drives behavior | same — here it moves *solve rate*, peaking at the mix arm |

The two are consistent: in both, **a scarce budget is the behavioral lever**. On the question of
frontier-model budget-(in)difference the picture is paper-dependent and worth stating precisely:
in disposition-bench Opus tools ~the same across all four cap arms, and on AIME **Paper I** Opus
solves 11 across all three budget arms — *but Paper II shows that "Opus is budget-indifferent" was
an artifact of Paper I containing a code-necessary problem (P9).* On Paper II, where no problem
requires code, **less code monotonically helps Opus** (by-hand 12 ≫ mix 8 ≫ all-code 6). So the
accurate cross-setting statement is the unifying principle above — code pays only when it captures
an otherwise-unsolvable problem — not a blanket "frontier model ignores the budget." AIME also adds
the contest-difficulty regime where *all-code* freedom actively *hurts* the weakest model (thrash)
— a sharper version of the headline's "Haiku builds narrow one-offs and over-applies them."

## Why AIME (and not GSM-Hard or MATH)

- **GSM-Hard** was tried first (parameterizable like CREATOR) but its reference programs
  **corrupt golds under resampling** — search-loop references return garbage and multi-role
  numbers use stale constants — so the variation pipeline only covered ~14/20 problems cleanly
  and some golds were wrong. See [[disposition-bench-math-too-easy]].
- **MATH** (competition subset) is **too easy / insight-not-computation** (L5 ~88% solvable by
  hand) and **contaminated**.
- **AIME 2026** needs no parameterization (clean integer golds), spans an interesting
  hand-vs-code difficulty range (P9's simulation genuinely needs compute; P1–P8 are
  hand-tractable for the stronger models), and is the freshest → least contaminated.

## Reproduction

```bash
# all-code arm (budget = N): generous scripting
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.run_aime_session \
    --models haiku sonnet opus --paper I --budget 15 --token-cap 200000 --max-tokens 2048

# mix arm: scarce budget
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.run_aime_session \
    --models haiku sonnet opus --paper I --budget 5 --token-cap 200000 --max-tokens 2048

# by-hand arm: scripting tools stripped, only submit_answer
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.run_aime_session \
    --models haiku sonnet opus --paper I --no-scripts --token-cap 200000 --max-tokens 2048

# Paper II: same three arms with --paper II

# anchor-effect test: decouple the announced budget from the enforced cap
# (announce "4" while actually enforcing 15 — neither binds Sonnet, so only the told number varies)
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.run_aime_session \
    --models sonnet --paper II --budget 15 --announce-budget 4 --token-cap 200000 --max-tokens 2048
```

Dataset: `external/aime_2026.jsonl` (30 problems; Paper I = first 15, Paper II = last 15). Runs:
`runs/aime_disposition_*/sessions.jsonl` (+ `config.json` per run). `--budget` is the *enforced*
script cap; `--announce-budget` overrides only the number shown in the prompt (default = `--budget`).
Paper-I golds: P1=277, P2=62, P3=79, P4=70, P5=65, P6=441, P7=396, P8=244, P9=29, P10=156, P11=896,
P12=161, P13=39, P14=681, P15=83. (Paper-II golds listed
[above](#paper-ii-golds).)

## Cost

Both papers' grids plus the Sonnet budget micro-study ≈ **$30–35** at the 200k cap (est.; runs
log only `spent_tokens` = uncached input + output, so this assumes ~55k output/run + a ~15%
cache-read uplift). Per run: Haiku ~$0.5, Sonnet ~$1.45, Opus ~$2.5. The Paper-I grid alone is
~$13; Paper II's three-model grid adds ~$13; the Sonnet budget reps (b3 ×5, b4 ×4) add ~$13.

## Caveats & next directions

- **Mostly n = 1 per cell** (exception: the Sonnet Paper-II budget micro-study has 4–5 reps).
  The *robust* effects replicate across both papers and many runs — all-code is worst everywhere;
  mix ≫ all-code; Haiku thrashes to a 2/15 floor; less-code-helps-Opus on a hand-tractable set.
  The *fragile* claims are the within-noise ones — the `mix > by-hand` edge (real only when a
  code-necessary problem exists) and any difference among Sonnet's non-binding budgets (≈ noise,
  σ ≈ 1.5).
- **The 200k cap binds for nearly every cell** (exception: Haiku-by-hand on Paper I) — so most
  cells mix a behavioral effect with a truncation artifact (problems left unreached are coverage
  losses, not wrong answers). A roomier cap would separate "would have solved" from "ran out of
  room."
- **The announcement/anchor hypothesis is open and the test is built but unrun** — see the
  [Sonnet micro-study](#sonnet-script-budget-micro-study). `--announce-budget` decouples told-vs-enforced.
- **Opus's by-hand arms force-advanced past a problem** (burned turns reasoning without
  submitting; Paper I past P10, Paper II reaching P13), illustrating that the cap interacts with
  the driver's force-advance rule.
