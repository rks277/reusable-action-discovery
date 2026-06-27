# AIME 2026 — Script-Budget Disposition

*Last updated 2026-06-26.* Code:
[`run_aime_session.py`](../scripts/creator/tool_disposition_benchmark/run_aime_session.py).
Companion to [Tool-Disposition Benchmark — Findings](disposition-bench.md); this is a
contest-math probe of the **same** disposition question on a different problem distribution.

## What this is

A spin-off of the [disposition benchmark](disposition-bench.md) that swaps the synthetic
CREATOR-arithmetic problems for **real contest problems** — [MathArena AIME
2026](https://huggingface.co/datasets/MathArena/aime_2026) — and turns the **script budget into
the manipulated variable**. The question is no longer "does the model build a reusable kit"
(AIME problems share no reusable tool — each is a one-off), but: *given a fixed token cap, how
does constraining the model's freedom to write code change how many problems it gets right?*

Why AIME: integer answers 0–999 graded by **exact match** (no sig-fig machinery, no gold
recomputation), and it is the **freshest** contest set available, so contamination is minimal.
We use **Paper I = the first 15 problems**. The harness is reused verbatim — same `driver`
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

## Relation to the disposition benchmark

| | [disposition-bench](disposition-bench.md) (headline) | this (AIME) |
|---|---|---|
| Problems | synthetic CREATOR arithmetic, parameterized | real AIME 2026 contest, integer |
| Grading | exact match @ D sig figs | exact integer match |
| Reusable tool exists? | yes (the point — a kit of primitives) | no (each problem one-off) |
| Manipulated variable | difficulty (magnitude × D), token cap | **script budget** (all-code / mix / by-hand) |
| Headline disposition result | tool reliance **inverse** with capability | all-code column is a **normal** ladder; budget is a lever only sub-frontier |
| Shared mechanism | binding-but-not-suffocating budget drives behavior | same — here it moves *solve rate*, peaking at the mix arm |

The two are consistent: in both, **a scarce budget is the behavioral lever and the frontier
model is budget-indifferent** (disposition-bench: Opus tools ~the same across all four cap arms;
AIME: Opus solves 11 across all three budget arms). AIME adds the contest-difficulty regime
where the *all-code* freedom actively *hurts* the weakest model (thrash) — a sharper version of
the headline's "Haiku builds narrow one-offs and over-applies them."

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
```

Dataset: `external/aime_2026.jsonl` (30 problems; Paper I = first 15). Runs:
`runs/aime_disposition_*/sessions.jsonl` (+ `config.json` per run). Paper-I golds: P1=277,
P2=62, P3=79, P4=70, P5=65, P6=441, P7=396, P8=244, P9=29, P10=156, P11=896, P12=161, P13=39,
P14=681, P15=83.

## Cost

The whole grid (10 runs) ≈ **$13** at the 200k cap (est.; runs log only `spent_tokens` =
uncached input + output, so this assumes ~55k output/run + a ~15% cache-read uplift). Per run:
Haiku ~$0.5, Sonnet ~$1.45, Opus ~$2.5.

## Caveats & next directions

- **n = 1 per cell.** The cross-model *ordering* (mix > by-hand > all-code for Haiku/Sonnet;
  flat Opus; normal all-code ladder) holds across eight+ runs, but each number is a single
  sample. **Reps** on the peak cells, and **Paper II** as a fresh-15 replication, are the
  obvious confirmations.
- **The 200k cap binds for everyone except Haiku-by-hand** — so most cells mix a behavioral
  effect with a truncation artifact (problems left unreached are coverage losses, not wrong
  answers). A roomier cap would separate "would have solved" from "ran out of room."
- **Opus's by-hand arm force-advanced past P10** (burned turns reasoning without submitting),
  illustrating that the cap interacts with the driver's force-advance rule.
