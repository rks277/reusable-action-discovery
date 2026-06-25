# Tool-Disposition Benchmark — Findings

*Last updated 2026-06-25.* Code: [`scripts/creator/tool_disposition_benchmark/`](../scripts/creator/tool_disposition_benchmark/).

## What this measures

A CREATOR-derived benchmark for tool **disposition** — not just *can* a model create and
use a tool, but *does it choose to*, and is that choice well-calibrated. A model works through
many distinct hard-arithmetic problems in **one persistent session** with a **scarce script
budget** and **precision-graded** answers, and we observe whether it builds a small kit of
general, reusable scripts and reaches for them appropriately — versus hand-grinding, or burning
its budget on one-off solvers.

## Design

- **One session, N distinct problems.** Problems (distinct CREATOR formulas: perimeters,
  interest, time-estimates, fluid dynamics, …) are presented one at a time. Scripts written
  with `write_script` **persist across all problems** and can be recalled (`list_scripts` /
  `read_script`) and rerun (`run_script`) on new inputs. Method-neutral prompt: code vs. by-hand
  is presented as an equally valid free choice.
- **Global write budget = 0.1N** (refused once exhausted). At N=20 that is **2 scripts for the
  whole session** — far too few to write one solver per problem, so the rational play is a few
  *general* primitives, reused.
- **Precision grading.** Each problem states a required number of significant figures `D`;
  an answer is correct iff it matches the gold rounded to `D` sig figs (exact match, replacing
  CREATOR's 1% tolerance). Tools: `write_script`, `run_script`, `list_scripts`, `read_script`,
  `submit_answer`. In-process (no MCP); runs Claude and vLLM models via the shared `RawChat`.

### Two difficulty knobs

| Knob | Controls | Effect |
|---|---|---|
| **`magnitude`** | size/messiness of the input numbers (scales `hard_resample`; `m=1.0` = 3–4 digit ints) | whether the arithmetic is **feasible by hand at all** |
| **`D`** (sig figs) | required answer precision | how much an **approximation must nail** to pass |

**Calibration finding:** by-hand solve rate is ~0.12 **flat across D** at `m=1.0` — the binding
difficulty is *magnitude* (multiplying 4-digit numbers), not the sig-fig count. Lowering to
`m=0.01` (2–3 digit inputs) raises by-hand to ~0.4–0.5, putting models in the interesting regime
where hand-computation is *possible but demanding*. All results below use **`m=0.01`, `D=6`** so
the by-hand baseline sits near 50% and tool use is a genuine choice, not a forced move.

## Metrics

- **Solve rate** — fraction correct at `D` sig figs.
- **Efficiency** — P(solve | used ≥1 script on that problem). Reported against P(solve | solved
  by hand) for contrast.
- **Reusability** — mean number of distinct problems each written script was applied to.
- **Recognition** — fraction of script-used problems the model **cannot** solve by hand
  (re-run with tools off). High = tools reached for on genuinely hard problems (necessity);
  low = gratuitous tool use.
- Descriptors: scripts written vs. budget, problems used/no-script, run calls per problem
  (chaining depth), tokens.

## Headline result (N=80 per model: 4 seeds × 20, Haiku/Sonnet/Opus)

**Parameters used for all headline runs:**

| Parameter | Value | Meaning |
|---|---|---|
| `D` (sig figs) | **6** | required answer precision; graded by exact match at 6 sig figs |
| `magnitude` | **0.01** | input-size scale (~2–3 digit numbers; by-hand baseline ≈50%) |
| write budget | **2 scripts/session** | `0.1·N` rounded, with N=20 → 2; refused once used up |
| `N` | **20 problems/seed** | 4 seeds (0,1,2,3) → 80 pooled per model |
| seeds | **0, 1, 2, 3** | each a distinct 20-problem sample at the same D/magnitude |
| reps | **1** per seed | one session per (model, seed) |
| token cap | **300,000 / session** | announced (in prompt; `tokens_remaining` shown) |
| per-call `max_tokens` | **2048** | output cap per model turn |
| `max_turns` | **300** | `max(60, 15·N)`; backstop, rarely binds |
| grading | exact match @ D sig figs | `grading.correct_to_sigfigs` (not CREATOR's 1% tolerance) |
| models | `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`, `claude-opus-4-8` | |

Mean ± SE is across the four seeds; pooled rows aggregate all 80 problems.

| Metric | Haiku 4.5 | Sonnet 4.6 | Opus 4.8 |
|---|---|---|---|
| **Solve rate** (mean±SE) | 0.33 ± 0.03 | 0.50 ± 0.02 | 0.50 ± 0.02 |
| **Efficiency** P(solve \| used script) | 0.36 | 0.47 | 0.22 |
| P(solve \| by hand) | 0.30 | 0.52 | **0.61** |
| **Recognition** (tool-use necessity) | 0.64 | 0.66 | **0.78** |
| Reusability (mean±SE) | 5.3 ± 1.2 | 5.1 ± 1.6 | 3.1 ± 0.3 |
| Problems that **used** a script | 36 / 80 | 32 / 80 | **23 / 80** |
| Problems with **no** script call | 44 / 80 | 48 / 80 | **57 / 80** |
| run_script calls / problem (mean±SE) | 0.53 ± 0.12 | 0.48 ± 0.14 | 0.31 ± 0.03 |
| Tokens (4 seeds) | 994k | 1,020k | 804k |

Per-seed solve rate (x/20) — stable, no outliers:

| | seed0 | seed1 | seed2 | seed3 |
|---|---|---|---|---|
| Haiku | 6 | 8 | 5 | 7 |
| Sonnet | 10 | 10 | 9 | 11 |
| Opus | 10 | 9 | 10 | 11 |

## Finding: tool reliance is *inverse* with capability

Three monotonic trends across the capability ladder:

1. **By-hand skill rises** — P(solve | by hand) **0.30 → 0.52 → 0.61**. Opus is ~2× the
   hand-calculator Haiku is at 6 sig figs.
2. **Tool reliance falls** — problems where a script was used: **36 → 32 → 23** of 80.
   Opus hand-grinds 57/80 and spends the **fewest tokens** (804k) precisely because it tools
   least.
3. **The strongest model's tool use is the most justified** — recognition **0.64 → 0.66 →
   0.78**. When Opus does reach for a script it is nearly always on a problem it genuinely
   cannot do by hand (and even tooled it cracks only 22% of those — its hardest problems).

**Sonnet and Opus tie on solve rate (0.50) by opposite strategies.** Sonnet under-grinds
(by-hand 0.52) and leans on a *general* reusable primitive (e.g. a 103-char `basic_calc` reused
on 10 problems in one session); Opus grinds confidently (by-hand 0.61) and tools only when
forced. Haiku trails (0.33) because it builds *narrow* one-offs (e.g. a problem-0-specific
`fence_cost`) and over-applies them to ill-fitting problems, so its tool use is inefficient
(P(solve|script)=0.36) and its hand-grinding is weak (0.30).

This reproduces the capability-inverse disposition seen elsewhere in this repo (ToolWorld /
WoodWorld recognition-inversion and build-proportion findings) in a new
arithmetic-precision setting: **higher capability → less tool reliance, better-targeted tool
use.**

## Ablation: is the announced token budget itself shaping behavior?

The headline runs *tell* the model its token budget (the cap is in the system prompt and every
tool result reports `tokens_remaining`). To test whether that framing is part of what we measure,
we ran a **no-cap arm**: identical task, but the model is told nothing about a token budget and
tool results omit `tokens_remaining`. A silent 500k safety ceiling bounds cost only — **no model
approached it**, so this is genuine unconstrained behavior. Both arms use the **same 2 seeds**
(0,1; N=40/model) for an apples-to-apples comparison — so the "budgeted" column here is the
seed-0/1 subset of the headline, not the full N=80.

| | Haiku | | Sonnet | | Opus | |
|---|---|---|---|---|---|---|
| **metric** | budgeted | no-cap | budgeted | no-cap | budgeted | no-cap |
| Solve rate | 0.35 | 0.38 | 0.50 | 0.47 | 0.47 | 0.50 |
| **Efficiency** P(solve\|used script) | 0.44 | 0.25 | 0.52 | 0.27 | 0.18 | 0.25 |
| P(solve\|by hand) | 0.27 | 0.41 | 0.47 | 0.55 | 0.59 | 0.61 |
| Recognition | 0.56 | 0.75 | 0.52 | 0.64 | 0.73 | 0.58 |
| Reusability (mean problems/script) | 5.5 | 2.5 | 5.5 | 4.8 | 3.0 | 3.5 |
| Problems used / no script | 18/22 | **8/32** | 21/19 | **11/29** | 11/29 | 12/28 |
| run_script / problem | 0.55 | 0.28 | 0.60 | 0.42 | 0.30 | 0.38 |
| Tokens (2 seeds) | 501k | 396k | 537k | 578k | 362k | 370k |

**Solve rate is unaffected** — the budget framing isn't propping up any model's score.

**But it shifts disposition toward tooling for the sub-frontier models, not Opus.** With the budget
hidden, Haiku and Sonnet both tool *less* and hand-grind *more* (Haiku: 18→8 problems tooled,
by-hand 0.27→0.41, and ~100k fewer tokens; Sonnet: 21→11, by-hand 0.47→0.55). **Opus is
budget-indifferent** — essentially identical either way (it never noticed the cap in the budgeted
arm, and behaves the same without it), consistent with it being the confident hand-grinder. So the
announced budget acts as a mild nudge to economize via tools, strongest below the frontier; the
capability-inverse pattern survives without it but is partly *amplified* by the budget framing.

The secondary metrics move consistently with that read: for Haiku and Sonnet the no-cap arm shows
lower reusability (5.5→2.5, 5.5→4.8) and lower Efficiency (0.44→0.25, 0.52→0.27) — when they tool
only the problems they *must*, those are the harder ones, so script success drops — while
Recognition rises (0.56→0.75, 0.52→0.64): their rarer tool use is more often genuinely necessary.
Opus's secondary metrics barely move except a noisier Recognition (0.73→0.58, over small
tool-used subsets).

**Caveat — high seed variance.** Sonnet's per-seed tool use swings hard in *both* arms (budgeted
15/20 & 6/20; no-cap 4/20 & 7/20), so the pooled effect is real but noisier than any single seed
suggests. Treat the direction (announced budget → more tooling, sub-frontier) as suggestive, not
settled — more seeds are needed for error bars on the magnitude.

No-cap runs: `runs/tool_disposition_20260625_005025` (s0), `_010316` (s1). Reproduce by adding
`--no-token-cap --safety-cap 500000` to the sweep command.

## Reproduction

```bash
# 1. (optional) calibrate the by-hand difficulty knob toward ~50%
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.calibrate \
    --model haiku --sample 16 --sig-figs 6 --magnitudes 0.01 0.05 0.2 1.0

# 2. run a sweep (builds/caches the dataset on first use); repeat per seed
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.run_sweep \
    --models haiku sonnet opus --n 20 --sig-figs 6 --magnitude 0.01 --seed 0 \
    --token-cap 300000 --concurrency 3

# 3. recognition pass (re-runs script-used problems with tools off; edits sessions.jsonl in place)
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.recognition runs/tool_disposition_<ts>/

# 4. per-run metric table
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.analyze runs/tool_disposition_<ts>/
```

Runs used for N=80 above: `runs/tool_disposition_20260624_234319` (Haiku+Sonnet s0),
`_20260624_235415` (Opus s0), `_20260625_000028` (s1), `_20260625_001111` (s2),
`_20260625_001113` (s3). Datasets cached in
`scripts/creator/tool_disposition_benchmark/datasets/`.

## Caveats

- **N=80 per model, 1 rep per problem-set.** SE is over four problem-samplings (seeds), which
  captures problem-sampling noise; it does not separate model stochasticity (use `--reps` for
  that). Recognition for the smaller tool-using subsets is still the noisiest metric.
- **`used_script` = "ran a script on that problem,"** not "the submitted answer came from it"
  (a model can run a script and still answer by hand). Efficiency/recognition inherit this
  definition.
- **Sonnet sometimes hits the 300k token cap** (it tools and chains the most); Opus never does.
  Capping truncates the tail of a session — a confound for the most tool-heavy runs.
- Single difficulty cell (`m=0.01, D=6`). The disposition split is expected to widen toward
  high-magnitude/high-D (everyone must tool) and collapse toward low-D (everyone eyeballs);
  a magnitude × D grid is the natural next experiment.

## Cost

Whole N=80 × 3-model study ≈ **$13–15** (Haiku ~$1.6, Sonnet ~$4.9, Opus ~$6.4 of session
tokens, plus recognition passes). Opus is the cheapest *per session* here only because it tools
least; per token it remains ~5× Haiku.

## Next directions

- **Magnitude × D grid** — trace where the capability-inverse split widens/collapses.
- **OSS ladder** (Qwen / Llama via the vLLM path) — is the inversion frontier-specific here too?
- **Budget sweep** — vary 0.1N to find where the scarcity constraint starts to bind behavior.
