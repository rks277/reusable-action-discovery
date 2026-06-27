# Tool-Disposition Benchmark — Findings

*Last updated 2026-06-25.* Code: [`scripts/creator/tool_disposition_benchmark/`](../scripts/creator/tool_disposition_benchmark/).

> **Spin-off:** [AIME 2026 — Script-Budget Disposition](aime-disposition.md) runs the same
> harness on real contest problems and makes the **script budget** the manipulated variable. It
> finds the scarce-budget lever (the [token-cap ablation](#ablation-does-the-token-cap-shape-behavior--and-is-it-the-announcement-or-the-tightness)
> below) moves the *solve rate* — peaking at a "mix" arm — for the sub-frontier models, while
> Opus stays budget-indifferent.

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

### Isolated by-hand baseline (no tools, no session, no budget)

The cleanest "can the model get the right number cold" measurement: each problem is presented as
its **own independent single-turn call** — no tools, no MCP, no persistent session, no token
budget, and no other problems in context. Graded by exact match at `D=6`; pooled over seeds 0+1
(N=40). This is the same per-problem isolated-call method the recognition pass uses, and it is the
**standard baseline** the tool-using metrics are read against. Script:
[`byhand_eval.py`](../scripts/creator/tool_disposition_benchmark/byhand_eval.py).

| Model | Isolated by-hand (seeds 0+1, N=40) |
|---|---|
| Haiku 4.5 | **0.325** (seed0 6/20, seed1 7/20) |
| Sonnet 4.6 | **0.500** (10/20, 10/20) |
| Opus 4.8 | **0.525** (11/20, 10/20) |

Monotonic with capability, and consistent with the headline `P(solve | by hand)` (0.30 / 0.52 /
0.61). The headline figure is *higher* for Opus because it is conditioned on the problems Opus
*chose* to do by hand (its confident/easier ones), whereas this forces **all** problems by hand —
including the hard ones it would normally tool — so the denominator is unfiltered. Note this is the
**isolated** rate; the **in-session** by-hand rate (all N in one running conversation, still no
tools) is a distinct measurement that adds long-context effects.

**Isolated ≈ in-session (within noise).** Running all 20 problems back-to-back in one shared
conversation (no tools, no budget;
[`byhand_session.py`](../scripts/creator/tool_disposition_benchmark/byhand_session.py)):

| Model | isolated | in-session | Δ | avg resp len |
|---|---|---|---|---|
| Haiku | 0.325 | 0.350 (seed0 6/20, seed1 8/20) | +0.025 | ~900 |
| Sonnet | 0.500 | 0.425 (9/20, 8/20) | −0.075 | ~560 |
| Opus | 0.525 | 0.475 (10/20, 9/20) | −0.050 | ~210 |

With N=40/condition the SE of the difference is ~0.11, so **no model's Δ is significant** — the
in-session and isolated by-hand rates agree within noise at every capability level (Haiku trends
slightly up, Sonnet/Opus slightly down). So the isolated baseline is a reasonable stand-in for
in-session by-hand solvability, though the small opposite-sign trends mean a long-context effect
can't be fully ruled out without more seeds/reps.

**Spontaneous code: none, at any capability level.** With no tool available, *none* of the three
models ever writes code in its prose (0 code blocks / Python keywords across all 120 turns) — they
hand-grind longhand arithmetic. Script-writing is a response to the `write_script` *affordance*,
not something the models fake in text when it's absent — removing the tool cleanly removes the
behavior. **Verbosity is capability-inverse:** mean response length falls Haiku ~900 → Sonnet ~560
→ Opus ~210 chars; the stronger model reaches the same (or better) by-hand answer in far fewer
words.

## Metrics

- **Solve rate** — fraction correct at `D` sig figs.
- **Efficiency** — P(solve | used ≥1 script on that problem). Reported against P(solve | solved
  by hand) for contrast.
- **Persistence** — mean number of distinct problems each script was *run on* (breadth of
  application; the old "reusability").
- **Reusability** — mean number of distinct problems each script was *beneficial* toward (its
  correct output became the submitted answer) — breadth of *successful* reuse. Always ≤ persistence;
  the gap is tool misfire (a script run widely but rarely producing the answer).
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
| Persistence (run on; mean±SE) | 5.3 ± 1.2 | 5.1 ± 1.6 | 3.1 ± 0.3 |
| Reusability (beneficial; mean±SE) | 1.6 ± 0.5 | 1.3 ± 0.6 | 0.5 ± 0.2 |
| Problems that **used** a script | 36 / 80 | 32 / 80 | **23 / 80** |
| Problems with **no** script call | 44 / 80 | 48 / 80 | **57 / 80** |
| run_script calls / problem (mean±SE) | 0.53 ± 0.12 | 0.48 ± 0.14 | 0.31 ± 0.03 |
| Tokens / seed | 249k | 255k | 201k |

Per-seed solve rate (x/20) — stable, no outliers:

| | seed0 | seed1 | seed2 | seed3 |
|---|---|---|---|---|
| Haiku | 6 | 8 | 5 | 7 |
| Sonnet | 10 | 10 | 9 | 11 |
| Opus | 10 | 9 | 10 | 11 |

Scripts written across the four seeds (budget = 2/session; one Sonnet session used only 1):

| Model | scripts (×count over 4 seeds) |
|---|---|
| Haiku | `fence_cost` ×4 (generic `sides×length×cost` multiplier), `project_time_estimate` ×2, `reynolds_number` ×2 |
| Sonnet | `reynolds` ×3, `compound_interest` ×2, `combination` ×1 (`comb(n,r)`), `basic_calc` ×1 (a multiplier, seed 0 only) |
| Opus | `reynolds`/`re` ×4 (`4ρQ/(πDμ)`), `compound`/`ci` ×2 (`P((1+r)^t−1)`), `frustum` ×2 (`πh/3·(R²+Rr+r²)`) |

The inventory is itself capability-ordered: Haiku spends its scarce scripts on a generic
**multiplier** (it needs help multiplying), Opus **exclusively** on hard special-function formulas
(it multiplies in its head), Sonnet in between. See *What each model builds scripts for* below for
how this plays out on the product-shaped problems.

## Finding: tool reliance is *inverse* with capability

Three monotonic trends across the capability ladder:

1. **By-hand skill rises** — P(solve | by hand) **0.30 → 0.52 → 0.61**. Opus is ~2× the
   hand-calculator Haiku is at 6 sig figs.
2. **Tool reliance falls** — problems where a script was used: **36 → 32 → 23** of 80.
   Opus hand-grinds 57/80 and spends the **fewest tokens** (201k/seed) precisely because it tools
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

## Ablation: does the token cap shape behavior — and is it the *announcement* or the *tightness*?

The headline runs *tell* the model its token budget (cap in the system prompt + `tokens_remaining`
on every tool result). To separate two confounded factors — whether the cap is **announced** at
all, and how **tight** it is — we ran four arms on the **same 2 seeds** (0,1; N=40/model):

1. **200k announced** — an *overly* tight announced cap; every model's session runs right up to it.
2. **300k announced** — the headline condition (cap in prompt, `tokens_remaining` shown). The
   "300k announced" column is the seed-0/1 subset of the headline, not the full N=80.
3. **500k announced** — cap announced as in #2, but at the roomy 500k ceiling.
4. **500k silent** — model told *nothing* about a budget, `tokens_remaining` omitted; a silent
   500k ceiling bounds cost only.

**No model approached 500k** in either 500k arm (max ~362k), so that ceiling never binds; the 300k
cap binds only Sonnet, and the **200k cap binds everyone** (all three spend ~190–205k/seed, i.e. at
the cap). Everything else is held constant across all four: `D`=6, `magnitude`=0.01, write
budget=2/session, N=20/seed, seeds 0–1, 1 rep, `max_tokens`=2048, `max_turns`=300, exact-match
grading, same three models.

| | 200k announced | 300k announced | 500k announced | 500k silent |
|---|---|---|---|---|
| **Haiku** | | | | |
| Solve | 0.30 | 0.35 | 0.35 | 0.38 |
| Efficiency P(solve\|script) | 0.43 | 0.44 | 0.23 | 0.25 |
| P(solve\|by hand) | 0.23 | 0.27 | 0.41 | 0.41 |
| Recognition | 0.71 | 0.56 | 0.69 | 0.75 |
| Persistence | 3.8 | 5.5 | 3.8 | 2.5 |
| Reusability (beneficial) | 1.2 | 2.0 | 0.5 | 0.5 |
| Beneficial (LLM judge) | 0.36 | 0.44 | 0.15 | 0.25 |
| Used / no-script | 14/26 | 18/22 | 13/27 | 8/32 |
| run_script/problem | 0.45 | 0.55 | 0.38 | 0.28 |
| Tokens / seed | 190k | 250k | 227k | 198k |
| **Sonnet** | | | | |
| Solve | 0.47 | 0.50 | 0.50 | 0.47 |
| Efficiency P(solve\|script) | 0.33 | 0.52 | 0.42 | 0.27 |
| P(solve\|by hand) | 0.54 | 0.47 | 0.54 | 0.55 |
| Recognition | 0.75 | 0.52 | 0.67 | 0.64 |
| Persistence | 3.5 | 5.5 | 4.0 | 4.8 |
| Reusability (beneficial) | 0.5 | 1.8 | 0.3 | 0.0 |
| Beneficial (LLM judge) | 0.17 | 0.29 | 0.08 | 0.00 |
| Used / no-script | 12/28 | 21/19 | 12/28 | 11/29 |
| run_script/problem | 0.45 | 0.60 | 0.50 | 0.42 |
| Tokens / seed | 205k | 268k | 276k | 289k |
| **Opus** | | | | |
| Solve | 0.47 | 0.47 | 0.45 | 0.50 |
| Efficiency P(solve\|script) | 0.20 | 0.18 | 0.18 | 0.25 |
| P(solve\|by hand) | 0.57 | 0.59 | 0.55 | 0.61 |
| Recognition | 0.60 | 0.73 | 0.73 | 0.58 |
| Persistence | 2.8 | 3.0 | 3.0 | 3.5 |
| Reusability (beneficial) | 0.2 | 0.3 | 0.3 | 0.3 |
| Beneficial (LLM judge) | 0.10 | 0.09 | 0.09 | 0.08 |
| Used / no-script | 10/30 | 11/29 | 11/29 | 12/28 |
| run_script/problem | 0.28 | 0.30 | 0.30 | 0.38 |
| Tokens / seed | 190k | 181k | 188k | 185k |

**Solve rate is invariant** to both the announcement and the cap size, for every model — the cap
isn't propping up any score.

The arms decompose the budget effect into clean contrasts:

- **Announcement effect** (500k silent → 500k announced, same non-binding ceiling): **small.**
  Sonnet 11→12 problems tooled, Opus 11→11, only Haiku moves (8→13). At a roomy ceiling, merely
  *telling* the model it has a budget barely changes disposition.
- **Tightness effect** (500k announced → 300k announced, both announced): **this is the real
  lever.** Sonnet jumps 12→21 tooled and Haiku 13→18. The 300k cap is tight enough that Sonnet
  actually *hits* it (307k, truncated) — and racing a binding budget is what pushes it to
  build-and-reuse a tool.
- **Tightness is non-monotonic — 300k is a sweet spot, not a floor** (300k → 200k): squeezing
  *further* does **not** keep raising tooling; it falls back. Sonnet drops 21→12 and Haiku 18→14,
  i.e. back to roughly the *roomy* 500k-announced level. At 200k the cap binds so early that it
  **truncates the session** before much build-and-chain can happen (all three spend ~190–205k/seed,
  at the cap), and the squeeze pushes the model to *conserve* — hand-grind rather than spend tokens
  writing+running scripts. So tooling is an inverted-U in tightness, peaking near 300k; solve rate
  is flat across all of it (0.30–0.47), so the cap never props up a score.

So the earlier two-arm reading ("announced budget → more tooling") **conflated announcement with
tightness**: it's a *binding-but-not-suffocating* ceiling (~300k here), not the announcement per se,
that drives sub-frontier tooling. A roomy budget behaves like no budget; an overly tight one
truncates the very behavior it was meant to elicit.

**Opus is budget-indifferent across all four arms** — ~10–12/40 tooled, solve ~0.45–0.50,
~181–195k tokens/seed regardless. Consistent with it being the confident hand-grinder that ignores the
cap entirely.

Secondary metrics track tool *volume*: when a model tools fewer problems (the 500k and 200k arms) it
tools only the harder ones, so **Efficiency** falls and **Recognition** rises (its rarer tool use is
more often genuinely necessary — Haiku/Sonnet recognition is actually *highest* at 200k, 0.71/0.75);
**Persistence** falls with fewer tool calls, and both benefit metrics — **Reusability** (beneficial)
and **Beneficial (LLM judge)** — are low throughout and **peak at 300k** (Haiku 2.0/0.44, Sonnet
1.8/0.29), sagging at both the roomy 500k arms and the over-tight 200k. Haiku and Sonnet follow
this; Opus barely moves on any of them (judge benefit ~0.08–0.10 everywhere).

**Caveats.** (a) Two seeds; Sonnet's per-seed tool use is high-variance, so treat magnitudes as
suggestive. (b) The 300k arm partly *truncates* Sonnet (it hits the cap), and the 200k arm
truncates *all three* models — so those columns mix a behavioral effect with a truncation artifact.
This cuts both ways: it reinforces that a binding cap (not the announcement) drives the 500k→300k
rise, but it also means the 300k→200k fall is partly mechanical (less session left to tool in), not
purely a conserve-tokens choice.

Runs — 200k announced: `runs/tool_disposition_20260625_171628` (s0), `_171918` (s1); 500k silent:
`_005025` (s0), `_010316` (s1); 500k announced: `_102039` (s0), `_102041` (s1). Reproduce the 200k
arm with `--token-cap 200000`; the silent arm with `--no-token-cap --safety-cap 500000`; the 500k
announced arm with `--token-cap 500000`.

## Were the scripts load-bearing?

`used_script` only means a script was *run* on a problem, not that the answer *came* from it — so
P(solve | used script) overstates the script's causal contribution. To check, we parsed every
transcript and, for each problem where a script ran, compared the script's `return_value`(s) and
the submitted answer against the gold (at D=6). Definitions, per script-used problem:

- **script returned correct** — some `run_script` call returned the gold value;
- **answer from a script** — the submitted answer equals one of the script's return values;
- **Beneficial** — *solved* **and** the submitted answer came from a script return that was itself
  correct (i.e. the script did the work).

Pooled over the three arms (script-used problems only):

| Arm | n used | solved | script returned correct | answer from a script | **Beneficial** |
|---|---|---|---|---|---|
| 300k-announced | 50 | 0.42 | 0.34 | 0.60 | **0.32** |
| 500k-silent | 31 | 0.26 | 0.10 | 0.45 | **0.10** |
| 500k-announced | 36 | 0.28 | 0.11 | 0.42 | **0.11** |

**Mostly, the script calls were *not* load-bearing.** A run returned the correct value on only
~10% of script-used problems at the roomy 500k ceiling (34% even under the tight 300k budget), and
the answer was both correct *and* from the script on just 10–32%. Two failure modes dominate:
(1) the model writes a general/narrow script and feeds it the wrong inputs/operation for the
specific problem, so it runs cleanly but computes the wrong thing; (2) the model then *submits that
wrong output anyway* — "answer from a script" (0.42–0.60) far exceeds "script returned correct"
(0.10–0.34), so **tool-trust is miscalibrated**: wrong answers are often wrong *because* the model
deferred to a buggy script.

### Per model (script-used problems)

| Arm / Model | n used | **Beneficial** | answer from script | script returned correct | solved by hand despite a run |
|---|---|---|---|---|---|
| **300k-announced** Haiku | 18 | **0.44** | 0.78 | 0.50 | 0 |
| Sonnet | 21 | **0.33** | 0.57 | 0.33 | 4 |
| Opus | 11 | 0.09 | 0.36 | 0.09 | 1 |
| **500k-silent** Haiku | 8 | 0.25 | 0.75 | 0.25 | 0 |
| Sonnet | 11 | **0.00** | 0.36 | 0.00 | 3 |
| Opus | 12 | 0.08 | 0.33 | 0.08 | 2 |
| **500k-announced** Haiku | 13 | 0.15 | 0.46 | 0.15 | 1 |
| Sonnet | 12 | 0.08 | 0.42 | 0.08 | 4 |
| Opus | 11 | 0.09 | 0.36 | 0.09 | 1 |

Three profiles:

- **Haiku — most script-dependent; benefit swings hardest with budget.** Trusts scripts heavily
  (answer-from-script 0.46–0.78) and almost never overrides one by hand (solved-by-hand ≈ 0). Under
  the binding 300k budget it commits to a tool that fits → beneficial 0.44 (every tooled solve came
  from the script); at roomy budgets its scripts misfire → 0.15–0.25.
- **Sonnet — benefits *only* when the binding budget forces a good general tool.** Beneficial 0.33
  at 300k (its reusable `basic_calc`), but **0.00** at 500k-silent (its casual scripts returned the
  gold zero times) and 0.08 at 500k-announced. It hand-overrides the most (3–4 solved-by-hand per
  arm) when a script output looks wrong. So Sonnet's heavy tool *use* only converts to *benefit*
  under the tight budget.
- **Opus — scripts barely help in any arm (~0.09).** It tools only its hardest problems (recognition
  0.73), the scripts return correct ~1-in-11, and it cracks only 2–3 of them. Tools are a last
  resort that mostly fails too — consistent with it being the hand-grinder that gains little from
  tooling.

Net: the tight 300k budget's main effect is to raise the *quality/benefit* of the sub-frontier
models' tools (they invest in one they then rely on), not their tendency to trust outputs — which
is miscalibrated across all models and arms.

**Method caveat.** "Beneficial" requires the final answer to equal a *single* correct script
output, so it undercounts genuine *partial-work chaining* (a primitive computes an intermediate the
model then combines by hand — the behavior the benchmark is meant to reward). Runs per
script-used problem are only ~1.0–1.1 here, so chaining is rare and the undercount small, but it is
why Sonnet's `basic_calc` **reusability** (beneficial) sits well below its **persistence** (run-on)
count. (This benefit-attributed count is exactly the **reusability** metric in the tables above;
**persistence** is the older run-on count.)

### LLM-judge validation

To check the exact-match metric (and catch any partial-work it misses), a Haiku judge
(`judge.py`) reviewed every script-used transcript — problem, gold, script code, each run's
inputs→output, the model's reasoning, and the submitted answer — and ruled whether a script's
output materially fed the correct answer (directly or as an intermediate). The judged rate matches
exact-match almost exactly: **0.30 vs 0.32** (300k), **0.10 vs 0.10** (500k-silent), **0.11 vs
0.11** (500k-announced) — a single Sonnet case differs (the model ran a script but its reasoning
shows it derived and verified the answer independently, so the script wasn't relied on). So the
exact-match `beneficial` numbers are robust, and there is **negligible hidden partial-work benefit**
to credit — consistent with the rarity of chaining.

> **Judge reliability note.** A first pass let the judge re-decide correctness; Haiku made
> significant-figure errors (e.g. calling `2060480` a wrong rounding of `2060478`, though they are
> equal at 6 sig figs) and penalized correct *generic* computations for having another problem's
> variable names — both spurious downgrades. The fix was to pass the grader's `solved` verdict as
> ground truth (judge must not re-check arithmetic) and instruct that reusing a generic script
> counts regardless of its labels. With that, Haiku is a reliable judge here; the un-corrected
> prompt is not.

## What each model builds scripts for (and how it handles "products")

The scarce 2-scripts/session budget makes *what* a model spends its scripts on revealing. Across
the four headline seeds (≈8 scripts/model):

| Model | scripts written (×count over 4 seeds) | character |
|---|---|---|
| Haiku | `fence_cost` ×4, `project_time_estimate` ×2, `reynolds_number` ×2 | workhorse is a generic 3-number **multiplier** (`sides×length×cost`), rebuilt nearly every seed |
| Sonnet | `reynolds` ×3, `compound_interest` ×2, `combination` ×1, `basic_calc` ×1 | mostly **hard special-function** formulas; one general multiplier (`basic_calc`), built only on seed 0 |
| Opus | `reynolds`/`re` ×4, `compound`/`ci` ×2, `frustum` ×2 | **exclusively** hard formulas (`4ρQ/(πDμ)`, `P((1+r)^t−1)`, `πh/3·(R²+Rr+r²)`); never a multiplier |

The inventory is capability-ordered: Haiku spends scarce scripts on a *multiplier* (it needs help
multiplying); Opus spends them only on π / exponential / geometry formulas (it multiplies in its
head); Sonnet sits between (and its high persistence was largely the seed-0 `basic_calc`, a
single-seed artifact — consistent with its high seed variance).

This shows up directly in **how each handles the ~6 product-shaped problems per set** (power
`V·I·t`, parallelogram area, volume, currency, …). Reading the reasoning transcripts:

- **Haiku — repurposes a narrow script as a calculator.** It derives each problem's formula, notices
  it's a product, and calls `fence_cost` to do the multiply (mapping the problem's numbers onto the
  script's `SIDES/LENGTH/FENCE_COST` inputs). It never *says* "I'll reuse `fence_cost`" — the reuse
  is functional/implicit — and it over-trusts the result. When a problem isn't a product (Reynolds),
  it runs the script, sees it's wrong, and pivots.
- **Sonnet — reaches the same outcome by building a general tool up front.** On the product problems
  it runs `basic_calc` (a deliberately generic multiplier) rather than repurposing a narrow one,
  reserves `reynolds`/`compound` for the hard formulas, and sometimes just multiplies by hand
  (e.g. `54×22×94` written out).
- **Opus — never tools products at all.** It hand-computes every product
  (`74×17×94 = 1258×94 = 118252`) and writes scripts *only* for formulas beyond hand-arithmetic
  (Reynolds, compound interest, frustum volume).

So the capability-inverse disposition is visible at the level of *which problems get tooled*: the
stronger the model, the more it confines tooling to genuinely hard formulas and does the routine
arithmetic itself.

> **Dataset-construction caveat (product over-sampling).** `build_dataset` takes the first-N
> feasible items in CC.jsonl index order, and the early items are product-heavy: **~30%** of an
> ordered N=20 set is an exact product of its inputs, vs **~17%** in a seeded random sample
> (`--shuffle`; product counts 6,6 ordered vs 5,2,3 shuffled across seeds). Order-selection roughly
> doubles the opportunity for one multiply-script to blanket many problems, inflating measured
> persistence and reusability (for Haiku/Sonnet). Use `--shuffle` for a representative sample; the
> capability-ordered *direction* of the result is unaffected, but absolute persistence/reusability numbers on
> ordered sets are upper-ish bounds.

## Reproduction

```bash
# 1. (optional) calibrate the by-hand difficulty knob toward ~50%
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.calibrate \
    --model haiku --sample 16 --sig-figs 6 --magnitudes 0.01 0.05 0.2 1.0

# 2. run a sweep (builds/caches the dataset on first use); repeat per seed
#    add --shuffle for a representative random sample instead of product-heavy first-N-in-order
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.run_sweep \
    --models haiku sonnet opus --n 20 --sig-figs 6 --magnitude 0.01 --seed 0 \
    --token-cap 300000 --concurrency 3

# 3. recognition pass (re-runs script-used problems with tools off; edits sessions.jsonl in place)
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.recognition runs/tool_disposition_<ts>/

# 4. per-run metric table
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.analyze runs/tool_disposition_<ts>/

# 5. (optional) LLM-judge: was each script call load-bearing? -> writes judge.jsonl
PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.judge runs/tool_disposition_<ts>/ --judge haiku
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
