# How This Project Differs from UltraTool

> Findings grounded in a deep-research pass (2026-06-21): 5 search angles, 14
> sources fetched, 64 claims extracted, 25 adversarially verified (3-vote), 24
> confirmed 3-0, 1 killed. UltraTool claims below are verified against the primary
> source ([arXiv:2401.17167](https://arxiv.org/abs/2401.17167), ACL 2024 Findings)
> and its [code](https://github.com/JoeYing1019/UltraTool). Companion to
> [lit-review-tool-creation-disposition.md](lit-review-tool-creation-disposition.md)
> and [CREATOR-fork-plan.md](CREATOR-fork-plan.md).

## Bottom line

UltraTool is the prior benchmark closest to this project — it has an explicit
**tool-creation** phase and sweeps many models across sizes. But it measures the
**orthogonal axis**: tool creation is **forced/instructed every instance** and
scored as **ability/quality**, and creation performance scales **monotonically
*with* capability** (GPT-4 best). This project measures **disposition** —
`P(builds | could profit)` when the model is *free not to build* — and finds it
scales **inversely** with capability (Haiku 0.95 > Sonnet 0.80 > Opus 0.60).
UltraTool therefore does not contradict the inversion; it measures a different
variable, and its monotonic-ability result actually *strengthens* the disposition
story (stronger models can build better — so when they don't, it's
recognition/disposition, not inability).

## What UltraTool actually is (verified)

UltraTool decomposes real-world tool utilization into **three aspects → six
scored dimensions** and maps cleanly onto the tool-learning pipeline taxonomy
(planning → tool selection → tool calling → response generation) from the survey
[arXiv:2405.17935](https://arxiv.org/abs/2405.17935):

| Aspect | What it scores | How |
|---|---|---|
| **Planning** | task decomposition into sub-steps | LLM-judge / structural match |
| **Tool Creation** — *Awareness* | does the model detect that existing tools are insufficient and a **new** tool is needed | **gap-detection accuracy vs. a gold label** (`cal_acc_for_aware`, 0/1 KeyValue accuracy) |
| **Tool Creation** — *Creation* | quality of the created tool definition | **LLM-judge across five quality dimensions** |
| **Usage** (selection / calling / etc.) | correct tool choice, arguments, invocation | accuracy vs. gold |

Verified specifics (all 3-0):

- **Creation is FORCED, not spontaneous.** Per §3.3 the model is *required to
  create* the lacking tool; the task is prescribed for every applicable instance.
  There is no condition in which the model freely chooses whether building pays.
- **Everything is scored as per-stage ability/correctness** — accuracy against gold
  annotations, or LLM-judged quality. No metric scores a *choice* under a real
  build-cost / reuse-payoff tradeoff.
- **Model sweep & per-model creation scores:** GPT-4 is best overall
  (~76.04 Chinese / ~74.58 English tool-utilization). On tool creation
  specifically (Table 2): **GPT-4 ≈ 65.55, Qwen-7B ≈ 19.40, LLaMA2-7B ≈ 3.24** —
  i.e., creation ability tracks capability, with weak small models scoring near
  the floor.
- **Scaling is monotonic-positive** (the strongest model wins; capability helps).
  *Caveat from verification:* a strictly-ordered version of this claim
  (7B < 13B < 70B with no exceptions) was **killed (1-2)** — Mistral-7B
  over-performs its parameter count, so strict size-ordering among small
  open-source models does not hold. The **general trend runs with capability**;
  the **clean per-parameter monotonicity does not.**
- **No inverse-capability analysis anywhere**, and **disposition is never isolated
  from ability.**

## The nearest neighbor — and why it still isn't disposition

UltraTool's **Awareness** dimension is the closest thing in the literature to a
recognition metric: it asks whether the model *notices* that no existing tool
suffices. But it is scored as **classification accuracy against a gold "you should
create here" label**, inside a pipeline that then *forces* creation. That differs
from this project's recognition axis in three decisive ways:

1. **Conditioned, not free.** Awareness is evaluated where the gold says a tool is
   needed; the model never faces the live option of *not* building when building
   would pay (or building when it wouldn't).
2. **Accuracy, not propensity.** It grades match-to-gold, not `P(builds)` as a
   behavioral tendency — so it cannot reveal a model that *can* recognize the need
   but *declines* to act, which is exactly the ToolWorld/Opus failure mode.
3. **No cost/reuse economics.** There is no resource cost to building and no
   amortization benefit to reuse, so "should I build?" has a single gold answer
   rather than a genuine tradeoff.

## Difference table

| Dimension | UltraTool | This project (ToolWorld / WoodWorld) |
|---|---|---|
| **Creation trigger** | forced/instructed every instance (§3.3) | spontaneous; build action offered but not required |
| **Measurement target** | ability / quality per pipeline stage | **disposition**: `P(builds \| could profit)` |
| **Recognition type** | discriminative awareness vs. gold label | **generative** recognition of an unbuilt-tool opportunity |
| **Capability scaling** | **monotonic-positive** (GPT-4 best) | **inverse** (Haiku 0.95 > Sonnet 0.80 > Opus 0.60) |
| **Ability vs. disposition** | conflated (never isolated) | **isolated** via spontaneous-vs-forced manipulation |
| **Setting** | static, real-world-scenario QA, tools/goal given | sequential environment with exploration, build-cost, reuse payoff |
| **Mechanism for failure** | not studied | confabulation / under-recognition (not inability) |

## Why the contrast is favorable, not threatening

UltraTool is the strongest "someone already did this" objection to the project,
and the deep-research pass resolves it cleanly:

- It establishes that **creation ability scales up with capability** (GPT-4 ≫
  Qwen-7B ≫ LLaMA2-7B). That rules out the alternative explanation that strong
  models build less *because they can't* — corroborating the project's
  "under-recognition, not inability" finding from an independent external source.
- Because UltraTool **forces** creation and scores **ability**, the project's
  variable — the **free choice** to build and its **inverse** scaling — is left
  entirely unmeasured. The orthogonality is structural, not a matter of degree:
  UltraTool has no condition and no metric that could surface it.

## Caveats

- Exact Table 2 figures (GPT-4 65.55 / Qwen-7B 19.40 / LLaMA2-7B 3.24) and the
  76.04/74.58 overall scores are from verified secondary extraction of the paper's
  tables; confirm against the published PDF/repo before quoting in a paper.
- The monotonic-with-capability claim is robust at the trend level but **not** at
  strict parameter-count granularity (Mistral-7B exception) — state it as
  "creation ability rises with capability," not "monotone in parameter count."
- The disposition-vs-ability framing is this project's interpretive lens; UltraTool
  does not describe itself in these terms.

## Sources

| Source | Role |
|---|---|
| [UltraTool — arXiv:2401.17167](https://arxiv.org/abs/2401.17167) | primary (paper) |
| [UltraTool code — JoeYing1019/UltraTool](https://github.com/JoeYing1019/UltraTool) | primary (metrics: `cal_acc_for_aware`) |
| [ACL Findings 2024 entry](https://aclanthology.org/2024.findings-acl.259/) | primary (publication) |
| [Tool Learning survey — arXiv:2405.17935](https://arxiv.org/abs/2405.17935) | primary (pipeline taxonomy) |
| [alphaXiv 2401.17167v2](https://www.alphaxiv.org/abs/2401.17167v2), [HF papers](https://huggingface.co/papers/2401.17167) | corroborating |
