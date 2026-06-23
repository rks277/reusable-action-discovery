# Annotated Bibliography

> Every external source consulted for the reusable-action-discovery
> (WoodWorld / ToolWorld / CREATOR-fork) project, grouped by the role it played.
> Compiled 2026-06-22 from the project's deep-research passes and design docs.
> Companion `.bib` file: [refs.bib](refs.bib).
>
> Provenance: sources were gathered across three adversarially-verified
> deep-research passes — tool-creation/disposition
> ([lit-review-tool-creation-disposition.md](lit-review-tool-creation-disposition.md)),
> inverse-scaling/confabulation
> ([inverse-scaling-agentic-confabulation.md](inverse-scaling-agentic-confabulation.md)),
> and the UltraTool contrast ([project-vs-ultratool.md](project-vs-ultratool.md)) —
> plus the open-world-discovery landscape scan in
> [6-16-summary.md](6-16-summary.md).

---

## 1. Tool-creation & library/abstraction-learning systems
*Consulted to establish the project's whitespace: in every prior system, building a
reusable tool is **forced or structurally mandated**, and only ability/utility is
measured — never disposition (`P(builds | could profit)`) isolated from ability.*

- **LATM — Large Language Models as Tool Makers** (`2305.17126`).
  Consulted as the clean *anti-finding*: it hard-codes that tool-making "requires
  more sophisticated capabilities" and assigns it to the stronger model (GPT-4),
  the exact opposite of our measured inverse-capability disposition. Also the
  closest thing in the literature to an inverse-capability framing (strong maker +
  weak user = all-strong at lower cost), but as cost/ability stratification, not
  disposition.

- **CREATOR — Tool Creation for Disentangling Abstract & Concrete Reasoning**
  (`2305.14318`).
  The base system our CREATOR-fork is built on. Consulted for its four-stage
  pipeline (Creation, Decision, Execution, Rectification) and to show its Creation
  stage is *forced* every instance and its "disentangling" is abstract-vs-concrete
  reasoning, not disposition-vs-ability. See
  [CREATOR-fork-plan.md](CREATOR-fork-plan.md).

- **ReGAL — Refactoring Programs to Discover Generalizable Abstractions**
  (`2401.16467`).
  Consulted as the "LLMs lack the disposition" premise made explicit: it learns
  reusable functions by refactoring a *pre-existing corpus*, designed to compensate
  for missing disposition rather than measure it.

- **LILO — Learning Interpretable Libraries by Compressing & Documenting Code**
  (`2310.19791`).
  Consulted as the unconditional-compression case: `compress()` runs every loop and
  the model cannot opt out; measures % held-out tasks solved.

- **DreamCoder** (ACM DOI `10.1145/3453483.3454080`, PLDI 2021).
  Consulted as the symbolic wake-sleep abstraction baseline whose "spontaneity" is
  architectural (MDL objective), not an LLM choice — only loosely comparable.

- **Voyager — An Open-Ended Embodied Agent with LLMs** (`2305.16291`).
  The closest system in modality to WoodWorld/ToolWorld and the only curiosity-rich
  one. Consulted because `add_skill()` auto-fires on task success — recognition is
  clamped to a side effect, so even this embodied setting can't speak to
  disposition (self-verify ablation −73%).

- **Alita** (`2505.20286`).
  Consulted as the nearest "self-initiated" creation (via capability
  self-assessment), but disposition and ability are collapsed into one pipeline and
  only accuracy (GAIA) is reported.

- **TroVE — Inducing Verifiable & Efficient Toolboxes** (`2410.20274`).
  Consulted for the reuse-failure result: learned functions reused in just **3 of
  3,201** test questions — an Efficiency-collapse finding.

- **Compute-matched re-evaluation of tool-library gains** (`2507.22069`).
  Consulted as the source tracing claimed library-learning gains to
  self-correction / self-consistency, *not* reuse (library-sharing ablation
  matched/exceeded baseline, Bonferroni p<0.05).

- **LEGO-Prover case study** (`2504.03048`).
  Consulted as corroboration of the reuse-failure cluster: LEGO-Prover reused
  exactly one lemma, once.

## 2. UltraTool — the nearest-neighbor benchmark
*Consulted as the strongest "someone already did this" objection, and to show it
measures the orthogonal axis (forced creation, scored as ability, monotonic with
capability). See [project-vs-ultratool.md](project-vs-ultratool.md).*

- **UltraTool** (`2401.17167`, ACL 2024 Findings).
  Primary source. Consulted for its three-aspect / six-dimension decomposition; its
  *Awareness* dimension is the closest literature analogue to our recognition axis
  but is scored as classification accuracy vs. a gold label inside a forced-creation
  pipeline. Verified Table-2 per-model creation scores (GPT-4 ≈ 65.55 ≫ Qwen-7B ≈
  19.40 ≫ LLaMA2-7B ≈ 3.24) establish that *creation ability* rises with capability
  — corroborating "under-recognition, not inability."
- **UltraTool code** (`JoeYing1019/UltraTool`, GitHub).
  Consulted for the actual metric implementation (`cal_acc_for_aware`).
- **ACL Findings 2024 entry** (`aclanthology.org/2024.findings-acl.259`).
  Consulted to confirm publication venue.
- **Tool Learning with Foundation Models — survey** (`2405.17935`).
  Consulted for the tool-learning pipeline taxonomy (planning → selection → calling
  → response) that UltraTool maps onto.

## 3. Inverse / U-shaped scaling
*Consulted to name and ground the inverse-capability scaling we observe (Haiku >
Sonnet > Opus) and to separate the capability axis from the budget axis. See
[inverse-scaling-agentic-confabulation.md](inverse-scaling-agentic-confabulation.md).*

- **Inverse Scaling Prize — McKenzie et al. 2023, TMLR** (`2306.09479`).
  Consulted as the empirical foundation that bigger ≠ better (11 datasets) and for
  its four causes of inverse scaling.
- **Inverse scaling can become U-shaped — Wei et al. 2022** (`2211.02011`).
  Consulted for the U-shaped reversal (4/11 stayed inverse, 6/11 became U-shaped at
  540B) — the basis for treating capability as the axis where a U could appear and
  for the ≥3-capability-point test design (Haiku → Sonnet → Opus).

## 4. Sycophancy, calibration & prior-over-context
*Consulted as the documented "ingredients" of the Opus confabulation mode —
truth-overriding that worsens with scale/RLHF — and to bound what is *not* novel.*

- **Discovering LM Behaviors w/ Model-Written Evals — Perez et al. 2022**
  (`2212.09251`).
  Consulted for sycophancy scaling with size and RLHF (framed as inverse scaling).
- **Towards Understanding Sycophancy — Sharma et al. 2023** (`2310.13548`).
  Consulted for sycophancy from preference-model optimization sacrificing
  truthfulness. (Difference from our finding: sycophancy targets the *user's*
  belief; Opus overrides truth for its *own* prior fiction.)
- **GPT-4 Technical Report — OpenAI 2023** (`2303.08774`).
  Consulted for RLHF degrading calibration (ECE ~10× worse post-training).
- **Factuality enhancement vs. context-faithfulness** (`2404.00216`).
  Consulted for the finding that strengthening parametric knowledge makes models
  overlook input context.
- **Situated Faithfulness / calibrated trust** (`2410.14675`).
  Consulted both for the prior-vs-observation trust-calibration framing (exactly
  what Opus fails) and as a candidate grounding intervention for future work.

## 5. Agentic / tool-use hallucination
*The field our confabulation finding sits in; consulted to show existing taxonomies
**cover** the behavior definitionally but do not document it with a positive
capability gradient (so the compound appears novel).*

- **MIRAGE-Bench** (`2507.21017`).
  Consulted for its three unfaithfulness axes (esp. "unfaithful to environment
  observations") and for **"presumptive hallucination"** — the nearest prior art to
  fabricating user/environment turns, but framed as an instruction-tuning bias, NOT
  a capability gradient.
- **Agent-hallucination survey** (`2509.18970`).
  Consulted for the "perception hallucinations" and "execution hallucinations"
  category names.
- **AgentHallu — Jan 2026** (`2601.06818`).
  Consulted to show tool-use hallucination is the field's hardest open problem
  (best model 11.6% on tool-use hallucinations) — the exact locus of our finding.
  *Caveat: dated near/at the knowledge cutoff.*
- **Contested context-vs-scale source** (`2603.09654`).
  Consulted during verification; **two claims sourced here were refuted** (1-2
  votes) — the context-faithfulness-vs-scale direction is genuinely contested.
  *Caveat: future-dated arXiv ID — eyeball before citing.*

## 6. Open-world RL / embodied tool discovery
*Consulted to map the landscape of "build/discover a tool to make progress" and
confirm our niche — obfuscated, un-enumerated affordance discovery tied to model
capability — is unclaimed. See [6-16-summary.md](6-16-summary.md).*

- **Crafter — Hafner 2021** (`2109.06780`).
  2D Minecraft-like with a 22-achievement tech tree where tool crafting *is* the
  progression. Tightest match to "build a tool to progress."
- **Craftax / Craftax-Classic — Matthews et al. 2024** (`2402.16801`).
  JAX rewrite of Crafter (~250× faster); explicitly an open-ended skill-discovery
  benchmark.
- **MineDojo — Fan et al. 2022** (`2206.08853`).
  Thousands of Minecraft tasks, many requiring tool fabrication, with a CLIP-based
  learned reward.
- **NetHack Learning Environment — Küttler et al. 2020** (`2006.13760`).
  Extreme exploration with deep, discoverable item/tool affordances.
- **SmartPlay** (`2310.01557`) and **contrastive-achievement Crafter work**
  (`2307.03486`).
  Consulted as the bridge: they already benchmark *LLM agents* on these discovery
  tech trees.
- **OpenAI hide-and-seek — Baker et al. 2019** (`1909.07528`).
  The canonical "agents discover tool use no one told them about" (ramp use,
  ramp-locking, box-surfing) — including strategies the researchers didn't know the
  environment supported.
- **PHYRE — Bakhtin et al. 2019** (`1908.05656`).
  Physical-reasoning tool placement: place a body so a goal state emerges after
  simulation.
- **Virtual Tools game — Allen et al. 2020** (`2312.10728`).
  Choose one of several tools to place; heavy exploration emphasis.
- **KinDER — 2026** (`2604.25788`).
  25 procedurally-generated robot-reasoning envs, one explicitly tool use.
  *Caveat: future-dated arXiv ID.*
- **CRAFT** (`2309.17428`).
  Consulted in the tool-creation taxonomy: write/define a tool for a *stated*
  problem — closest LLM bucket, but the agent is told the task and that a tool
  should be made.

---

## Named but not separately fetched
The following tool-*use* / *selection* benchmarks were named in the taxonomy in
[project-vs-ultratool.md](project-vs-ultratool.md) to delimit our scope (they
enumerate tools → not discovery), but were not fetched as primary sources:
**ToolBench, Berkeley Function-Calling Leaderboard, API-Bank, τ-bench, ToolEmu,
MetaTool**.

## General caveats on these sources
- Several recent IDs are future-dated (`2601.x`, `2603.x`, `2604.x`) and should be
  eyeballed before formal citation; the well-established named-system findings rest
  on solid IDs.
- The C·R·E and inverse-capability mappings are *interpretive overlays* — the source
  papers do not frame themselves in those terms.
- MineDojo and NetHack arXiv IDs (`2206.08853`, `2006.13760`) are standard
  references not present verbatim in our docs; verify before quoting.
- UltraTool Table-2 figures come from verified *secondary* extraction — confirm
  against the published PDF before quoting in a paper.
