# LLM Tool Creation via Code: Benchmarks, Methods & the Disposition Gap

> Deep-research literature survey (2026-06-20). 21 sources fetched, 98 claims
> extracted, 25 adversarially verified (3-vote, all confirmed 3-0, none killed),
> synthesized to 10 findings. Generated to position the reusable-action-discovery
> (WoodWorld/ToolWorld) project against prior work.

## Bottom line

Across **every** major tool-creation, library/abstraction-learning, and
embodied-skill-creation system surveyed, building a reusable tool is **forced or
structurally mandated — never measured as a spontaneous disposition**. None
isolates *recognition* (does the model decide on its own that a reusable tool
would pay off) from *ability* (can it write one). They uniformly report end-task
accuracy or abstraction utility instead.

This is precisely the whitespace this project occupies: measuring
`P(builds | could profit)` as a disposition variable, separate from build
ability — and finding it scales *inversely* with capability
(Haiku 0.95 > Sonnet 0.80 > Opus 0.60).

## How each system mandates creation

| System | Citation | Creation is… | What it measures | Domain |
|---|---|---|---|---|
| **LATM** | [2305.17126](https://arxiv.org/abs/2305.17126) | A prescribed pipeline phase; "tool maker" role **assigned to the stronger model** (GPT-4) because tool-making "requires more sophisticated capabilities" | End-task accuracy, **cost** | Big-Bench |
| **CREATOR** | [2305.14318](https://arxiv.org/abs/2305.14318) | Instructed "Creation" stage with demonstrations, every instance | Accuracy; frames variation as "tool creation **ability**" | MATH, TabMWP, etc. |
| **TroVE** | [2410.20274](https://arxiv.org/html/2410.20274v1) | Induced toolbox step | Accuracy + library size | Programmatic/math |
| **ReGAL** | [2401.16467](https://arxiv.org/abs/2401.16467) | Unconditional refactoring of an existing corpus | Program-prediction accuracy | LOGO, Date, TextCraft, MATH, TabMWP |
| **LILO** | [2310.19791](https://arxiv.org/abs/2310.19791) | `compress()` runs **every loop**, model can't opt out | % held-out tasks solved | REGEX, CLEVR, LOGO |
| **DreamCoder** | [ACM 3453483](https://dl.acm.org/doi/10.1145/3453483.3454080) | Mandatory wake-sleep abstraction phase (symbolic, not an LLM) | Compression/generalization | Inductive prog., drawing, physics |
| **Voyager** | [2305.16291](https://arxiv.org/abs/2305.16291) | `add_skill()` auto-fires on task success — side effect, no payoff deliberation | Exploration/task progress | Minecraft (JS skills) |
| **Alita** | [2505.20286](https://arxiv.org/abs/2505.20286) | Self-initiated via capability self-assessment, but disposition + ability collapsed into one pipeline | Accuracy only (GAIA 75% pass@1) | Web/code agent (MCP/Python) |

## C·R·E decomposition (project-framework overlay)

The project decomposes a tool episode into three axes — **Curiosity** (act to
discover affordances / resolve uncertainty before committing), **Recognition**
(spontaneously recognize that building/abstracting a reusable tool pays off), and
**Efficiency** (execute, and actually amortize the built tool, without redundant
work). ToolWorld is the fundamental case: it isolates **generative recognition**
(conceive-and-combine) and finds it *inverts* with capability. Laying every
surveyed system over these axes makes the gap concrete — **R is forced or clamped
in all of them; only C and E are ever measured.**

A recognition sub-distinction matters: **generative** R = construct a tool that
isn't on offer (ToolWorld combine, CREATOR abstract-a-function); **discriminative**
R = select the right tool from a provided menu (WildToolBench). ToolWorld's
inversion is a claim about *generative* R specifically.

| System | Curiosity (C) | Recognition (R) | Efficiency (E) |
|---|---|---|---|
| **ToolWorld** *(this project)* | shallow, non-limiting (exploration has slack) | **generative** combine/build — *measured, and inverts* (Haiku 0.95 > Sonnet 0.80 > Opus 0.60) | high, ~flat; Haiku over-acts post-build |
| **CREATOR** | absent — fully observed, single-shot | **generative** abstract-a-function — **forced** every instance | weak — only solution correctness, no trace |
| **WildToolBench** | clarify-under-uncertainty (epistemic ask-user, not exploration) | **discriminative** select-from-menu + abstain on Chat | strong — parallel / sequential / dependency-flow / arg-filling |
| **LATM** | n/a (Big-Bench given) | **assigned to the *stronger* model by fiat** — the *anti-inversion* premise; never the model's choice | cost-amortization: weak user reuses maker's tool; dispatcher routes by cache-hit/coverage |
| **TroVE** | n/a | **forced** toolbox-induction step | reuse *is* measured and **collapses** — 3 / 3,201 |
| **ReGAL** | n/a (refactors a fixed corpus) | **removed** — abstractions extracted from existing code; premise = "LLMs lack the disposition" | downstream program-prediction accuracy (abstraction utility) |
| **LILO** | n/a | **unconditional** `compress()` every loop — cannot opt out | % held-out solved; Stitch symbolic compression as the amortizer |
| **DreamCoder** | n/a | **mandatory** wake-sleep abstraction — *symbolic, not an LLM* | MDL compression *is* the objective (compression/generalization) |
| **Voyager** | **present and driving** — automatic curriculum + Minecraft exploration | `add_skill()` **auto-fires** on success; no payoff deliberation | skills stored as a side effect; reuse incidental; self-verify ablation −73% |
| **Alita** | mild — MCP-Brainstorming / open-source retrieval | self-initiated via capability self-assessment — *triggered but not isolated/measured* | reuse of constructed MCP tools; reported only as GAIA accuracy |

Three observations this overlay surfaces:

- **LATM is the clean anti-finding.** It hard-codes the assumption that generative
  recognition *requires* capability (tool-making "requires more sophisticated
  capabilities" → assign to GPT-4) — the exact opposite of ToolWorld's measured
  inversion. Same axis, opposite predicted direction, never tested as a disposition.
- **Voyager is the only curiosity-rich system** and the closest in modality to
  ToolWorld/WoodWorld — yet it clamps R to an automatic side effect, so its
  embodied setting still can't speak to disposition.
- **TroVE (and the reuse-failure cluster) is an Efficiency-collapse result**: the
  library gets built but is essentially never amortized, so even the E column —
  the one these systems *do* measure — fails to establish payoff-driven abstraction.

*(This overlay is interpretive, like the inverse-capability mapping noted in
Caveats — the papers do not frame themselves in C·R·E terms.)*

## Detailed findings (all verified 3-0)

### Tool-creation methods

- **LATM** ([2305.17126](https://arxiv.org/abs/2305.17126)) — Closed-loop two-phase
  framework: a "tool maker" LLM crafts reusable Python tools
  (Tool Proposing + Verification + Wrapping), a "tool user" LLM applies them.
  Tool-making is **explicitly assigned to a powerful model** ("Recognizing that
  tool-making requires more sophisticated capabilities, we assign this task to a
  powerful... model"). The optional dispatcher routes only by cache-hit/coverage
  (does a tool already exist?), **not** by recognizing payoff — so it is not
  spontaneous creation. The model is never measured on whether it would *choose*
  to build a tool.

- **LATM cost tradeoff (inverse-capability adjacent)** — Abstract verbatim:
  "With GPT-4 as the tool maker and GPT-3.5 as the tool user, LATM demonstrates
  performance equivalent to using GPT-4 for both roles, but with a significantly
  reduced inference cost." This is the closest thing in the literature to an
  inverse-capability framing, but it is a **cost/ability stratification**, not a
  disposition finding (within-paper, Big-Bench, not independently replicated).

- **CREATOR** ([2305.14318](https://arxiv.org/abs/2305.14318)) — Genuine
  tool-*creation* (not selection): LLMs "create their own tools using
  documentation and code realization." Four stages (Creation, Decision,
  Execution, Rectification). But the Creation stage "explicitly instruct[s] LLMs
  with demonstrative examples to create tools" for every instance — **forced**.
  The titular "disentangling" is *abstract-vs-concrete reasoning*, **not**
  disposition-vs-ability. Outcomes attributed to varying "tool creation
  abilities."

### Library / abstraction learning

- **ReGAL** ([2401.16467](https://arxiv.org/abs/2401.16467)) — Gradient-free
  method learning reusable functions via **code refactorization of a pre-existing
  corpus**. Abstractions are extracted from existing code, not spontaneously
  created. Measures downstream program-prediction accuracy (e.g. CodeLlama-13B
  +11.5% LOGO). Premise: LLMs "lack the global view needed to develop useful
  abstractions" — explicitly designed to **compensate for** missing disposition,
  never to measure it.

- **LILO** ([2310.19791](https://arxiv.org/abs/2310.19791)) — LLM synthesis +
  Stitch symbolic compression + AutoDoc. Algorithm 1 runs `compress()`
  ("Generate new abstractions") **unconditionally every iteration** — the model
  cannot opt out. Primary metric: % held-out tasks solved (REGEX 93.2% vs
  DreamCoder 43.9%; CLEVR 96.8%; LOGO 48.9%). No experiment lets the model decide
  whether to abstract.

- **DreamCoder** ([ACM 3453483](https://dl.acm.org/doi/10.1145/3453483.3454080))
  — Wake-sleep loop alternately extends the language with new symbolic
  abstractions and trains the neural net; E-graph refactoring builds a
  progressively deepening library. Abstraction is the loop's built-in objective
  (minimizing description length). **Symbolic neuro-guided synthesis, not an LLM
  agent** — its "spontaneity" is architectural, so only loosely comparable.

### Embodied / agentic skill creation

- **Voyager** ([2305.16291](https://arxiv.org/abs/2305.16291)) — Ever-growing
  skill library of executable code. Loop: automatic curriculum proposes task →
  generate code → iterative env-feedback/self-verification → on success
  `skill_manager.add_skill(code)` **automatically**. No deliberation about
  reusability/payoff; storage is a side effect of solving curriculum-proposed
  tasks. Ablating self-verification dropped performance ~73% — creation depends
  on the framework, not the model's disposition.

- **Alita** ([2505.20286](https://arxiv.org/abs/2505.20286)) — Autonomously
  constructs/refines/reuses MCP tools (Python) from open source via MCP
  Brainstorming → ScriptGeneratingTool → CodeRunningTool. Creation **is**
  self-initiated by capability self-assessment (closer to spontaneous than
  human-instructed), **but** the paper conflates disposition and ability into one
  pipeline and measures only accuracy (GAIA 75.15% pass@1, 87.27% pass@3;
  MathVista 74.00%; PathVQA 52.00%). No experiment isolates recognition from
  ability. ("Embodied" is a slight overreach — it's a web/code agent.)

## Two findings most relevant to this project

**1. Learned "tool libraries" are rarely actually reused.** Verified 3-0:
TroVE reused learned functions in just **3 of 3,201** test questions; LEGO-Prover
reused **exactly one lemma, once**. A compute-matched re-evaluation
([2507.22069](https://arxiv.org/abs/2507.22069)) traces the claimed gains to
**self-correction / self-consistency, not reuse** (library-sharing ablation
matched/exceeded baseline, Bonferroni p<0.05, two MATH splits). Corroborated by a
LEGO-Prover case study ([2504.03048](https://arxiv.org/abs/2504.03048)). So these
systems don't even establish that spontaneous, payoff-driven abstraction is
happening.

**2. The closest thing to inverse-capability scaling is LATM's labor split** —
strong maker + weak user matches all-strong at lower cost. But this is cost/ability
stratification, not disposition. No paper shows *disposition* scaling with
capability, let alone scaling *inversely*.

## The gap, stated precisely

No surveyed system includes a **spontaneous-vs-forced manipulation** (e.g.,
offering a build action without requiring it) or any **build-disposition metric
independent of build-ability**. Every system mandates creation:

- LATM / CREATOR / Alita make it an instructed or assigned pipeline stage.
- ReGAL / LILO / DreamCoder run an unconditional refactoring/compression pass
  every loop.
- Voyager stores skills automatically on task success.

Metrics are uniformly end-task accuracy or abstraction utility. This directly
maps to the **recognition-vs-ability** framing — existing work measures
ability/utility and treats tool-making as a capability/cost-allocation question
(LATM even assigns it to the stronger model), leaving the disposition axis and
any inverse-capability scaling of disposition **unmeasured**.

## Open questions (whitespace this project can fill)

1. Does any benchmark explicitly manipulate spontaneous-vs-forced tool creation
   (offering but not requiring a build action) and report a disposition metric
   distinct from accuracy? None surfaced — a genuine gap.
2. Does tool-creation *disposition* scale inversely with capability the way the
   WoodWorld/ToolWorld results suggest? Surveyed papers show only ability/cost
   stratification (LATM), not disposition scaling.
3. Is the library-reuse-failure finding (TroVE/LEGO-Prover) specific to
   math/theorem-proving benchmarks, or would payoff-driven reuse appear in
   high-sub-task-recurrence embodied settings (e.g. Voyager)?
4. How should a benchmark operationalize "payoff recognition" — e.g. contrasting
   build-cost vs. expected amortized reuse — so a model's *choice* to build is
   scored independently of whether the build *succeeds*?

## Caveats

- DreamCoder is symbolic synthesis, not an LLM agent — only loosely comparable.
- "Spontaneous" is nuanced for Alita and Voyager: both let the agent *initiate*
  creation (Alita via capability self-assessment, Voyager via curriculum), but
  neither *measures* disposition separately from ability. The gap claim is about
  absence of measurement/isolation, not absence of any autonomous trigger.
- The inverse-capability mapping is an interpretive overlay onto the source
  framing, not a claim the papers make in those terms.
- The library-reuse-failure results, while replicated, are specific to math/
  theorem-proving benchmarks (TroVE on MATH, LEGO-Prover on miniF2F) and may not
  generalize to all library-learning settings.
- Newer systems (Alita, May 2025; re-evaluation papers, 2025) are recent and
  lightly replicated.
- A few benchmark-angle sources surfaced with future-dated arXiv IDs
  (e.g. 2603.x, 2604.x) that should be eyeballed before citing; the **named-system
  findings above all rest on well-established IDs** and are high-confidence.
- Not an exhaustive census — other benchmarks may partially address the gap.

## Primary sources

| System / topic | arXiv / DOI |
|---|---|
| LATM | [2305.17126](https://arxiv.org/abs/2305.17126) |
| CREATOR | [2305.14318](https://arxiv.org/abs/2305.14318) |
| ReGAL | [2401.16467](https://arxiv.org/abs/2401.16467) |
| LILO | [2310.19791](https://arxiv.org/abs/2310.19791) |
| DreamCoder | [10.1145/3453483.3454080](https://dl.acm.org/doi/10.1145/3453483.3454080) |
| Voyager | [2305.16291](https://arxiv.org/abs/2305.16291) |
| Alita | [2505.20286](https://arxiv.org/abs/2505.20286) |
| TroVE reuse analysis | [2410.20274](https://arxiv.org/html/2410.20274v1) |
| Compute-matched re-evaluation | [2507.22069](https://arxiv.org/abs/2507.22069) |
| LEGO-Prover case study | [2504.03048](https://arxiv.org/abs/2504.03048) |
