# Budget / Tool-Disposition — Related Work & Direction Map

*Compiled 2026-06-27.* Companion to [AIME 2026 — Script-Budget Disposition](aime-disposition.md)
and the [Tool-Disposition Benchmark](disposition-bench.md). This document surveys the literature
around our central result — *that constraining a model's tool/compute use can **raise** solve
rate, not just cut cost* — and tracks, for each paper, **which of its proposed future directions
have since been done (by later work or by us) and which remain open**, with emphasis on directions
closest to ours.

## Our work, in one paragraph (the anchor)

An in-session tool-disposition benchmark: a model solves 15 AIME-2026 contest problems in **one
session** under **two coupled budgets** — a hard total **token cap** and a separate, announced
**script-writing budget** (how many code scripts it may write) — graded by exact match. Findings:
(1) a **tighter script budget can increase solve rate** by reallocating the fixed token budget
from code-iteration toward broader problem coverage; (2) unconstrained "all-code" is the worst arm
— **catastrophic for the weakest model** (it debug-loops/thrashes) but **deadweight for the
strongest**; (3) **code pays only when a problem is genuinely computation-necessary**, else
removing code is best, most so for the strongest hand-solver; (4) the budget is *announced*,
raising an **announce-vs-enforce** question (test built via `--announce-budget`, not yet run);
(5) effects **interact with capability** (inverse: stronger model tools less / is hurt less).

## Headline verdict

The *general* phenomenon (constraining tools/compute can improve accuracy) is **well-precedented**
across several literatures, and more directions are taken than earlier drafts credited — To-Call
spans 3B–120B *and* does budget allocation, so **capability-spanning and budget allocation are NOT
our white space.** What genuinely remains: the **iterative-code retry-DEPTH axis** (how many attempts
on a problem — `n*`) and the **capability-graded verify/rewrite spiral**, neither of which can exist
in atomic-call settings (To-Call) and neither framed as *disposition* under a shared budget (vs
forced refinement in self-refine). The announce-vs-enforce lever (E0) is also unresolved elsewhere.
*(See the To-Call novelty correction below — capability/budget framings were over-claimed and are
corrected throughout.)*

---

## Direction-status map (the core table)

Each row is a future direction **proposed by some paper**; the status column says whether it has
been **done by later work**, **done by us**, or **remains open**. Sorted by closeness to our work.

| Direction (proposed by) | Status | Who did it / why open |
|---|---|---|
| **Jointly manage token limits *and* tool-call budgets** — *BATS, [2511.17006](https://arxiv.org/abs/2511.17006) App. E* | **DONE (later work + us)** | Coupled dual budgets `(B_tool, B_tok)` realized in [2605.05701](https://arxiv.org/abs/2605.05701) (search agents) and [2603.12634](https://arxiv.org/abs/2603.12634) (BAVT); **we** couple a *token cap* with a *code-script* budget specifically (BATS itself only budgets tool calls and *deprioritizes* tokens). |
| **Principled budget *allocation* under a cap** — *BATS App. E* | **DONE (later work); partly OPEN for us** | [2605.00737](https://arxiv.org/abs/2605.00737) §5 "affordability" ranks instances by marginal utility; [2601.08815](https://arxiv.org/abs/2601.08815) §6.1 gives proportional/equal/negotiated allocation + shared-pool reallocation. **Open:** empirical *across-problem* reallocation of one token pool (coverage-vs-depth) in a contest-math session — our setting, not yet formalized as allocation. |
| **Use code/tools *only when computation is necessary* (selective/adaptive)** — *implied by [2604.05404](https://arxiv.org/abs/2604.05404), [2605.26414](https://arxiv.org/abs/2605.26414)* | **DONE (method) / DONE (us, empirically)** | [2502.12022](https://arxiv.org/abs/2502.12022) (TATA) trains per-instance CoT-vs-TIR selection (fewer executions, higher accuracy). **We** show the *disposition* version: code pays only on computation-necessary problems, else hurts. |
| **Why tool calls yield *negative utility*** — *[2605.00737](https://arxiv.org/abs/2605.00737) §5* | **PARTLY DONE (us + TIR-inefficiency work)** | [2604.05404](https://arxiv.org/abs/2604.05404) catalogs four over-use patterns (confirmatory use, tool-mixing…) correlating higher tool cost with lower correctness. **We** add a concrete mechanism: under a token cap, code-iteration *thrash* starves coverage (Haiku's debug-loops). Still **open**: a predictive theory of when a call will be net-negative. |
| **Difficulty-dependent optimal compute (interior optimum)** — *[2604.10739](https://arxiv.org/abs/2604.10739), [2508.13141](https://arxiv.org/abs/2508.13141)* | **DONE for reasoning tokens; OPEN for tool/code budget** | Per-difficulty optima established for *CoT length* (easy ~1.5K vs hard ~8K tokens; CODA [2603.08659](https://arxiv.org/abs/2603.08659) allocates by difficulty). **Open / ours:** the same inverted-U for a *code/script* budget under a token cap (our "code-necessary problem" is the tool-side analog). |
| **Iterative-code retry-DEPTH + capability-graded spiral** — *no atomic-call or single-shot work has this* | **OPEN — our clearest contribution** | To-Call (atomic web search) and 2410.01748 (single-shot code) have no iteration, so no spiral. Self-refine has depth but as forced refinement, not disposition under a shared budget, and not capability-graded. **Our** "iterative all-code catastrophic for weakest (spiral), deadweight for strongest" lives here. |
| **Capability/model-size × benefit-of-tools interaction** — *[2605.26414](https://arxiv.org/abs/2605.26414), [2410.01748](https://arxiv.org/abs/2410.01748), To-Call* | **PARTLY taken — NOT our clearest contribution** | To-Call already spans 3B–120B (per-model estimators) and 2410.01748 finds code helps small models more — so "capability matters for tool decisions" is partly precedented. Our narrow remainder: capability as a *generalizing* input (leave-one-model-out) and as the *moderator of the spiral* — not the bare existence of a capability effect. |
| **Announce a budget vs. merely enforce it (is awareness causal?)** — *core to BATS, [2508.17196](https://arxiv.org/abs/2508.17196); contested by CODA* | **OPEN (our built-but-unrun test)** | BATS/BudgetThinker show *announced* budget signals help; CODA argues for *learned implicit* allocation over announcing; [2601.08815](https://arxiv.org/abs/2601.08815) §5.2 distinguishes soft-prompt vs hard-monitor ("token elasticity"). **Nobody isolates announcement alone as a causal lever on solve rate** — exactly our `--announce-budget` decoupling experiment (built, not yet run). |
| **Context/history pruning to improve task success** — *BATS App. E* | **DONE (later work)** | [2606.10209](https://arxiv.org/abs/2606.10209) prunes to last ~5 tool calls → completion 71%→79% while cutting ~64% tokens (a direct "less context, better accuracy" analog to our "tighter budget, higher solve"). |
| **Native function-calling / selective RL to curb overthinking** — *[2502.08235](https://arxiv.org/abs/2502.08235) §6.3* | **DONE (later work)** | ASPO ([2508.19201](https://arxiv.org/abs/2508.19201)) and TATA-style RL/SFT shape tool-invocation behavior; early-exit detectors (RCPD, [2508.17627](https://arxiv.org/abs/2508.17627)) cut tokens ~44% preserving accuracy. |
| **Heterogeneous / multi-dimensional tool pricing** — *BAVT [2603.12634](https://arxiv.org/abs/2603.12634) §6; Agent Contracts* | **OPEN** | Our scripts are uniform-cost; differential tool costs unstudied here. |

---

## By strand

### A. Budget-aware tool use

- **BATS — [2511.17006](https://arxiv.org/abs/2511.17006)** (the paper the user flagged). *Thesis:*
  agents lack **budget awareness** and plateau as the tool-call budget grows; an announced
  "remaining budget" tracker fixes it. **Future work (App. E, folded into Limitations):** (i) *joint
  token + tool-call budgets* — "managing multiple resource constraints jointly… token limits…
  and tool-call budgets"; (ii) *resource allocation* — "principled budget allocation strategies";
  (iii) smarter context management. **Status:** (i) **done** by BAVT / Inference-Time Budget Control
  and **by us** (token cap × script budget); (ii) **done** by To-Call §5 and Agent Contracts §6.1;
  (iii) **done** by Less-Context-Better-Agents. *Caveat for us:* BATS explicitly says **tighter
  budgets don't inherently help — awareness does**, the cleanest counter-framing to our "tighter
  script budget raises solve rate," and the reason our announce-vs-enforce test matters.
- **To Call or Not to Call — [2605.00737](https://arxiv.org/abs/2605.00737).** Decomposes each call
  into necessity/utility/affordability; **optimal selective calling beats always-call**. The
  **affordability** axis *is* budget allocation across instances (top-K under a K-call budget). Tests
  **6 models, 3B–120B** (per-model hidden-state estimators); smaller models gain more.
  > **Novelty correction (verified 2026-06-29).** To-Call is **NOT capability-blind** (spans 3B–120B)
  > and **does budget allocation** (top-K instances). So **neither "capability-indexed" nor "budget
  > allocation" is our novelty.** Its limits — and our actual differentiators — are that its tool
  > calls are **atomic** (one web search/instance): **no retry-DEPTH / iteration**, **no spiral**, and
  > **per-model estimators** (not a capability-*generalizing* predictor). It is also **web-search
  > only (no code)**. Our defensible slice: the **iterative-code retry-depth axis + the
  > capability-graded verify/rewrite spiral** (which atomic-call work cannot produce). Lead with that;
  > do not lead with capability or budget allocation.
- **BudgetThinker — [2508.17196](https://arxiv.org/abs/2508.17196).** Announced **control tokens**
  give token-budget control over CoT. **Notable caution for us:** on **AIME-2024** specifically,
  length-restriction gave *no* gain ("simply restricting the generation length does not necessarily
  yield better solutions") — but its lever is *CoT length*, not a *code budget under a token cap*,
  so it is a contrast, not a refutation. No explicit future-work section.
- **Inference-Time Budget Control — [2605.05701](https://arxiv.org/abs/2605.05701)** and
  **Spend Less, Reason Better / BAVT — [2603.12634](https://arxiv.org/abs/2603.12634).** Both
  realize **coupled (tool-call, token) budgets** — structurally our two-budget idea, in
  search/QA, not code. BAVT shows **5 tool calls beating 20** in EM (accuracy, not just cost — our
  finding 1). Both *announce* the budget in-prompt. Neither does code-execution or across-problem
  reallocation. Future work: lightweight critics; heterogeneous tool pricing; long-horizon tasks.
- **Agent Contracts — [2601.08815](https://arxiv.org/abs/2601.08815).** Formal resource bounds;
  §6.1 has explicit **cross-agent allocation** (proportional / shared-pool reallocation) and §5.2
  the **soft-announce vs hard-enforce** distinction ("token elasticity") — the two ideas nearest
  our allocation + announce-vs-enforce questions, but formal/prescriptive, not an empirical
  solve-rate study. Limitation it shares with us: a single expensive call can't be stopped
  mid-flight (token spend known only post-hoc).

### B. Overthinking / test-time compute

- **The Danger of Overthinking — [2502.08235](https://arxiv.org/abs/2502.08235).** Agentic
  SWE-bench; an overthinking score anti-correlates with success, and **selecting the
  lowest-overthinking sample beats the high-reasoning baseline at lower cost** (our finding 1, via
  *selection* not a hard cap). Capability interaction: smaller models overthink more.
  **Tension:** it finds *more* o1 reasoning effort *reduces* agentic overthinking — opposite sign
  to our "verbose reasoning is self-limiting under a token cap"; reconcile via setting (multi-turn
  agentic vs single-shot capped contest math). **Future work:** native function-calling + selective
  RL (since done by ASPO/TATA).
- **When More Thinking Hurts — [2604.10739](https://arxiv.org/abs/2604.10739).** Negative marginal
  returns past ~12K tokens; **per-difficulty optima** (easy ~1.5K, hard ~8K) — the closest analog
  to our inverted-U/sweet-spot, but for reasoning tokens, open-weight ~32B only. **Future work:**
  other domains, proprietary models, naturalistic (non-forced) elicitation, causal mechanism.
- **OptimalThinkingBench — [2508.13141](https://arxiv.org/abs/2508.13141).** Benchmarks over- and
  under-thinking jointly; **no model balances them.** **Open future work:** routers between
  fast/slow modes. Difficulty-dependence is the framing (supports our finding 3 shape).

### C. Code-vs-reasoning & tool-integrated reasoning

- **Reasoning, Code, or Both? — [2605.26414](https://arxiv.org/abs/2605.26414).** Strongest direct
  ally for our finding (3): **code execution is *less* robust than CoT** ("the interpreter executes
  the wrong solution with certainty"), and it **uses Claude Haiku 4.5** — the same model where our
  all-code arm thrashes. **Future work (it explicitly lists):** run across *multiple models* and
  *pinpoint where code helps vs hurts* — **which our capability sweep (Haiku/Sonnet/Opus) and
  code-necessary-vs-not split partially do.** Limitation: single model, single perturbation subset.
- **Not All LLM Reasoners Are Created Equal — [2410.01748](https://arxiv.org/abs/2410.01748).**
  Two-hop "compositional GSM" exposes a reasoning gap hitting **small models hardest**, and finds
  **code helps small models most** — the **opposite** of our "all-code catastrophic for the
  weakest." Key citable **tension** (resolution: AIME difficulty + script-debug-loop dynamic vs
  one-shot GSM code).
- **TATA — [2502.12022](https://arxiv.org/abs/2502.12022).** Trains **aptitude-matched adaptive
  CoT/TIR** selection: higher accuracy at fewer code executions — the method realization of "use
  code only when warranted," and notes base models of *different sizes* differ in CoT/TIR tendency
  (capability interaction). **Future work:** step-level (not instance-level) selection; RL; beyond
  math.
- **Understanding Tool-Integrated Reasoning — [2508.19201](https://arxiv.org/abs/2508.19201).**
  Theory: tools **strictly expand** the feasible support — "breaking the capability ceiling… by
  unlocking strategies otherwise impossible or intractably verbose" — and benefits extend even to
  *abstract-insight* (not just compute-heavy) problems; introduces ASPO to push *earlier* tool use.
  **Direct tension with our finding (3):** they argue tools (almost) always help in *capability*
  terms; we find tools are *deadweight/harmful* in *realized solve rate under a token budget* when
  the problem isn't computation-necessary. Reconciliation: **feasibility ≠ disposition-under-budget**
  — tools expand what's *possible*; a token cap makes their *cost* lower realized coverage.
- **Beyond Accuracy: Inefficiency Patterns in TIR — [2604.05404](https://arxiv.org/abs/2604.05404).**
  Four over-use failure modes; **higher tool cost ⇒ lower correctness**; frontier models over-tool
  and pay efficiency cost (resonates with our "deadweight for the strongest"). A diagnostic *metric*
  (PTE), not an enforced budget. No explicit future-work section.

---

## What our work uniquely occupies (white space)

*(Reordered post-correction; lead with the genuinely-unclaimed items.)*

1. **The iterative-code retry-DEPTH axis + capability-graded verify/rewrite spiral.** To-Call's calls
   are atomic and 2410.01748 is single-shot — no iteration, so no spiral. Self-refine has depth but
   forced, not as disposition under a shared budget, and not capability-graded. **This is the cleanest
   slice.**
2. **Single-shot vs iterative as the reconciling knob** — explains why 2410.01748 (single-shot)
   finds code-helps-weak while we (iterative) find code-hurts-weak; same phenomenon, different protocol.
3. **Announcement-as-causal-lever, isolated.** The announce-vs-enforce axis is *named* (Agent
   Contracts) and *assumed-beneficial* (BATS, BudgetThinker) but never **isolated** with enforcement
   held fixed — our `--announce-budget` decoupling (E0) is the first clean test.
4. **The "code-necessary problem" switch + retry-depth `n*`** as what decides whether a budget helps —
   the tool-side analog of the reasoning-token difficulty-optimum, with a *depth* dimension To-Call lacks.

**NOT our white space (corrected — see To-Call note):** capability-spanning (To-Call: 3B–120B),
budget allocation across instances (To-Call: top-K), retry-depth-helps (self-refine), interior
compute optimum (overthinking lit). Capability enters only as a *moderator of the spiral* and as a
*generalizing* predictor input (LOMO) — not as a bare "capability matters" claim.

## Tensions to address (cite, don't ignore)

- **BATS: "tightness doesn't help, awareness does."** ⇒ run the announce-vs-enforce test before
  claiming tightness is causal.
- **[2410.01748]: code helps small models most** vs our **all-code catastrophic for Haiku** ⇒
  attribute to AIME difficulty + script-debug-loop thrash (a *process* failure, not code-per-se).
- **[2508.19201]: tools strictly expand capability** vs our **code as deadweight** ⇒ feasibility
  (their claim) ≠ realized solve rate under a token budget (ours).
- **[2502.08235]: more reasoning curbs overthinking** vs our **verbose reasoning self-limits under
  a cap** ⇒ multi-turn agentic vs one-shot capped contest math.

> **Chosen direction:** we are pursuing the net-negative-call theory, anchored on capability — see
> the [research plan](net-negative-plan.md).

## Remaining open directions (candidates for us)

- Run the **announce-vs-enforce** experiment (resolves the BATS tension; fills white-space #3).
- Formalize **across-problem token reallocation** as a budget-allocation policy (BATS direction (ii)
  in *our* coverage-vs-depth setting; connect to To-Call affordability + Agent-Contracts pooling).
- A **difficulty/computation-necessity × budget** grid — the tool-side inverted-U
  ([2604.10739](https://arxiv.org/abs/2604.10739) analog).
- **OSS capability ladder** to test whether the inverse-capability interaction is frontier-specific
  (mirrors our CREATOR/ToolWorld frontier findings).

---

### Caveats on this survey
- Two arXiv ids resolved oddly: **2508.17627** (listed as "Stop Spinning Wheels") returned a paper
  titled *"The Evolution of Thought… Reasoning Completion Point"* (RCPD early-exit) — likely a
  retitle/version drift; the extracted content is the paper *at that id*. **2604.10739**'s id is as
  supplied. Treat both ids as provisional.
- Extractions were done by parallel sub-agents from arXiv HTML/PDF; future-work/limitation
  paraphrases are faithful but not exhaustive — verify exact quotes against the source before
  citing in a paper.
