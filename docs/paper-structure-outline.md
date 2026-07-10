# Paper Structure: Reusable Tool Creation as Online Investment

**Status:** writing scaffold based on the active project record as of 2026-07-10.  
**Target:** main-track NLP/ML conference paper.  
**Status note:** the publication-grade Qwen tool-transfer rerun, paired Opus tool cell, and minimum
GPT cross-family breadth replication are complete. Bracketed notes now mark writing choices or
optional scope widening, not required empirical gates.

## 0. Recommended paper identity

### Working title

**Code Before Evidence: Measuring Online Tool-Investment Capability in LLM Agents**

Alternative titles:

- **Building Before Evidence: A Capability Boundary in LLM Tool Investment**
- **When Building a Tool Hides—or Preserves—the Allocation Problem**
- **Can LLM Agents Treat Reusable Tool Creation as Online Investment?**

The first title makes the benchmark and measured capability primary rather than presupposing universal
failure. It accommodates both the large framing shifts and GPT-5.6 Sol's cross-frame preservation.
Avoid “premature commitment,” “knowing–doing gap,” universal “LLMs cannot allocate,” and any title
that implies capability scaling from the two GPT points.

### One-sentence thesis

Reusable tool creation tests whether an agent can preserve an online allocation policy when investment
requires constructive action: several competent models collapse to economically insensitive first-sight
commitment, while GPT-5.6 Sol preserves reserve across the same framing change, showing that the
capability is difficult, consequential, and achievable rather than a universal model failure.

### Claim hierarchy

The paper should make claims in this order:

1. **Formalization:** reusable tool creation is an online investment decision under uncertain future reuse and a scarce irreversible build budget.
2. **Capability measure:** paired abstract and code-required conditions distinguish possessing an allocation policy from preserving it during constructive tool use.
3. **Controlled dissociation:** with matched streams and disclosed information, Haiku reserves in an abstract allocation frame but builds immediately in a reusable-script frame.
4. **Mechanism within the Haiku benchmark:** requiring code emission is sufficient to reproduce the full first-sight commitment effect; tool-call modality, realistic problem content, correctness stakes, and a hand-solving escape valve are not required.
5. **Economic invariance in the affected model:** Haiku's abstract allocation responds to build charges, while its code-required commitment remains near 100% first-sight across eager, selective/wait, and never-build regimes.
6. **Discriminative boundary:** the shift recurs in Opus and GPT-5.4-mini but not in competence-qualified GPT-5.6 Sol, which preserves reserve across R0 and R2c. The benchmark is therefore difficult but not impossible.
7. **Acquisition without activation:** a reward-learned Qwen reserve policy generalizes across abstract vocabularies but does not reliably activate in reusable script creation.

Do not make model scaling, universal LLM behavior, or regret magnitude the organizing claim. GPT-5.6
establishes achievability and discriminative validity under the locked protocol; it does not establish
that general capability, scale, or a particular training method caused the preservation.

### Recommended result order

Lead with the Haiku same-information dissociation, then explain it with the framing ladder and economic
surface. Present RL afterward as a convergent case study, not as the paper’s foundation. End with the
competence-gated model panel: replication in GPT-5.4-mini followed by preservation in GPT-5.6 Sol.
This turns the final result into the benchmark's payoff—the measured failure is common across the tested
models but avoidable—without pretending that two GPT points define a scaling law.

---

## Abstract draft

Creating a reusable tool is an investment: an agent pays a fixed cost now in exchange for uncertain
future reuse. We introduce a paired benchmark that tests whether language-model agents preserve online
allocation behavior when the same investment decision moves from an abstract keep/pass frame to
reusable code construction. Claude Haiku reserves budget in the abstract frame but makes 100% of its
script builds on first sight, compared with 28% abstract first-sight commitment. A preregistered framing
ladder localizes this shift to required code emission: a declarative solver claim yields 39% first-sight
commitment, while requiring code—without correctness-dependent reward or a hand-solving option—raises
it to 100%. Across build charges spanning eager, selective, and never-build regimes, Haiku's abstract
choices adjust to the economics, whereas its code-required choices remain at 98–100% first sight,
including when never building is optimal. The shift replicates in Opus and, after a preregistered
competence gate, in GPT-5.4-mini (16.7% abstract versus 87.9% code-required). In contrast,
competence-qualified GPT-5.6 Sol preserves reserve across the same comparison (19.4% versus 27.8%;
paired interval includes zero), demonstrating that the tested capability is achievable rather than
structurally precluded by code generation. In a separate proof-of-possibility case study, a Qwen-14B
policy trained from reward acquires abstract reserve but remains eager in reusable script creation.
Together, these results establish online tool investment as a discriminative capability boundary:
constructive framing can suppress an available allocation policy, but a capable model can preserve it.

**Before submission:** tighten this draft to the venue word limit and verify every value against the
final matched-q8 Qwen and locked GPT analyses.

---

# 1. Introduction

## 1.1 Reusable tools create an online investment problem

Scaffold:

- Open with the deployment setting: agents can write scripts, macros, API wrappers, or skills that cost effort now but may reduce future work.
- State the missing decision: before recurrence is known, the agent must decide whether to build now, wait for evidence, or never build.
- Contrast this with conventional tool-use evaluation, which asks whether an agent can select or execute an existing tool.
- Give a simple motivating example: three available script writes, a stream of initially unfamiliar problem types, and only some types recurring often.
- Explain the failure cost: eager construction consumes scarce budget on one-off types; excessive waiting misses amortization.

Desired endpoint:

> Reusable tool creation is not only a coding problem; it is an online resource-allocation problem whose optimal action depends on uncertain future reuse.

## 1.2 Why behavior in the tool frame is hard to interpret

Scaffold:

- A model that builds eagerly might lack the allocation policy.
- It might possess the policy but fail to recognize that script creation instantiates it.
- It might understand the economics but be pulled toward solving the current problem.
- Ordinary tool benchmarks cannot separate these explanations because representation, action modality, coding, and immediate task reward change together.
- Introduce the paper’s strategy: hold streams and information fixed, then vary framing in controlled steps.
- State the criterion for a meaningful capability measure: competent models must be able to pass, while
  weaker or context-bound policies can fail under the same locked comparison.

## 1.3 Main findings

Use one short paragraph or five compact bullets:

1. **Same-information dissociation:** Haiku first-sight commitment is 28% in the abstract A2 condition and 100% in the paired tool condition.
2. **Code-emission sufficiency:** R0/R1/R2 remain at 28%/17%/39%; adding required code in R2c raises commitment to 100%, matching full script construction.
3. **Economic invariance:** R2c remains 98–100% first-sight over `K ∈ {0,10,24}`, while R0 becomes cautious as building becomes selective or dominated.
4. **Measured failure breadth:** Opus reproduces the frontier dissociation, and GPT-5.4-mini shows a
   preregistered +75.8-point paired first-sight shift after passing the abstract competence gate;
5. **Achievable preservation:** GPT-5.6 Sol passes the same gate and preserves reserve in R2c, showing
   only +8.3 points with an interval crossing zero. Qwen separately shows that acquisition in the
   abstract frame does not guarantee activation during reusable construction.

## 1.4 Contributions

State five contributions:

- An online tool-investment formulation and discriminative benchmark with paired streams, scarce writes, uncertain recurrence, and explicit behavioral timing metrics.
- A same-information test distinguishing absent allocation competence from framing-dependent non-expression.
- A causal mechanism study combining a preregistered framing ladder, a code-required control, and an economic response surface.
- A reward-learning case study showing that acquisition in an abstract frame does not imply activation during reusable construction.
- A preregistered competence-gated replication in GPT, with seed-paired uncertainty and
  both a large framing shift and successful cross-frame preservation. The latter establishes that the
  benchmark measures a difficult but achievable capability.

Do not list every diagnostic as a separate contribution.

Suggested page budget: 0.9–1.1 pages.

---

# 2. Related Work

Keep this section compact and organized around distinctions the experiments need.

## 2.1 LLM tool use, tool making, and skill libraries

Scaffold:

- Cite tool-use and tool-making systems, especially work that amortizes an expensive tool over many future instances.
- Explain that prior systems typically assume recurrence, build up front, or optimize post-hoc retrieval, pruning, and maintenance.
- Position this paper at the missing decision point: whether to author an irreversible reusable artifact before future reuse is known.
- Distinguish online build timing from library drift and skill retirement.

## 2.2 Cost-aware agents and exploration

Scaffold:

- Cover budget-aware tool use, cost-aware model/tool selection, exploration–exploitation, and stopping.
- Distinguish one-shot spending for the current item from a fixed construction cost amortized over a future stream.
- Connect calibration-before-action approaches to possible mitigation, not to the paper’s novel object.

## 2.3 Online algorithms and rent-or-buy decisions

Scaffold:

- Introduce ski-rental and online rent-or-buy as the formal precedent.
- State what is imported: fixed cost, uncertain duration/reuse, and delayed evidence.
- State what is new: constructive actions generated by LLM agents, multiple latent problem classes, a scarce global write budget, and a controlled framing comparison.
- Avoid implying that the benchmark is a direct textbook ski-rental instance in every detail.

## 2.4 Framing, content effects, and policy transfer

Scaffold:

- Situate the same-information comparison among work on abstraction gaps, content effects, elicitation, and context-bound learned policies.
- Make clear that the method is established but the domain—online reusable-tool investment—is new.
- Relate the RL result to out-of-distribution policy transfer and asymmetric/privileged critics.

Suggested page budget: 0.7–0.9 pages.

---

# 3. Reusable Tool Creation as Online Investment

## 3.1 Decision process

Define:

- A horizon of `T` problems drawn from `N` latent problem types.
- A global budget of `B` irreversible builds.
- At each first or subsequent occurrence of a type, the agent may decline, build, or reuse an existing artifact.
- Building incurs a one-time charge or opportunity cost; reuse yields future value.
- Type frequencies are initially unknown, so the agent must infer recurrence online.

Include a compact mathematical formulation:

- History `h_t`, remaining budget `b_t`, latent type `c_t`, and action `a_t`.
- Utility as solved/collected value minus build charges.
- The policy objective over the finite horizon.

Keep the notation sufficient for the experiments; move full dynamic-programming details to the appendix.

## 3.2 Benchmark instantiation

Scaffold:

- Describe the eight numeric problem families and uniform-hard construction.
- Explain hot and trap types, `T=60`, typical `N=8`, and early-trap stress condition.
- Define the abstract urn frame and reusable-script frame as views of the same class stream.
- Explain why hand solving is approximately unavailable in the hardened pool and how script correctness is calibrated.
- State which streams are paired across conditions.

## 3.3 Information conditions

Scaffold:

- Define no-N as the ecologically realistic condition.
- Define A2 as exact disclosure of `N`.
- Explain that A2 is the main causal condition because model and reference receive the same type-count information.
- Do not claim that matching `N` removes every prompt difference; it removes the identified information asymmetry.

## 3.4 Outcome measures and reference policies

Lead measures:

- First-sight commitment: proportion of realized commitments made on occurrence one.
- First-sight commitment hazard: commitments on eligible first-sighting turns divided by eligible first-sighting turns.
- Lateness: commitment occurrence minus first occurrence.
- Trap allocation and budget exhaustion.

References:

- Use the online Bayesian Dirichlet policy only as a labelled behavioral comparator in the original urn task.
- Use the exact hindsight net optimum for economic-surface regret and explicitly call it an upper-bound/offline reference, not an online policy.
- Define regret currencies separately: balls or net points for the economic experiments; avoid cross-task magnitude comparisons.

## 3.5 Hypotheses

State the hypotheses before presenting results:

- H1: disclosing `N` improves abstract allocation but does not eliminate eager script construction.
- H2: the largest framing-ladder discontinuity occurs when code emission becomes required.
- H3: abstract commitment changes with build charge, whereas code-required commitment is less economically elastic.
- H4: an allocation policy learned in the abstract frame need not transfer to reusable script creation.
- H5: among models that pass the abstract competence gate, code-required preservation is neither
  guaranteed nor impossible; the paired benchmark can expose model-level capability differences.

Suggested page budget: 0.9–1.1 pages.

---

# 4. Experimental Protocol

## 4.1 Models and conditions

Scaffold:

- List exact model identifiers and versions.
- Explain why Haiku is the primary controlled subject: demonstrated abstract competence, clean paired tool behavior, and tractable repeated evaluation.
- Explain why Qwen-14B is the learning subject: adequate script competence but absent baseline allocation policy.
- Add Opus and the additional model family after publication-gate runs.
- State decoding, context, tool-choice, and quantization settings.

## 4.2 Pairing, seeds, and effective sample size

Scaffold:

- State that the core Haiku comparisons use 12 canonical paired streams.
- Clarify that hundreds of frame-cell sessions reuse these 12 stream clusters; the effective independent stream sample is 12, not the raw session count.
- Describe held-out Qwen seeds and disjoint training seeds.
- Report seed-clustered uncertainty and all seed-level points.
- Emphasize effect sizes and consistency over asymptotic p-values.

## 4.3 Mechanical-validity safeguards

Scaffold:

- Stream-hash assertions and byte-identical class sequences.
- Distinct transport-error and model-decision failure labels.
- No behavioral-outcome-driven reruns; only mechanical failures trigger repair.
- Token-cap checks, malformed/unknown tool-call accounting, and preregistered unresolved-decision thresholds.
- Briefly disclose the tool-choice and prompt/token-cap bugs found during ladder development, with full chronology in the appendix.

## 4.4 Statistical reporting

Scaffold:

- Paired seed-level differences.
- Seed-clustered bootstrap intervals.
- Numerators and denominators for commitment percentages.
- Separate confirmatory outcomes from post-hoc diagnostics such as R2c code correctness.
- No treating turns or repeated cells on the same stream as independent observations.

Suggested page budget: 0.6–0.8 pages.

---

# 5. Same Information, Different Allocation

This is the first and cleanest results section.

## 5.1 Core 2×2 design

Present:

- Frame: abstract urn versus reusable script.
- Information: no-N versus A2.
- Identical streams, budget, horizon, and disclosed `N` in A2.

Primary figure:

- A four-cell plot of first-sight commitment and lateness.
- Show paired seed points, not only means.
- Put regret in a small secondary panel or appendix.

## 5.2 N disclosure reveals a framing dissociation

Core result:

- Haiku abstract: 53% to 28% first-sight after `N` disclosure.
- Haiku tool: 100% to 100%.
- In A2, lateness is 1.19 abstract versus 0 tool.
- The model uses recurrence information in the abstract frame but not during reusable script creation.

Language discipline:

- Say “matches the specified Bayesian comparator on these streams” rather than “provably optimal.”
- Say “same disclosed type-count information,” not “all cognition held constant.”
- Lead with timing behavior, not regret.

## 5.3 Transcript evidence: solving now displaces investing for later

Scaffold:

- Summarize the tool-frame rationale pattern: immediate problem solving, reactive reuse, retrospective budget awareness, no prospective reserve reasoning.
- Include two short transcript excerpts: one abstract wait decision and one eager script build.
- Use transcripts as mechanism-consistent qualitative evidence, not as the causal identification by themselves.

## 5.4 Abstract competence and constructive preservation are distinct axes

Scaffold:

- Use the abstract condition as a competence gate rather than interpreting every eager tool policy alike.
- Separate three observed states:
  - policy absence or non-expression in both frames, as in baseline Qwen;
  - abstract competence with framing-dependent suppression, as in Haiku, Opus, and GPT-5.4-mini;
  - abstract competence with cross-frame preservation, as in GPT-5.6 Sol.
- Briefly preview the completed paired Opus and GPT results, reserving full statistics for Section 9.
- Explain why the third state matters: it shows that the benchmark is passable and that code generation
  does not mechanically force first-sight commitment.
- Do not claim smooth capability scaling or identify the training feature responsible for preservation.

Suggested page budget: 1.0–1.2 pages.

---

# 6. What Triggers Eager Construction?

## 6.1 Preregistered framing ladder

Define the rungs:

- R0: abstract free-text keep/pass.
- R1: abstract keep/pass tool calls.
- R2: real-problem declarative solver claim, with no code.
- R3: full reusable-script construction.

State the causal limit:

- This is a coarse ladder, not a complete factorial design.
- The preregistered criterion is the earliest adjacent jump of at least 30 percentage points.

Primary figure:

- Horizontal ladder plot with first-sight percentage and paired seed points.
- Annotate adjacent changes: `−11`, `+22`, `+61` percentage points.

## 6.2 Tool calls and realistic problem content do not reproduce the full effect

Present:

- R0/R1/R2 first-sight: 28%/17%/39%.
- Neither modality alone nor declarative commitment crosses the preregistered threshold.
- Keep the R1→R2 action-label/content confound explicit.

## 6.3 Requiring code is sufficient

Introduce R2c:

- Same binary claim/skip loop and payoff as R2.
- No hand-solving option.
- No correctness-dependent reward.
- Only added requirement: provide non-empty solver code.

Present:

- R2c first-sight 100%, lateness 0, matching R3.
- Preferred claim: “requiring code emission is sufficient to reproduce the observed first-sight commitment effect.”
- Avoid broad psychological language such as “construction burden causes myopia” unless carefully qualified as benchmark-local shorthand.

## 6.4 Code quality reveals current-instance fixation

Treat as a secondary, post-hoc diagnostic:

- Only 2% of tested R2c instances are correct.
- Most claims contain structurally plausible but instance-specific code with hard-coded incidental parameters.
- This converges with the allocation finding: the model produces code for the current instance rather than a reusable abstraction.
- Do not make correctness part of the causal R2-versus-R2c comparison, because payoff was intentionally unconditional.

Suggested page budget: 0.9–1.1 pages.

---

# 7. Code-Required Commitment Is Economically Inelastic

## 7.1 Design: budget × build charge

Scaffold:

- Compare R0 and R2c on identical streams.
- Cross `B ∈ {1,3,5}` with the plotted charge axis `K ∈ {0,10,24}`.
- Explain the regimes:
  - `K=0`: building hot classes readily pays.
  - `K=10`: selective/wait regime; hot classes pay, traps do not.
  - `K=24`: never building is exactly optimal on all canonical streams.
- Mention the retained `K=20` near-never data only in the appendix or extended analysis.

## 7.2 Exact economic reference

Scaffold:

- Define the exact hindsight net optimum: choose up to `B` classes with largest positive `size_c − K`.
- State that it is prior-free and cap-free.
- Be explicit that it is an offline upper bound used for realized net regret, not an online policy available to the model.
- Retain the online Bayesian comparator only for timing intuition where tractable.

## 7.3 Abstract allocation responds to price

Present:

- R0 first-sight hazard falls as charge rises.
- At `K=10`, R0 waits: first-sight 0–16%, lateness 1.56–3.75.
- At `K=24`, R0 is still imperfect but substantially less eager.
- Describe R0 as economically sensitive, not optimal.

## 7.4 Code-required commitment remains pinned to first sight

Present:

- R2c is 98–100% first-sight over the surface and exactly 100% in the genuine wait-band cells.
- At `K=24`, it builds despite the proof that every build is net-negative.
- At `K=10`, it commits immediately even though selective evidence gathering is valuable.
- Regret grows with charges and budget because eager commitments spend more of the charged budget.

Primary figure:

- Use `fig_economic_response_surface` with clear labels for R0, R2c, and exact hindsight optimum.
- Caption must state that all conditions reuse 12 paired stream clusters.

Main interpretation:

> The code-required frame does not merely shift a threshold; it largely removes behavioral elasticity to visible build economics.

Suggested page budget: 1.0–1.2 pages.

---

# 8. Reward-Learned Reserve Remains Context-Bound

Frame this as a case study supporting the recognition account, not as a robust RL contribution.

## 8.1 Why train Qwen-14B?

Scaffold:

- Baseline Qwen-14B is eager in both frames and therefore lacks the target policy.
- It has sufficient script competence to make transfer meaningful.
- Training asks whether reward can install reserve, followed by a zero-shot activation test in script creation.

## 8.2 Training setup and disclosure

Scaffold:

- Per-decision PPO/QLoRA, 20 outer steps, no demonstrations.
- Training occurs only in the abstract urn frame.
- The critic receives privileged true-rate information during training.
- One successful training run is reported.
- State explicitly that this is a proof of possibility, not evidence of training robustness, sample efficiency, or ordinary non-privileged learnability.
- Move pilot history and implementation detail to the appendix.

## 8.3 Reserve is acquired in the training frame

Current result:

- Held-out first-sight falls from 75% to 32%.
- Balls collected rises from 87% to 101% of the Bayesian reference.
- Lateness rises from 0.375 to 0.903.
- Show paired seed points and clarify that training seeds are disjoint.

## 8.4 The learned policy does not activate in reusable script creation

Use the publication-grade rerun here:

- Matched quantization.
- Calibrated script correctness.
- Robust empty-fence retry/session handling.
- At least 24 paired seeds.
- Report first-sight and lateness as primary; sustained engagement and regret as diagnostics.

Final result (publication-grade rerun, 2026-07-10; matched q8_0, n=24 paired seeds, EFR4, per-tag
calibrated `a_script`):

> The same checkpoint remains **95% first-sight in the tool frame** (58/61 builds, lateness 0.131) despite
> **32%** in the urn, and is indistinguishable from the base control (95%, 59/62). Across all 48 sessions
> the tool channel is clean (0 malformed / 0 unknown / 0 refused, 0/24 token-cap hits), so build-count
> (2.5/seed) and regret (RL-final 1821±396, base 1909±417, at calibrated `a_script` 0.94/0.89) are
> reported as de-biased diagnostics; first-sight and lateness remain the lead. This is the confounder-clean
> version of the transfer boundary: same model, same quant, same disclosed information, opposite behavior
> across frames.

## 8.5 Boundary tests: vocabulary versus action modality

Scaffold:

- Free-text reskins preserve reserve: pooled RL 19% versus base 89%.
- Isomorphic keep/pass tool calls retain only partial transfer: RL 62% versus base 99%, with strong vocabulary dependence.
- Therefore the wall is not absolute and is not explained by literal ball vocabulary.
- The strongest supported interpretation is a boundary around decision modality and constructive script context.
- Keep the cauldron-specific speculation out of the main text; report it in the appendix.

Suggested page budget: 0.9–1.1 pages.

---

# 9. A Difficult but Achievable Capability

This section is the benchmark-level payoff. It should remain short, but it must treat GPT-5.6 Sol as
positive evidence that cross-frame preservation is achievable—not merely as a failed replication.

## 9.1 The framing failure extends beyond the primary model

Present:

- Opus under matched A2 abstract and tool conditions: 0% abstract versus 100% tool first-sight,
  with identical canonical streams and the hardened magnitude setting.
- GPT-5.4-mini after the locked abstract competence gate: 16.7% R0 versus 87.9% R2c, paired
  difference +75.8 points, bootstrap 95% CI [+54.5, +93.9].
- These results show that Haiku's dissociation is neither model-specific nor confined to a weaker
  capability tier.
- Preserve GPT-5.4-mini's seed-level heterogeneity: one wait-y R2c seed and one zero-commit seed.

## 9.2 GPT-5.6 Sol preserves allocation across construction

Present the result as a successful capability demonstration under the same preregistered test:

- `gpt-5.6-sol` passes the R0 gate: 19.4% first-sight and 98.5% of comparator payoff.
- In paired R2c it remains selective: 27.8% first-sight, paired difference +8.3 points,
  bootstrap 95% CI [−11.1, +27.8].
- Operational interpretation: under the locked prompt, information, and stream contract, Sol retains
  the timing policy that the affected models lose when code becomes required.
- Construct-validity implication: code emission does not mechanically induce eagerness, and the metric
  is not an impossible trap. It distinguishes failure from successful integration of allocation and
  construction.
- Scientific implication: the central phenomenon is a capability boundary, not a universal law.
- Causal limit: this comparison does not show that generic capability or scale produced preservation.
  Differences in post-training, reasoning policy, architecture, or tool-use experience remain live.

## 9.3 Interpreting the model panel

Synthesize the three-state pattern:

- **Absent policy:** baseline Qwen is eager in both frames.
- **Possessed but suppressed policy:** Haiku, Opus, and GPT-5.4-mini reserve abstractly but become
  sharply eager under code-required construction.
- **Preserved policy:** GPT-5.6 Sol reserves in both frames.

This taxonomy is more informative than a single pooled model effect. It separates whether a model has
the allocation policy from whether it can carry that policy into constructive action.

Do not:

- Rank the models by overall intelligence from this benchmark.
- Infer monotonic scaling or inverse scaling from two GPT points.
- Claim that Sol is economically optimal; the GPT panel tests cross-frame timing preservation, not the
  full Haiku budget-by-charge response surface.
- Average model effects into a universal “LLM” coefficient that erases the capability boundary.

Scope-widening work can test more frontier and open-weight families under the same prompt-information
contract and seed pairing, then ask whether preservation predicts performance on naturalistic agent tasks.
It is not required for the present bounded claim.

Suggested page budget: 0.4–0.7 pages. If space is tight, fold this into Sections 5 and 6.

---

# 10. Discussion

## 10.1 Constructive preservation is a distinct capability

Scaffold:

- Haiku, Opus, GPT-5.4-mini, and GPT-5.6 Sol demonstrate the policy in the abstract frame.
- Qwen demonstrates that a policy can be acquired from reward.
- Code-required construction suppresses that policy in Haiku, Opus, and GPT-5.4-mini, but not
  detectably in GPT-5.6 Sol under the locked protocol.
- Distinguish policy possession, policy acquisition, and policy activation.
- Treat Sol's preservation as evidence that activation during construction is a measurable capability,
  not merely the absence of a statistically significant failure.
- The benchmark is most useful as a conditional diagnostic: first establish abstract competence, then
  measure whether timing survives constructive reframing.

## 10.2 Why code generation may suppress investment reasoning

Offer bounded interpretations:

- Immediate problem-solving priors dominate prospective allocation.
- Emitting code invokes a reactive coding-assistant mode.
- Construction may compress “solve now” and “invest for later” into one salient action.
- R2c rules out correctness stakes and hand-solving availability as necessary causes, but does not identify a neural mechanism.
- Mark representation-level explanations as future work.

## 10.3 Implications for agent design

Scaffold:

- Separate the “should we build?” decision from “build the artifact now.”
- Maintain an explicit recurrence and remaining-budget ledger.
- Require evidence or predicted reuse before authoring persistent tools.
- Evaluate tool-building systems on delayed utility and library opportunity cost, not only immediate task success.
- Present these as design implications, not tested mitigations.

## 10.4 Implications for capability evaluation

Scaffold:

- Evaluate online investment separately from coding correctness and existing-tool selection.
- Require a competence gate so eager behavior can be attributed to policy absence versus suppression.
- Include both abstract and constructive conditions; either condition alone misses the dissociation.
- Reward benchmarks that admit successful preservation rather than treating universal model failure as
  evidence of difficulty.
- Test whether this metric predicts library quality, resource allocation, and long-horizon agent
  performance in naturalistic environments.

## 10.5 Implications for training

Scaffold:

- In-frame success does not guarantee cross-frame activation.
- Training distributions may need to include constructive tool contexts, not only abstract allocation.
- Because the reported critic is privileged and only one training run is available, avoid claims about the best training algorithm.

Suggested page budget: 0.6–0.8 pages.

---

# 11. Limitations

Use a dedicated section rather than scattering all caveats.

## 11.1 Synthetic environment and task scope

- One numeric stream environment with engineered hot/trap recurrence.
- Real task types may be ambiguous, nonstationary, and only partially reusable.
- R2c payoff is intentionally independent of code correctness, so it isolates code emission rather than realistic artifact value.
- Naturalistic task streams remain the most important external-validity extension.

## 11.2 Model and sample breadth

- The core causal ladder is centered on Haiku.
- Twelve canonical streams are repeatedly crossed with conditions; raw session count is not independent sample size.
- Opus and GPT-5.4-mini extend the observed failure, while GPT-5.6 Sol demonstrates preservation; this
  is enough to establish heterogeneity and achievability, not a capability trend.
- The benchmark has not yet been validated as a predictor of broader agent performance.

## 11.3 Reference policies

- The Bayesian urn comparator is prior-dependent and mismatched to the fixed hot/trap generator.
- The economic hindsight optimum is exact but unavailable online.
- Behavioral timing is therefore the primary cross-frame outcome.

## 11.4 RL evidence

- One successful training run.
- Privileged critic.
- Quantization and session-pathology issues in the original transfer evaluation; describe how the final rerun addresses them.
- No claim of robust retraining, non-privileged learnability, or training efficiency.

## 11.5 Prompt and harness sensitivity

- Tool choice and prompt format affect deliberation.
- The ladder is coarse rather than fully factorial.
- The R1→R2 action-label/content contrast remains bundled.
- Mechanical fixes were transparently logged but occurred during benchmark development.

Suggested page budget: 0.4–0.6 pages.

---

# 12. Conclusion

One paragraph:

- Restate reusable tool creation as an online investment problem.
- Summarize the central dissociation, code-emission trigger, economic invariance, and learned-policy activation boundary.
- State the positive benchmark result: GPT-5.6 Sol preserves reserve across the locked framing change,
  proving that online allocation during constructive action is achievable by a capable model.
- End with the scientific implication: online tool investment is a distinct capability worth measuring,
  and the affected agents may need explicit separation between deciding to invest and executing construction.

Suggested page budget: 0.2–0.3 pages.

---

# Recommended Main-Paper Figures

## Figure 1: Task and same-information dissociation

- Left: one stream shown as abstract colors and numeric problem types.
- Center: decision timelines for reserve versus first-sight construction.
- Right: Haiku 2×2 first-sight/lateness with paired seed points.
- Purpose: explain the benchmark and establish the core result immediately.

## Figure 2: Framing ladder and R2c

- R0, R1, R2, R2c, R3 in conceptual order.
- First-sight percentage with uncertainty and individual seeds.
- Annotate what changes between adjacent rungs.
- Purpose: localize the discontinuity to required code emission.

## Figure 3: Economic response surface

- `B × K` panels or lines for R0, R2c, and exact hindsight optimum.
- Main axis `K={0,10,24}`.
- First-sight hazard primary; regret secondary.
- Purpose: show economic sensitivity versus invariance.

## Figure 4: Learned policy and transfer boundary

- Left: base versus RL in original abstract, held-out vocabularies, isomorphic tool calls, and reusable script frame.
- Right: competence-gated R0/R2c model panel for GPT-5.4-mini and GPT-5.6 Sol, with Opus summarized
  separately because it uses the A2 urn/tool endpoint.
- First-sight percentage with paired seed uncertainty and channel-validity annotations.
- Purpose: show both forms of boundary evidence—acquisition without constructive activation and a
  model that successfully preserves allocation across construction.

If the venue permits only three main figures, combine Figures 1 and 2 or move the vocabulary panel to the appendix.

---

# Recommended Main-Paper Tables

## Table 1: Experimental conditions

Include frame, information, action interface, real problem content, code required, correctness-dependent payoff, hand-solving option, model, and paired seeds.

## Table 2: Core numerical results

Include only the main behavioral measures for:

- Haiku A2 abstract versus tool.
- Ladder R0/R1/R2/R2c/R3.
- Qwen base/RL abstract versus tool.
- Opus and the competence-gated GPT panel, visually distinguishing framing-shift replication from
  cross-frame preservation.

Avoid duplicating every plotted economic cell.

## Table 3: Scope and caveats

Prefer placing this in the appendix unless venue norms favor a compact robustness table.

---

# Appendix Structure

## A. Full benchmark specification

- Problem families and stream generator.
- Prompt text and information disclosures.
- Tool schemas and session transitions.

## B. Reference-policy details

- Bayesian DP formulation and cap validation.
- Prior-mismatch discussion.
- Proof of the exact hindsight net optimum.
- K=24 never-build proof.

## C. Framing-ladder preregistration and deviations

- Locked threshold.
- Rung parity checks.
- Tool-choice and prompt/token-cap repair chronology.
- Abstract `claim_solver` control as unrun future work.

## D. Complete economic results

- All `K={0,10,20,24}` cells.
- Net points, regret, first-sight hazard, lateness, and zero-commit incidence.
- All seed-level points and clustered intervals.

## E. RL method and pilot history

- Critic features and privileged-information disclosure.
- PPO objective and training hyperparameters.
- Step-0 diagnostic and training curves.
- Explicit statement that only one successful training run is analyzed.

## F. Tool-transfer session diagnostics

- Empty-fence pathology.
- Retry procedure.
- Valid/malformed/unknown calls.
- Quantization and calibration details.
- Representative transcripts.

## G. Vocabulary and modality probes

- Per-vocabulary seed-level results.
- Cauldron collapse.
- Unparsed counts and prompt correction.

## H. Model breadth

- Opus paired cells.
- Additional-family competence gates.
- Per-model prompts, settings, and results.

## I. Reproducibility

- Artifact manifest.
- Exact commands.
- Seed lists.
- Model versions and dates.
- Missing historical Qwen artifacts and which publication-grade reruns replace them.

---

# Writing Rules for the Final Manuscript

Use:

- “Code-required commitment” or “required code emission.”
- “First-sight commitment” and “lateness” as primary outcomes.
- “Matches the specified Bayesian comparator” for the original urn reference.
- “Exact hindsight net optimum” for the economic upper bound.
- “Proof-of-possibility RL case study.”
- “Within this benchmark” whenever making the mechanism claim.
- “Cross-frame preservation” for GPT-5.6 Sol.
- “Difficult but achievable capability” when summarizing the competence-gated model panel.
- “Capability boundary” only when immediately paired with the observed model-level heterogeneity.

Avoid:

- “All LLMs fail.”
- “The framing wall is absolute.”
- “π* is the true optimum.”
- “Construction burden alone” without immediately defining it as required code emission.
- “216 independent trials”; use “216 sessions over 12 paired stream clusters.”
- “RL robustly learns” or “RL transfers poorly in general.”
- “The model understands” based only on a transcript; use behavioral descriptions.
- “Sol proves scaling” or “stronger models recover”; one model establishes achievability, not the cause.
- “GPT-5.6 is a non-replication” without also stating the substantive positive result: reserve is
  preserved under required code emission.
- “Sol is economically optimal”; the GPT panel does not include the full charge response surface.

---

# Condensed Eight-Page Allocation

- Introduction: 1.0 page
- Related work: 0.7 page
- Formalization + protocol: 1.2 pages
- Same-information result: 0.9 page
- Framing ladder: 1.0 page
- Economic surface: 1.0 page
- RL + breadth: 1.2 pages
- Discussion, limitations, conclusion: 1.0 page

If space becomes tight, preserve the core sequence:

1. same-information dissociation;
2. code-emission sufficiency;
3. economic invariance;
4. competence-gated model panel, including GPT-5.6 Sol preservation;
5. RL acquisition-without-activation.

If another cut is required, compress RL before cutting the GPT preservation result. Move the no-N
condition, full Qwen ladder, detailed regret tables, K=20 cells, idle-tail diagnostics, and
vocabulary-specific results to the appendix.

---

# Material to Exclude from the Main Narrative

The repository contains several valuable earlier research threads, but combining them with this paper would weaken the causal story:

- **ToolWorld/WoodWorld build-versus-grind experiments** and `fig_build_vs_grind_region_2n`: related construct-and-exploit behavior, but a different estimand from online build timing.
- **AIME script-budget sweeps:** useful motivation for cost-sensitive tool use, but too weakly replicated and operationally different for the main evidence chain.
- **CREATOR capability and confabulation threads:** informative project history, but cross-family effects are heterogeneous and do not support the paper’s framing claim directly.
- **Superseded SFT/adapter-transfer attempts:** retain only as internal history unless needed to explain why the final intervention uses PPO.
- **Old capability-law narratives:** do not revive inverse-scaling or monotonic-capability claims that later experiments falsified.

At most, mention these as motivation or companion evidence in related work or an appendix. The main
paper should remain the July online-investment program: same-information dissociation, code-emission
mechanism, economic invariance, competence-gated failure versus preservation, and acquisition without
activation.
