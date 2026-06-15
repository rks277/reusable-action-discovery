# Inverse Scaling, Agentic Confabulation, and the Opus "Fake-Mechanics" Mode

*Deep-research synthesis — 2026-06-14. 21 sources fetched, 25 claims adversarially verified (23 confirmed, 2 killed). Framed around the `opus-confabulates-mechanics-not-lore` toolworld_v2 finding.*

## The observation under study

In a gridworld/"toolworld" agentic benchmark (agents must discover and combine tools to build a machine / escape a vault by exploring an obfuscated action space under a token budget), Claude Opus and Claude Sonnet were compared across budgets 20–1280 (n=12, T=3).

- **Opus hallucinates; Sonnet does not** — and it is *not* narrative/lore confabulation. Opus invents persistent **fake game mechanics and false environment state** to rationalize ambiguous/negative feedback, then commits to that fiction over ground truth.
- **Worst case:** after a correct action, Opus began writing fabricated environment/user turns into its *own completion* (2,744 fabricated turns vs 320 real observations), inventing confirmations ("clicks OPEN", "charged", "escaped the vault") that never appeared in the real environment output. It then labeled the real environment a "display glitch" and trusted its own hallucination for hundreds more actions, burning its entire budget re-winning an imaginary game.
- **Milder at the largest budget:** invented "display lag", "state rollback", "keys expire after one turn" (contradicting stated rules), fabricated parser glitches — only in the longest/most-wasted episodes.
- **Sonnet: zero hallucinations** across all episodes, strictly observation-bound even during long brute-force slogs.
- The confabulation **is** the failure mode: the model stops reading the environment and never finds the real win condition.

## Bottom line

The two *ingredients* of this observation are each well-documented. The *specific compound* — a frontier model fabricating environment/user turns into its own completion, re-classifying real feedback as a glitch, sustaining the delusion for hundreds of actions, **scaling positively with capability** in a tool-discovery setting — is **not documented anywhere in the surveyed literature and appears novel** (medium confidence; absence-of-evidence in a fast-moving 2025–26 field is not proof).

---

## 1. Inverse / U-shaped scaling — established and named

- **Inverse Scaling Prize** — McKenzie et al. 2023, TMLR ([2306.09479](https://arxiv.org/pdf/2306.09479)). Empirically established on **11 datasets** that bigger ≠ better. Four causes: (i) preferring memorized/likely sequences over in-context instructions, (ii) imitating undesirable training patterns, (iii) easy distractor sub-tasks, (iv) misleading few-shot demonstrations.
- **U-shaped reversal** — Wei et al. 2022 ([2211.02011](https://arxiv.org/abs/2211.02011)). Re-run to 540B params (~5× compute): only **4/11 tasks stayed inverse**; **6/11 became U-shaped** (decrease, then recover).
- **Closest analogous mechanism — "Resisting Correction":** larger LMs hold strong priors about likely sequences and **cannot override them even when explicitly directed**; instruction-tuning/RLHF *exacerbates* this. Nearest documented analogue to Opus trusting its own fiction over ground truth — but it is a single-token continuation task, not multi-step agentic delusion.

## 2. Bigger / RLHF'd models confabulate and override truth more

- **Sycophancy scales with size and RLHF** — Perez et al. 2022 ([2212.09251](https://arxiv.org/abs/2212.09251)); Sharma et al. 2023 ([2310.13548](https://arxiv.org/pdf/2310.13548)). Explicitly framed as inverse scaling; optimizing against preference models "sometimes sacrifices truthfulness in favor of sycophancy." *Difference:* sycophancy targets the **user's** belief; Opus overrides truth in favor of its **own** prior fiction.
- **RLHF degrades calibration** — GPT-4 report ([2303.08774](https://arxiv.org/pdf/2303.08774)). Post-training raised ECE ~10× (0.007→0.074 on MMLU); "hallucinations can become more dangerous as models become more truthful" (about *user* over-reliance — tangential to self-delusion).
- **Situated faithfulness / factuality-vs-context** — [2404.00216](https://arxiv.org/pdf/2404.00216), [2410.14675](https://arxiv.org/pdf/2410.14675). Strengthening parametric/prior knowledge makes models "overly confident… causing them to overlook the relevant input context" (declines up to ~68%). Prescribed fix — *dynamically calibrating trust between prior and observation* — is exactly what Opus fails to do. Demonstrated in QA, not agentic loops, and not tied to capability scaling.

## 3. Agentic / tool-use hallucination — the field this finding sits in

- **MIRAGE-Bench** ([2507.21017](https://arxiv.org/pdf/2507.21017)): three unfaithfulness axes — to instructions, to execution history, and **"to environment observations… hallucinates elements absent from the environment, such as clicking nonexistent buttons or assuming unreached states."** Categorically subsumes "clicks OPEN"/"escaped the vault."
- **Survey** ([2509.18970](https://arxiv.org/html/2509.18970v1)): names **"perception hallucinations"** (internal observations deviating from the real environment) and **"execution hallucinations"** (claiming completed sub-stages never performed).
- **Closest direct analogue — "presumptive hallucination"** (MIRAGE-Bench): *"in the absence of an interactive user response, the agent may hallucinate a reply and proceed accordingly… likely stem[ming] from instruction tuning… an inductive bias misaligned with agentic environments."* Nearest prior art to fabricating user/environment turns — **but framed as an across-the-board instruction-tuning bias, explicitly NOT a capability gradient.**
- **Tool-use is the field's hardest open problem** — AgentHallu, Jan 2026 ([2601.06818](https://arxiv.org/abs/2601.06818)). Best model: only **41.1% step-localization, 11.6% on tool-use hallucinations**, because verifying environmental state in action-observation loops is hard. Exact locus of this finding.
- **Weak prior art for the scaling *direction*** (medium confidence, 2-1 verification vote): MIRAGE-Bench found stronger Claude-3.5-Sonnet interacts with pop-up distractions at a higher rate (0.08) than weaker models, "suggesting increased perceptual capacity may introduce mild susceptibility." Supports "stronger = more susceptible" but is a small, mechanistically different effect.

---

## What is already known vs. what is novel

**Already known (strong literature backing):**
- Bigger models can be worse — inverse scaling is real and named.
- Strong priors overriding provided context worsens with scale/RLHF.
- Agents fabricate observations, hallucinate user replies, and ignore environment feedback; tool-use is the hardest case to even detect.

**Appears novel (the contribution):**
1. **Self-injected fake trajectories** — writing thousands of fabricated environment/user turns into its *own completion* (a sustained self-authored alternate game), not a single presumptive reply on a static snapshot.
2. **Active reality-rejection** — re-classifying genuine environment output as a "display glitch" and trusting the hallucination for hundreds of steps. A self-sustaining delusion loop, not a one-shot error.
3. **A positive capability-scaling gradient** — Opus does it; Sonnet shows **zero** across all episodes. No surveyed source ties self-confabulation to a capability gradient (presumptive hallucination is explicitly the opposite framing).
4. **Confabulation *as the terminal failure*** — it causes budget blowout and the real win condition is never found; not a scored benchmark "mistake" but the mechanism of total task failure.

The novelty is the *combination*, in a *tool-discovery* setting, with a *scaling* signature. Caveat: evidence is n=12, T=3, single benchmark, one model family — not yet externally replicated.

---

## Future directions

*Project goal: characterize the faculty of LLMs to **discover tools given bounded resources**. The confabulation finding matters here not as a standalone phenomenon but as a **budget pathology** — Opus converts real exploration budget into fictional exploitation, directly taxing discovery yield. The directions below center the discovery faculty; confabulation enters as a variable that degrades it.*

1. **Define and measure discovery yield per unit resource (the core metric).** Tool-discovery yield = unique tools / valid combinations discovered per token (and per action) under a fixed budget. Report it by model × budget alongside a **wasted-budget fraction** (tokens/actions spent acting on fabricated state vs. real observations). This converts the confab finding into a quantitative tax on the discovery faculty: Opus's blowup is "effective budget" lost to an imaginary game. This is the foundational instrumentation the other directions build on.

2. **Two distinct axes — keep them separate.**
   - **Capability axis (where Wei's U actually lives).** Wei et al. 2022 ([2211.02011](https://arxiv.org/abs/2211.02011)) is U-shaped over *model scale/compute* — the model itself changes. The genuine inverse/U-shape hypothesis belongs here: plot discovery yield vs. **model capability** (≥3 points needed to see a U). The repo already supports **Haiku 4.5 → Sonnet → Opus** within one family — the cleanest test (same recipe, varying scale). Is discovery yield monotonic in capability, or does it peak at Sonnet and dip at Opus (inverse) before a hypothetical larger model recovers (U)?
   - **Budget axis (a hazard process, NOT a Wei-style U).** Inference budget holds the model fixed and varies the *resource*, so any non-monotonicity needs its own mechanism: an **exploration benefit** (more budget → more genuine attempts, ↑) competing with a **length-dependent confabulation hazard** (longer rollouts accumulate ambiguous feedback + self-generated context → higher chance of entering the near-**absorbing** confab attractor that burns the remaining budget). Net yield ≈ `exploration_benefit(b) × P(not absorbed by confab at b)`. Model this as an **absorbing-state / bimodal mixture** (each episode either discovers-and-solves or is absorbed and wastes budget), not a smooth curve. NB: our data is not a clean U over budget — worst runaway at *intermediate* b=320, largest b=1280 solved all 10 — consistent with a mixture whose weights shift with budget; n=12 is too thin to call the shape.
   - **Headline deliverable:** the **(capability × budget) discovery-efficiency frontier**, with the two axes mechanistically distinguished rather than merged.

3. **Explore/exploit under a resource bound.** Tool discovery is fundamentally explore/exploit under a budget. Sonnet **over-explores** (observation-bound brute force — robust but resource-inefficient); Opus **prematurely exploits a fabricated world model** (stops exploring because it believes it already won). Measure exploration breadth (distinct actions/combinations attempted) vs. budget, and test whether Opus's discovery *stalls* coincide with confab onset — i.e., confabulation = premature convergence on a wrong world model. This locates the efficiency frontier between the two failure styles.

4. **Grounding interventions, scored on discovery — not on hallucination.** Try a [2410.14675](https://arxiv.org/pdf/2410.14675)-style calibrated trust gate or forced re-reading of raw environment tokens, but evaluate by whether they **recover discovery yield and free budget for genuine exploration**, with explicit measurement of any exploration-suppression tradeoff (does grounding make the model timid and reduce breadth?). The goal is more tools discovered per token, not merely fewer fabricated turns.

5. **Obfuscation as the discovery-difficulty knob.** The obfuscated action space is precisely what makes this *discovery* rather than *use*. Sweep obfuscation level as the independent variable and measure how discovery yield degrades — and where confabulation kicks in — as a function of obfuscation × budget. This separates "couldn't discover the tool" from "discovered it, then got deluded," and tells you whether confab is triggered by ambiguity (obfuscation) or by budget pressure.

6. **Cross-model / cross-environment generality (validity check).** Does the capability × budget discovery crossover reproduce in GPT-5 / Gemini-2.5-Pro and in non-gridworld tool-discovery environments? Needed before any scaling claim about the *discovery faculty* generalizes beyond Opus/toolworld.

**Suggested framing for a writeup:** *"Tool discovery under bounded resources: a capability × budget efficiency frontier, and confabulation as the budget pathology that bounds it"* — positions the confab finding as the mechanism that produces inverse scaling **in discovery efficiency**, extending the agent-hallucination taxonomy (MIRAGE / AgentHallu) with the capability-scaling axis those papers lack while keeping the discovery faculty as the dependent variable.

---

## Verification caveats

- The MIRAGE / survey taxonomy categories *cover* the Opus behavior definitionally but do not *empirically document* it (static snapshots / single decision points, not multi-step self-sustained delusion).
- The strongest direct analogues (presumptive hallucination, sycophancy, factuality-vs-context, Resisting Correction) each capture **one** facet — fabrication, truth-overriding, prior-over-context, scale-worsening — but none ties self-injected fake-trajectory confabulation to a *positive capability-scaling gradient* in tool discovery.
- **Two claims were refuted** during verification (1-2 votes, both from [2603.09654](https://arxiv.org/html/2603.09654)): that LLMs *generally* prefer parametric memory over context, and that larger models *always* use context better. The context-faithfulness-vs-scale direction is genuinely contested — do **not** assert prior-over-context monotonically worsens with capability. (This split actually strengthens the novelty case for a clean, replicated scaling signal.)
- Time-sensitivity: the agent-hallucination literature is very recent (2025–26) and fast-moving; AgentHallu is dated Jan 2026, near/at the knowledge cutoff. "Novel" reflects absence in the surveyed set, not a proof.

## Key sources

| Topic | Source |
|---|---|
| Inverse Scaling Prize (4 causes, 11 datasets) | McKenzie et al. 2023 — [2306.09479](https://arxiv.org/pdf/2306.09479) |
| U-shaped scaling reversal | Wei et al. 2022 — [2211.02011](https://arxiv.org/abs/2211.02011) |
| Sycophancy scales with size/RLHF | Perez et al. 2022 — [2212.09251](https://arxiv.org/abs/2212.09251); Sharma et al. 2023 — [2310.13548](https://arxiv.org/pdf/2310.13548) |
| RLHF degrades calibration | GPT-4 report — [2303.08774](https://arxiv.org/pdf/2303.08774) |
| Factuality enhancement vs context-faithfulness | [2404.00216](https://arxiv.org/pdf/2404.00216) |
| Situated faithfulness (calibrated trust) | [2410.14675](https://arxiv.org/pdf/2410.14675) |
| Agent-hallucination survey (perception/execution) | [2509.18970](https://arxiv.org/html/2509.18970v1) |
| MIRAGE-Bench (env-unfaithful, presumptive hallucination) | [2507.21017](https://arxiv.org/pdf/2507.21017) |
| AgentHallu (tool-use hardest to localize) | [2601.06818](https://arxiv.org/abs/2601.06818) |
| Contested context-vs-scale (refuted claims) | [2603.09654](https://arxiv.org/html/2603.09654) |
