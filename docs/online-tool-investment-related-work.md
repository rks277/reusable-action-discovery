# Related work — online tool investment / when-to-build

Companion to `online-tool-investment-plan.md`. Positions our finding — **LLMs build
reusable tools eagerly on first sight of a problem, without waiting for evidence that the
problem type recurs; the failure is uniform across the capability ladder and immune to
explicit disclosure of the recurrence structure** — against the literature, and marks the
one slice that is genuinely unclaimed.

The one-line contribution to defend: *an online rent-vs-buy (ski-rental) formalization of
irreversible reusable-tool **creation** under unknown future reuse, with model **regret vs.
the online-optimal** as the headline metric.* This is a formalization + measurement
contribution, **not** a "we discovered LLMs commit early" discovery — that framing is taken.

Full-text reads done 2026-07-01/02. Neighbors sort into five clusters; none owns the
online build-timing decision.

---

## 1. Premature commitment / myopia (phrase-neighbors)

- **Mehta et al., "When Agents Commit Too Soon: Diagnosing Premature Commitment in LLM
  Agents"** (arXiv:2606.22936). Owns the *phrase*, but the object is **internal
  representational commitment** — cross-run hidden-state convergence at a fixed reasoning
  step (interpretive collapse on a single QA problem: HotpotQA/StrategyQA, ReAct, step-4
  hidden state; AUROC ~0.97). No tools, no stream, no amortization, no external irreversible
  act. **Overlap = the phrase + "persists across models" only.** Cite in one sentence;
  distinguish: their commitment is intra-trajectory representational collapse on one query,
  ours is premature irreversible tool construction under uncertain future reuse.
  **Do not use "premature commitment" as our headline noun — Mehta owns it.**

- **"When Greedy Wins: Emergent Exploitation Bias in Meta-Bandit LLM Training"**
  (arXiv:2509.24923) — *full text read*. RL/meta-bandit training induces a systematic
  **greedy exploitation bias**: LLMs implicitly learn to commit to early-observed
  high-reward arms rather than learn Thompson/UCB-style exploration ("a form of amortized
  learning where the model trades theoretical optimality for computational efficiency").
  The bias appears roughly **uniform across model scales** early in training. This is our
  strongest **mechanistic** neighbor — a training-origin story for *why* models don't wait —
  and a hook for our mitigation section. But the act is **arm selection**, not constructive
  tool-building, and there is no fixed-cost/amortization/ski-rental structure. Cite as the
  mechanism precedent; distinguish on the object (select an arm vs. fabricate a reusable
  tool) and the economics (no rent-vs-buy break-even).

## 2. Exploration/exploitation & value-of-information (framework-neighbors)

- **"Do LLM Agents Have Regret?"** (arXiv:2403.16843) — regret is external regret vs. the
  best fixed action in hindsight; action = pick from a fixed set. LLM-regret precedent, but
  no construction, no fixed-cost amortization, no ski-rental. Our regret is vs. an online
  rent-vs-buy optimum over an irreversible costly build.
- **Krishnamurthy et al. 2024** (arXiv:2403.15371) — MAB, stateless repeated arm selection,
  under-exploration. **Capability-graded** (only GPT-4 + CoT + summarized history explores;
  weaker models fail) — a reason to keep uniformity as a *measured result*, not the headline
  claim. Adjacent framework, disjoint object.
- **Su & Cardie 2025** (arXiv:2605.25284) — single-turn QA, recognize-ambiguity-but-answer
  instead of asking. Asking is **free**: no cost, no irreversibility, no amortization. =
  info-*acquisition* failure, not info-*investment*. One-sentence cite.

## 3. Cost-aware exploration as mitigation (intervention-neighbors)

- **"Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents"** (arXiv:2602.16699) —
  *full text read*. Agent first estimates confidence, then chooses an action cost (call an
  expensive model / use a tool vs. accept a cheap approximation); evaluated on cost-accuracy
  tradeoff with regret against a cost-aware optimal. **Key distinction:** the "investment"
  is a **one-shot per-decision spend** (pay more compute *now* for better info on *this*
  item) — it is not an irreversible artifact whose cost is **amortized over a future stream**
  of items. Their calibrate-then-act loop is exactly the shape of the *mitigation* we would
  test (assess recurrence before authoring); cite as the intervention prior, distinguish on
  the amortization structure.
- **"Look Before You Leap: Autonomous Exploration for LLM Agents"** (arXiv:2605.16143) —
  explore-then-act: a goal-free exploration phase producing a knowledge summary before
  acting. Another mitigation-arm neighbor (gather evidence first). Same distinction: no
  amortized artifact.
- **CaRT** (arXiv:2510.08517) — when-to-stop / calibrated acting. Adjacent VOI/timing prior.

## 4. Tool/skill-library creation & amortization (mechanism-neighbors — closest cluster)

- **LATM, "Large Language Models as Tool Makers"** (arXiv:2305.17126) — the amortized
  tool-creation *mechanism*: an expensive maker builds a Python tool once, a cheap user
  reuses it across many instances ("once created, a tool can be reused across hundreds of
  instances without regeneration, amortizing the cost of the expensive maker"). This is the
  **cost model we invoke** — but LATM never studies the online *timing* of when to build; it
  assumes a recurring task and builds up front. We supply the missing decision axis.
- **Self-evolving skill-library wave (2026)** — all treat over-accumulation as a
  **post-hoc pruning / maintenance** problem, never the online authoring-timing decision:
  - **"Library Drift: Diagnosing and Fixing a Silent Failure Mode in Self-Evolving LLM Skill
    Libraries"** (arXiv:2605.19576) — *full text read*. Claude Opus 4.7 on MBPP+ hard-100;
    Router→Solver→Grader→Critic→Curator loop. Failure = "unbounded skill accumulation
    without outcome-driven lifecycle management causes retrieval degradation, false-positive
    injections, and performance stagnation." Fix = the "Ratchet Recipe" (outcome-driven
    **retirement**, active-cap of 50 skills, authoring-style prior); +0.33 on solve.
    **Confirmed: post-hoc curation only — no delaying of authorship, no one-off discussion,
    no regret / ski-rental / rent-vs-buy.** Skills are only synthesized *after* ≥3 shared
    failure patterns (i.e. recurrence is assumed already observed). This is the paper closest
    in spirit and the one to distinguish most carefully: they prune a library that grew too
    fast; we study the *decision to author before recurrence is known*, with regret.
  - **"Skill Drift Is Contract Violation"** (arXiv:2605.10990), **SkillBrew**
    (arXiv:2605.29440), **SkillOps** (arXiv:2605.13716) — variations on curation / drift
    repair / bank-level multi-objective optimization. Same gap: maintenance, not timing.
  - **TroVE, LILO** — grow-then-trim skill induction for program synthesis. Trimming, not the
    online build-now-vs-wait decision.

## 5. Online algorithms / ski-rental (the formalism — no LLM)

- **Sequential Ski Rental** (arXiv:2104.06050) and multislope / tail-risk / ML-advised
  variants (0802.2832, 2308.05067, 2508.06809) — the rent-vs-buy formalism, competitive
  ratio 2 (deterministic) / ~1.58 (randomized), and the "buy after renting B/R days" optimal
  threshold that our reference policy uses. Zero LLM contact. **No prior work applies
  ski-rental to LLM tool creation** — this is the formal frame we import.

## 6. Overconfidence / calibration (disposition-neighbors)

- **"Agentic Uncertainty Reveals Agentic Overconfidence"** (arXiv:2602.06948),
  **"The Confidence Dichotomy: ... Miscalibration in Tool-Use Agents"** (arXiv:2601.07264),
  **CalVerT** (arXiv:2606.21777) — RLHF degrades calibration → agents act on thin evidence
  and over-trust parametric knowledge. Supports our *interpretation* ("acts before evidence
  justifies it") but studies answer/tool-call confidence, not investment timing. Background
  cite.

---

## Positioning summary (guardrails for related-work + intro)

**Surviving, unclaimed contribution (confirmed after full-text reads):** an online
rent-vs-buy / ski-rental formalization of reusable-**tool creation** under unknown reuse,
where the irreversible act is a *constructive build* (not answer / arm / stop — every
premature-commitment neighbor uses answer/arm/stop; none uses tool-fabrication), with model
**regret vs. the online-optimal** as the headline metric, shown to be **disclosure-immune**
and to hold **across the capability ladder**.

Do:
- Frame as **formalization + measurement**: the one place the field studies the
  build-timing decision itself, with an economics (ski-rental) the skill-library papers lack.
- Cite **LATM** (cost model) and **Library Drift / TroVE** (build-mechanism + over-eager
  accumulation) as the closest neighbors, then distinguish on the *online timing decision +
  regret*.
- Cite **When Greedy Wins** as the mechanism/training-origin precedent and mitigation hook;
  **Calibrate-Then-Act / Look-Before-You-Leap** as the intervention priors.

Don't:
- Headline **"premature commitment"** (Mehta owns the phrase).
- Headline **"uniform across capability"** as the *discovery* — bandit priors are
  capability-graded and our own runs falsified an inverse-capability story
  ([[sonnet-builds-oneoffs-when-hard]]). Keep uniformity as a measured result, not the claim.
- Over-claim that Calibrate-Then-Act "does ski-rental" — it does cost-aware per-action
  spend, not amortized irreversible construction.
