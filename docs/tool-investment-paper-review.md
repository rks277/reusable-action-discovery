# Review: "Tools as Resources: Measuring Online Tool-Investment Capability in LLM Agents"

Rating: **8/10**

Last updated 2026-07-11, after extending the core Haiku/Opus/GPT panel from
N=12 to N=24 canonical seeds (2000–2023) and regenerating every dependent
table, statistic, and figure.

## Strengths

- The core design — pairing identical latent class streams across an
  abstract (keep/pass) frame and a constructive (write-a-tool) frame with
  matched disclosed information — is a clean identification strategy. It
  separates "policy absent" from "policy present but suppressed by framing,"
  which ordinary tool-use benchmarks conflate.
- The construction-only control (R2c: require code, decouple payoff from
  correctness) is the strongest single piece of evidence in the paper. It
  isolates code emission itself as sufficient, ruling out tool-call
  modality, correctness incentive, and hand-solving escape valves as
  necessary causes.
- Self-policing is well above the norm for benchmark papers: explicit
  seed-count caveats, a prior-dependent Bayesian comparator flagged as such,
  a single-run RL result with a privileged critic labeled as
  proof-of-possibility not a robust result, unmeasured GPT R3 cells reported
  as unmeasured rather than assumed.
- GPT-5.6 Sol's preservation result does real argumentative work: it
  forecloses the trivial "code generation mechanically forces eagerness"
  explanation.
- Preregistration (competence gates, a 30-point localization rule) guards
  against post-hoc storytelling.

## What changed with the N=24 seed extension

The original submission's headline numbers rested on 12 canonical seeds;
every core table now reflects 12 original + 12 extension seeds
(2000–2023), run under the identical generator, prompts, and harness, at a
real cost of $20.08. Three findings from the extension are worth flagging
directly (see `docs/tool-investment.tex` for exact figures):

1. **The headline Haiku result held up almost exactly.** R3-minus-R0 first
   sight: +72.2 points at both N=12 and N=24; the interval tightened from
   [+61.1,+83.3] to [+65.3,+79.2]. This is the best-case outcome for a seed
   extension — same point estimate, narrower interval.
2. **One claim weakened and was corrected, not smoothed over.** Opus's R0
   urn behavior was 0% first-sight at N=12 (giving a degenerate
   [+100,+100] paired interval); at N=24 it is 4% (3/72), giving a real,
   non-degenerate interval [+91.7,+100.0]. The paper's "Opus reserves
   completely" language has been corrected to "reserves almost completely."
   This is a case where more data caught an overstatement baked into a
   small sample, and the fix is now in the text rather than left for a
   reviewer to find.
3. **One qualitative claim sharpened.** The R1-vs-R0 (tool-call modality)
   difference was ambiguous at N=12 ([-30.6,+8.3], included zero). At N=24
   it is entirely negative ([-26.4,-2.8]): tool-call modality alone
   produces a small but real *additional* damper on first-sight
   commitment, not a null effect. This doesn't change the paper's causal
   story (the eagerness jump still localizes to R2→R2c) but it's a more
   precise supporting fact than what shipped originally.

One methodological asymmetry is now flagged rather than silently present:
the no-N urn/tool comparison (Table 2's other two columns) was not part of
the approved seed-extension run list and remains N=12. The paper now says
this explicitly in the protocol section and the table caption, rather than
implying uniform N=24 across every table.

## Remaining weaknesses

- N=24 is still a modest independent-stream count for the size of the
  claims being made, though it is double what shipped originally and the
  paper is explicit about not treating repeated sessions as additional
  independent samples.
- The framing ladder is still coarse (4 rungs, not a full factorial), and
  the R1→R2 contrast still bundles action-label and content changes — the
  paper flags this in Limitations but a cleaner R1.5 rung would settle it.
- The Qwen RL case study remains a single successful run with a privileged
  critic; it was already at N=24 and unaffected by this extension.
- GPT R3 cells remain unmeasured (cost-capped), so the GPT comparison
  supports claims about code *emission*, not preservation through
  execution and debugging.
- No LaTeX toolchain was available in this environment to compile-check the
  document after editing; brace/environment balance was verified
  programmatically but a full compile has not been confirmed.

## Bottom line

This is a well-identified, self-aware, appropriately hedged empirical paper
whose central claim survived a real replication check essentially intact,
and whose one weaker claim (Opus's "complete" abstract reservation) was
caught and corrected by the same check rather than papered over. The
ceiling on the score is the still-modest stream count and the coarseness of
the framing ladder, not conceptual or methodological sloppiness.
