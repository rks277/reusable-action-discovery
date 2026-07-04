# The same-information optimal policy: an exact finite-horizon belief-state DP

The reference policy `π*` for the online tool-investment benchmark is the **exact Bayes-optimal
policy** for the finite-horizon, budgeted build problem, computed by dynamic programming over the
learner's belief state. It has **no relaxation and no `N → ∞` assumption**: it is the literal optimum
against the same information the model has. This document defines it, states the structure of the
optimal policy, explains why a Whittle/index construction cannot substitute for it at our scale, and
describes how it is computed and used. **Implemented** in `exact_dp.py` (`ExactDP`) and wired into the
scorer as `skirental_scorer.exact_pistar_report`; validated lossless against a brute-force DP and
certified exact-at-scale via the count-cap (§6).

Companion to `online-tool-investment-working-notes.md` (§ "Reference policy π\*").

---

## 1. The decision problem

A single session presents `T` problems one at a time. Each problem is an i.i.d. draw from an unknown
distribution over `N` problem **types** (families). The learner knows only `{N, T, B}` and that the
problems are i.i.d. from *some* distribution over `N` types — **not** which types are frequent, nor
the frequent/rare split. On each problem the learner either:

- **hand-solves** it, or
- **builds** a reusable tool for that type (writing + running a script), consuming one of `B`
  irreversible write-budget units, or
- **reuses** an already-built tool for that type.

The objective is to maximize total expected utility over the session. Building is **irreversible**
(a tool, once written, persists and is reused for free thereafter) and **budget-limited** (`B`
builds for the whole session). This is a Bayesian sequential decision problem: the learner must
decide *when* to spend scarce, irreversible builds while learning the type distribution online.

**Utilities (one-utility cost model; see the design note and `skirental_scorer.Costs`).** A correct
answer is worth `R`; each token costs `λ`. An action's utility is `R·(accuracy) − λ·(tokens)`:

```
        u_hand  = R · a_hand   − λ · h          # hand-solve  (a_hand = P(correct by hand))
        u_build = R · a_script − λ · (C + r)    # write tool + first run  (a_script ≈ 1)
        u_reuse = R · a_script − λ · r           # call an existing tool
```

`a_hand` is per-type (measured, A0 forced-hand condition); `h, C, r` are token counts (`h, C`
measured; `r` a small reuse-call constant); `R, λ` set the value/compute exchange rate. For the
hard families (`a_hand ≈ 0`) with the A0 calibration this gives `u_hand ≈ −98.7`, `u_build ≈ 49.2`,
`u_reuse ≈ 80.0`. Two consequences matter:

- `u_hand < 0`: a hand attempt on a hand-infeasible problem earns ~nothing yet still costs tokens.
- `u_build > u_hand`: building solves the current problem correctly, so it beats a failed hand
  attempt **even for a one-off**. The only thing that makes building the 2nd/3rd type suboptimal is
  the scarce, irreversible **budget** — so the entire tension is budget allocation, which the DP
  weighs exactly.

---

## 2. The belief model

The unknown type distribution `(p_1, …, p_N)` is given a **symmetric Dirichlet(α)** prior — the
*same-information* prior: structure-agnostic, encoding no knowledge of the hot/rare split. Given
occurrence counts `c = (c_1, …, c_N)` over `s` elapsed slots, the Dirichlet–multinomial **posterior
predictive** for the next draw is

```
        p_i(c, s) = (α + c_i) / (N·α + s).                                    (2.1)
```

`α` is a pseudo-count (prior strength): `α → 0` ⇒ one sighting nearly certifies recurrence
(rich-get-richer); `α → ∞` ⇒ sightings barely move the belief (near-uniform). We use `α = 1`
(Laplace); α-sensitivity is discussed in the design note. Note (2.1) makes recurrence *informative*:
a type seen more often has a higher predictive, so "has it recurred?" is the evidence the policy acts
on.

---

## 3. The exact DP

**State** `(s, c, B, 𝔟)`: `s` = elapsed slots, `c` = per-type counts, `B` = builds remaining,
`𝔟 ⊆ {1..N}` = already-built types.

> **State invariant: `Σ_i c_i = s`.** The predictive (2.1) is a valid distribution iff the number of
> type-observations equals the number of elapsed slots (then `Σ_i(α + c_i) = Nα + s`, the
> denominator, so the `p_i` sum to 1). Every reachable state maintains this because the recursion
> only advances `s → s+1` together with `c → c + e_i`. **Violating the invariant leaks probability
> mass and silently corrupts the value function** — it is the easiest bug to introduce when querying
> `V` at a hand-constructed state, and it must be asserted in code and tests.

**Value function.** `V(s, c, B, 𝔟)` is the maximum expected total utility from slot `s` to the
horizon `T`. Nature draws the next type `i` with probability `p_i(c, s)`; the learner then acts on
the revealed type:

```
  V(s, c, B, 𝔟) = Σ_i p_i(c, s) · Q_i ,          with  c' := c + e_i ,

              ⎧ u_reuse + V(s+1, c', B, 𝔟)                                  if i ∈ 𝔟   (reuse)
        Q_i = ⎨ max{ u_hand  + V(s+1, c', B,   𝔟),                          if i ∉ 𝔟
              ⎩       u_build + V(s+1, c', B−1, 𝔟 ∪ {i}) }  (build only if B > 0)
```

with terminal condition `V(T, ·, ·, ·) = 0`. The `max` is the **only** decision: for an unbuilt type,
*hand-solve and keep the state* vs. *spend one irreversible build*. A built type is absorbing — it
yields `u_reuse` on every future occurrence with no further choice. `u_build` includes solving the
*current* problem with the freshly written tool (building is not merely a future-reuse investment).

A build/reserve decision is read off by comparing the two branches of `Q_i` at the state of interest,
with **both continuations evaluated at the same `c'`** so the comparison is honest.

---

## 4. Structure of the optimal policy

Solving the DP exactly on tractable instances (§6) reveals a clean and interpretable structure, all
of which is *emergent* from (3), not imposed:

1. **Build on demonstrated recurrence, not on arrival.** For a brand-new type appearing at any slot
   `s ≥ 1`, the *reserve* branch strictly dominates — the optimum hand-solves it and keeps the
   budget. A type seen `≥ 1` time before (its 2nd or later sighting) is built on sight. The sole
   first-sight build is the very first slot `s = 0`, and even that is marginal.

   This is *not* impatience-avoidance for its own sake: arriving early is genuine evidence of a high
   rate (in a skewed i.i.d. stream a first-seen type is usually frequent), so the immediate build
   value is real. It is dominated only because **waiting one slot costs at most one forfeited
   occurrence, while spending the irreversible budget on an unconfirmed type risks the whole budget
   unit** — and with `B` scarce relative to the number of candidate types, that option value wins.

2. **The optimal decision is context-dependent.** Whether to build a type on its 2nd sighting depends
   on the **remaining budget** and on **what other types have done**: with the last build in hand and
   a competitor already seen several times (clearly hotter), the optimum *reserves* for the
   competitor rather than build the type in front of it — the identical `(k, s)` for that type builds
   in one context and reserves in another. The optimal policy is a rule over the *joint* state, not a
   per-type rule.

Qualitatively, the optimum implements "**hand-solve until a type proves it recurs, then build the
demonstrated winners in temporal order until the budget is spent**" — the evidence-gathering behavior
the eager model fails to exhibit. See the design note for the empirical contrast (the model builds
the first `B` distinct types on sight, lateness 0; the optimum waits).

---

## 5. Why not a Whittle / index construction

An earlier version of `π*` used the **Whittle index** from a Lagrangian relaxation: relax the hard
budget to a per-build price `λ`, solve each type's build decision independently against that price,
and tune `λ` so expected builds `= B`. It is attractive because it is `O(T)` and separable — but it
is **wrong at our scale**, for two linked reasons:

- **It is only asymptotically (`N → ∞`) optimal.** The relaxation prices the budget with a single
  *constant* `λ`, which cannot represent the *time-varying option value* of holding an irreversible
  budget. At `N ≈ 12` the relaxation gap is material: the constant-price rule builds a type at its
  *first sighting* whenever the sighting is early, whereas the exact optimum reserves (§4.1). The
  Whittle rule builds at first sight ≈44% of the time; the optimum essentially never does past slot 0.

- **It is structurally the wrong object.** The Whittle index is **per-type and context-free** — a
  type's decision depends only on its own `(k, s)` and the constant `λ*`. But the optimal decision is
  **context-dependent** (§4.2): it flips on remaining budget and competing types. No constant-price
  index — and no per-type `(k, s)` threshold, however tuned, including "suppress first-sight builds"
  or a rolling/re-solved `λ` — can reproduce a decision that depends on the joint state. This is not
  a tuning error to be priced away; it is intrinsic to the relaxation.

Consequence: regret measured against the (weaker) Whittle policy is a *valid but loose* lower bound;
the exact DP is stronger, so the honest same-information regret is larger. We therefore use the exact
DP as the reference. (The index literature remains the reason to *expect* an index-like policy to be
near-optimal for large `N`; it simply is not the exact finite-`N` object we need.)

---

## 6. Computation: exact, and tractable at scale (implemented, `exact_dp.py`)

**Naively intractable.** The state carries the full count *vector* `c`, so the reachable state count
is `Σ_{s≤T} C(s + N − 1, N − 1)` times built-configurations — `≳ 10^{13}` at `N = 12, T = 60`. The
uncapped vector DP is feasible only to `T ≈ 40` (`N=8,T=14` in seconds; `N=12,T=40` ~4.8M states/40s;
`N=14,T=24` already blows up — memory, not time).

**Exchangeability reduction.** Two facts shrink it. (1) `#built = B − budget` is deterministic
(budget only drops on a build), so the state needs no separate built-set beyond which count-parts are
marked built. (2) The tree is shallow: at most `B` build decisions per trajectory, and once
`budget = 0` the continuation is **closed-form** — by the Pólya-urn martingale property every future
draw of a type at count `c` has marginal probability `(α+c)/(Nα+s)`, so the remaining value is
`(T−s)/(Nα+s) · [u_reuse·built_mass + u_hand·(rest)]`. This terminates the recursion after the ≤B
build decisions. By type-symmetry `V` depends on `c` only through the **multiset** of counts, so the
canonical state is `(built-count-multiset, unbuilt-count-multiset)` with `s` and `budget` derived.

`ExactDP` implements this (memoized). Validated **lossless**: the canonical root value equals the
brute-force vector DP to `< 1e-6` on six small `(N,T,B)` instances × α ∈ {0.5, 1, 2}.

**The count-cap makes `T = 60` exact and cheap.** Even canonicalized, tracking unbuilt counts up to
`T` still proliferates (partitions of `s`). `ExactDP(cap=K)` caps *unbuilt* counts at `K`: a type that
reaches `K` sightings while unbuilt is **force-built** on its next arrival, so unbuilt parts never
exceed `K`. This is itself a **feasible same-info policy** ("build any type that recurs `K` times"),
so `V(K)` is a valid lower bound, monotone non-decreasing in `K`. A K-sweep at `N=12,T=60,B=3`
**certifies it lossless here**:

| K | states | time | `V(K)` |
|---|---|---|---|
| 2 | 138K | 0.8s | −897.845 |
| **3** | **542K** | **3.1s** | **−897.835** |
| 4 | 1.6M | 9.7s | −897.835 |
| 5 | 3.9M | 51s | −897.835 |
| 6 | 7.9M | 208s | −897.835 |

`V(3) = V(4) = V(5) = V(6)` **exactly** (and cap=2 matches the *uncapped* exact at `T ≤ 40`). Since
`V(·)` is monotone and the root value is a positively-weighted sum over reachable states, root-value
equality forces per-state equality — so **cap=3 is the exact optimum** at ~540K states / ~3s in pure
Python, zero ε. The cap is a *state-space bound*, **not** the build rule (§4: π\* builds on the 2nd
sighting, far inside the cap; the cap-forced 4th-sighting build essentially never fires on-path).
This certification is **regime-specific** ({N,T,B,α,costs}); re-running the K-sweep re-certifies if the
cost model or α changes — if the tail stopped being flat we would raise `K`.

**Utilities are scalar (uniform a_hand).** The same-info reference treats types exchangeably, so
`ExactDP` uses a single representative `a_hand` (the pool mean) for its *decisions*; the realized
**value** of a policy's builds is scored separately with the per-family utilities in
`skirental_scorer` (§7). The forward pass (`ExactDP.policy_builds`) was cross-checked: its realized
value averaged over belief-sampled (Pólya-urn) streams matches `root_value()` within Monte-Carlo error.

---

## 7. Use as the reference and the regret it defines

Wired into the scorer as `skirental_scorer.exact_pistar_report` (called from `score_run`; pass a
prebuilt `ExactDP` via `pistar_dp=` to reuse the value table — a design constant of
`{N,T,B,cap,costs}` — across a seed sweep). Let `V*` be the exact-DP policy's value (played forward on
a realized stream, valued over full realized class sizes) and `V_model` the value of the model's
realized builds on the same stream. The reported quantity is

```
        regret(model) = V* − V_model ,
```

the shortfall of the model against a policy with **identical information**. Because `π*` is the exact
same-information optimum (not a relaxation), this is the honest same-information regret, not a bound.
It is complemented by the per-instance **clairvoyant** value (which additionally knows the realized
counts); `(clairvoyant − V*)` is the intrinsic, irreducible price of online uncertainty, and
`(V* − V_model)` is the model's excess regret over optimal use of its own information.

**Result (3-seed Haiku dry-run, re-scored with the exact reference):** regret ≈ **2293/seed**
(positive on all three seeds), π\* builds **0 traps/seed** (never baited — it waits for recurrence)
vs the model's **1 trap/seed** built on sight; `(clairvoyant − V*)` ≈ 150–350 (online uncertainty is
cheap here). This roughly doubles the regret the retired Whittle reference reported (≈1263) and, unlike
Whittle, is positive on every seed — the exact optimum is strong enough to beat the model even on the
seed where its eager builds happened to land well.

---

## References

- R. Bellman (1957). *Dynamic Programming.* Princeton University Press. (Finite-horizon backward
  induction — the exact DP of §3.)
- M. H. DeGroot (1970). *Optimal Statistical Decisions.* McGraw-Hill. (Bayesian sequential decision
  problems; Dirichlet–multinomial conjugacy and the posterior predictive (2.1).)
- P. Whittle (1988). *Restless bandits: activity allocation in a changing world.* J. Appl. Probab.
  25A, 287–298. (The index construction we reject at finite `N`, §5.)
- R. R. Weber & G. Weiss (1990). *On an index policy for restless bandits.* J. Appl. Probab. 27(3),
  637–648. (Asymptotic — `N → ∞` — optimality of the Whittle index; why it does not apply at `N ≈ 12`.)
- D. B. Brown & J. E. Smith (2020). *Index policies and performance bounds for dynamic selection
  problems.* Management Science 66(7), 3029–3050. (Finite-horizon LP relaxation / index bounds — the
  relaxation family, for contrast with the exact DP.)
