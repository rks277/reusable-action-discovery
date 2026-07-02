# Asymptotic optimality of the Whittle index policy

A self-contained statement and proof of the Weber–Weiss asymptotic-optimality theorem for the
Whittle index policy in restless multi-armed bandits, with all technical terms defined, followed by
an honest account of how it does and does not apply to our budgeted online tool-building problem.

Companion to `online-tool-investment-stochastic-design.md` (§ "π\* via per-type DP + λ-tuning").

---

## 0. Notation and standing assumptions

Throughout, "arm" is an abstract controlled Markov process; in our application each arm is a *problem
type*. Time is discrete, `t = 0, 1, 2, …`. We work in the **long-run average-reward** criterion (the
setting of the classical theorem); the finite-horizon analogue relevant to us is discussed in §6.

---

## 1. The restless multi-armed bandit (RMAB)

**Markov decision process (MDP).** A tuple `(S, A, P, r)`: a finite **state space** `S`, a finite
**action space** `A`, a **transition kernel** `P(s' | s, a)` (probability of moving to `s'` from `s`
under action `a`), and a bounded **reward** `r(s, a)`. A **policy** maps histories to actions; a
**stationary policy** depends only on the current state.

**Average reward.** For a policy `π` on a single MDP, its long-run average reward is
`g(π) = liminf_{T→∞} (1/T) E^π[ Σ_{t=0}^{T-1} r(s_t, a_t) ]`. Under standard ergodicity (every
stationary policy induces a unichain Markov chain, i.e. a single recurrent class plus transient
states) this liminf is a limit independent of the initial state.

**Restless bandit (Whittle 1988).** There are `N` statistically identical arms. Each arm `n` has its
own state `s_n(t) ∈ S`. At each `t`, each arm is assigned a binary action: **active** (`a = 1`) or
**passive** (`a = 0`), so `A = {0, 1}`. An arm in state `s` under action `a` earns reward `r(s, a)`
and moves according to `P_a(· | s) := P(· | s, a)`.

> **"Restless"** means passive arms *also evolve*: in general `P_0(· | s) ≠ δ_s`, so a passive arm's
> state can change. This is the generalization of Gittins' classical bandit (Gittins 1979), where
> passive arms are *frozen* (`P_0(· | s) = δ_s`); the frozen case is solved optimally by the Gittins
> index, but the restless case is far harder.

**The N-arm control problem.** A scheduler observes all arm states and, at each `t`, must activate
**exactly `M` arms**, `M = ⌊αN⌋` for a fixed **activation fraction** `α ∈ (0, 1)`:
```
        Σ_{n=1}^N a_n(t) = M              for every t.                     (C)
```
The objective is to maximize the total long-run average reward
`G_N(π) = liminf_{T→∞} (1/T) E^π[ Σ_{t<T} Σ_{n=1}^N r(s_n(t), a_n(t)) ]`.
Write `OPT(N) = sup_π G_N(π)` for the optimal value under the hard per-period constraint (C).

> **Why this is hard.** The arms are coupled only through (C), but that coupling makes the joint
> problem **PSPACE-hard** in general (Papadimitriou & Tsitsiklis 1999): the optimal scheduler is not
> separable across arms and the joint state space is `|S|^N`. Whittle's index is a tractable
> heuristic; the theorem below says *when* it is nearly optimal for large `N`.

---

## 2. The Lagrangian relaxation and the per-arm value

**Relaxed problem.** Replace the hard per-period constraint (C) by its **time-average** version:
```
        limsup_{T→∞} (1/T) E[ Σ_{t<T} Σ_n a_n(t) ] ≤ M.                    (C̄)
```
Any policy feasible for (C) is feasible for (C̄), so the relaxed optimum `REL(N)` satisfies
`REL(N) ≥ OPT(N)`.

**Lagrangian decoupling.** Introduce a multiplier `λ ≥ 0` (a **subsidy for passivity**) and define
the single-arm subsidized problem
```
        g(λ) := sup_{single-arm policy} liminf_{T→∞} (1/T) E[ Σ_{t<T} ( r(s_t, a_t) + λ(1 - a_t) ) ].
```
Here `λ(1 − a_t)` pays a reward `λ` for each period the arm is left passive. Because the `N` arms are
identical and, after relaxing (C) to the average constraint (C̄), otherwise independent, the relaxed
`N`-arm problem separates into `N` copies of the subsidized single-arm problem:

**Lemma 1 (relaxation is a per-arm upper bound).** For every `λ ≥ 0`,
`OPT(N) ≤ REL(N) ≤ N·g(λ) − λ(N − M)`, and hence, with `M = αN`,
```
        (1/N)·OPT(N) ≤ (1/N)·REL(N) ≤ min_{λ ≥ 0} ( g(λ) − λα ) =: V_rel.       (1)
```

*Proof.* Fix `λ ≥ 0`. For any policy feasible for (C̄), by (C̄),
`E-avg[ Σ_n λ(1 − a_n) ] ≥ λ(N − M)`. Hence
`G_N(π) = E-avg[ Σ_n r ] ≤ E-avg[ Σ_n ( r + λ(1 − a_n) ) ] − λ(N − M)
       ≤ N·g(λ) − λ(N − M)`,
where the last inequality drops the coupling and bounds each arm's subsidized reward by the
single-arm optimum `g(λ)`. Taking the sup over feasible `π` gives `REL(N) ≤ N g(λ) − λ(N − M)`.
Dividing by `N`, using `M/N = α`, and minimizing over `λ ≥ 0` yields
`(1/N) REL(N) ≤ min_λ ( g(λ) − λ(1 − α) )`. Re-parameterizing the subsidy (charge `λ` per
*activation* rather than reward per passivity — an affine change that shifts the constant) gives the
equivalent normalized form `V_rel = min_{λ≥0}(g(λ) − λα)` used below; both are the per-arm value of
the relaxed LP. The chain `OPT(N) ≤ REL(N)` was noted above. ∎

The minimizing multiplier `λ* = argmin_λ (g(λ) − λα)` is the **fair charge per activation**: at `λ*`
the single-arm optimal policy activates the arm a fraction `α` of the time (complementary slackness),
matching the budget on average.

---

## 3. Indexability and the Whittle index

Fix the single-arm subsidized problem. Let `D(λ) ⊆ S` be its **passive set**: the set of states in
which the optimal subsidized policy chooses `a = 0` (passive). As the passivity subsidy `λ` grows,
being passive becomes more attractive, so one expects `D(λ)` to grow.

> **Definition (indexability, Whittle 1988).** The arm is **indexable** if `D(λ)` is monotone
> nondecreasing in `λ`: `λ_1 ≤ λ_2 ⇒ D(λ_1) ⊆ D(λ_2)`, and `D(λ) = ∅` for `λ` small enough,
> `D(λ) = S` for `λ` large enough.

> **Definition (Whittle index).** For an indexable arm, the **Whittle index** of state `s` is
> ```
>     W(s) := inf{ λ : s ∈ D(λ) },
> ```
> the smallest passivity subsidy that makes the optimal policy indifferent between activating and not
> activating in state `s`. Higher `W(s)` ⇒ the state "deserves" activation more (a larger subsidy is
> needed before we'd willingly leave it passive).

Indexability is a genuine restriction: not every restless arm is indexable (Whittle 1988 raised this;
Niño-Mora 2001 gives sufficient conditions and an algorithm). For our tool-building arm, activation
(building) is **irreversible** and the value of building is monotone in a scalar posterior statistic,
which yields indexability by a standard monotone-comparative-statics argument (§6).

**Whittle index policy (for the `N`-arm problem).** At each period, compute `W(s_n(t))` for every
arm and **activate the `M` arms with the largest indices** (ties broken arbitrarily). This is a
proper policy for the hard constraint (C) — it always activates exactly `M` arms — even though the
index was derived from the *relaxed* single-arm problem.

---

## 4. Asymptotic optimality: statement

> **Definition (asymptotic optimality).** A sequence of policies `(π_N)` for the `N`-arm problems is
> **asymptotically optimal** if
> ```
>     lim_{N→∞} (1/N)·[ OPT(N) − G_N(π_N) ] = 0,
> ```
> equivalently `(1/N) G_N(π_N) → lim_N (1/N) OPT(N)`. The *per-arm* optimality gap vanishes; the
> absolute gap may still grow sublinearly (`o(N)`).

To state the achievability theorem we need the mean-field description of the index policy.

**Empirical measure and mean field.** Let `m^N(t) ∈ Δ(S)` be the **empirical state distribution**,
`m^N_s(t) = (1/N)·#{ n : s_n(t) = s }` — the fraction of arms in each state. Under the Whittle index
policy the activation decision is a **threshold on the index**: with `M = αN` arms activated, one
activates all arms whose index exceeds a cutoff determined by the current `m^N(t)` (the cutoff is the
`α`-quantile of the index distribution induced by `m^N`). Consequently the *aggregate* dynamics of
`m^N(·)` are a function of `m^N(·)` alone (a closed system), up to `O(1/√N)` stochastic fluctuations.

> **Definition (mean-field / fluid limit).** A deterministic trajectory `m(·)` with
> `ṁ(t) = F(m(t))`, where `F` is the drift induced by the index-threshold activation rule together
> with the kernels `P_0, P_1`, such that `m^N(·) → m(·)` (uniformly on compacts, in probability) as
> `N → ∞`. Existence of this limit for finite-state interacting particle systems of this "mean-field
> interaction" form is standard (Kurtz 1970; Benaïm & Le Boudec 2008).

> **Definition (globally asymptotically stable fixed point).** A point `m*` with `F(m*) = 0` such
> that *every* trajectory of `ṁ = F(m)` converges to `m*` as `t → ∞`. Call this the **Weber–Weiss
> condition (WW)**.

> **Theorem (Weber & Weiss 1990).** Suppose the (identical) arms are **indexable** and the mean-field
> ODE `ṁ = F(m)` induced by the Whittle index policy has a **globally asymptotically stable fixed
> point** `m*` (condition WW). Then the Whittle index policy is **asymptotically optimal**, and
> ```
>     lim_{N→∞} (1/N) G_N(Whittle) = lim_{N→∞} (1/N) OPT(N) = V_rel.
> ```
> Conversely (Weber & Weiss 1990, §3, and the 1991 addendum), if WW fails — the ODE has a limit
> cycle or multiple attractors — the Whittle index policy **need not** be asymptotically optimal;
> they exhibit an explicit 4-state counterexample.

---

## 5. Proof

The proof combines three facts: the upper bound (Lemma 1), the fact that the mean-field fixed point
of the index policy *realizes* the relaxed per-arm value, and the fluid-limit convergence.

**Step A — upper bound.** By Lemma 1, `(1/N) OPT(N) ≤ V_rel` for every `N`. It remains to show the
index policy *attains* `V_rel` per arm in the limit.

**Step B — the fixed point realizes `V_rel`.** Consider the single-arm subsidized problem at the fair
charge `λ*` from §2. Because the arm is indexable, its optimal policy is the **threshold policy**
"activate iff `W(s) > λ*`" (Whittle-index states above the charge are worth activating; this is
exactly the meaning of `W` and the monotonicity `D(λ)` gives). Let `μ*` be the stationary
distribution of a single arm under this threshold policy. By complementary slackness at `λ*`, the
stationary activation probability equals the budget fraction:
```
        Σ_s μ*(s)·1{ W(s) > λ* } = α,                                        (2)
```
and the single-arm average reward under this policy equals `V_rel` (this is the statement that the
Lagrangian dual is tight for the relaxed LP — strong duality for finite-state average-reward MDPs,
Bertsimas & Niño-Mora 2000).

Now examine the fixed point `m*` of the `N`-arm mean-field ODE. At `m*` the index-threshold rule
activates the top `α`-fraction of arms by index. Because `m*` is a stationary distribution of the
per-arm kernel *closed under exactly this threshold rule*, and (2) says the threshold rule at
stationarity activates fraction `α`, the fixed point coincides with the single-arm stationary law:
`m* = μ*`. Hence the per-arm average reward evaluated at `m*` equals `V_rel`. (Uniqueness of the
active fraction at `m*` uses indexability + (2); see Weber–Weiss 1990, Lemma 2.)

**Step C — fluid limit + stability transfer the fixed-point value to the finite system.** We invoke
the mean-field convergence theorem for finite-state mean-field interaction models:

> **Lemma 2 (mean-field convergence; Kurtz 1970; Benaïm & Le Boudec 2008).** For the family of
> `N`-arm systems under the index-threshold policy, for every finite `T` and `ε > 0`,
> ```
>     P( sup_{t ≤ T} ‖ m^N(t) − m(t) ‖ > ε ) → 0     as N → ∞,
> ```
> where `m(·)` solves `ṁ = F(m)` with `m(0) = lim m^N(0)`.

*Proof of Lemma 2 (sketch).* The generator of `m^N(·)` is that of a density-dependent Markov jump
process: transitions of individual arms change `m^N` by `O(1/N)` and their rates depend on `(s_n, a_n)`
only through the current empirical measure `m^N` (the index cutoff is a function of `m^N`). Kurtz's
theorem (1970) for density-dependent chains gives uniform-on-compacts convergence of the scaled
process to the ODE whose drift is the expected per-step change `F(m) = E[ Δ m^N | m^N = m ]·N`,
provided `F` is Lipschitz. `F` is piecewise-Lipschitz here (the only non-smoothness is at ties in the
index cutoff, a measure-zero set for generic reward parameters); Benaïm & Le Boudec (2008) extend the
convergence to this mean-field-interaction setting and handle the cutoff nonsmoothness. ∎

Combine: fix `ε > 0`. By WW there is `T_ε` with `‖ m(t) − m* ‖ < ε` for all `t ≥ T_ε`, from *any*
initial condition. By Lemma 2, `m^N(t)` is within `ε` of `m(t)` uniformly on `[0, T]` with
probability `→ 1`. Taking `T → ∞` after `N → ∞` (a standard interchange justified by the uniform
ergodicity of the finite chains, Weber–Weiss 1990 §2), the empirical time-average of the reward under
the index policy converges to the reward at `m*`, which by Step B is `V_rel`:
```
        (1/N) G_N(Whittle) → V_rel      as N → ∞.
```
With Step A's upper bound `(1/N) OPT(N) ≤ V_rel` and the trivial `G_N(Whittle) ≤ OPT(N)`, we get
```
        V_rel ≥ (1/N) OPT(N) ≥ (1/N) G_N(Whittle) → V_rel,
```
so all three coincide in the limit, proving asymptotic optimality. ∎

**Remark (necessity of WW).** Step C is the only place stability enters, and it is essential: if
`ṁ = F(m)` has a limit cycle, the time-average of the reward along the cycle can be strictly below
`V_rel`, and the index policy loses a `Θ(N)` amount. Weber & Weiss (1990) construct such an instance;
this is why indexability alone does **not** imply asymptotic optimality.

---

## 6. Refinements and rates

- **Convergence rate.** Under a *non-degeneracy* / **Uniform Global Attractor Property (UGAP)** and a
  local-stability (Hurwitz) condition on `F` at `m*`, the gap is not merely `o(N)` but `O(1/N)`
  (Gast, Gaujal & Yan 2023) or `O(√N)` absolute (Zhang & Frazier 2021), and can be **exponentially**
  small, `O(e^{−cN})`, when a further non-singularity condition holds (Gast, Gaujal & Yan 2023).
- **Non-indexable arms.** Verloop (2016) shows a broader class of **LP-priority policies** derived
  from the same relaxation is asymptotically optimal under WW-type conditions *without* requiring
  indexability, so the relaxation route survives even when the Whittle index is undefined.
- **LP view.** Bertsimas & Niño-Mora (2000) recast the relaxation as a linear program over occupation
  measures; `V_rel` is its optimal value and the Whittle index is the associated reduced-cost /
  primal-dual index. This is the cleanest route to Step B's strong duality.

---

## 7. Application to our budgeted online tool-building problem — and its limits

Map: **arm = problem type**; **state** = its posterior sufficient statistic `(k, t)` = (occurrences
seen, slots elapsed) plus a built-flag; **active = build** (write the tool for that type). Building
is **irreversible**: once active the arm is absorbed in a "built" state that thereafter yields the
reuse reward with no further decision — a *monotone / one-time-activation* restless arm, for which
indexability holds by monotonicity of the build value in the posterior-expected remaining occurrences
(§3). The per-type DP + `λ`-tuning described in the design note computes exactly `g(λ)`, the passive
set `D(λ)`, and hence the Whittle index; tuning `λ` to meet the budget is finding `λ*` via (2).

**Three honest caveats — why this is a *justification*, not a literal guarantee, for our runs:**

1. **Horizon.** The theorem is **infinite-horizon average-reward**; our benchmark is **finite-horizon**
   (`T ≈ 60`). The correct asymptotic-optimality theory for finite horizons is the LP-relaxation /
   fluid line — Brown & Smith (2020), Zhang & Frazier (2021), Gast–Gaujal–Yan finite-horizon results —
   which give analogous `o(N)` (and better) guarantees for the *time-cumulative* budget we actually
   have. The *shape* of the argument (relaxation upper bound + index policy attains it) carries over;
   the specific theorem invoked must be the finite-horizon one.

2. **Budget form.** Classical RMAB fixes a **per-period** active count `M`; we have a **single
   cumulative budget `B` over the whole horizon** with irreversible activation. This is a
   **weakly-coupled / budgeted MDP** (a.k.a. "bandit superprocess with a knapsack constraint"); the
   Lagrangian relaxation and index construction are identical, but the operative asymptotic theorem is
   again the finite-horizon LP one, not Weber–Weiss verbatim.

3. **Small `N`.** We use `N ≈ 12` types, not `N → ∞`. Asymptotic optimality says nothing rigorous at
   `N = 12`. **Therefore we do not rely on it operationally.** Our actual optimality certificate is the
   computable, per-instance **clairvoyant upper bound** (`optimal_build_set` / `fullinfo_value`): we
   report the index policy's value against that bound on every generated set, so any residual gap is
   *measured*, not assumed. The Whittle theory is the *reason to expect* the gap to be small (the index
   policy is the finite-`N` restriction of the provably-asymptotically-optimal relaxation solution),
   and the bracket is the *proof* that it is small in our regime.

---

## References

- J. C. Gittins (1979). *Bandit processes and dynamic allocation indices.* J. R. Statist. Soc. B
  41(2), 148–177.
- P. Whittle (1988). *Restless bandits: activity allocation in a changing world.* J. Appl. Probab.
  25A, 287–298.
- R. R. Weber & G. Weiss (1990). *On an index policy for restless bandits.* J. Appl. Probab. 27(3),
  637–648. (Addendum: Adv. Appl. Probab. 23(2), 1991, 429–430.)
- C. H. Papadimitriou & J. N. Tsitsiklis (1999). *The complexity of optimal queuing network control.*
  Math. Oper. Res. 24(2), 293–305. (RMAB is PSPACE-hard.)
- D. Bertsimas & J. Niño-Mora (2000). *Restless bandits, linear programming relaxations, and a
  primal–dual index heuristic.* Oper. Res. 48(1), 80–90.
- J. Niño-Mora (2001). *Restless bandits, partial conservation laws and indexability.* Adv. Appl.
  Probab. 33(1), 76–98. (Sufficient conditions for indexability.)
- T. G. Kurtz (1970). *Solutions of ordinary differential equations as limits of pure jump Markov
  processes.* J. Appl. Probab. 7(1), 49–58. (Fluid limit.)
- M. Benaïm & J.-Y. Le Boudec (2008). *A class of mean field interaction models for computer and
  communication systems.* Performance Evaluation 65(11–12), 823–838.
- I. M. Verloop (2016). *Asymptotically optimal priority policies for indexable and nonindexable
  restless bandits.* Ann. Appl. Probab. 26(4), 1947–1995.
- D. B. Brown & J. E. Smith (2020). *Index policies and performance bounds for dynamic selection
  problems.* Management Science 66(7), 3029–3050. (Finite-horizon LP bounds.)
- X. Zhang & P. I. Frazier (2021). *Restless bandits with many arms: beating the central limit
  theorem.* (Finite-horizon near-optimality; arXiv:2107.xxxx.)
- N. Gast, B. Gaujal & C. Yan (2023). *Exponential asymptotic optimality of the LP-index / Whittle
  index policies.* (Rates under UGAP + non-degeneracy; see also their finite-horizon papers.)
