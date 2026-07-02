"""Ski-rental scorer for the tool-amortization benchmark.

Converts a completed session (per-slot actions) + the stream's hidden labels into m*-relative
decision errors and compute-matched regret.

Cost model (one utility): a correct answer is worth R; every token costs lambda. Per action:
  hand:   u_hand  = R*a_hand - lambda*h         (a_hand from A0 forced-hand; h = hand tokens)
  build:  u_build = R*a_script - lambda*(C+r)    (write + first run; a_script ~ 1 from A0)
  reuse:  u_reuse = R*a_script - lambda*r         (call an existing tool)
Break-even horizon:  m* = lambda*C / s,  s = R*(1-a_hand) + lambda*(h-r)   (build iff m_f >= m*).
Regret includes lambda*tokens BY CONSTRUCTION -> compute-matched (answers the TroVE-matched skeptic).

OPTIMAL POLICY IS PLUGGABLE.
  - `fullinfo_value` (clairvoyant, known horizon): a valid UPPER BOUND. Implemented.
  - `online_value_provisional` (unknown horizon): a *PROVISIONAL* deterministic ski-rental baseline
    (build at ceil(m*)). This is NOT the true online optimum against an unknown horizon
    DISTRIBUTION — that is an online-distribution-learning problem to be filled in after the
    literature review. Regret-vs-online is reported but flagged provisional.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.skirental_scorer   # self-test
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

AUTHOR_ACTIONS = ("build", "rebuild")   # both author a tool; 'rebuild' = a redundant re-author
ACTIONS = ("hand", "build", "reuse", "rebuild")


@dataclass
class Costs:
    """Measured constants. a_hand is per-family (at the run's magnitude). Tokens in tokens; R and
    lambda set the value/compute exchange rate (report m* across a lambda range)."""
    a_hand: dict[str, float]
    h: float                    # mean hand-solve tokens
    C: float                    # mean tool-write tokens
    r: float                    # mean tool-call tokens
    R: float = 100.0            # value of a correct answer
    lam: float = 0.1            # per-token cost
    a_script: float = 1.0
    default_a_hand: float = 0.3  # for families not A0-calibrated (e.g. one-off pool)

    def ah(self, fam: str) -> float:
        return self.a_hand.get(fam, self.default_a_hand)

    def u_hand(self, fam: str) -> float:
        return self.R * self.ah(fam) - self.lam * self.h

    def u_build(self) -> float:
        return self.R * self.a_script - self.lam * (self.C + self.r)

    def u_reuse(self) -> float:
        return self.R * self.a_script - self.lam * self.r

    def m_star(self, fam: str) -> float:
        s = self.R * (1 - self.ah(fam)) + self.lam * (self.h - self.r)
        return (self.lam * self.C / s) if s > 0 else float("inf")


def _util(action: str, correct, tokens, costs: Costs) -> float:
    """Realized utility of one action. `correct` may be 0/1 (real transcript) or a float in [0,1]
    (expected value, used by the synthetic simulator). tokens=None -> use the action's constant."""
    if tokens is None:
        tokens = {"hand": costs.h, "build": costs.C + costs.r,
                  "rebuild": costs.C + costs.r, "reuse": costs.r}[action]
    return costs.R * float(correct) - costs.lam * tokens


# --------------------------------------------------------------------- reference policies
def fullinfo_value(fam: str, size: int, costs: Costs) -> float:
    """Clairvoyant known-horizon optimum: max(all-hand, build-on-first-then-reuse). Upper bound."""
    all_hand = size * costs.u_hand(fam)
    build_first = costs.u_build() + (size - 1) * costs.u_reuse()
    return max(all_hand, build_first)


def online_value_provisional(fam: str, size: int, costs: Costs) -> float:
    """PROVISIONAL online baseline (NOT the true optimum vs an unknown horizon distribution):
    deterministic ski-rental — hand-solve until member t=ceil(m*), then build & reuse the rest.
    Placeholder until the online-distribution-learning literature is consulted."""
    t = max(1, math.ceil(costs.m_star(fam)))
    if size < t:
        return size * costs.u_hand(fam)
    return (t - 1) * costs.u_hand(fam) + costs.u_build() + (size - t) * costs.u_reuse()


# --------------------------------------------------------------------- realizable ONLINE reference
def _posterior_expected_p(k: int, t: int, support: list[float],
                          prior: list[float] | None = None) -> float:
    """E[p | this type was seen k times in the first t iid slots], posterior over the discrete
    `support` of the pool's per-type appearance probabilities (prior uniform over the support unless
    given). Bernoulli-count likelihood p^k (1-p)^(t-k) (the binomial coeff cancels). This is exactly
    the information the online policy has -- it does NOT know which type is frequent, it infers each
    type's rate from the observed prefix, like the model."""
    if not support:
        return 0.0
    prior = prior or [1.0] * len(support)
    ws, num = 0.0, 0.0
    for pj, pr in zip(support, prior):
        pj = min(max(pj, 0.0), 1.0)
        like = (pj ** k) * ((1.0 - pj) ** (t - k))
        w = pr * like
        ws += w
        num += w * pj
    return (num / ws) if ws > 0 else (sum(support) / len(support))


def online_reference(actions: list[dict], costs: Costs, budget: int | None = None,
                     min_repeats: int = 2, support: list[float] | None = None) -> dict:
    """Realizable ONLINE reference for the STOCHASTIC design (problems drawn i.i.d. from a pool of
    types). The policy sees the same thing the model does -- a stream of typed problems, one at a
    time -- and knows only that types are drawn from a distribution (NOT which type is frequent, NOT
    the horizon). It gathers evidence before spending an irreversible, budget-limited build:

      * a type is BUILD-ELIGIBLE only after it has been seen `min_repeats` times (default 2: one
        confirmatory repeat) -- so scarce builds are never wasted on singletons / rarely-seen types;
      * eligible types are built in the temporal order they become eligible, until the write budget
        is spent -- because high-rate types recur sooner and more often, they become eligible first
        and win the budget; low-rate types recur late (after the budget is gone) or never.

    This is the key contrast with the model: the reference WAITS for a repeat (mean lateness >= 1) and
    RESERVES its budget for demonstrated recurrence; the eager model builds on first sight (lateness
    0) and burns the budget on whatever it sees first. It is myopic (build-now-vs-never at the
    eligibility point) and greedy on the budget, so it builds weakly EARLIER / less selectively than
    the true optimum -> regret vs this reference is a conservative LOWER BOUND on regret vs the true
    online optimum. `support` (pool probabilities) is accepted for the posterior readout only; the
    build rule keys off observed repeats and is prior-free."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for a in actions:
        by_class[a["class_id"]].append(a)
    remaining_budget = budget if budget is not None else len(by_class)
    stream = sorted(actions, key=lambda a: a["slot_index"])
    seen: dict[int, int] = defaultdict(int)
    built_at: dict[int, int] = {}          # cid -> class_position at which the reference built
    ep_at_build: dict[int, float] = {}
    for a in stream:
        cid = a["class_id"]
        seen[cid] += 1
        if cid in built_at or seen[cid] < min_repeats or remaining_budget <= 0:
            continue
        # eligible now (>= min_repeats sightings) and budget free -> build, reserving it for this
        # demonstrated-recurring type. (Under the m*<1 accuracy regime the per-type profitability
        # test is always met once it recurs, so the binding constraint is the budget, not m*.)
        built_at[cid] = seen[cid]
        remaining_budget -= 1
        if support is not None:
            ep_at_build[cid] = _posterior_expected_p(seen[cid], a["slot_index"] + 1, support)

    total, per_class = 0.0, {}
    for cid, acts in by_class.items():
        fam, size = acts[0]["family"], acts[0]["class_size"]
        bpos = built_at.get(cid)
        if bpos is None:
            val = size * costs.u_hand(fam)
        else:
            val = ((bpos - 1) * costs.u_hand(fam) + costs.u_build()
                   + (size - bpos) * costs.u_reuse())
        total += val
        per_class[cid] = {"built": bpos is not None, "build_position": bpos, "value": val,
                          "lateness": (bpos - 1) if bpos is not None else None,
                          "ep_at_build": ep_at_build.get(cid)}
    lats = [v["lateness"] for v in per_class.values() if v["lateness"] is not None]
    return {"total_value": total, "classes": per_class,
            "mean_lateness": (sum(lats) / len(lats)) if lats else float("nan"),
            "n_built": len(built_at)}


def _build_gain(fam: str, size: int, costs: Costs) -> float:
    """Value of building-then-reusing this class over solving all its members by hand."""
    all_hand = size * costs.u_hand(fam)
    build_first = costs.u_build() + (size - 1) * costs.u_reuse()
    return build_first - all_hand


def optimal_build_set(metas: list[tuple], costs: Costs, budget: int) -> set:
    """Budget-constrained clairvoyant optimum. When building buys so much accuracy that the per-class
    break-even m* < 1 (build-everything looks optimal), the REAL scarcity is the write budget: with
    at most `budget` tools, the optimum spends them on the `budget` classes with the largest
    build-gain (a 0/1-knapsack with unit weights -> just top-k). metas: (class_id, family, size).
    Returns the class_ids the optimum builds."""
    gains = [(_build_gain(fam, size, costs), cid) for cid, fam, size in metas]
    gains = [(g, cid) for g, cid in gains if g > 0]
    gains.sort(reverse=True)
    return {cid for _, cid in gains[:budget]}


# --------------------------------------------------------------------- scoring
def score_class(actions: list[dict], costs: Costs, opt_build: bool | None = None) -> dict:
    """Score one class's action records (each: action, correct, tokens?, class_position). actions
    need not be sorted. `opt_build`: whether the BUDGET-constrained optimum builds this class
    (None -> fall back to the unconstrained per-class m* rule, `size >= m*`)."""
    acts = sorted(actions, key=lambda a: a["class_position"])
    fam = acts[0]["family"]
    size = acts[0]["class_size"]
    model_value = sum(_util(a["action"], a["correct"], a.get("tokens"), costs) for a in acts)

    authored = [a for a in acts if a["action"] in AUTHOR_ACTIONS]
    built = len(authored) > 0
    build_time = authored[0]["class_position"] if built else None
    lateness = (build_time - 1) if built else None
    reuse_count = sum(a["action"] == "reuse" for a in acts)
    rebuild_count = sum(a["action"] == "rebuild" for a in acts)

    mstar = costs.m_star(fam)
    pays = size >= mstar
    should_build = pays if opt_build is None else opt_build
    if should_build and built:
        decision = "correct-build"
    elif should_build and not built:
        decision = "wrongly-skipped"
    elif (not should_build) and built:
        decision = "wrongly-built"
    else:
        decision = "correct-skip"

    # optimal value for THIS class under the chosen optimum (budget-constrained if opt_build given)
    all_hand = size * costs.u_hand(fam)
    build_first = costs.u_build() + (size - 1) * costs.u_reuse()
    opt_value = (build_first if should_build else all_hand) if opt_build is not None \
        else fullinfo_value(fam, size, costs)
    on = online_value_provisional(fam, size, costs)
    return {"family": fam, "size": size, "m_star": mstar, "pays": pays,
            "opt_build": should_build,
            "built": built, "build_time": build_time, "lateness": lateness,
            "reuse_count": reuse_count, "rebuild_count": rebuild_count,
            "model_value": model_value, "decision": decision,
            "regret_fullinfo": fullinfo_value(fam, size, costs) - model_value,
            "regret_budget": opt_value - model_value,
            "regret_online_provisional": on - model_value}


def score_stream(actions: list[dict], costs: Costs, budget: int | None = None,
                 online_min_repeats: int = 2, support: list[float] | None = None) -> dict:
    """actions: normalized per-slot records with class_id, class_size, class_position, family,
    action, correct, tokens?. `budget`: if given, decisions/regret are scored against the
    BUDGET-CONSTRAINED clairvoyant optimum (build the `budget` highest-gain classes); else the
    unconstrained per-class m* rule. `online_min_repeats`/`support` parameterize the realizable
    online reference (see online_reference). Returns per-class rows + an aggregate summary."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for a in actions:
        by_class[a["class_id"]].append(a)
    opt_set = None
    if budget is not None:
        metas = [(cid, acts[0]["family"], acts[0]["class_size"]) for cid, acts in by_class.items()]
        opt_set = optimal_build_set(metas, costs, budget)
    cids = list(by_class.keys())
    rows = [score_class(by_class[cid], costs,
                        opt_build=(cid in opt_set) if opt_set is not None else None)
            for cid in cids]

    # realizable online reference (budget-aware, evidence-gathering) -> per-class regret_online
    online = online_reference(actions, costs, budget=budget,
                              min_repeats=online_min_repeats, support=support)
    for cid, r in zip(cids, rows):
        oc = online["classes"].get(cid, {})
        r["online_built"] = oc.get("built")
        r["online_build_position"] = oc.get("build_position")
        r["regret_online"] = oc.get("value", r["model_value"]) - r["model_value"]

    def _rate(num, den):
        return (num / den) if den else float("nan")

    recurring = [r for r in rows if r["size"] >= 2]
    oneoffs = [r for r in rows if r["size"] == 1]
    counts = defaultdict(int)
    for r in rows:
        counts[r["decision"]] += 1
    eligible_reuse = sum(max(0, r["size"] - 1) for r in rows if r["built"])
    agg = {
        "n_classes": len(rows),
        "decision_counts": dict(counts),
        "build_rate_recurring": _rate(sum(r["built"] for r in recurring), len(recurring)),
        "build_rate_oneoff": _rate(sum(r["built"] for r in oneoffs), len(oneoffs)),
        "n_oneoffs_built": sum(r["built"] for r in oneoffs),
        "reuse_rate": _rate(sum(r["reuse_count"] for r in rows), eligible_reuse),
        "total_rebuilds": sum(r["rebuild_count"] for r in rows),
        "mean_lateness": _rate(sum(r["lateness"] for r in rows if r["lateness"] is not None),
                               sum(1 for r in rows if r["lateness"] is not None)),
        "total_regret_fullinfo": sum(r["regret_fullinfo"] for r in rows),
        "total_regret_budget": sum(r["regret_budget"] for r in rows),
        "total_regret_online_provisional": sum(r["regret_online_provisional"] for r in rows),
        "total_regret_online": sum(r["regret_online"] for r in rows),
        "online_mean_lateness": online["mean_lateness"],
        "online_n_built": online["n_built"],
        "budget": budget,
    }
    return {"classes": rows, "aggregate": agg}


# --------------------------------------------------------------------- constants from A0
def costs_from_a0(run_dir: str, model: str, magnitude: int,
                  R: float = 100.0, lam: float = 0.1, r: float = 200.0) -> Costs:
    """Build Costs from an a0_oracle_gap results.jsonl: a_hand per family (mean 'correct' of hand
    records at `magnitude`), h = mean hand tokens, C = mean build tokens. r is not measured by A0
    (reuse happens in-session) -> pass a small constant; refine from session run_script tokens."""
    recs = [json.loads(l) for l in Path(run_dir, "results.jsonl").read_text().splitlines() if l.strip()]
    recs = [x for x in recs if x["model"] == model and x["magnitude"] == magnitude]
    a_hand, htoks, ctoks = {}, [], []
    by_fam = defaultdict(lambda: {"hand": [], "build": []})
    for x in recs:
        by_fam[x["family"]][x["condition"]].append(x)
    for fam, d in by_fam.items():
        if d["hand"]:
            a_hand[fam] = sum(z["correct"] for z in d["hand"]) / len(d["hand"])
            htoks += [z["tokens"] for z in d["hand"] if z.get("tokens")]
        ctoks += [z["tokens"] for z in d["build"] if z.get("tokens")]
    h = sum(htoks) / len(htoks) if htoks else 3000.0
    C = sum(ctoks) / len(ctoks) if ctoks else 3000.0
    return Costs(a_hand=a_hand, h=h, C=C, r=r, R=R, lam=lam)


# --------------------------------------------------------------------- transcript adapter (best-effort)
def actions_from_session(session: dict, slots: list[dict]) -> list[dict]:
    """Map a persistent-session harness record (its `records[]`) + the stream `slots` (labels) into
    normalized action records.

    BUILD is attributed by the AUTHORING event: a slot whose record logged `scripts_authored` (a
    `write_script` fired while that problem was on screen) is a 'build' for that class ('rebuild' if
    the class already authored one). This is robust to (a) truncation cutting off a freshly-built
    tool before it is reused, and (b) the model running a tool on a DIFFERENT family (exploration /
    flailing) -- neither of which is a build for the visited class. Running an existing script with
    no new authoring -> 'reuse'; nothing -> 'hand'.

    Fallback: older runs (recorded before `scripts_authored` existed) use the legacy proxy = a
    script's FIRST appearance in `scripts_run` marks its authoring slot."""
    recs = {r.get("idx"): r for r in session.get("records", [])}
    has_authoring = any("scripts_authored" in (r or {}) for r in recs.values())
    seen_scripts: set = set()
    class_built: set = set()
    out = []
    for s in sorted(slots, key=lambda z: z["slot_index"]):
        rec = recs.get(s["slot_index"])
        cid = s["class_id"]
        if has_authoring:
            authored = (rec.get("scripts_authored") if rec else None) or []
            if authored:
                action = "rebuild" if cid in class_built else "build"
                class_built.add(cid)
            elif rec and rec.get("used_script"):
                action = "reuse"
            else:
                action = "hand"
        else:                              # legacy run-based proxy
            if not rec or not rec.get("used_script"):
                action = "hand"
            else:
                ran = rec.get("scripts_run") or []
                new = [sc for sc in ran if sc not in seen_scripts]
                if new:
                    action = "rebuild" if cid in class_built else "build"
                    class_built.add(cid)
                else:
                    action = "reuse"
                seen_scripts.update(ran)
        out.append({"slot_index": s["slot_index"], "class_id": cid, "family": s["family"],
                    "class_size": s["class_size"], "class_position": s["class_position"],
                    "action": action, "correct": bool(rec["correct"]) if rec else False,
                    "tokens": None})
    return out


# --------------------------------------------------------------------- pi* same-info reference
def model_builds_from_actions(actions: list[dict]) -> dict:
    """{class_id: build_position (1-based class_position of the first authoring) or None} -- the
    model's ACTUAL build decisions, from its authoring events. Feeds `value_of_builds`, which values
    them analytically over full realized class sizes (so a truncated session is extrapolated: reuse
    the rest if built, hand-solve the rest if not) -- killing the truncation confound."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for a in actions:
        by_class[a["class_id"]].append(a)
    out = {}
    for cid, acts in by_class.items():
        authored = sorted((a for a in acts if a["action"] in AUTHOR_ACTIONS),
                          key=lambda a: a["class_position"])
        out[cid] = authored[0]["class_position"] if authored else None
    return out


def pistar_report(slots: list[dict], costs: Costs, budget: int, N: int, T: int, pool: list[str],
                  a_hands: dict, magnitude: int, model_builds: dict, alpha: float = 1.0,
                  price: float | None = None, n_sim: int = 150) -> dict:
    """Value the model's ACTUAL builds and pi*'s builds ANALYTICALLY over full realized class sizes
    (no truncation), on one stochastic stream. pi* is the SAME-INFORMATION reference (Whittle-index
    DP under the exchangeable prior; knows only {N,T,B}). Reports:
      * regret_lb = value(pi*) - value(model)  -- the headline CONSERVATIVE LOWER BOUND on the
        model's regret (pi* has identical info to the model, so a true optimum is >= pi*);
      * clairvoyant_gap = value(clairvoyant) - value(pi*) -- intrinsic price of online uncertainty;
      * role-keyed bait: how many TRAP types (ground-truth low-rate) each policy built.
    `price`: pass a pre-tuned Whittle price to reuse across seeds (it's a design constant, a function
    of {N,T,B,pool}); None -> tune here."""
    from scripts.creator.tool_disposition_benchmark.pi_star import (
        tune_price, _regions_for_pool, policy_builds, value_of_builds, clairvoyant_builds)
    if price is None:
        price = tune_price(costs, pool, a_hands, N, T, budget, magnitude, alpha=alpha, n_sim=n_sim)
    regions = _regions_for_pool(costs, a_hands, price, T, alpha, N)
    pi_builds = policy_builds(slots, regions, budget)

    role = {s["class_id"]: s.get("role") for s in slots}

    def _lat(builds):
        ls = [b - 1 for b in builds.values() if b is not None]
        return (sum(ls) / len(ls)) if ls else float("nan")

    def _traps(builds):
        return sum(1 for cid, b in builds.items() if b is not None and role.get(cid) == "trap")

    def _hots(builds):
        return sum(1 for cid, b in builds.items() if b is not None and role.get(cid) == "hot")

    v_model = value_of_builds(slots, model_builds, costs)
    v_star = value_of_builds(slots, pi_builds, costs)
    v_clair = value_of_builds(slots, clairvoyant_builds(slots, budget), costs)
    return {
        "price": price,
        "value_model": v_model, "value_pistar": v_star, "value_clairvoyant": v_clair,
        "regret_lb": v_star - v_model,          # headline: model's regret lower bound
        "clairvoyant_gap": v_clair - v_star,    # intrinsic online-uncertainty price
        "model_n_built": sum(1 for b in model_builds.values() if b is not None),
        "pistar_n_built": sum(1 for b in pi_builds.values() if b is not None),
        "model_lateness": _lat(model_builds), "pistar_lateness": _lat(pi_builds),
        "model_traps_built": _traps(model_builds), "model_hots_built": _hots(model_builds),
        "pistar_traps_built": _traps(pi_builds), "pistar_hots_built": _hots(pi_builds),
    }


# --------------------------------------------------------------------- score a real stream run
def score_run(run_dir: str, a0_dir: str, model: str, magnitude: int,
              R: float = 100.0, lam: float = 0.1, r: float = 200.0,
              pistar_price: float | None = None) -> dict:
    """Load a stream run (stream.json labels + sessions.jsonl transcript), build costs from the A0
    calibration, map the transcript to actions, score, and print. Returns the score dict."""
    slots = json.loads(Path(run_dir, "stream.json").read_text())
    rows = [json.loads(l) for l in Path(run_dir, "sessions.jsonl").read_text().splitlines() if l.strip()]
    session = next((x for x in rows if x.get("model_key") == model or x.get("model", "").endswith(model)), None)
    if session is None or session.get("error"):
        raise RuntimeError(f"no clean session for {model}: {session and session.get('error')}")
    costs = costs_from_a0(a0_dir, model, magnitude, R=R, lam=lam, r=r)

    budget = None
    cfg_path = Path(run_dir, "config.json")
    if cfg_path.exists():
        budget = json.loads(cfg_path.read_text()).get("budget")

    actions = actions_from_session(session, slots)
    res = score_stream(actions, costs, budget=budget)

    print(f"costs: a_hand={ {k: round(v,2) for k,v in costs.a_hand.items()} }  "
          f"h={costs.h:.0f} C={costs.C:.0f} r={costs.r:.0f} R={costs.R} lam={costs.lam}")
    print(f"m* per family: { {f: round(costs.m_star(f),2) for f in costs.a_hand} }\n")
    print(f"budget-constrained optimum builds the top-{budget} classes by build-gain\n")
    print(f"{'family':<16}{'size':>5}{'m*':>6}{'optB':>6}{'built':>6}{'btime':>6}{'reuse':>6}"
          f"{'rebld':>6}{'decision':>16}{'regret_B':>11}")
    for c in sorted(res["classes"], key=lambda z: (-z["size"], z["family"])):
        print(f"{c['family']:<16}{c['size']:>5}{c['m_star']:>6.1f}{str(c['opt_build']):>6}"
              f"{str(c['built']):>6}{str(c['build_time']):>6}{c['reuse_count']:>6}"
              f"{c['rebuild_count']:>6}{c['decision']:>16}{c['regret_budget']:>11.1f}")
    a = res["aggregate"]
    print(f"\naggregate: {a['decision_counts']}")
    print(f"  build_rate(recurring)={a['build_rate_recurring']:.2f}  "
          f"build_rate(one-off)={a['build_rate_oneoff']:.2f} ({a['n_oneoffs_built']} built)  "
          f"reuse_rate={a['reuse_rate']:.2f}  rebuilds={a['total_rebuilds']}  "
          f"mean_lateness={a['mean_lateness']:.2f}")
    print(f"  total regret vs BUDGET-optimum={a['total_regret_budget']:.1f}  "
          f"(vs unconstrained-fullinfo={a['total_regret_fullinfo']:.1f})")
    print(f"  total regret vs wait-one-repeat (extra-info: knows rare=one-off)="
          f"{a['total_regret_online']:.1f}  "
          f"(waits: mean_lateness={a['online_mean_lateness']:.2f} vs "
          f"model {a['mean_lateness']:.2f}; built {a['online_n_built']})")

    # ---- pi* SAME-INFO reference: analytic value over FULL realized sizes (kills truncation) ----
    meta_path = Path(run_dir, "meta.json")
    if meta_path.exists() and budget is not None:
        meta = json.loads(meta_path.read_text())
        pool = list(meta["assignment"].keys())
        a_hands = {f: costs.ah(f) for f in pool}
        model_builds = model_builds_from_actions(actions)
        rep = pistar_report(slots, costs, budget, meta["N"], meta["T"], pool, a_hands,
                            meta.get("magnitude", magnitude), model_builds, price=pistar_price)
        res["pistar"] = rep
        print(f"\n  pi* SAME-INFO reference (analytic over full realized sizes; price={rep['price']:.1f}):")
        print(f"    value: model={rep['value_model']:.1f}  pi*={rep['value_pistar']:.1f}  "
              f"clairvoyant={rep['value_clairvoyant']:.1f}")
        print(f"    REGRET vs pi* (LOWER BOUND) = {rep['regret_lb']:.1f}   "
              f"clairvoyant gap = {rep['clairvoyant_gap']:.1f}")
        print(f"    builds: model={rep['model_n_built']} "
              f"(traps={rep['model_traps_built']} hots={rep['model_hots_built']} "
              f"lateness={rep['model_lateness']:.2f})  vs  "
              f"pi*={rep['pistar_n_built']} (traps={rep['pistar_traps_built']} "
              f"hots={rep['pistar_hots_built']} lateness={rep['pistar_lateness']:.2f})")
    return res


# --------------------------------------------------------------------- self-test (synthetic, no model)
def _simulate(slots: list[dict], policy: str, costs: Costs) -> list[dict]:
    """Synthetic policies for testing. `correct` emitted as EXPECTED value (float): a_hand for hand,
    1.0 for a tool -> deterministic, so regret assertions are exact."""
    out = []
    for s in slots:
        fam, pos, size = s["family"], s["class_position"], s["class_size"]
        if policy == "always_hand":
            action, correct = "hand", costs.ah(fam)
        elif policy == "eager":                       # build on first sighting, reuse after (builds one-offs too)
            action, correct = ("build", 1.0) if pos == 1 else ("reuse", 1.0)
        elif policy == "rebuild":                     # author every time, never reuse
            action, correct = ("build", 1.0) if pos == 1 else ("rebuild", 1.0)
        elif policy == "fullinfo":                    # clairvoyant: build iff size>=m*
            if size >= costs.m_star(fam):
                action, correct = ("build", 1.0) if pos == 1 else ("reuse", 1.0)
            else:
                action, correct = "hand", costs.ah(fam)
        else:
            raise ValueError(policy)
        out.append({"slot_index": s["slot_index"], "class_id": s["class_id"], "family": fam,
                    "class_size": size, "class_position": pos, "action": action,
                    "correct": correct, "tokens": None})
    return out


def _selftest():
    from scripts.creator.tool_disposition_benchmark.stream_builder import StreamSpec, build_stream
    costs = Costs(a_hand={"product3": 0.5}, h=1000, C=4200, r=100, R=100.0, lam=0.1,
                  default_a_hand=0.5)   # one-off family uses the fallback -> same m*
    ms = costs.m_star("product3")
    assert abs(ms - 3.0) < 1e-9, ms
    print(f"m* = {ms:.2f}  (product3 size4 pays; size1 one-off does not)\n")

    spec = StreamSpec(recurring=[("product3", 4)], n_one_offs=1, one_off_difficulty="easy",
                      magnitude=10, arrival="spread", seed=1)
    slots = build_stream(spec)

    def rec_row(res):   # the recurring class (size >= 2)
        return next(r for r in res["classes"] if r["size"] >= 2)

    def dis_row(res):   # the one-off (size 1)
        return next(r for r in res["classes"] if r["size"] == 1)

    for pol in ("always_hand", "eager", "rebuild", "fullinfo"):
        res = score_stream(_simulate(slots, pol, costs), costs)
        rec, dis = rec_row(res), dis_row(res)
        print(f"{pol:<12} recurring(size4): {rec['decision']:<15} regret_fi={rec['regret_fullinfo']:+7.1f} "
              f"rebuilds={rec['rebuild_count']} | one-off: {dis['decision']:<14} "
              f"regret_fi={dis['regret_fullinfo']:+7.1f} | totRegret={res['aggregate']['total_regret_fullinfo']:+7.1f}")

    # assertions
    ah = score_stream(_simulate(slots, "always_hand", costs), costs)
    eg = score_stream(_simulate(slots, "eager", costs), costs)
    rb = score_stream(_simulate(slots, "rebuild", costs), costs)
    assert rec_row(ah)["decision"] == "wrongly-skipped" and rec_row(ah)["regret_fullinfo"] > 0
    assert dis_row(ah)["decision"] == "correct-skip" and abs(dis_row(ah)["regret_fullinfo"]) < 1e-9
    assert rec_row(eg)["decision"] == "correct-build" and abs(rec_row(eg)["regret_fullinfo"]) < 1e-9
    assert dis_row(eg)["decision"] == "wrongly-built" and dis_row(eg)["regret_fullinfo"] > 0
    assert rec_row(rb)["regret_fullinfo"] > rec_row(eg)["regret_fullinfo"]
    assert rec_row(rb)["rebuild_count"] == 3
    print("\nself-test OK")


def _selftest_stochastic():
    """STOCHASTIC design: problems drawn i.i.d. from a heavy-tailed pool of types, with a binding
    write budget. Verifies the online reference (a) WAITS -- builds only after a repeat, mean
    lateness >= 1 -- and (b) BEATS the eager (build-on-first-sight) model, because eager burns its
    scarce budget on whatever it sees first (often a rare type) while the reference reserves it for
    demonstrated-recurring (high-rate) types. This is the paper's core comparison in miniature."""
    import random as _r
    print("\n================ stochastic reference self-test ================")
    # Heavy tail with GENUINE TRAPS: 3 hot types (expected count >> m*, worth a tool) + 10 rare
    # "trap" types (expected count ~1-2 < m*, NOT worth a tool). Budget < #hot forces SELECTION:
    # spend scarce builds on demonstrated-hot types, never on a trap. This is the regime where eager
    # (build-on-first-sight, first-come budget) is genuinely wrong -- it burns builds on early traps
    # and on un-vetted types, starving the hot ones. (See threshold note: m* ~ 2.8 below.)
    types = ([("hotA", 0.24), ("hotB", 0.20), ("hotC", 0.16)]
             + [(f"trap{i}", 0.04) for i in range(10)])   # 10 traps @ ~2.4 expected occ each
    fams = [f for f, _ in types]
    support = [p for _, p in types]
    T, BUDGET = 60, 3                       # 3 builds; 3 hot types -> exactly binding, no slack
    costs = Costs(a_hand={f: 0.4 for f in fams}, h=1000, C=4200, r=100, R=100.0, lam=0.1,
                  default_a_hand=0.4)       # s=150, m*=lam*C/s=2.8 -> traps (<=2 occ) are unprofitable

    def make_actions(draw, policy):
        seen, sizes = defaultdict(int), {c: draw.count(c) for c in set(draw)}
        built, budget_left, out = set(), BUDGET, []
        for i, c in enumerate(draw):
            seen[c] += 1
            fam = fams[c]
            if policy == "eager":           # build on FIRST sight, budget spent first-come
                if seen[c] == 1 and budget_left > 0:
                    act, budget_left = ("build", 1.0), budget_left - 1
                    built.add(c)
                elif c in built:
                    act = ("reuse", 1.0)
                else:
                    act = ("hand", costs.ah(fam))
            elif policy == "hand":
                act = ("hand", costs.ah(fam))
            else:
                raise ValueError(policy)
            out.append({"slot_index": i, "class_id": c, "family": fam, "class_size": sizes[c],
                        "class_position": seen[c], "action": act[0], "correct": act[1],
                        "tokens": None})
        return out

    eager_reg, ref_lat, ref_gt_eager = [], [], 0
    n = 60
    for seed in range(n):
        rng = _r.Random(seed)
        draw = rng.choices(range(len(types)), weights=support, k=T)
        eager_acts = make_actions(draw, "eager")
        res = score_stream(eager_acts, costs, budget=BUDGET, support=support)
        a = res["aggregate"]
        eager_reg.append(a["total_regret_online"])
        ref_lat.append(a["online_mean_lateness"])
        if a["total_regret_online"] > 0:
            ref_gt_eager += 1

    import statistics as _st
    mreg, mlat = _st.mean(eager_reg), _st.mean(ref_lat)
    print(f"seeds={len(eager_reg)}  eager-model regret vs online reference: "
          f"mean={mreg:.1f}  (>0 on {ref_gt_eager}/{len(eager_reg)} seeds)")
    print(f"online reference mean_lateness={mlat:.2f} (waits ~1 repeat)  vs  eager model lateness=0")
    assert mlat >= 0.99, f"reference should wait >=1 repeat, got {mlat}"
    assert mreg > 0, f"eager model should have positive MEAN regret vs reference, got {mreg}"
    # NOTE: per-seed win-rate is only ~55-60%, NOT ~100%: the myopic reference pays a lateness-1
    # timing cost, so on lucky draws where eager happens to build only hot types its lateness-0
    # timing edges out the reference. The robust claim is the positive MEAN regret + the lateness gap.
    print("stochastic self-test OK")


if __name__ == "__main__":
    _selftest()
    _selftest_stochastic()
