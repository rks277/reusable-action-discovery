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


def score_stream(actions: list[dict], costs: Costs, budget: int | None = None) -> dict:
    """actions: normalized per-slot records with class_id, class_size, class_position, family,
    action, correct, tokens?. `budget`: if given, decisions/regret are scored against the
    BUDGET-CONSTRAINED clairvoyant optimum (build the `budget` highest-gain classes); else the
    unconstrained per-class m* rule. Returns per-class rows + an aggregate summary."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for a in actions:
        by_class[a["class_id"]].append(a)
    opt_set = None
    if budget is not None:
        metas = [(cid, acts[0]["family"], acts[0]["class_size"]) for cid, acts in by_class.items()]
        opt_set = optimal_build_set(metas, costs, budget)
    rows = [score_class(acts, costs,
                        opt_build=(cid in opt_set) if opt_set is not None else None)
            for cid, acts in by_class.items()]

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


# --------------------------------------------------------------------- score a real stream run
def score_run(run_dir: str, a0_dir: str, model: str, magnitude: int,
              R: float = 100.0, lam: float = 0.1, r: float = 200.0) -> dict:
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


if __name__ == "__main__":
    _selftest()
