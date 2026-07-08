"""RL Phase 1 reward (docs/rl-finetuning-plan.md "Phase 1: RL on urn, from base model").

reward = raw balls collected -- the urn's literal, stated objective. No reference policy, no DP, no
normalization. This SUPERSEDES an earlier `1 - regret/regret_eager` design (both terms computed against
`exact_dp.ExactDP`, the pi* same-info reference) for two independent reasons, both argued in the doc's
"Reward, revised" section:

  1. `rl_train.py::compute_advantages` already does exact group-relative advantage normalization
     (z-score within each G-sized group of rollouts sharing one stream) -- so the reward's absolute
     scale never reaches the gradient, only its ordering within a group does. A reference-policy-based
     rescaling was solving a problem GRPO's own baselining already solves.
  2. pi* (symmetric-Dirichlet(alpha) belief) is Bayes-optimal for a smooth continuous-draw world, not
     this benchmark's fixed discrete hot/trap split, and is measurably BEATEN in expectation by a
     two-line wait-for-one-repeat heuristic on the real generator -- so it was not even a trustworthy
     reference to normalize against. See the doc for the measured numbers.

pi*/wait2/eager/clairvoyant remain valid, useful EVALUATION yardsticks (is the trained policy's
disposition wait2-like? how far from clairvoyant?) -- just not baked into the reward that drives the
gradient.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.rl_reward   # self-test
"""
from __future__ import annotations


def episode_reward(slots: list[dict], kept: dict) -> dict:
    """kept: {class_id: class_position} as returned by `urn_session.run_episode`'s row (int keys --
    caller must NOT have stringified them through a JSON round-trip, or cast back to int first; only
    KEPT colors are present, no None values). reward = total balls collected: for each kept color, its
    occurrences from the keep position onward (current + all future same-color draws); un-kept colors
    collect nothing -- exactly what the model is told to maximize."""
    sizes: dict[int, int] = {}
    for s in slots:
        sizes[s["class_id"]] = sizes.get(s["class_id"], 0) + 1
    balls = sum(sizes[cid] - pos + 1 for cid, pos in kept.items() if pos is not None)
    return {"reward": float(balls), "balls": balls}


def per_decision_rewards(slots: list[dict], transcript: list[dict]) -> list[float]:
    """docs/rl-ppo-credit-assignment-spec.md §2: decompose the scalar episode reward into one term per
    KEEP/PASS decision, aligned 1:1 with `transcript` (== one entry per assistant turn in `row["messages"]`
    -- `urn_session.run_episode` appends exactly one user+assistant pair per decision, never per raw draw,
    so index i here IS turn i there; no separate index-mapping needed downstream).

    A KEEP at `class_position` out of `class_size` total occurrences contributes exactly
    `class_size - class_position + 1` (itself plus every future same-color draw, auto-collected free);
    a PASS contributes exactly 0. This is not reward shaping -- summing these across the episode
    reproduces `episode_reward`'s scalar exactly (see `_selftest`), because that scalar was already this
    sum in disguise (`_balls_collected`/`episode_reward` group the same per-color terms by class instead
    of by decision)."""
    sizes: dict[int, int] = {}
    for s in slots:
        sizes[s["class_id"]] = sizes.get(s["class_id"], 0) + 1
    return [float(sizes[t["class_id"]] - t["class_position"] + 1) if t["decision"] == "KEEP" else 0.0
            for t in transcript]


def builds_to_transcript(slots: list[dict], builds: dict, B: int) -> list[dict]:
    """Reconstruct a decision-turn list (the subset of `urn_session.run_episode`'s `transcript` fields
    `per_decision_rewards`/the critic's feature extraction need: slot/class_id/class_position/decision)
    from a `builds` dict ({class_id: class_position_kept_at, or None}) as produced by `pi_star.py`'s
    heuristics (`eager_builds`/`wait_k_builds`/`clairvoyant_builds`). Lets self-tests and the critic's
    standalone validation (docs/rl-ppo-credit-assignment-spec.md §10 step 2) exercise real decision
    sequences without an LLM call -- mirrors `run_episode`'s own decision loop exactly: skip draws of an
    already-kept color (auto-collected, not a decision), stop once the keep budget is exhausted."""
    kept: set[int] = set()
    budget_left = B
    out = []
    for s in sorted(slots, key=lambda z: z["slot_index"]):
        cid, pos = s["class_id"], s["class_position"]
        if cid in kept:
            continue
        if budget_left == 0:
            break
        decision = "KEEP" if builds.get(cid) == pos else "PASS"
        out.append({"slot": s["slot_index"], "class_id": cid, "class_position": pos, "decision": decision})
        if decision == "KEEP":
            kept.add(cid)
            budget_left -= 1
    return out


def _selftest():
    from scripts.creator.tool_disposition_benchmark.stream_builder import (
        StochasticStreamSpec, build_stochastic_stream)
    from scripts.creator.tool_disposition_benchmark.urn_session import UNIFORM, N, T, B, MAG, G
    from scripts.creator.tool_disposition_benchmark.pi_star import (
        eager_builds, wait_k_builds, clairvoyant_builds)

    slots, _meta = build_stochastic_stream(StochasticStreamSpec(
        families=UNIFORM, n_hot=B, T=T, budget=B, guarantee_trap_early=G, magnitude=MAG, seed=2000))

    def r(builds):
        return episode_reward(slots, {cid: v for cid, v in builds.items() if v is not None})["reward"]

    none_r, eager_r, wait2_r, clair_r = r({}), r(eager_builds(slots, B)), \
        r(wait_k_builds(slots, B, 2)), r(clairvoyant_builds(slots, B))
    assert none_r == 0.0, none_r                 # never keeping collects nothing
    assert eager_r > none_r, (eager_r, none_r)   # keeping something beats keeping nothing
    assert clair_r >= wait2_r >= 0, (clair_r, wait2_r)
    print(f"none={none_r:.0f}  eager={eager_r:.0f}  wait2={wait2_r:.0f}  clairvoyant={clair_r:.0f}")

    # per-decision decomposition (docs/rl-ppo-credit-assignment-spec.md §2): sum of per-decision
    # rewards must reproduce the scalar exactly, on every policy above (0 keeps, all-eager, wait2).
    for name, builds in [("none", {}), ("eager", eager_builds(slots, B)),
                         ("wait2", wait_k_builds(slots, B, 2))]:
        kept = {cid: v for cid, v in builds.items() if v is not None}
        transcript = builds_to_transcript(slots, builds, B)
        decomposed = sum(per_decision_rewards(slots, transcript))
        scalar = episode_reward(slots, kept)["reward"]
        assert decomposed == scalar, (name, decomposed, scalar)
    print("per-decision decomposition matches scalar reward on none/eager/wait2 -- OK")
    print("rl_reward self-test OK")


if __name__ == "__main__":
    _selftest()
