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
    print("rl_reward self-test OK")


if __name__ == "__main__":
    _selftest()
