"""RL Phase 1 rollout collection (docs/rl-finetuning-plan.md "Phase 1: RL on urn, from base model").

Thin wrapper around `urn_session.run_episode` -- no new serving-layer code, no vLLM, no tool-calling
bypass, just G-way sampling-mode concurrency over many seeds against whatever `client`/`model` the
caller points at (Ollama, serving the current outer step's checkpoint).

Seeds: a dedicated disjoint range (9000+), never reused across outer steps within one pilot run (fresh
training streams each step, not overfit to a small fixed pool) -- disjoint from eval (2000-2023),
tool_bridge (4000+), urn SFT (5000+), anchor (6000+), mechanics_bridge (7000+), error_recovery (8000+).

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.rl_rollout   # self-test (fake client)
"""
from __future__ import annotations

import asyncio

from scripts.creator.tool_disposition_benchmark.stream_builder import (
    StochasticStreamSpec, build_stochastic_stream)
from scripts.creator.tool_disposition_benchmark.urn_session import render_system, run_episode

RL_SEED_START = 9000


async def collect_batch(client, model, seeds: list[int], G: int, *, N: int, T: int, B: int,
                        pool: list[str], magnitude: int, temperature: float,
                        announce_n: bool = False, conc: int = 8, guarantee_trap_early: float = 1.0
                        ) -> list[dict]:
    """One seed -> ONE stream, shared (same object) across its G group members -- required for GRPO's
    group-relative advantage, which compares G outcomes from the SAME starting condition. Returns a
    flat list of {seed, slots, row}, `len(seeds)*G` entries, `conc`-limited concurrency across the
    whole batch (not just within a group -- Ollama serializes per `OLLAMA_NUM_PARALLEL` regardless of
    how the caller shapes the gather)."""
    system = render_system(T, B, N, announce_n)
    units = []
    for seed in seeds:
        slots, _meta = build_stochastic_stream(StochasticStreamSpec(
            families=pool, n_hot=B, T=T, budget=B, guarantee_trap_early=guarantee_trap_early,
            magnitude=magnitude, seed=seed))
        units.extend({"seed": seed, "slots": slots} for _ in range(G))

    sem = asyncio.Semaphore(conc)

    async def _run(unit):
        async with sem:
            row = await run_episode(client, model, unit["slots"], T=T, B=B, system=system,
                                    temperature=temperature)
            return {**unit, "row": row}

    return await asyncio.gather(*(_run(u) for u in units))


def group_by_seed(results: list[dict]) -> dict[int, list[dict]]:
    """{seed: [G results]} -- the shape the GRPO group-relative advantage needs."""
    groups: dict[int, list[dict]] = {}
    for r in results:
        groups.setdefault(r["seed"], []).append(r)
    return groups


def _selftest():
    from scripts.creator.tool_disposition_benchmark.urn_session import UNIFORM, N, T, B, MAG

    class FakeClient:
        """Alternates KEEP/PASS by call count so groups aren't degenerate -- enough to exercise the
        batching/grouping shape without needing a real model."""
        def __init__(self):
            self.n = 0
            self.last_usage = {"input_tokens": 10, "output_tokens": 5}

        async def chat(self, model, system, messages, max_tokens, temperature):
            self.n += 1
            return "DECISION: KEEP" if self.n % 3 == 0 else "DECISION: PASS"

    async def main():
        seeds = [RL_SEED_START, RL_SEED_START + 1]
        results = await collect_batch(FakeClient(), "fake", seeds, G=4, N=N, T=T, B=B, pool=UNIFORM,
                                      magnitude=MAG, temperature=0.7, conc=4)
        assert len(results) == len(seeds) * 4
        groups = group_by_seed(results)
        assert set(groups) == set(seeds)
        for seed, g in groups.items():
            assert len(g) == 4
            assert all(r["slots"] is g[0]["slots"] for r in g), "group members must share ONE stream object"
        print(f"collected {len(results)} episodes across {len(groups)} seeds, "
              f"{len(next(iter(groups.values())))} per group")
        print("rl_rollout self-test OK")

    asyncio.run(main())


if __name__ == "__main__":
    _selftest()
