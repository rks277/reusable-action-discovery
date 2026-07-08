"""Box-side diagnostic (2026-07-08, RL Phase 1 pilot investigation): sample N_ROLLOUTS episodes at a
given temperature against a served Ollama checkpoint on ONE fixed stream, and print each rollout's
raw KEEP/PASS decision sequence + whether it hit `parse_decision`'s default-PASS fallback. Used to
diagnose the pilot v2 checkpoint's eager-mode collapse (6/8 identical `KKK` rollouts at temp=0.7, 0
parse failures) and to sanity-check temperature 1.0/1.2 as a fix for insufficient rollout diversity
before committing a full pilot run to it -- see docs/rl-finetuning-plan.md "Pilot v2, take 1".

Requires a checkpoint already served under MODEL_TAG via Ollama (merge_lora -> convert_hf_to_gguf ->
`ollama create`, same pipeline as `rl_urn_pilot.serve_checkpoint`).

  PYTHONPATH=. python scripts/box/diag_rl_pilot_rollouts.py [temperature]   # default 0.7
"""
import asyncio, sys
from dotenv import load_dotenv
load_dotenv()

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.stream_builder import StochasticStreamSpec, build_stochastic_stream
from scripts.creator.tool_disposition_benchmark.urn_session import UNIFORM, N, T, B, MAG, G, render_system, run_episode

MODEL_TAG = "qwen-rl-diag:latest"   # tag the checkpoint under diagnosis was `ollama create`d as
SEED = 9175
N_ROLLOUTS = 8
TEMPERATURE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7


async def main():
    print(f"=== temperature={TEMPERATURE} ===")
    slots, meta = build_stochastic_stream(StochasticStreamSpec(
        families=UNIFORM, n_hot=B, T=T, budget=B, guarantee_trap_early=G, magnitude=MAG, seed=SEED))
    system = render_system(T, B, N, False)
    client = RawChat()

    results = []
    for i in range(N_ROLLOUTS):
        row = await run_episode(client, MODEL_TAG, slots, T=T, B=B, system=system, temperature=TEMPERATURE)
        decisions = [t["decision"] for t in row["transcript"]]
        results.append(row)
        print(f"rollout {i}: kept={row['kept']} collected={row['collected']} unparsed={row['unparsed']} "
              f"decisions={''.join('K' if d=='KEEP' else 'p' for d in decisions)}")

    print("\n--- diversity check ---")
    kept_signatures = [tuple(sorted(r["kept"].items())) for r in results]
    print(f"distinct kept-signatures across {N_ROLLOUTS} rollouts: {len(set(kept_signatures))}")
    for sig in set(kept_signatures):
        print(f"  {sig}: {kept_signatures.count(sig)}/{N_ROLLOUTS}")
    total_unparsed = sum(r["unparsed"] for r in results)
    print(f"total unparsed (default-PASS fallback) turns across all rollouts: {total_unparsed}")

    print("\n--- sample raw transcript (rollout 0, first 4 turns) ---")
    for t in results[0]["transcript"][:4]:
        print(f"[{t['how']}] decision={t['decision']}  reply={t['reply']!r}")

asyncio.run(main())
