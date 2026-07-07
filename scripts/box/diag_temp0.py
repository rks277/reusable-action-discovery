"""One-off diagnostic (2026-07-07): does the early garbage-token derailment seen in the corpus-fix /
Adapter F tool evals (docs/qwen-finetune-transfer-plan.md "Transcript-level investigation") persist
under greedy decoding (temperature=0)? If it does, the checkpoint's learned distribution itself puts
a garbage token in first place at that context -- not sampling variance. If it goes away, sampling
variance at the default temp=0.7/top_p=0.8/top_k=20 is the proximate trigger.

Capped at ~15 problems (max_turns=30) per seed -- the question is answered as soon as we see whether
derailment occurs, no need to run a full 60-problem session.

  PYTHONPATH=. python -u -m scripts.box.diag_temp0 --model qwen-ft-pistar-corpusfix:latest --seeds 2000 2001
"""
from __future__ import annotations
import argparse, asyncio, json
from pathlib import Path
from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.driver import run_session
from scripts.creator.tool_disposition_benchmark.session_state import SessionState
from scripts.creator.tool_disposition_benchmark.stream_builder import StochasticStreamSpec, build_stochastic_stream
from scripts.creator.tool_disposition_benchmark.run_stream_session import slots_to_problems
from scripts.creator.tool_disposition_benchmark.family_kit import set_profile

UNIFORM = ["lcg", "modpow", "continued_frac", "crt_solve", "josephus", "quadratic_map_mod",
           "xorshift_steps", "matrix_power_mod"]
N, T, B, G, MAG = len(UNIFORM), 60, 3, 1.0, 100

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001])
ap.add_argument("--max-turns", type=int, default=30)
ap.add_argument("--out", default="runs/diag_temp0")
args = ap.parse_args()


async def run_one(client, seed):
    slots, meta = build_stochastic_stream(StochasticStreamSpec(
        families=UNIFORM, n_hot=B, T=T, budget=B, guarantee_trap_early=G, magnitude=MAG, seed=seed))
    state = SessionState(problems=slots_to_problems(slots), budget=B)
    state.announce_recurrence = True
    row = await run_session(client, args.model, state, token_cap=50_000, max_tokens=4096,
                            max_turns=args.max_turns, announce_cap=True,
                            stop_on_budget_exhausted=True, temperature=0.0)
    return row


async def main():
    load_dotenv(); set_profile(args.model)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    client = RawChat()
    for seed in args.seeds:
        print(f"=== seed {seed} (temperature=0, max_turns={args.max_turns}) ===", flush=True)
        row = await run_one(client, seed)
        (out / f"seed_{seed}.json").write_text(json.dumps(row, indent=2))
        t = row["transcript"]
        derailed_at = None
        for i, m in enumerate(t):
            if m.get("role") == "assistant" and (m.get("content") or "").strip() and not m.get("tool_calls"):
                derailed_at = i
                break
        print(f"  n_messages={len(t)}  n_tool_calls={row['n_tool_calls']}  "
              f"n_malformed={row['n_malformed_tool_calls']}  n_unknown={row['n_unknown_tool_calls']}")
        if derailed_at is not None:
            print(f"  DERAILED at msg idx {derailed_at}: {t[derailed_at]['content'][:80]!r}")
        else:
            print("  no non-tool-call assistant content found (clean run over this window)")
        for i, m in enumerate(t):
            role = m.get("role")
            c = (m.get("content") or "")[:70]
            tc = bool(m.get("tool_calls"))
            print(f"    {i:>3} {role:<9} {'TOOL' if tc else '    '} {c!r}")

if __name__ == "__main__":
    asyncio.run(main())
