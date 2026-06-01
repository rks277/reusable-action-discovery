"""Smoke one budgeted episode each for gpt-5 and gemini-2.5-pro + replay verify."""
import asyncio, json
from dotenv import load_dotenv
from scripts.toolworld_v2 import run
from scripts.replay_toolworld import verify


async def main():
    load_dotenv()
    for model in ["gpt-5", "gemini-2.5-pro"]:
        result, trace = await run(model, n=8, n_types=3, relabel_seed=0,
                                  drop_seed=0, hint=True, max_turns=80, budget=36)
        row = {"model": model, "n": 8, "n_types": 3, "hint": True, "budget": 36,
               "relabel_seed": 0, "drop_seed": 0, "labels": result["labels"],
               "actions": [x["action"] for x in trace],
               "agent_texts": [x["agent_text"] for x in trace],
               "obs": [x["obs"] for x in trace],
               "solved": result["solved"], "total_actions": result["total_actions"]}
        ok, msg = verify(json.loads(json.dumps(row)))
        built = any("fuse into" in o for o in row["obs"])
        n_noop = sum(1 for t in trace if not t["agent_text"].strip())
        print(f"\n=== {model}: actions={result['total_actions']} solved={result['solved']} "
              f"built={built} empty_texts={n_noop} replay_ok={ok}: {msg}")
        assert ok


asyncio.run(main())
