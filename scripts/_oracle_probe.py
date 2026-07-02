"""Quick 3x3 probe: N x difficulty, 1 rep, sequential, all 3 models."""
import asyncio, json, sys, time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_oracle_llm import run                    # noqa: E402
from scripts.oracle_sweep_config import MODELS, CONCURRENCY, budget_for  # noqa: E402

N_VALUES    = [4, 8, 12, 16, 20]   # linear step 4
DIFF_VALUES = [14, 16, 18, 20, 22] # linear step 2
REPS        = 1


async def main():
    load_dotenv()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs/oracle") / f"oracle_probe_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"; out_path.write_text("")
    cells = [(prov, model, n, d, rep)
             for prov, model in MODELS
             for n in N_VALUES for d in DIFF_VALUES for rep in range(REPS)]
    print(f"probe: {len(cells)} rollouts -> {out_path}", flush=True)

    sems = {p: asyncio.Semaphore(CONCURRENCY.get(p, 4)) for p, _ in MODELS}
    lock = asyncio.Lock()

    async def one(prov, model, n, d, rep):
        async with sems[prov]:
            t0 = time.time()
            try:
                result, _ = await run(model, n=n, difficulty=d, seed=rep,
                                      budget=budget_for(d, n=n), sequential=True)
                row = {**result, "elapsed_s": round(time.time() - t0, 2)}
            except Exception as e:
                row = {"model": model, "n": n, "difficulty": d, "seed": rep,
                       "error": f"{type(e).__name__}: {e}",
                       "elapsed_s": round(time.time() - t0, 2)}
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                tc = len(row.get("tool_calls") or [])
                sub = row.get("submit")
                m = [k for k in ("haiku", "sonnet", "opus") if k in model]
                m = m[0] if m else model[:6]
                print(f"  {m:6s} N={n:<2} d={d:<2} calls={tc:<2} "
                      f"sub={'OK' if sub and sub.get('correct') else ('X' if sub else '-')} "
                      f"reason={row.get('stopped_reason', 'err')} "
                      f"{'ERR' if row.get('error') else ''}", flush=True)

    await asyncio.gather(*(one(*c) for c in cells))

    # quick summary table
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    print(f"\n{'model':8s} {'N':>3s} {'d':>3s} {'calls':>5s} {'sub_ok':>6s} {'reason'}")
    for r in sorted(rows, key=lambda r: (r["model"], r["n"], r["difficulty"])):
        m = next((k for k in ("haiku", "sonnet", "opus") if k in r["model"]), r["model"][:6])
        tc = len(r.get("tool_calls") or [])
        sub = r.get("submit")
        print(f"{m:8s} {r['n']:>3d} {r['difficulty']:>3d} {tc:>5d} "
              f"{'YES' if sub and sub.get('correct') else 'NO':>6s} "
              f"{r.get('stopped_reason', '?')}")
    print(f"\nDone. {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
