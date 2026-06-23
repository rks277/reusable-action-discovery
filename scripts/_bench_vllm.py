"""Quick concurrency benchmark for a vLLM OpenAI-compatible server.

Aggregate throughput vs concurrent requests, realistic ~2k-token prompt, small
generation. Usage:
  VLLM_BASE_URL=http://127.0.0.1:8003/v1 python -m scripts._bench_vllm Qwen/Qwen2.5-72B-Instruct
"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL = sys.argv[1]
LEVELS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["8", "16", "32"])]
URL = os.environ["VLLM_BASE_URL"].rstrip("/") + "/chat/completions"

FILLER = ("You searched the vault. Out fell a small dull object and a faintly active "
          "component; they lie on the ground. You hold several items and have opened "
          "no doors yet. Recall prior turns and the budget remaining. ") * 60
PROMPT = FILLER + "\nGiven this state, output exactly one action like: examine <object>."


def one_call(_):
    body = json.dumps({"model": MODEL, "max_tokens": 32,
                       "messages": [{"role": "user", "content": PROMPT}]}).encode()
    t = time.time()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer vllm"})
    r = json.load(urllib.request.urlopen(req))
    u = r.get("usage", {})
    return time.time() - t, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def run_level(c):
    with ThreadPoolExecutor(max_workers=c) as ex:
        t = time.time()
        res = list(ex.map(one_call, range(c)))
        wall = time.time() - t
    ptok = sum(r[1] for r in res); gtok = sum(r[2] for r in res)
    lat = [r[0] for r in res]
    return c, wall, sum(lat) / len(lat), (ptok + gtok) / wall


one_call(0)  # warmup
print(f"{'C':>3} {'wall_s':>7} {'mean_lat':>9} {'tot_tps':>9}")
base = None
for c in LEVELS:
    c, wall, mlat, tps = run_level(c)
    base = base or tps
    print(f"{c:>3} {wall:>7.1f} {mlat:>9.1f} {tps:>9.0f}  ({tps/base:.1f}x)", flush=True)
