"""Run any sweep under a hard USD cost cap.

Launches the sweep as a child process group, polls its episodes.jsonl as rows
are written, estimates cumulative cost with analyze_beach_sweep.cost_of (the same
Anthropic list pricing + cache accounting the analyzer uses), and SIGTERMs the
whole group the instant cumulative cost crosses CAP. In-flight episodes already
dispatched when the cap trips may still bill a little -- that overshoot is
unavoidable (the API calls are already out), but no new episodes are launched.

Usage:
  python scripts/_beach_sweep_capped.py <cap_usd> <run_glob> <module> [args...]
e.g.
  python scripts/_beach_sweep_capped.py 15 'runs/beach_v2_sweep_*' \
      scripts.run_beach_v2_sweep --model claude-haiku-4-5-20251001
"""

import glob
import json
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_beach_sweep import cost_of

POLL = 2.0
CAP = float(sys.argv[1])
RUN_GLOB = sys.argv[2]
SWEEP_CMD = sys.argv[3:]  # module + its args, run via `python -u -m <module> ...`

before = set(glob.glob(RUN_GLOB))
proc = subprocess.Popen([sys.executable, "-u", "-m", *SWEEP_CMD],
                        start_new_session=True)


def newest_run_dir():
    new = set(glob.glob(RUN_GLOB)) - before
    return max(new, key=os.path.getmtime) if new else None


def cum_cost(path):
    total, n = 0.0, 0
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partially-written final line; counted next poll
                n += 1
                if r.get("error"):
                    continue
                c = cost_of(r.get("usage") or {}, r["model"])
                if c:
                    total += c
    except FileNotFoundError:
        pass
    return total, n


def kill():
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass


run_dir = None
for _ in range(60):                      # wait up to 30s for the run dir to appear
    if proc.poll() is not None:
        break
    run_dir = newest_run_dir()
    if run_dir:
        break
    time.sleep(0.5)

ep = os.path.join(run_dir, "episodes.jsonl") if run_dir else None
print(f"[watchdog] hard cap ${CAP:.0f}; monitoring {ep}", flush=True)

while True:
    rc = proc.poll()
    cost, n = cum_cost(ep) if ep else (0.0, 0)
    if cost > CAP:
        print(f"[watchdog] cumulative ${cost:.2f} > ${CAP:.0f} after {n} eps "
              f"-> KILLING sweep now", flush=True)
        kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            kill()
        final, fn = cum_cost(ep)
        print(f"[watchdog] killed. final logged cost ${final:.2f} over {fn} "
              f"episodes (a few in-flight episodes may have added to this).",
              flush=True)
        sys.exit(0)
    if rc is not None:
        print(f"[watchdog] sweep completed on its own. total cost ${cost:.2f} "
              f"over {n} episodes.", flush=True)
        sys.exit(0)
    print(f"[watchdog] {n} eps logged, ${cost:.2f} so far", flush=True)
    time.sleep(POLL)
