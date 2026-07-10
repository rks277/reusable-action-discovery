"""External cost watchdog for OpenAI API sweeps (gpt-5 grid_v3, etc.).

Polls a running sweep's episodes.jsonl, sums the per-episode token `usage`, converts
to USD at editable per-1M prices, and KILLS the sweep process if the running total
crosses --cap. External by design: the sweep script has no kill switch; this process
is the sole enforcer, so a bug here can never silently disable the cap in the sweep.

Cost model (matches the usage schema written by lomekwi.raw_chat):
    usage = {input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
             reasoning_tokens, calls}
  - input_tokens is the TOTAL prompt (cached subset included), so uncached = input - cache_read
  - output_tokens already includes reasoning_tokens (don't double-count)
    cost = uncached*P_IN + cache_read*P_CACHED + output*P_OUT   (per 1M)

Only COMPLETED episodes carry usage, so the total lags in-flight episodes by up to
(concurrency) episodes -- a small, bounded overshoot past the cap. Set the cap with
that headroom in mind.

gpt-5 default prices are a BEST-RECOLLECTION (~Aug 2025 launch); VERIFY against the
current OpenAI pricing page and override with --p-in/--p-cached/--p-out if needed.

Usage:
  python scripts/cost_watchdog.py --dir runs/grid_v3/dense/grid_sweep_v3_gpt-5_<ts> \
      --cap 60 [--match "run_grid_sweep_v3 --model gpt-5"] [--interval 30] \
      [--p-in 1.25 --p-cached 0.125 --p-out 10.0]
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _one_file(ep_path: Path):
    """(input, output, cache, n) summed over a single episodes.jsonl."""
    inp = out = cache = n = 0
    if not ep_path.is_file():
        return 0, 0, 0, 0
    with ep_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                u = (json.loads(line).get("usage") or {})
            except json.JSONDecodeError:
                continue  # partial trailing line mid-write; skip this poll
            inp += u.get("input_tokens", 0) or 0
            out += u.get("output_tokens", 0) or 0
            cache += u.get("cache_read_tokens", 0) or 0
            n += 1
    return inp, out, cache, n


def read_cost(ep_paths, p_in: float, p_cached: float, p_out: float):
    """Return (usd, n_episodes, tok_summary) summed over one or more episodes.jsonl.

    Prices are applied uniformly; when the set spans cheaper models (e.g. gpt-5-nano)
    alongside gpt-5, using gpt-5 prices for all is a conservative OVER-estimate, which
    only makes the cap trip earlier -- the safe direction for a spend guard."""
    if isinstance(ep_paths, (str, Path)):
        ep_paths = [ep_paths]
    inp = out = cache = n = 0
    for p in ep_paths:
        i, o, c, k = _one_file(Path(p))
        inp += i; out += o; cache += c; n += k
    uncached = max(0, inp - cache)
    usd = uncached / 1e6 * p_in + cache / 1e6 * p_cached + out / 1e6 * p_out
    return usd, n, {"input": inp, "output": out, "cache_read": cache}


def resolve_paths(args) -> list[Path]:
    """episodes.jsonl paths from --dir (repeatable) and/or --glob, minus --exclude."""
    paths: list[Path] = []
    for d in (args.dir or []):
        paths.append(Path(d) / "episodes.jsonl")
    if args.glob:
        for d in sorted(_glob.glob(args.glob)):
            if args.exclude and args.exclude in d:
                continue
            paths.append(Path(d) / "episodes.jsonl")
    # de-dup preserving order
    seen, uniq = set(), []
    for p in paths:
        if str(p) not in seen:
            seen.add(str(p)); uniq.append(p)
    return uniq


def find_sweep_pids(match: str) -> list[int]:
    """PIDs whose command line contains `match`, excluding this watchdog itself."""
    try:
        out = subprocess.run(["pgrep", "-f", match], capture_output=True, text=True)
    except FileNotFoundError:
        return []
    pids = []
    for tok in out.stdout.split():
        try:
            pid = int(tok)
        except ValueError:
            continue
        if pid != os.getpid() and pid != os.getppid():
            pids.append(pid)
    return pids


def kill_sweep(match: str, log) -> None:
    pids = find_sweep_pids(match)
    if not pids:
        log(f"WARN: no process matched {match!r}; nothing to kill (already exited?)")
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            log(f"SIGTERM -> {pid}")
        except ProcessLookupError:
            pass
    time.sleep(5)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            log(f"SIGKILL -> {pid}")
        except ProcessLookupError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", help="run dir(s) with episodes.jsonl (repeatable)")
    ap.add_argument("--glob", help="glob of run dirs to sum, e.g. 'runs/.../grid_sweep_v3_gpt-5*'")
    ap.add_argument("--exclude", help="substring; drop matching dirs from --glob (e.g. 'gpt-5-mini')")
    ap.add_argument("--cap", type=float, required=True, help="USD spend cap (combined)")
    ap.add_argument("--match", default="run_grid_sweep_v3",
                    help="pgrep -f pattern identifying the sweep process(es) to kill")
    ap.add_argument("--interval", type=float, default=30.0, help="poll seconds")
    ap.add_argument("--p-in", type=float, default=1.25, help="$/1M uncached input")
    ap.add_argument("--p-cached", type=float, default=0.125, help="$/1M cached input")
    ap.add_argument("--p-out", type=float, default=10.0, help="$/1M output (reasoning incl)")
    ap.add_argument("--flag-dir", default=".", help="where to write COST_CAP_TRIPPED.flag")
    args = ap.parse_args()

    if not args.dir and not args.glob:
        ap.error("need at least one of --dir or --glob")
    flag = Path(args.flag_dir) / "COST_CAP_TRIPPED.flag"

    def log(msg):
        print(f"[watchdog {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"cap=${args.cap:.2f}  match={args.match!r}  "
        f"prices(in/cached/out)=${args.p_in}/${args.p_cached}/${args.p_out} per 1M "
        f"(uniform => conservative for cheaper models)")

    while True:
        eps = resolve_paths(args)
        usd, n, tok = read_cost(eps, args.p_in, args.p_cached, args.p_out)
        pct = usd / args.cap * 100 if args.cap else 0
        log(f"episodes={n}  spent=${usd:.2f} / ${args.cap:.2f} ({pct:.0f}%)  "
            f"in={tok.get('input',0):,} out={tok.get('output',0):,} cache={tok.get('cache_read',0):,}")
        if usd >= args.cap:
            log(f"CAP EXCEEDED (${usd:.2f} >= ${args.cap:.2f}) -- killing sweep")
            kill_sweep(args.match, log)
            flag.write_text(f"tripped at ${usd:.2f} after {n} episodes "
                            f"({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
            log(f"wrote {flag}; exiting")
            return
        # stop watching once the sweep process is gone (finished cleanly)
        if n > 0 and not find_sweep_pids(args.match):
            log(f"sweep process gone; final spend ${usd:.2f} over {n} episodes. exiting")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
