"""Probe the count-cap approximation: (Run 2) cross-check V(cap) vs the exact uncapped DP where exact
is feasible (T<=40), then (Run 1) tractability + convergence of V(cap) at the real size N=12,T=60.

  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.probe_cap
"""
from __future__ import annotations
import time
from scripts.creator.tool_disposition_benchmark.exact_dp import ExactDP

UH, UB, UR = -98.7, 49.2, 80.0
N, B, ALPHA = 12, 3, 1.0


def run2_crosscheck():
    print("=== Run 2: cap vs EXACT (uncapped) root value, where exact is feasible ===", flush=True)
    for T in (30, 35, 40):
        exact = ExactDP(UH, UB, UR, N, T, B, ALPHA, cap=None).root_value()
        print(f"  T={T}: exact(uncapped) = {exact:.3f}", flush=True)
        for K in (2, 3, 4, 5):
            t0 = time.time()
            vk = ExactDP(UH, UB, UR, N, T, B, ALPHA, cap=K).root_value()
            print(f"      cap K={K}: V={vk:9.3f}   gap-to-exact={vk - exact:+8.4f}   "
                  f"({time.time()-t0:.1f}s)", flush=True)
    print(flush=True)


def run1_scale():
    print("=== Run 1: tractability + convergence at N=12, T=60, B=3 ===", flush=True)
    prev = None
    for K in (2, 3, 4, 5, 6):
        dp = ExactDP(UH, UB, UR, N, 60, B, ALPHA, cap=K)
        t0 = time.time()
        vk = dp.root_value()
        dt = time.time() - t0
        inc = "" if prev is None else f"  increment={vk - prev:+.4f}"
        print(f"  cap K={K}: V={vk:9.3f}  states={len(dp._cache):>12,}  time={dt:7.1f}s{inc}",
              flush=True)
        prev = vk


if __name__ == "__main__":
    run2_crosscheck()
    run1_scale()
    print("probe done", flush=True)
