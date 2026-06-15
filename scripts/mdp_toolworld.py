"""Backward value iteration for the tool-world MDP over state (d, b, m).

State: d = doors still locked, b = budget (actions) left, m = machine built?
Objective: V(d,b,m) = max probability of opening all d remaining doors within b
actions. Solved by backward value iteration (LaValle, planning/node54): sweep b
from 0 upward (smaller-budget states first), so every transition lands on an
already-computed value.

Dynamics
--------
Machine built (m=1): deterministic exploit, 2 actions/door (operate machine on a
  door -> its key; use key on door -> opens). So V(d,b,1) = 1[b >= 2d].

No machine (m=0): V(d,b,0) = max(grind, build).
  grind (1 action, coupon collector): a uniformly random key opens a still-locked
    door w.p. d/D. grind = (d/D) V(d-1,b-1,0) + (1-d/D) V(d,b-1,0).
    (Clearing d doors this way takes ~ D*(H_D - H_{D-d}) actions in expectation.)
  build (commit, then exploit): build = P(K <= b - 2d), where K = G + R is the
    number of actions to build the machine:
      G = searches to collect all T byproduct types  (coupon collector over T),
      R = combine attempts to hit the recipe, ~ Uniform{1..C(T,2)} (one of the
          C(T,2) candidate pairs is correct).
    After K actions the machine exists with b-K budget left, then exploit needs
    2d, so success iff K <= b - 2d.

Assumptions (both make BUILD conservative, i.e. a lower bound on its value):
  - keys that drop while gathering parts are ignored (doors don't open during build);
  - all T types are gathered before any pair is tested.

Usage: PYTHONPATH=. python -m scripts.mdp_toolworld [--D 12 --T 3 --B 200 --plot]
"""

from __future__ import annotations

import argparse
from math import comb


def build_K_cdf(T: int, kmax: int) -> list[float]:
    """CDF of K = G + R up to kmax. G ~ time to collect all T coupons (uniform
    draws); R ~ Uniform{1..C(T,2)}. Returns cdfK with cdfK[x] = P(K <= x)."""
    M = comb(T, 2)                       # candidate recipe pairs (one correct)
    gmax = kmax                          # G's tail beyond kmax can't help P(K<=kmax)
    # coupon-collector CDF: P(all T seen within g draws)
    #   = sum_{k=0}^{T} (-1)^k C(T,k) ((T-k)/T)^g
    cdfG = []
    for g in range(gmax + 1):
        s = sum((-1) ** k * comb(T, k) * ((T - k) / T) ** g for k in range(T + 1))
        cdfG.append(min(1.0, max(0.0, s)))
    pmfG = [cdfG[g] - (cdfG[g - 1] if g else 0.0) for g in range(gmax + 1)]
    # K = G + R: convolve pmfG with the uniform recipe-search cost
    pmfK = [0.0] * (kmax + 1)
    for g, pg in enumerate(pmfG):
        if pg == 0.0:
            continue
        for r in range(1, M + 1):
            x = g + r
            if x <= kmax:
                pmfK[x] += pg / M
    cdfK, c = [0.0] * (kmax + 1), 0.0
    for x in range(kmax + 1):
        c += pmfK[x]
        cdfK[x] = c
    return cdfK


def solve(D: int = 12, T: int = 3, B: int = 200):
    """Backward value iteration. Returns dict of arrays indexed [d][b]:
    V (optimal, m=0), pol (optimal first action), grind_only, build_only."""
    cdfK = build_K_cdf(T, B + 1)

    def CDFK(x: int) -> float:                       # P(K <= x), clamped
        if x < 0:
            return 0.0
        return cdfK[x] if x < len(cdfK) else cdfK[-1]

    def build_val(d: int, b: int) -> float:          # commit-to-build value
        return CDFK(b - 2 * d)

    V = [[0.0] * (B + 1) for _ in range(D + 1)]       # optimal, no machine
    G = [[0.0] * (B + 1) for _ in range(D + 1)]       # grind-only baseline
    pol = [["-"] * (B + 1) for _ in range(D + 1)]
    for b in range(B + 1):
        V[0][b] = G[0][b] = 1.0
        pol[0][b] = "win"
    for b in range(B + 1):                            # backward: small budget first
        for d in range(1, D + 1):
            if b == 0:
                pol[d][b] = "fail"
                continue
            p = d / D
            grind = p * V[d - 1][b - 1] + (1 - p) * V[d][b - 1]
            build = build_val(d, b)
            if build > grind + 1e-12:
                V[d][b], pol[d][b] = build, "build"
            else:
                V[d][b], pol[d][b] = grind, "grind"
            if V[d][b] <= 1e-12:
                pol[d][b] = "fail"   # value 0: unwinnable, not a real "grind" choice
            G[d][b] = p * G[d - 1][b - 1] + (1 - p) * G[d][b - 1]   # grind-only
    build_only = [build_val(D, b) for b in range(B + 1)]
    return {"V": V, "pol": pol, "grind_only": G, "build_only": build_only,
            "D": D, "T": T, "B": B, "cdfK": cdfK}


def first_at_least(seq, thresh):
    return next((i for i, v in enumerate(seq) if v >= thresh), None)


def report(res: dict) -> None:
    D, T, B = res["D"], res["T"], res["B"]
    V, pol, Go, Bo = res["V"], res["pol"], res["grind_only"], res["build_only"]
    Vstart = [V[D][b] for b in range(B + 1)]
    # E[K] from the cdf
    cdfK = res["cdfK"]
    pmf = [cdfK[x] - (cdfK[x - 1] if x else 0.0) for x in range(len(cdfK))]
    EK = sum(x * p for x, p in enumerate(pmf))

    print(f"Tool-world MDP  D={D} doors, T={T} types  (C(T,2)={comb(T,2)} pairs, "
          f"E[K_build]={EK:.2f} actions, exploit=2D={2*D})\n")
    print("Start state (all D locked, no machine): success probability vs budget")
    print(f"{'b':>5} {'grind-only':>11} {'build-only':>11} {'OPTIMAL':>9} "
          f"{'1st move':>9}")
    print("-" * 50)
    shown = [b for b in (12, 20, 24, 30, 36, 40, 48, 60, 80, 120, 160)
             if b <= B] + [B]
    for b in sorted(set(shown)):
        print(f"{b:>5} {Go[D][b]:>11.3f} {Bo[b]:>11.3f} {Vstart[b]:>9.3f} "
              f"{pol[D][b]:>9}")

    print("\nThresholds on the start state (optimal play):")
    for thr in (0.5, 0.9, 0.99, 0.999):
        b = first_at_least(Vstart, thr)
        print(f"  P(success) >= {thr:<5} at budget b = {b}")
    bswitch = next((b for b in range(B + 1) if pol[D][b] == "build"), None)
    print(f"  start-state first move flips grind -> build at b = {bswitch}")
    bg = first_at_least([Go[D][b] for b in range(B + 1)], 0.99)
    bb = first_at_least(Bo, 0.99)
    print(f"  grind-only reaches 0.99 at b = {bg};  build-only at b = {bb}")

    # Policy structure: for each d, the budget band where build is the optimal move.
    print("\nOptimal first move by (d, b)  [where build beats grind, m=0]:")
    print(f"{'d':>3}  {'build-preferred budget band':>30}")
    for d in range(1, D + 1):
        bs = [b for b in range(B + 1) if pol[d][b] == "build"]
        band = f"{bs[0]}..{bs[-1]}" if bs else "(never)"
        print(f"{d:>3}  {band:>30}")


def make_plot(res: dict, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    D, T, B = res["D"], res["T"], res["B"]
    V, pol, Go, Bo = res["V"], res["pol"], res["grind_only"], res["build_only"]
    bmax = min(B, 100)
    bs = list(range(bmax + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.plot(bs, [Go[D][b] for b in bs], label="grind only", color="seagreen")
    ax1.plot(bs, [Bo[b] for b in bs], label="build only", color="indianred")
    ax1.plot(bs, [V[D][b] for b in bs], label="optimal (max)", color="navy", lw=2)
    ax1.axvline(2 * D, color="0.5", ls="--", lw=0.8, label=f"2D={2*D} (exploit floor)")
    ax1.set_xlabel("budget b (actions)"); ax1.set_ylabel("P(open all D doors)")
    ax1.set_title(f"Start state success vs budget (D={D}, T={T})")
    ax1.set_ylim(-0.03, 1.03); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # policy heatmap: 0=fail, 1=grind, 2=build
    code = {"fail": 0, "grind": 1, "build": 2, "win": 1, "-": 0}
    M = np.array([[code[pol[d][b]] for b in bs] for d in range(1, D + 1)])
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#dddddd", "seagreen", "indianred"])
    ax2.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=2,
               extent=[0, bmax, 1, D])
    ax2.set_xlabel("budget b (actions)"); ax2.set_ylabel("doors still locked d")
    ax2.set_title("Optimal first move (green=grind, red=build, grey=hopeless)")
    fig.tight_layout()
    fig.savefig(out, dpi=150); fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"\nwrote {out} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=12)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--plot", metavar="OUT.png", default=None)
    args = ap.parse_args()
    res = solve(args.D, args.T, args.B)
    report(res)
    if args.plot:
        make_plot(res, args.plot)


if __name__ == "__main__":
    main()
