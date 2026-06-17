"""Replay verification for woodworld episodes (mirrors scripts/replay_toolworld.py).

The world is deterministic given (labels, gather_seed, n, hint): gather is the
ONLY stochastic action, and its RNG is seeded identically to the runner, so
replaying the recorded actions recomputes every observation byte-for-byte. This
is what guarantees the metrics can be recomputed post-hoc (see
analyze_woodworld.py), and it is the key correctness check for the stochastic
gather (the RNG must be consumed once per gather, in the recorded order).

Usage:
  python -m scripts.replay_woodworld runs/woodworld_sweep_*/episodes.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from scripts.woodworld import GATHER_PROB, State, make_world, world_from_labels


def replay(row: dict) -> tuple[State, list[str]]:
    """Reconstruct the world from recorded labels (preferred; scheme-independent)
    and replay row['actions'], recomputing every observation. Mirrors the runner's
    State construction exactly (same gather_rng seeding + recorded gather_prob, so
    the stochastic gather reproduces byte-exact)."""
    gp = row.get("gather_prob", GATHER_PROB)
    labels = row.get("labels")
    if labels:
        world = world_from_labels(labels, row["n"], gp)
    else:  # legacy fallback: re-derive (depends on ambient obfuscation scheme)
        world = make_world(row["relabel_seed"], row["n"], gp)
    s = State(world=world,
              gather_rng=random.Random(row["gather_seed"] * 6151 + 1),
              hint=row.get("hint", True))
    recomputed = []
    for i, act in enumerate(row["actions"]):
        s.turn = i
        verb = act[0]
        if verb == "gather":
            obs = s.gather()
        elif verb in ("craft", "combine"):
            obs = s.craft(act[1])
        elif verb == "use":
            obs = s.use(act[1])
        else:
            obs = f"<unknown verb {verb!r}>"
        recomputed.append(obs)
    return s, recomputed


def verify(row: dict) -> tuple[bool, str]:
    """Replay and check recomputed obs == recorded obs, and final solved match."""
    if row.get("error"):
        return True, "skipped (errored episode)"
    s, recomputed = replay(row)
    recorded = row["obs"]
    if len(recomputed) != len(recorded):
        return False, f"length mismatch: {len(recomputed)} vs {len(recorded)}"
    for i, (a, b) in enumerate(zip(recomputed, recorded)):
        if a != b:
            return False, (f"obs mismatch at action {i}:\n  recomputed: {a!r}\n"
                           f"  recorded:   {b!r}")
    if s.solved() != row["solved"]:
        return False, f"solved mismatch: replay={s.solved()} recorded={row['solved']}"
    return True, f"ok ({len(recomputed)} actions, solved={s.solved()})"


def main():
    path = Path(sys.argv[1])
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n_ok = n_fail = 0
    for row in rows:
        ok, msg = verify(row)
        tag = (f"{row.get('model', '?')[:28]:28s} N={row.get('n')} "
               f"seed=({row.get('relabel_seed')},{row.get('gather_seed')})")
        if ok:
            n_ok += 1
            print(f"  [OK]   {tag}  {msg}")
        else:
            n_fail += 1
            print(f"  [FAIL] {tag}\n{msg}")
    print(f"\nReplay verification: {n_ok} ok, {n_fail} failed (of {len(rows)} rows).")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
