"""Replay verification for tool-world episodes.

The world is deterministic given (relabel_seed, drop_seed, n, hint). This module
reconstructs a fresh State from a logged JSONL row's seeds and feeds the row's
recorded actions back through it, recomputing every observation. If the
recomputed observations match the recorded ones byte-for-byte, the world state
has reconstructed exactly -- which is what guarantees the four metrics can be
recomputed POST-HOC from the trace (see analyze_toolworld.py).

Usage:
  python -m scripts.replay_toolworld runs/toolworld_sweep_*/episodes.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from scripts.run_toolworld_llm import State as StateV1, make_world as make_world_v1
from scripts.toolworld_v2 import (State as StateV2, make_world as make_world_v2,
                                  world_from_labels)

_V2_LABEL_KEYS = ("door_base", "key_base", "machine", "types", "recipe")


def replay(row: dict) -> tuple[object, list[str]]:
    """Reconstruct the world from row['*_seed'] and replay row['actions'].

    Mirrors the runner's State construction exactly (same drop_rng seeding).
    Dispatches to the v2 (multi-type) world when the row carries 'n_types',
    else the v1 (single-byproduct) world. Returns the final State and the list
    of recomputed observations. Uses the action index as the turn stamp; the
    original trace is ordered, so relative ordering (t_star / build) is kept.
    """
    if "n_types" in row:
        labels = row.get("labels") or {}
        # Prefer the recorded labels so replay does not depend on the ambient
        # obfuscation scheme (make_world reads a mutable global); fall back to
        # re-deriving them for older rows that predate label recording.
        if all(k in labels for k in _V2_LABEL_KEYS):
            world = world_from_labels(labels, row["n"])
        else:
            world = make_world_v2(row["relabel_seed"], row["n"], n_types=row["n_types"])
        State = StateV2
    else:
        world = make_world_v1(row["relabel_seed"], row["n"])
        State = StateV1
    s = State(world=world,
              drop_rng=random.Random(row["drop_seed"] * 6151 + 1),
              hint=row.get("hint", True))
    recomputed = []
    for i, act in enumerate(row["actions"]):
        s.turn = i
        verb = act[0]
        if verb == "examine":
            obs = s.examine(act[1])
        elif verb == "combine":
            obs = s.combine(act[1], act[2])
        elif verb == "use":
            obs = s.use(act[1], act[2])
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
        return False, (f"solved mismatch: replay={s.solved()} "
                       f"recorded={row['solved']}")
    return True, f"ok ({len(recomputed)} actions, solved={s.solved()})"


def main():
    path = Path(sys.argv[1])
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n_ok = n_fail = 0
    for j, row in enumerate(rows):
        ok, msg = verify(row)
        tag = f"{row.get('model', '?')[:28]:28s} n={row.get('n')} " \
              f"seed=({row.get('relabel_seed')},{row.get('drop_seed')})"
        if ok:
            n_ok += 1
            print(f"  [OK]   {tag}  {msg}")
        else:
            n_fail += 1
            print(f"  [FAIL] {tag}\n{msg}")
    print(f"\nReplay verification: {n_ok} ok, {n_fail} failed "
          f"(of {len(rows)} rows).")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
