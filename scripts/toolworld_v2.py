"""Multi-type construction variant (v2) of the obfuscated tool world.

WHY v2: in v1 (run_toolworld_llm.py) there is ONE generic byproduct and
combine(bp,bp) -> machine. Construction was trivially discoverable ("I hold
several identical junk items; combine is a verb; try it"), so models built the
tool ~always, hint-independently, and built_tool carried no signal (flat
calibration curve). See validate_toolworld_v2.py for the cost re-derivation.

v2 hardens CONSTRUCTION (01_world_mechanics.md "construction-hardness knob"):
  - T distinct byproduct TYPES, each its own nonsense label.
  - examine(door) drops ONE byproduct of a UNIFORMLY RANDOM type + the usual
    random door-key (keys are coupon-collector exactly as v1 -> BRUTE UNCHANGED).
  - the machine needs a SPECIFIC unordered pair of DISTINCT types {A,B}
    (randomized per episode). combine(x,y) builds iff {type(x),type(y)}=={A,B};
    every other combine -> "nothing happens". Now "which two components
    combine?" is a genuine search (C(T,2)+T candidates, one correct).

Brute baseline, the machine interface (use(machine,door)->key; use(key,door)
->open), the index correspondence, and "the agent is told nothing" are all
UNCHANGED from v1. Only invariant #3 (generic byproduct) is amended, with the
cost model re-derived in validate_toolworld_v2.py.

This module is self-contained (its own make_world/State/run/replay) so v1 stays
byte-reproducible: assign()'s labels depend on the element list, which differs.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from lomekwi.obfuscation import assign
from lomekwi.raw_chat import RawChat
# reuse v1's parser + system prompt verbatim (verbs/format are identical)
from scripts.run_toolworld_llm import ACT, SYS, extract, parse  # noqa: F401


def recipe_indices(relabel_seed: int, n_types: int) -> tuple[int, int]:
    """The two DISTINCT byproduct-type indices that build the machine this
    episode. Deterministic in relabel_seed so the world replays exactly."""
    rng = random.Random(relabel_seed * 2654435761 + 7)
    a, b = rng.sample(range(n_types), 2)
    return tuple(sorted((a, b)))


def make_world(relabel_seed: int, n: int, n_types: int = 3) -> dict:
    elements = ["door", "key", "machine"] + [f"bp{i}" for i in range(n_types)]
    labels = assign(elements, seed=relabel_seed)
    types = [labels[f"bp{i}"] for i in range(n_types)]
    ri, rj = recipe_indices(relabel_seed, n_types)
    return {
        "n": n,
        "n_types": n_types,
        "door_base": labels["door"],
        "key_base": labels["key"],
        "machine": labels["machine"],
        "types": types,                      # all byproduct-type labels
        "recipe": sorted([types[ri], types[rj]]),  # the two that combine
        "doors": [f"{labels['door']}_{i}" for i in range(1, n + 1)],
    }


@dataclass
class State:
    world: dict
    drop_rng: random.Random
    hint: bool = True
    byproducts: dict = field(default_factory=dict)  # type_label -> count held
    held_keys: set = field(default_factory=set)
    has_machine: bool = False
    opened: dict = field(default_factory=dict)       # door_idx -> (method, turn)
    machine_used_on: set = field(default_factory=set)
    turn: int = 0

    # --- helpers ---
    def door_idx(self, tok: str):
        b = self.world["door_base"]
        m = re.fullmatch(rf"{re.escape(b)}_(\d+)", tok)
        return int(m.group(1)) if m else None

    def key_idx(self, tok: str):
        b = self.world["key_base"]
        m = re.fullmatch(rf"{re.escape(b)}_(\d+)", tok)
        return int(m.group(1)) if m else None

    def key_tok(self, i: int) -> str:
        return f"{self.world['key_base']}_{i}"

    # --- actions ---
    def examine(self, tok: str) -> str:
        di = self.door_idx(tok)
        if di is not None and 1 <= di <= self.world["n"]:
            ty = self.world["types"][self.drop_rng.randrange(self.world["n_types"])]
            j = self.drop_rng.randint(1, self.world["n"])
            self.byproducts[ty] = self.byproducts.get(ty, 0) + 1
            self.held_keys.add(j)
            return (f"You search {tok}. Out falls {self.key_tok(j)} and a {ty}. "
                    f"You now hold {self.byproducts[ty]} {ty}(s).")
        if tok in self.world["types"]:
            if self.hint:
                return (f"A small dull {tok}. It feels faintly active, and the "
                        f"sensation shifts when held alongside a different kind.")
            return f"A small dull {tok}. Nothing remarkable."
        if tok == self.world["machine"] and self.has_machine:
            return (f"A {self.world['machine']}. It has a recess; you are unsure "
                    f"what it does.")
        if self.key_idx(tok) is not None:
            return f"{tok}: a small fitted object."
        return f"You examine {tok}: nothing of note."

    def combine(self, a: str, b: str) -> str:
        types = self.world["types"]
        if a in types and b in types:
            # need to actually hold the components
            have_a = self.byproducts.get(a, 0)
            have_b = self.byproducts.get(b, 0)
            enough = (have_a >= 1 and have_b >= 1) if a != b else have_a >= 2
            if not enough:
                return f"You combine {a} and {b}: nothing happens."
            if a != b and sorted([a, b]) == self.world["recipe"]:
                if not self.has_machine:
                    self.has_machine = True
                    return (f"You combine a {a} and a {b}. They fuse into a "
                            f"{self.world['machine']} (new). It persists.")
                return f"You already hold a {self.world['machine']}."
            return f"You combine {a} and {b}: nothing happens."
        return f"You combine {a} and {b}: nothing happens."

    def use(self, item: str, target: str) -> str:
        di = self.door_idx(target)
        if di is None or not (1 <= di <= self.world["n"]):
            return f"There is no {target}."
        if di in self.opened:
            return f"{target} is already open."
        if item == self.world["machine"]:
            if not self.has_machine:
                return f"You don't have a {self.world['machine']}."
            self.held_keys.add(di)
            self.machine_used_on.add(di)
            return (f"You operate the {self.world['machine']} on {target}. It "
                    f"yields {self.key_tok(di)}. (the {self.world['machine']} "
                    f"remains with you.)")
        ki = self.key_idx(item)
        if ki is not None:
            if ki not in self.held_keys:
                return f"You don't have {item}."
            if ki == di:
                method = "machine" if di in self.machine_used_on else "drop"
                self.opened[di] = (method, self.turn)
                return f"You use {item} on {target}: it fits. {target} opens."
            return f"You use {item} on {target}: it does not fit."
        if item in self.world["types"]:
            return f"You press the {item} to {target}: nothing."
        return f"You use {item} on {target}: nothing happens."

    def solved(self) -> bool:
        return len(self.opened) == self.world["n"]

    def initial_obs(self) -> str:
        return (
            f"You are in a sealed vault. There are {self.world['n']} locked doors: "
            f"{', '.join(self.world['doors'])}.\n\n"
            f"GOAL: open ALL the doors, using as few actions as possible.\n\n"
            f"You may act on objects with these verbs (one action per turn, on its "
            f"own line):\n"
            f"  examine <object>      - inspect or search an object\n"
            f"  combine <a> <b>       - combine two objects you hold\n"
            f"  use <object> <door>   - use an object on a door\n"
            f"You hold nothing yet. Things you obtain persist unless stated otherwise."
        )


async def run(model: str, n: int = 8, relabel_seed: int = 0, drop_seed: int = 0,
              hint: bool = True, n_types: int = 3, max_turns: int = 120,
              budget: int | None = None):
    """budget: if set, a strict cap on the number of actions. The agent is told
    the cap and its remaining count each turn, so building the tool becomes an
    economic CHOICE (spend scarce actions hunting the recipe vs. brute-force).
    Does not touch world mechanics, so replay stays exact."""
    load_dotenv()
    client = RawChat()
    world = make_world(relabel_seed, n, n_types=n_types)
    s = State(world=world, drop_rng=random.Random(drop_seed * 6151 + 1), hint=hint)
    intro = s.initial_obs()
    if budget is not None:
        intro += (f"\n\nYou have a STRICT BUDGET of {budget} actions total. If the "
                  f"doors are not all open within {budget} actions, you fail. "
                  f"Spend them wisely.")
    msgs = [{"role": "user", "content": intro + "\n\nWhat do you do?"}]
    trace = []
    noop = 0
    usage_tot = {k: 0 for k in RawChat.USAGE_FIELDS}
    usage_tot["calls"] = 0
    for t in range(max_turns):
        s.turn = t
        try:
            text = await client.chat(model, SYS, msgs, max_tokens=1500)
        except Exception:
            text = ""
        if client.last_usage is not None:  # None on failed call -> skip
            for k, v in client.last_usage.items():
                usage_tot[k] += v
            usage_tot["calls"] += 1
        act = extract(text)
        if not act:
            noop += 1
            if noop >= 4:
                break
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "No parseable action. Use "
                      "examine/combine/use, one action on its own line."}]
            continue
        noop = 0
        if act[0] == "examine":
            obs = s.examine(act[1])
        elif act[0] == "combine":
            obs = s.combine(act[1], act[2])
        else:
            obs = s.use(act[1], act[2])
        done = s.solved()
        opened_n = len(s.opened)
        print(f"  [t{t+1}] {act} -> {obs[:74]} [{opened_n}/{n}]", flush=True)
        trace.append({"turn": t, "action": act, "obs": obs, "agent_text": text,
                      "usage": client.last_usage})
        out_of_budget = budget is not None and len(trace) >= budget
        if done or out_of_budget:
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": obs +
                      (f"\n\nAll {n} doors open. Done." if done
                       else f"\n\nBudget exhausted ({budget} actions). You fail.")}]
            break
        left = f", {budget - len(trace)} actions left" if budget is not None else ""
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": obs +
                  f"\n\n[{opened_n}/{n} doors open{left}] What next?"}]

    build_turn = next((x["turn"] for x in trace
                       if x["action"][0] == "combine" and "fuse" in x["obs"]), None)
    result = {
        "model": model, "n": n, "n_types": n_types,
        "relabel_seed": relabel_seed, "drop_seed": drop_seed, "hint": hint,
        "labels": {"door_base": world["door_base"], "key_base": world["key_base"],
                   "machine": world["machine"], "types": world["types"],
                   "recipe": world["recipe"]},
        "solved": s.solved(), "opened": len(s.opened),
        "total_actions": len(trace), "budget": budget,
        "built_machine": s.has_machine, "build_turn": build_turn,
        "usage": usage_tot,
    }
    print(f"\n  RESULT {model} n={n} T={n_types} seed=({relabel_seed},{drop_seed}): "
          f"solved={result['solved']} actions={result['total_actions']} "
          f"built={result['built_machine']} build_turn={build_turn}", flush=True)
    return result, trace


async def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-4-5-20251001"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    n_types = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    hint = "--no-hint" not in sys.argv
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / f"toolworld_v2_{model.replace('/', '_')}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    result, trace = await run(model, n=n, n_types=n_types, relabel_seed=0,
                              drop_seed=0, hint=hint)
    (out / "result.json").write_text(json.dumps({"result": result, "trace": trace},
                                                indent=2))
    print(f"\nSaved: {out/'result.json'}")


if __name__ == "__main__":
    asyncio.run(main())
