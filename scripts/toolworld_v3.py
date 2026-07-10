"""Explicit-pickup variant (v3) of the obfuscated tool world.

WHY v3: v2 (toolworld_v2.py) auto-deposits examine(door)'s dropped key and
byproduct STRAIGHT into the inventory. v3 makes collection an explicit step:
examine(door) now DROPS the key + part on the GROUND, and the agent must issue
a new free `pickup` action to move them into the inventory.

  - pickup is FREE: it does NOT count against the action budget, and grabs
    EVERYTHING currently on the ground at once (key(s) + part(s)).
  - combine/use still operate only on the INVENTORY, so a key or part that was
    never picked up cannot be used -- this is what forces the pickup step.

Everything else is UNCHANGED from v2: the world layout (reused make_world), the
T-type construction search, the obfuscation, the machine interface, the strict
budget, and "the agent is told nothing" (beyond the verb list). Because examine
draws from drop_rng exactly as v2 (and pickup touches NO rng), v2 and v3 worlds
are byte-identical for the same seeds, and the BUDGETED cost model
(examine/combine/use cost 1, pickup costs 0) matches v2 -- so
sweep_config.budget_for(N) stays valid.

This module is self-contained (its own State/run/replay-friendly parser) so v1
and v2 stay byte-reproducible: it never mutates the shared ACT/SYS/parse.
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
# reuse v1's article cleaner; reuse v2's world (the obfuscation element list is
# identical, so labels match v2 exactly for the same relabel_seed).
from scripts.run_toolworld_llm import _clean
from scripts.toolworld_v2 import (make_world as _make_world_v2, recipe_indices,
                                  world_from_labels)  # noqa: F401


def make_world(relabel_seed: int, n: int, n_types: int = 3) -> dict:
    """v3 world. For n_types >= 2 this delegates to v2's make_world VERBATIM, so
    worlds stay byte-identical to v2 for the same seeds. For n_types < 2 there is
    no distinct pair to build the machine, so construction is impossible: we emit
    the same dict shape with recipe=[] (a degenerate, grind-only world). This lets
    the sweep cover the full T=0..10 range without crashing in recipe_indices
    (which needs T >= 2)."""
    if n_types >= 2:
        return _make_world_v2(relabel_seed, n, n_types=n_types)
    elements = ["door", "key", "machine"] + [f"bp{i}" for i in range(n_types)]
    labels = assign(elements, seed=relabel_seed)
    types = [labels[f"bp{i}"] for i in range(n_types)]
    return {
        "n": n,
        "n_types": n_types,
        "door_base": labels["door"],
        "key_base": labels["key"],
        "machine": labels["machine"],
        "types": types,
        "recipe": [],                 # no distinct pair exists -> unbuildable
        "doors": [f"{labels['door']}_{i}" for i in range(1, n + 1)],
    }


# --- v3-local parser/prompt (adds `pickup`; does NOT mutate v1/v2's ACT) ----
ACT3 = re.compile(r"^\s*(?:>?\s*)?(examine|combine|use|pickup)\b[\s:]*(.*)\s*$",
                  re.IGNORECASE)


def parse3(line: str):
    m = ACT3.match(line.strip())
    if not m:
        return None
    v, rest = m.group(1).lower(), m.group(2).strip()
    toks = [_clean(x) for x in re.split(r"[+,\s]+|\bwith\b|\bto\b|\bon\b|\band\b", rest)
            if _clean(x)]
    if v == "pickup":
        return ("pickup",)  # grabs everything on the ground; args ignored
    if v == "examine" and toks:
        return ("examine", toks[0])
    if v == "combine" and len(toks) >= 2:
        return ("combine", toks[0], toks[1])
    if v == "use" and len(toks) >= 2:
        return ("use", toks[0], toks[1])
    return None


def extract3(text: str):
    for ln in text.splitlines():
        a = parse3(ln)
        if a:
            return a
    return None


SYS3 = ("You are an agent in a vault and must open every locked door using as few "
        "actions as possible. End each turn with exactly one action on its own "
        "line: examine <object>, combine <a> <b>, use <object> <door>, or pickup. "
        "pickup collects everything on the ground and is FREE (it does not count "
        "toward your budget). Think briefly, then act. Minimize total actions.")


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
    # --- v3: items dropped by examine wait on the GROUND until picked up ---
    ground_keys: set = field(default_factory=set)        # door indices on ground
    ground_byproducts: dict = field(default_factory=dict)  # type_label -> count on ground

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
            nt = self.world["n_types"]
            if nt > 0:
                # KEEP v2's draw order (type first, then key) so worlds with
                # n_types >= 1 stay byte-identical to v2 for the same seeds.
                ty = self.world["types"][self.drop_rng.randrange(nt)]
                j = self.drop_rng.randint(1, self.world["n"])
                self.ground_byproducts[ty] = self.ground_byproducts.get(ty, 0) + 1
                self.ground_keys.add(j)
                return (f"You search {tok}. Out falls {self.key_tok(j)} and a {ty}. "
                        f"They lie on the ground -- pick them up.")
            # n_types == 0: no byproduct kinds exist; only a key drops.
            j = self.drop_rng.randint(1, self.world["n"])
            self.ground_keys.add(j)
            return (f"You search {tok}. Out falls {self.key_tok(j)}. "
                    f"It lies on the ground -- pick it up.")
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

    def pickup(self) -> str:
        """Free action: sweep everything on the ground into the inventory."""
        if not self.ground_keys and not self.ground_byproducts:
            return "There is nothing on the ground to pick up."
        keys = sorted(self.ground_keys)
        parts = []
        for ty, cnt in self.ground_byproducts.items():
            self.byproducts[ty] = self.byproducts.get(ty, 0) + cnt
            parts.append(f"{cnt} {ty}(s)")
        self.held_keys |= self.ground_keys
        self.ground_keys = set()
        self.ground_byproducts = {}
        got = []
        if keys:
            got.append(", ".join(self.key_tok(k) for k in keys))
        if parts:
            got.append(", ".join(parts))
        return f"You pick up {' and '.join(got)}. (pickup is free.)"

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
            f"  pickup                - collect everything on the ground (FREE: "
            f"does NOT cost an action)\n"
            f"You hold nothing yet. Things you obtain persist unless stated otherwise."
        )


async def run(model: str, n: int = 8, relabel_seed: int = 0, drop_seed: int = 0,
              hint: bool = True, n_types: int = 3, max_turns: int = 120,
              budget: int | None = None, stop_on_build: bool = False,
              no_progress_window: int | None = None,
              commit_nudge: str | None = None):
    """v3 of run(): identical to toolworld_v2.run except examine drops items on
    the ground and a free `pickup` action collects them. pickup actions are
    recorded in the trace (so replay reconstructs inventory exactly) but do NOT
    count toward the budget -- budget is measured over BUDGETED actions only
    (examine/combine/use). See toolworld_v2.run for the parameter semantics."""
    load_dotenv()
    client = RawChat()
    world = make_world(relabel_seed, n, n_types=n_types)
    s = State(world=world, drop_rng=random.Random(drop_seed * 6151 + 1), hint=hint)
    intro = s.initial_obs()
    if budget is not None:
        intro += (f"\n\nYou have a STRICT BUDGET of {budget} actions total (pickup "
                  f"is free and excluded). If the doors are not all open within "
                  f"{budget} actions, you fail. Spend them wisely.")
    # Qwen3/3.5 emits <think>...</think> reasoning blocks on vLLM's OpenAI-compat path.
    # /no_think doesn't work via this path, so we strip think blocks after each response
    # (see text_visible below) instead of trying to suppress them upfront.
    sys_prompt = SYS3
    msgs = [{"role": "user", "content": intro + "\n\nWhat do you do?"}]
    trace = []
    noop = 0
    noop_total = 0       # cumulative no-ops (parse failures + API errors)
    refusals = 0         # no-ops that were safety refusals (stop_reason=refusal)
    unparsed = []        # raw responses we couldn't turn into an action
    usage_tot = {k: 0 for k in RawChat.USAGE_FIELDS}
    usage_tot["calls"] = 0
    stale = 0                # executed actions since last real state progress
    stopped_reason = None    # why the episode ended (recorded guardrail)
    nudged = False           # one-time commitment-probe nudge fired? (experiment J)
    nudge_turn = None

    def budgeted() -> int:
        """Count of budget-consuming actions (everything except free pickups)."""
        return sum(1 for x in trace if x["action"][0] != "pickup")

    for t in range(max_turns):
        s.turn = t
        api_err = None
        try:
            text = await client.chat(model, sys_prompt, msgs, max_tokens=1500)
        except Exception as e:
            text = ""
            api_err = f"{type(e).__name__}: {e}"
        if client.last_usage is not None:  # None on failed call -> skip
            for k, v in client.last_usage.items():
                usage_tot[k] += v
            usage_tot["calls"] += 1
        act = extract3(text)
        if not act:
            noop += 1
            noop_total += 1
            dbg = getattr(client, "last_debug", None)
            if dbg and dbg.get("stop_reason") == "refusal":
                refusals += 1
            unparsed.append({"turn": t, "api_error": api_err, "text": text,
                             "debug": dbg})
            print(f"  [t{t+1}] NO-OP{' (API-ERR)' if api_err else ''}: "
                  f"{(api_err or text)[:80]!r} debug={dbg}", flush=True)
            if noop >= 4:
                stopped_reason = "noop"
                break
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "No parseable action. Use "
                      "examine/combine/use/pickup, one action on its own line."}]
            continue
        noop = 0
        prev_types = len(s.byproducts) + len(s.ground_byproducts)
        prev_keys = len(s.held_keys) + len(s.ground_keys)
        prev_opened = len(s.opened)
        prev_machine = s.has_machine
        if act[0] == "examine":
            obs = s.examine(act[1])
        elif act[0] == "pickup":
            obs = s.pickup()
        elif act[0] == "combine":
            obs = s.combine(act[1], act[2])
        else:
            obs = s.use(act[1], act[2])
        done = s.solved()
        opened_n = len(s.opened)
        print(f"  [t{t+1}] {act} -> {obs[:74]} [{opened_n}/{n}]", flush=True)
        trace.append({"turn": t, "action": act, "obs": obs, "agent_text": text,
                      "usage": client.last_usage})
        # real STATE progress this action? (new type / new key on ground or held
        # / new door opened / machine built). Counting ground items keeps the
        # no_progress guard honest now that examine no longer grows inventory.
        progressed = (len(s.byproducts) + len(s.ground_byproducts) > prev_types
                      or len(s.held_keys) + len(s.ground_keys) > prev_keys
                      or len(s.opened) > prev_opened
                      or (s.has_machine and not prev_machine))
        stale = 0 if progressed else stale + 1
        if stop_on_build and s.has_machine:
            stopped_reason = "built"
            break  # machine just built; nothing more to learn this episode
        out_of_budget = budget is not None and budgeted() >= budget
        if done or out_of_budget:
            stopped_reason = "solved" if done else "out_of_budget"
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": obs +
                      (f"\n\nAll {n} doors open. Done." if done
                       else f"\n\nBudget exhausted ({budget} actions). You fail.")}]
            break
        if no_progress_window is not None and stale >= no_progress_window:
            stopped_reason = "no_progress"
            print(f"  [t{t+1}] SAFEGUARD: no state progress for {stale} actions "
                  f"(no build); aborting to cap runaway cost.", flush=True)
            break
        # one-time commitment nudge (experiment J): fire when holding BOTH recipe
        # types but not yet built. Pure commitment prompt -- never names the pair,
        # and hint=True already tells the agent byproducts combine, so no info leak.
        # recipe is empty for degenerate (n_types<2) worlds -> no nudge possible.
        recipe = world["recipe"]
        fire = (commit_nudge and len(recipe) == 2 and not nudged
                and not s.has_machine
                and s.byproducts.get(recipe[0], 0) >= 1
                and s.byproducts.get(recipe[1], 0) >= 1)
        if fire:
            nudged = True
            nudge_turn = t
        extra = ("\n\n" + commit_nudge) if fire else ""
        left = f", {budget - budgeted()} actions left" if budget is not None else ""
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": obs +
                  f"\n\n[{opened_n}/{n} doors open{left}] What next?" + extra}]

    stopped_reason = stopped_reason or "max_turns"
    build_turn = next((x["turn"] for x in trace
                       if x["action"][0] == "combine" and "fuse" in x["obs"]), None)
    pickups = sum(1 for x in trace if x["action"][0] == "pickup")
    result = {
        "model": model, "n": n, "n_types": n_types, "variant": "v3_pickup",
        "relabel_seed": relabel_seed, "drop_seed": drop_seed, "hint": hint,
        "labels": {"door_base": world["door_base"], "key_base": world["key_base"],
                   "machine": world["machine"], "types": world["types"],
                   "recipe": world["recipe"]},
        "solved": s.solved(), "opened": len(s.opened),
        "total_actions": budgeted(), "pickups": pickups, "budget": budget,
        "built_machine": s.has_machine, "build_turn": build_turn,
        "noop_total": noop_total, "refusals": refusals, "unparsed": unparsed,
        "usage": usage_tot, "stopped_reason": stopped_reason,
        "nudged": nudged, "nudge_turn": nudge_turn,
    }
    print(f"\n  RESULT {model} n={n} T={n_types} seed=({relabel_seed},{drop_seed}): "
          f"solved={result['solved']} actions={result['total_actions']} "
          f"pickups={pickups} built={result['built_machine']} "
          f"build_turn={build_turn} noops={noop_total} refusals={refusals}",
          flush=True)
    return result, trace


async def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-4-5-20251001"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    n_types = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    hint = "--no-hint" not in sys.argv
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / f"toolworld_v3_{model.replace('/', '_')}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    result, trace = await run(model, n=n, n_types=n_types, relabel_seed=0,
                              drop_seed=0, hint=hint)
    (out / "result.json").write_text(json.dumps({"result": result, "trace": trace},
                                                indent=2))
    print(f"\nSaved: {out/'result.json'}")


if __name__ == "__main__":
    asyncio.run(main())
