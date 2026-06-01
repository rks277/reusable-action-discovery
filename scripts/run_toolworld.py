"""Option-1 tool world: a constructed, persistent, REUSABLE tool vs an
always-available expensive brute-force route. Voyager's skill-library paradigm,
de-confounded (obfuscated, optional, multi-scale).

World:
  - Obfuscated primitives (ample, persistent, not consumed by combining).
  - A hidden combine chain builds a MASTER KEY (a persistent tool):
        combine A + B -> mid ;  combine mid + C -> masterkey
  - N locked targets ("wards").
  - Two ways to open each ward:
      TOOL ROUTE:  apply masterkey -> ward   (1 action; the key PERSISTS, so
                   once built it opens every ward at 1 action each)
      BRUTE ROUTE: each ward has its own hidden 2-primitive combo that opens it
                   (different per ward); discover by trial (expensive), per ward.
  - Goal: open all N wards in as few actions as possible.

Cost structure (N wards):
  TOOL : discover key-chain (~once) + 2 builds + N applies      -> cheap, scales well
  BRUTE: discover a 2-combo for EACH ward (~per ward)           -> expensive, N times

So building the key is the rational, calibrated choice when N is large; brute-
force is fine when N is tiny. Tool use is OPTIONAL (brute-force always works),
which is the point. We measure whether the agent builds & reuses the key.

Informative feedback => discovery is reliable (no hallucination).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.run_grammar import extract  # reuse the combine/define/invoke parser? we use our own


PRIMS = ["velax", "silex", "brann", "morwen", "dros", "kel", "pell", "garn"]


def make_world(seed: int, n_wards: int = 6):
    rng = random.Random(seed * 2654435761 % (2**32))
    p = PRIMS[:]
    rng.shuffle(p)
    # key chain: p0+p1 -> mid ; mid+p2 -> masterkey
    key_chain = {
        ("combine", frozenset({p[0], p[1]})): "keystone",
        ("combine", frozenset({"keystone", p[2]})): "masterkey",
    }
    # per-ward brute-force combos: each ward opened by combining two distinct prims
    ward_combos = {}
    pairs = []
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            pairs.append(frozenset({p[i], p[j]}))
    rng.shuffle(pairs)
    used = {frozenset({p[0], p[1]})}  # avoid colliding with key chain step 1
    wards = []
    for w in range(1, n_wards + 1):
        name = f"ward_{w}"
        wards.append(name)
        # pick a pair not equal to the key step-1 pair
        combo = next(c for c in pairs if c not in used)
        used.add(combo)
        ward_combos[name] = combo
    return {
        "seed": seed, "primitives": p, "key_chain": key_chain,
        "ward_combos": ward_combos, "wards": wards, "n_wards": n_wards,
        "key_chain_steps": [(p[0], p[1], "keystone"), ("keystone", p[2], "masterkey")],
    }


@dataclass
class ToolSession:
    world: dict
    have: set = field(default_factory=set)
    opened: set = field(default_factory=set)
    actions: int = 0
    built_key: bool = False
    key_uses: int = 0
    brute_opens: int = 0

    def __post_init__(self):
        if not self.have:
            self.have = set(self.world["primitives"])

    def intro(self):
        L = ["You are in a vault with several obfuscated items (ample supply; combining does not use them up):",
             "  " + ", ".join(sorted(self.world["primitives"])),
             "",
             f"There are {self.world['n_wards']} locked wards: {', '.join(self.world['wards'])}.",
             "GOAL: open ALL wards, using as few actions as possible.",
             "",
             "You can combine items (some combinations yield new items or open things; most do nothing).",
             "After each action you are told what happened. Items you create persist and can be reused.",
             "",
             "Actions (one per turn; combine and apply each cost 1 action):",
             "  combine <a> <b>        - combine two items",
             "  apply <item> <ward>    - apply an item to a ward",
             "",
             "Be efficient: minimize total actions across all wards."]
        return "\n".join(L)

    def combine(self, a, b):
        self.actions += 1
        if a not in self.have or b not in self.have:
            miss = [x for x in (a, b) if x not in self.have]
            return f"You lack: {', '.join(miss)}."
        res = self.world["key_chain"].get(("combine", frozenset({a, b})))
        if res:
            new = res not in self.have
            self.have.add(res)
            if res == "masterkey":
                self.built_key = True
            return f"You combine {a} + {b}. Yields {res} {'(NEW)' if new else '(already had)'}."
        return f"You combine {a} + {b}. Nothing happens."

    def apply(self, item, ward):
        self.actions += 1
        if ward not in self.world["wards"]:
            return f"There is no {ward}."
        if ward in self.opened:
            return f"{ward} is already open."
        if item not in self.have:
            return f"You don't have {item}."
        # TOOL route: masterkey opens any ward, persists
        if item == "masterkey":
            self.opened.add(ward); self.key_uses += 1
            done = len(self.opened) == self.world["n_wards"]
            return (f"You apply the masterkey to {ward}. It opens. (The masterkey remains in your "
                    f"inventory.) [{len(self.opened)}/{self.world['n_wards']} wards open]"
                    + (" **ALL WARDS OPEN.**" if done else ""))
        return f"You apply {item} to {ward}. Nothing happens."

    def brute_combine_on_ward(self, a, b, ward):
        """Brute route is expressed as: combining the ward's specific pair while
        'targeting' it. We fold this into combine when one arg is a ward."""
        self.actions += 1
        if ward in self.opened:
            return f"{ward} is already open."
        combo = self.world["ward_combos"].get(ward)
        if combo == frozenset({a, b}) and a in self.have and b in self.have:
            self.opened.add(ward); self.brute_opens += 1
            done = len(self.opened) == self.world["n_wards"]
            return (f"You force {ward} with {a}+{b}. It clicks open. "
                    f"[{len(self.opened)}/{self.world['n_wards']}]"
                    + (" **ALL WARDS OPEN.**" if done else ""))
        return f"You try {a}+{b} on {ward}. Nothing happens."


ACT = re.compile(r"^\s*(?:>?\s*)?(combine|apply|force)\b[\s:]*(.*)\s*$", re.IGNORECASE)


def _clean(t):
    return re.sub(r"^(a|an|the)\s+", "", t.strip().lower())


def parse(line):
    m = ACT.match(line.strip())
    if not m:
        return None
    v, rest = m.group(1).lower(), m.group(2).strip()
    toks = [_clean(x) for x in re.split(r"[+,\s]+|\bwith\b|\bon\b|\bto\b|\band\b", rest) if _clean(x)]
    if v == "apply" and len(toks) >= 2:
        return ("apply", toks[0], toks[1])
    if v in ("combine", "force"):
        # force <ward> with <a> <b>  OR combine <a> <b> [on <ward>]
        wards = [t for t in toks if t.startswith("ward")]
        items = [t for t in toks if not t.startswith("ward")]
        if wards and len(items) >= 2:
            return ("brute", items[0], items[1], wards[0])
        if len(items) >= 2:
            return ("combine", items[0], items[1])
    return None


def extract_action(text):
    for ln in text.splitlines():
        a = parse(ln.strip())
        if a:
            return a
    return None


SYS = ("You must open all locked wards in a vault as efficiently as possible. End each turn with "
       "exactly one action on its own line. Actions: combine <a> <b>; apply <item> <ward>; "
       "force <ward> with <a> <b>. Minimize total actions.")


async def run(model, n_wards=6, seed=1, max_turns=80):
    load_dotenv()
    client = RawChat()
    world = make_world(seed, n_wards)
    s = ToolSession(world=world)
    msgs = [{"role": "user", "content": s.intro() + "\n\nWhat do you do?"}]
    noop = 0
    for t in range(max_turns):
        try:
            text = await client.chat(model, SYS, msgs, max_tokens=1500)
        except Exception:
            text = ""
        act = extract_action(text)
        if not act:
            noop += 1
            if noop >= 4:
                break
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "No parseable action. Use combine/apply/force."}]
            continue
        noop = 0
        if act[0] == "combine":
            obs = s.combine(act[1], act[2])
        elif act[0] == "apply":
            obs = s.apply(act[1], act[2])
        elif act[0] == "brute":
            obs = s.brute_combine_on_ward(act[1], act[2], act[3])
        done = len(s.opened) == n_wards
        print(f"    [t{t+1}] {act} -> {obs[:74]}", flush=True)
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": obs + ("\n\nAll wards open. Done." if done else "\n\nWhat next?")}]
        if done:
            break
    # classify
    used_tool = s.built_key and s.key_uses >= 2  # built the key and reused it
    result = {
        "model": model, "seed": seed, "n_wards": n_wards,
        "total_actions": s.actions, "opened": len(s.opened),
        "built_key": s.built_key, "key_uses": s.key_uses, "brute_opens": s.brute_opens,
        "used_tool": used_tool,
    }
    print(f"\n  {model}: opened {len(s.opened)}/{n_wards} in {s.actions} actions | "
          f"built_key={s.built_key} key_uses={s.key_uses} brute_opens={s.brute_opens} "
          f"-> {'TOOL USE' if used_tool else ('partial' if s.built_key else 'BRUTE/none')}", flush=True)
    return result


async def main():
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-7"
    n_wards = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / f"toolworld_{model.replace('/','_')}_{ts}"; out.mkdir(parents=True, exist_ok=True)
    r = await run(model, n_wards=n_wards, seed=seed)
    (out / "result.json").write_text(json.dumps(r, indent=2))
    print(f"Result: {out/'result.json'}")


if __name__ == "__main__":
    asyncio.run(main())
