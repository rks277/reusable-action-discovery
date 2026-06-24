"""GATED coupon-collector WoodWorld -- coupon goal + ToolWorld's OBSTACLE STRUCTURE.

Motivation (tool-wood-discrepancy): the plain coupon WoodWorld matched ToolWorld's GOAL
(collect N distinct kinds) but still gave Haiku near-zero recognition (R=0.01 same-recipe,
0.08 distinct-recipe), and 7/8 builders never used the axe. The remaining structural gap we
isolated: in the coupon world the goal items are merely ENUMERATED (gather hands them out,
the axe is a random dispenser whose value is tail-only), whereas in ToolWorld they are
GATED behind locked obstacles and the tool is DIRECTED at a specific lock -- so building
immediately and obviously advances the goal ("build machine -> use on door -> open").

This env adds ToolWorld's gating to the coupon, holding everything else at the WoodWorld
value, so the ONLY new variable is the obstacle structure:

  - GOAL: collect all N distinct kinds of wood. Each kind w_i is LOCKED inside container
          c_i (the N containers are VISIBLE from turn 1, like ToolWorld's doors).
  - examine <container>: the grind/search path -- drops 1 uniformly-random KEY (coupon
          over the N keys) + 1 uniformly-random tree-resource (the build ingredient). Both
          prob 1. (Mirrors ToolWorld examine(door): random key + random byproduct.)
  - combine <a> <b>: stick+stick -> axe  ("same"); or r0+r1 -> axe ("distinct"). Persists.
  - use <axe> <container_i>: DIRECTED tool -- yields key_i (the key for THAT container).
          The axe persists. (Mirrors use(machine, door_i) -> key_i.)
  - use <key_i> <container_i>: opens c_i, revealing wood w_i (if the key matches). Else
          "does not fit". (Mirrors use(key_i, door_i) -> opens.)

So the grind bypass is the coupon-collector over keys (examine -> random key, open the
matching container), and the build path is the directed axe (build once, then 2 actions per
container: axe->key, key->open). E[grind] = N*H_N + N (== ToolWorld); budget = 1.2x that, so
building stays OPTIONAL (a disposition signal), never forced by the budget.

Reuses ToolWorld's parser (examine/combine/use, 2-arg use) + the WoodWorld obfuscation.
Run from repo root with PYTHONPATH=. so `lomekwi.*` and `scripts.*` resolve.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field

from dotenv import load_dotenv

from lomekwi.obfuscation import assign
from lomekwi.raw_chat import RawChat
from scripts.run_toolworld_llm import parse, extract   # examine/combine/use (2-arg use)


SYS = ("You are an agent in a workshop and must collect every different kind of a target "
       "material, each locked inside its own container, using as few actions as possible. "
       "End each turn with exactly one action on its own line: examine <object>, "
       "combine <a> <b> (exactly two items), or use <object> <container>. Think briefly, "
       "then act. Minimize total actions.")


# --- world construction + obfuscation -------------------------------------------
def _familize_resources(labels: dict, res_roles: list, relabel_seed: int) -> dict:
    """Relabel the T tree-resource roles as a shared-stem numbered family '{stem}{i}'
    (random stem + random index order, deterministic in relabel_seed), so the parts are a
    priori indistinguishable and the latent stick is hidden among them. Same helper as the
    plain coupon env; the stem avoids the wood/key/container/axe letters."""
    rng = random.Random(relabel_seed * 99991 + 17)
    labels = dict(labels)
    nonfam = set(labels.values())
    alph = list("abcdefghijkmnpqrstuvwxyz")          # drop l/o (look like 1/0)
    rng.shuffle(alph)
    stem = next(c for c in alph if c not in nonfam)
    order = list(res_roles)
    rng.shuffle(order)
    for i, it in enumerate(order, 1):
        labels[it] = f"{stem}{i}"
    return labels


def make_world(relabel_seed: int, n_kinds: int, n_types: int,
               obfuscate: bool = True, recipe_mode: str = "same") -> dict:
    """Per-episode world. `container_base`/`key_base`/`axe` + the N wood kinds get distinct
    single letters; containers = '{cbase}_i', keys = '{kbase}_i' (i=1..N). The T
    tree-resources become a look-alike family ('r0' = latent stick).

    recipe_mode: "same" -> axe = stick + stick (two of one look-alike); "distinct" ->
    axe = r0 + r1 (two distinct types, both required; r2..r{T-1} inert) -- ToolWorld-shaped.
    """
    assert n_kinds + 3 <= 24, "letter scheme: container+key+axe + N woods + stem must fit 26"
    wood_roles = [f"w{i}" for i in range(n_kinds)]
    res_roles = [f"r{i}" for i in range(n_types)]        # r0 = latent stick
    nonfam = ["container", "key", "axe"] + wood_roles
    if obfuscate:
        labels = assign(nonfam, seed=relabel_seed, scheme="letter")
        labels = _familize_resources(labels, res_roles, relabel_seed)
    else:
        labels = {**{e: e for e in nonfam}, **{r: r for r in res_roles}}
    wood_toks = [labels[w] for w in wood_roles]
    res_toks = [labels[r] for r in res_roles]
    if recipe_mode == "distinct":
        recipe_in = {labels["r0"]: 1, labels["r1"]: 1}
    else:
        recipe_in = {labels["r0"]: 2}
    # container i (1-based) holds wood kind wood_toks[i-1]; key i opens container i.
    return {
        "n_kinds": n_kinds, "n_types": n_types, "recipe_mode": recipe_mode,
        "labels": labels, "tok2latent": {v: k for k, v in labels.items()},
        "wood_toks": wood_toks, "wood_set": set(wood_toks),
        "res_toks": res_toks,
        "container_base": labels["container"], "key_base": labels["key"],
        "stick_tok": labels["r0"], "axe_tok": labels["axe"],
        "recipe_in": recipe_in, "ingredient_toks": list(recipe_in.keys()),
        "containers": [f"{labels['container']}_{i}" for i in range(1, n_kinds + 1)],
    }


# --- episode state ----------------------------------------------------------------
@dataclass
class State:
    world: dict
    rng: random.Random
    hint: bool = False
    inv: dict = field(default_factory=dict)            # resource/axe token -> count
    held_keys: set = field(default_factory=set)        # key indices held (1-based)
    opened: dict = field(default_factory=dict)         # container idx -> turn opened
    axe_used_on: set = field(default_factory=set)      # container idxs the axe keyed
    turn: int = 0
    gather_count: int = 0                              # examines (the grind/search verb)
    craft_attempts: int = 0
    craft_success: int = 0
    use_count: int = 0
    use_axe_count: int = 0                             # times the axe produced a key
    built_axe: bool = False
    build_turn: int | None = None
    held_ingredients: bool = False                     # ever satisfied the recipe (sticky)

    # --- index helpers ---
    def container_idx(self, tok: str):
        m = re.fullmatch(rf"{re.escape(self.world['container_base'])}_(\d+)", tok)
        return int(m.group(1)) if m else None

    def key_idx(self, tok: str):
        m = re.fullmatch(rf"{re.escape(self.world['key_base'])}_(\d+)", tok)
        return int(m.group(1)) if m else None

    def key_tok(self, i: int) -> str:
        return f"{self.world['key_base']}_{i}"

    def wood_of(self, container_i: int) -> str:
        return self.world["wood_toks"][container_i - 1]

    def held(self, tok: str) -> int:
        return self.inv.get(tok, 0)

    def _add(self, tok: str, k: int):
        self.inv[tok] = self.inv.get(tok, 0) + k

    def _consume(self, multiset: dict):
        for tok, c in multiset.items():
            self.inv[tok] = self.inv.get(tok, 0) - c
            if self.inv[tok] <= 0:
                self.inv.pop(tok, None)

    def distinct_held(self) -> int:
        return len(self.opened)                        # opened container == wood kind held

    def solved(self) -> bool:
        return len(self.opened) >= self.world["n_kinds"]

    def _check_held_ingredients(self):
        if self.held_ingredients:
            return
        if all(self.held(t) >= c for t, c in self.world["recipe_in"].items()):
            self.held_ingredients = True

    # --- actions ---
    def examine(self, tok: str) -> str:
        """Search a container: drops 1 random KEY (coupon over the N keys) + 1 random
        tree-resource. Target-independent drop, like ToolWorld examine(door)."""
        ci = self.container_idx(tok)
        if ci is not None and 1 <= ci <= self.world["n_kinds"]:
            self.gather_count += 1
            j = self.rng.randint(1, self.world["n_kinds"])
            self.held_keys.add(j)
            res = self.rng.choice(self.world["res_toks"])
            self._add(res, 1)
            opened_note = " (already open)" if ci in self.opened else ""
            return (f"You search {tok}{opened_note}. Out falls {self.key_tok(j)} and "
                    f"1 {res}. You now hold {self.held(res)} {res}.")
        if tok in self.world["res_toks"]:
            return f"A small {tok}. Nothing remarkable."          # nohint: opaque
        if tok == self.world["axe_tok"] and self.built_axe:
            return f"A {tok}. It has a working edge; you sense it could force a container."
        if self.key_idx(tok) is not None:
            return f"{tok}: a small fitted key."
        return f"You examine {tok}: nothing of note."

    def craft(self, a: str, b: str) -> str:
        """Combine exactly two held items. Only the recipe pair -> axe; else opaque."""
        self.craft_attempts += 1
        toks = [a, b]
        provided = Counter(toks)
        need = Counter(self.world["recipe_in"])
        if provided == need and all(self.held(t) >= c for t, c in need.items()):
            self._consume(dict(need))
            axe = self.world["axe_tok"]
            self._add(axe, 1)
            self.craft_success += 1
            if not self.built_axe:
                self.built_axe = True
                self.build_turn = self.turn
            made = ", ".join(f"{c} {t}" for t, c in need.items())
            return f"You combine {made} into 1 {axe} (new). It persists."
        return f"You try to combine {a} and {b}: nothing happens."

    def use(self, item: str, target: str) -> str:
        """use axe on container -> that container's key (directed); use key on container ->
        opens it, revealing the wood kind (if the key matches)."""
        self.use_count += 1
        ci = self.container_idx(target)
        if ci is None or not (1 <= ci <= self.world["n_kinds"]):
            return f"There is no {target}."
        if ci in self.opened:
            return f"{target} is already open."
        # the directed tool
        if item == self.world["axe_tok"]:
            if self.held(item) < 1:
                return f"You don't have a {item}."
            self.held_keys.add(ci)
            self.axe_used_on.add(ci)
            self.use_axe_count += 1
            return (f"You use the {item} on {target}. It yields {self.key_tok(ci)}. "
                    f"(the {item} remains with you.)")
        # a key
        ki = self.key_idx(item)
        if ki is not None:
            if ki not in self.held_keys:
                return f"You don't have {item}."
            if ki == ci:
                self.opened[ci] = self.turn
                wood = self.wood_of(ci)
                self._add(wood, 1)
                return (f"You use {item} on {target}: it fits. {target} opens, revealing "
                        f"1 {wood} (a kind of wood). You now hold it.")
            return f"You use {item} on {target}: it does not fit."
        return f"You use {item} on {target}: nothing happens."

    def initial_obs(self, budget: int | None) -> str:
        w = self.world
        s = (f"You are in a sealed workshop. There are {w['n_kinds']} locked containers: "
             f"{', '.join(w['containers'])}.\n\n"
             f"GOAL: collect at least one of EACH of the {w['n_kinds']} different kinds of "
             f"wood, using as few actions as possible. Each kind is locked inside one of "
             f"the containers above; different kinds are named differently.\n\n"
             f"You may act on objects with these verbs (one action per turn, on its own "
             f"line):\n"
             f"  examine <object>      - inspect or search an object\n"
             f"  combine <a> <b>       - combine exactly two items you hold\n"
             f"  use <object> <container> - use an item on a container\n"
             f"You hold nothing yet. Things you obtain persist unless stated otherwise.")
        if budget is not None:
            s += (f"\n\nYou have a STRICT BUDGET of {budget} actions total. If you do not "
                  f"hold all {w['n_kinds']} kinds within {budget} actions, you fail. "
                  f"Spend them wisely.")
        return s


# --- one episode ------------------------------------------------------------------
async def run(model: str, n_kinds: int, n_types: int, relabel_seed: int = 0,
              draw_seed: int = 0, hint: bool = False, max_turns: int = 300,
              budget: int | None = None, no_progress_window: int | None = None,
              stop_on_build: bool = False, obfuscate: bool = True,
              recipe_mode: str = "same", world_override: dict | None = None):
    """One gated coupon-collector episode. Mirrors coupon_woodworld.run but with
    ToolWorld's 2-arg use and gated mechanics. no_progress_window defaults to None: the
    coupon tail (duplicate keys waiting for the last container) is legitimately
    no-progress, so the budget binds instead."""
    load_dotenv()
    client = RawChat()
    world = world_override if world_override is not None else make_world(
        relabel_seed, n_kinds, n_types, obfuscate, recipe_mode)
    s = State(world=world, rng=random.Random(draw_seed * 6151 + 1), hint=hint)

    intro = s.initial_obs(budget)
    msgs = [{"role": "user", "content": intro + "\n\nWhat do you do?"}]
    trace = []
    noop = noop_total = refusals = 0
    unparsed = []
    usage_tot = {k: 0 for k in RawChat.USAGE_FIELDS}
    usage_tot["calls"] = 0
    stale = 0
    stopped_reason = None
    t_star = None

    def dispatch(act):
        if act[0] == "examine":
            obs = s.examine(act[1])
        elif act[0] in ("craft", "combine"):
            obs = s.craft(act[1], act[2])
        else:
            obs = s.use(act[1], act[2])
        s._check_held_ingredients()
        return obs

    for t in range(max_turns):
        s.turn = len(trace)
        if budget is not None and len(trace) >= budget:
            stopped_reason = stopped_reason or "out_of_budget"
            break
        api_err = None
        try:
            text = await client.chat(model, SYS, msgs, max_tokens=1500)
        except Exception as e:
            text = ""
            api_err = f"{type(e).__name__}: {e}"
        if client.last_usage is not None:
            for k, v in client.last_usage.items():
                usage_tot[k] += v
            usage_tot["calls"] += 1

        act = extract(text)
        if not act:
            noop += 1
            noop_total += 1
            dbg = getattr(client, "last_debug", None)
            if dbg and dbg.get("stop_reason") == "refusal":
                refusals += 1
            unparsed.append({"turn": t, "api_error": api_err, "text": text})
            print(f"  [t{t+1}] NO-OP{' (API-ERR)' if api_err else ''}: "
                  f"{(api_err or text)[:80]!r}", flush=True)
            if noop >= 4:
                stopped_reason = "noop"
                break
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "No parseable action. Use examine / "
                      "combine <a> <b> / use <item> <container>, one action on its own line."}]
            continue
        noop = 0

        prev_opened = len(s.opened)
        prev_keys = len(s.held_keys)
        prev_inv = len(s.inv)
        prev_built = s.built_axe
        prev_use_axe = s.use_axe_count
        obs = dispatch(act)
        if s.use_axe_count > prev_use_axe and t_star is None:
            t_star = len(trace)

        done = s.solved()
        opened_n = len(s.opened)
        print(f"  [t{t+1}] {act} -> {obs[:74]} [{opened_n}/{n_kinds}]", flush=True)
        trace.append({"turn": t, "action": list(act), "obs": obs, "agent_text": text,
                      "usage": client.last_usage})

        progressed = (len(s.opened) > prev_opened or len(s.held_keys) > prev_keys
                      or len(s.inv) > prev_inv or (s.built_axe and not prev_built))
        stale = 0 if progressed else stale + 1

        if stop_on_build and s.built_axe:
            stopped_reason = "built"
            break
        out_of_budget = budget is not None and len(trace) >= budget
        if done or out_of_budget:
            stopped_reason = "solved" if done else "out_of_budget"
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": obs +
                      (f"\n\nYou hold all {n_kinds} kinds. Done." if done
                       else f"\n\nBudget exhausted ({budget} actions). You fail.")}]
            break
        if no_progress_window is not None and stale >= no_progress_window:
            stopped_reason = "no_progress"
            print(f"  [t{t+1}] SAFEGUARD: no progress for {stale} actions; aborting.",
                  flush=True)
            break
        left = f", {budget - len(trace)} actions left" if budget is not None else ""
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": obs +
                  f"\n\n[{opened_n}/{n_kinds} kinds{left}] What next?"}]

    stopped_reason = stopped_reason or "max_turns"
    result = {
        "model": model, "task": "gated_woodworld",
        "n_kinds": n_kinds, "n_types": n_types, "recipe_mode": world.get("recipe_mode"),
        "relabel_seed": relabel_seed, "draw_seed": draw_seed,
        "hint": hint, "obfuscate": obfuscate, "labels": world["labels"],
        "solved": s.solved(), "distinct_held": s.distinct_held(),
        "total_actions": len(trace), "budget": budget,
        "built_axe": s.built_axe, "build_turn": s.build_turn, "t_star": t_star,
        "held_ingredients": s.held_ingredients,
        "gather_count": s.gather_count, "craft_attempts": s.craft_attempts,
        "craft_success": s.craft_success, "use_count": s.use_count,
        "use_axe_count": s.use_axe_count,
        "noop_total": noop_total, "refusals": refusals, "unparsed": unparsed,
        "stopped_reason": stopped_reason, "usage": usage_tot,
    }
    print(f"\n  RESULT {model} N={n_kinds} T={n_types} seed={relabel_seed} "
          f"recipe={world.get('recipe_mode')}: solved={result['solved']} "
          f"opened={len(s.opened)}/{n_kinds} actions={len(trace)}/{budget} "
          f"built={s.built_axe} build_turn={s.build_turn} axe_uses={s.use_axe_count} "
          f"reason={stopped_reason}", flush=True)
    return result, trace
