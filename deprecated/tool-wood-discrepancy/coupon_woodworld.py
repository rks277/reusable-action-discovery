"""Coupon-collector WoodWorld -- a WoodWorld whose GOAL STRUCTURE matches ToolWorld.

Motivation (tool-wood-discrepancy): the ToolWorld recognition inversion is a genuine
recognition failure, and the remaining structural difference vs WoodWorld is that
ToolWorld's goal items are DISTINCT (open N distinct doors / collect N distinct keys)
while WoodWorld's wood is FUNGIBLE (hold N identical wood). This env makes WoodWorld a
COUPON-COLLECTOR problem so its goal becomes "collect all N distinct kinds of wood" --
the direct analog of ToolWorld's "collect N distinct keys".

Mechanics (confirmed spec):
  - GOAL: hold >= 1 of EACH of the N distinct wood kinds (the world has exactly N).
  - gather: drops 1 uniformly-random wood-kind (the coupon) AND 1 uniformly-random
            tree-resource (one of T types). Both with probability 1.
  - The T tree-resources = 1 latent "stick" + (T-1) inert "apple" decoys, relabeled as
            a LOOK-ALIKE numbered family {stem}{i} (like the iso_recipe variant) so the
            agent must SEARCH which pair combines.
  - build:  stick + stick -> axe (non-hostile; sticks are free). The axe PERSISTS.
  - use axe: yields 1 uniformly-random wood-kind NOT yet held, w.p. 1 (eliminates the
            coupon-collector duplicate-draw waste -> the reusable tool's value).

This is a NEW self-contained env (the stock woodworld Mech/State assume a single fungible
goal_item) but reuses the obfuscation labeler, the woodworld action parser, and RawChat.

Run from repo root with PYTHONPATH=. so `lomekwi.*` and `scripts.*` resolve.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from dotenv import load_dotenv

from lomekwi.obfuscation import assign
from lomekwi.raw_chat import RawChat
from scripts.woodworld import parse, extract   # verbs gather/combine/use (correct set)


SYS = ("You are an agent in a workshop and must collect every different kind of a "
       "target material using as few actions as possible. End each turn with exactly "
       "one action on its own line: gather, combine <a> <b> (exactly two items), or "
       "use <item>. Think briefly, then act. Minimize total actions.")


# --- world construction + obfuscation -------------------------------------------
def _familize_resources(labels: dict, res_roles: list, relabel_seed: int) -> dict:
    """Relabel the T tree-resource roles as a shared-stem numbered family '{stem}{i}'
    (random stem + random index order, deterministic in relabel_seed), so the parts are
    a priori indistinguishable and the latent stick is hidden among them. Mirrors
    scripts/woodworld._familize; the stem avoids the wood/axe letters."""
    rng = random.Random(relabel_seed * 99991 + 17)
    labels = dict(labels)
    nonfam = set(labels.values())                       # wood + axe letters
    alph = list("abcdefghijkmnpqrstuvwxyz")             # drop l/o (look like 1/0)
    rng.shuffle(alph)
    stem = next(c for c in alph if c not in nonfam)
    order = list(res_roles)
    rng.shuffle(order)
    for i, it in enumerate(order, 1):
        labels[it] = f"{stem}{i}"
    return labels


def make_world(relabel_seed: int, n_kinds: int, n_types: int,
               obfuscate: bool = True, recipe_mode: str = "same") -> dict:
    """Per-episode world: N distinct wood-kind tokens + axe get distinct single letters;
    the T tree-resources become a look-alike family. `r0` is the latent stick.

    recipe_mode:
      "same"     -> axe = stick + stick (two of ONE look-alike). Default.
      "distinct" -> axe = r0 + r1 (TWO DISTINCT resource types, both required); the
                    remaining r2..r{T-1} are inert. Mirrors ToolWorld's "combine the two
                    byproduct types" -> the search is which PAIR of the T combine (C(T,2)).
    """
    assert n_kinds + 1 <= 23, "letter scheme caps at 26; n_kinds + axe must be <= 23"
    wood_roles = [f"w{i}" for i in range(n_kinds)]
    res_roles = [f"r{i}" for i in range(n_types)]        # r0 = latent stick
    nonfam = wood_roles + ["axe"]
    if obfuscate:
        labels = assign(nonfam, seed=relabel_seed, scheme="letter")   # distinct letters
        labels = _familize_resources(labels, res_roles, relabel_seed)
    else:
        labels = {**{e: e for e in nonfam}, **{r: r for r in res_roles}}
    wood_toks = [labels[w] for w in wood_roles]
    res_toks = [labels[r] for r in res_roles]
    if recipe_mode == "distinct":
        recipe_in = {labels["r0"]: 1, labels["r1"]: 1}   # two distinct types, both needed
    else:
        recipe_in = {labels["r0"]: 2}                    # stick + stick
    return {
        "n_kinds": n_kinds, "n_types": n_types, "recipe_mode": recipe_mode,
        "labels": labels, "tok2latent": {v: k for k, v in labels.items()},
        "wood_toks": wood_toks, "wood_set": set(wood_toks),
        "res_toks": res_toks,
        "stick_tok": labels["r0"], "axe_tok": labels["axe"],
        "recipe_in": recipe_in,
        "ingredient_toks": list(recipe_in.keys()),
    }


# --- episode state ----------------------------------------------------------------
@dataclass
class State:
    world: dict
    rng: random.Random                 # single RNG: gather draws + use-axe draw
    hint: bool = False
    inv: dict = field(default_factory=dict)            # token -> count held
    turn: int = 0
    gather_count: int = 0
    craft_attempts: int = 0
    craft_success: int = 0
    use_count: int = 0
    use_axe_count: int = 0
    built_axe: bool = False
    build_turn: int | None = None
    held_ingredients: bool = False                     # ever held >= 2 sticks (sticky)

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
        return sum(1 for t in self.world["wood_set"] if self.held(t) > 0)

    def solved(self) -> bool:
        return self.distinct_held() >= self.world["n_kinds"]

    def _check_held_ingredients(self):
        """Sticky: set once inventory satisfies the axe recipe's input multiset
        (recipe-agnostic -- stick+stick OR the two distinct ingredients)."""
        if self.held_ingredients:
            return
        if all(self.held(t) >= c for t, c in self.world["recipe_in"].items()):
            self.held_ingredients = True

    # --- actions ---
    def gather(self) -> str:
        """Draw 1 random wood-kind (the coupon) + 1 random tree-resource. Both prob 1.
        Two rng draws per call (deterministic in the episode seed)."""
        self.gather_count += 1
        coupon = self.rng.choice(self.world["wood_toks"])
        self._add(coupon, 1)
        res = self.rng.choice(self.world["res_toks"])
        self._add(res, 1)
        return (f"You gather. You find 1 {coupon} and 1 {res}. You now hold "
                f"{self.held(coupon)} {coupon} and {self.held(res)} {res}.")

    def craft(self, toks: list) -> str:
        """Combine exactly two held items. Only stick+stick -> axe; everything else is
        opaque ('nothing happens'). Axe persists; first build sets built_axe."""
        self.craft_attempts += 1
        if len(toks) != 2:
            return (f"You can only combine two items at a time (you tried {len(toks)}). "
                    "Combine exactly two.")
        provided = Counter(toks)
        need = Counter(self.world["recipe_in"])          # {stick: 2}
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
        return f"You try to combine {', '.join(toks)}: nothing happens."

    def use(self, item: str) -> str:
        """Use the axe -> 1 uniformly-random wood-kind you do NOT yet hold (w.p. 1). The
        axe is never consumed. Anything else does nothing."""
        self.use_count += 1
        if item == self.world["axe_tok"]:
            if self.held(item) < 1:
                return f"You don't have a {item}."
            unheld = [t for t in self.world["wood_toks"] if self.held(t) == 0]
            if not unheld:
                return (f"You use the {item}, but you already hold every kind. "
                        f"(the {item} remains with you.)")
            new = self.rng.choice(unheld)
            self._add(new, 1)
            self.use_axe_count += 1
            return (f"You use the {item}. It yields 1 {new} (a new kind). (the {item} "
                    f"remains with you.) You now hold {self.held(new)} {new}.")
        return f"You use {item}: nothing happens."

    def initial_obs(self, budget: int | None) -> str:
        w = self.world
        s = (f"You are in a sealed workshop.\n\n"
             f"GOAL: collect at least one of EACH of the {w['n_kinds']} different kinds "
             f"of wood that exist here (you currently hold none of any kind), using as "
             f"few actions as possible. Different kinds are named differently.\n\n"
             f"You may act with these verbs (one action per turn, on its own line):\n"
             f"  gather                - search for raw material\n"
             f"  combine <a> <b>       - combine exactly two items you hold\n"
             f"  use <item>            - operate or apply an item you hold\n"
             f"You hold nothing yet. Things you obtain persist unless stated otherwise.")
        if budget is not None:
            s += (f"\n\nYou have a STRICT BUDGET of {budget} actions total. If you do "
                  f"not hold all {w['n_kinds']} kinds within {budget} actions, you fail. "
                  f"Spend them wisely.")
        return s


# --- one episode ------------------------------------------------------------------
async def run(model: str, n_kinds: int, n_types: int, relabel_seed: int = 0,
              draw_seed: int = 0, hint: bool = False, max_turns: int = 300,
              budget: int | None = None, no_progress_window: int | None = None,
              stop_on_build: bool = False, obfuscate: bool = True,
              recipe_mode: str = "same", world_override: dict | None = None):
    """One coupon-collector episode. Mirrors scripts/woodworld.run.

    NB no_progress_window defaults to None (disabled): the coupon-collector TAIL is a long
    run of duplicate gathers waiting for the last rare kind -- legitimately "no progress" --
    so a window would falsely abort honest grinders. The budget (hard cap on total actions)
    binds instead.
    """
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
        if act[0] == "gather":
            obs = s.gather()
        elif act[0] in ("craft", "combine"):
            obs = s.craft(act[1])
        else:
            obs = s.use(act[1])
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
                     {"role": "user", "content": "No parseable action. Use gather / "
                      "combine <a> <b> / use <item>, one action on its own line."}]
            continue
        noop = 0

        prev_distinct = s.distinct_held()
        prev_inv = len(s.inv)
        prev_built = s.built_axe
        prev_use_axe = s.use_axe_count
        obs = dispatch(act)
        if s.use_axe_count > prev_use_axe and t_star is None:
            t_star = len(trace)

        done = s.solved()
        distinct = s.distinct_held()
        print(f"  [t{t+1}] {act} -> {obs[:74]} [{distinct}/{n_kinds}]", flush=True)
        trace.append({"turn": t, "action": list(act), "obs": obs, "agent_text": text,
                      "usage": client.last_usage})

        progressed = (distinct > prev_distinct or len(s.inv) > prev_inv
                      or (s.built_axe and not prev_built))
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
                  f"\n\n[{distinct}/{n_kinds} kinds{left}] What next?"}]

    stopped_reason = stopped_reason or "max_turns"
    result = {
        "model": model, "task": "coupon_woodworld",
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
    print(f"\n  RESULT {model} N={n_kinds} T={n_types} seed={relabel_seed}: "
          f"solved={result['solved']} distinct={s.distinct_held()}/{n_kinds} "
          f"actions={len(trace)}/{budget} built={s.built_axe} "
          f"build_turn={s.build_turn} axe_uses={s.use_axe_count} "
          f"reason={stopped_reason}", flush=True)
    return result, trace
