"""Woodworld: a build-or-fail tool-discovery task (obfuscated, leak-free).

A minimal sibling of scripts/toolworld_v2.py that isolates the SAME phenomenon
(does the agent recognize and build a persistent reusable tool, or brute-force?)
on a self-contained economic substrate, rather than the doors/keys world.

Mechanics (all latent names obfuscated per episode; the agent discovers them):
  - gather            -> +1 wood with prob GATHER_PROB (0.8), else +0. Stochastic.
  - combine <a> <b> ... -> combine items per an EDITABLE recipe table:
        2 wood           -> 4 sticks
        1 stick + 1 wood -> 1 axe
    Crafting consumes inputs, produces outputs.
  - use <item>        -> use the AXE -> +2 wood, and the axe is NOT consumed (it
    is the persistent reusable tool). Using anything else -> nothing. `use` never
    consumes.
  GOAL: hold >= N wood (net, in inventory) within a strict budget B = round(1.2 N).

Economics (see scripts/validate_woodworld.py for the derivation): gather alone
yields 0.8 wood/action, so reaching N by gathering needs ~1.25 N actions > B, i.e.
brute-force (barely) fails in expectation. Building the axe (a small upfront wood
+ craft cost, then +2 wood/use) is the rational path once N exceeds N* (see
scripts/validate_woodworld.py, which derives N* from the current recipe table).
The axe is the analog of toolworld's machine; built_axe is the calibration signal.

Obfuscation (lomekwi.obfuscation): per episode we draw a fresh uniformly-random
relabeling of the three latent roles {wood, stick, axe}. The goal names the
obfuscated wood token and gather yields it (so wood is learned immediately), but
the recipes and the axe's +2 effect are never told -- the agent discovers them by
acting. examine has no analog here; a subtle interaction nudge is available via
--hint. Results are reported as a mean over relabelings.

This module is self-contained (its own make_world/State/run) and mirrors
toolworld_v2 field-for-field where possible so the sweep/replay/analyze trio can
mirror the toolworld ones. State reads only obfuscated strings, so replay from
recorded labels is byte-exact regardless of the ambient obfuscation scheme.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from lomekwi.obfuscation import assign
from lomekwi.raw_chat import RawChat

# --- EDITABLE MECHANICS (single source of truth; latent names) --------------
# Edit these to change the task; make_world/world_from_labels/validate all read
# them. Names here are LATENT roles; each episode relabels them to nonsense tokens.
GATHER_PROB = 0.8
GATHER_YIELD = ("wood", 1)                       # (latent item, count) on success
RECIPES = [                                       # ordered; in/out are multisets
    {"in": {"wood": 2}, "out": {"stick": 4}},
    {"in": {"stick": 1, "wood": 1}, "out": {"axe": 1}},
]
USE_YIELDS = {"axe": ("wood", 2)}                 # latent item -> (yield item, n); never consumes
GOAL_ITEM = "wood"                                # hold >= N of this
TOOL_ITEM = "axe"                                 # the persistent reusable tool (instrumentation)
ELEMENTS = ["wood", "stick", "axe"]               # the latent roles to relabel


@dataclass(frozen=True)
class Mech:
    """A self-contained mechanics spec (one game variant). The module globals
    above define the BASE game; variants override fields to ablate structural
    factors (see VARIANTS). `elements` ORDER matters -- assign() is positional
    over a shuffled alphabet, so BASE.elements must equal ELEMENTS verbatim or
    every baseline label shifts. `decoy_items` are inert items gather drops
    (no recipe, `use` does nothing); they only widen the combine space."""
    gather_yield: tuple                # (latent item, count)
    recipes: tuple                     # tuple of {"in": {...}, "out": {...}}
    use_yields: dict                   # latent -> (yield latent, n)
    goal_item: str
    tool_item: str
    elements: tuple                    # latent roles to relabel (order-sensitive)
    decoy_items: tuple = ()            # inert items also dropped by gather (one-of-k, w.p. p)
    extra_drops: tuple = ()            # items each dropped by an INDEPENDENT Bernoulli(p) draw
                                       # per gather (e.g. a free side-ingredient + an inert decoy)
    family_items: tuple = ()           # items relabeled as a shared-stem numbered family
                                       # "{stem}{i}" (random stem + index order per episode),
                                       # so the parts look interchangeable -> the recipe becomes
                                       # a "which of these parts combine?" search (ToolWorld-style)


BASE = Mech(gather_yield=GATHER_YIELD, recipes=tuple(RECIPES), use_yields=dict(USE_YIELDS),
            goal_item=GOAL_ITEM, tool_item=TOOL_ITEM, elements=tuple(ELEMENTS))

# iso variant: ablate the narrow tree (3 inert apple decoys widen the combine
# space) AND the multilayer (single-step 2 wood -> axe, no sticks), KEEPING the
# hostile build (the combine still consumes 2 wood, the goal item). Isolates
# whether hostility alone drives WoodWorld's discovery bottleneck.
ISO_APPLES3 = Mech(
    gather_yield=("wood", 1),
    recipes=({"in": {"wood": 2}, "out": {"axe": 1}},),
    use_yields={"axe": ("wood", 2)},
    goal_item="wood", tool_item="axe",
    elements=("wood", "axe", "apple1", "apple2", "apple3"),
    decoy_items=("apple1", "apple2", "apple3"),
)

# non-hostile control: ablate the HOSTILE build. The tool is made from a FREE
# side-resource (sticks, gathered independently w.p. p) instead of the goal item
# (wood), so building is pure upside -- no wood is lost. Single-step (2 sticks ->
# axe) and widened with one inert apple decoy, matching iso_apples3 on everything
# BUT hostility. Expected ToolWorld-like (models build readily). gather drops wood,
# stick, apple each by an independent Bernoulli(p) draw; nothing but stick+stick
# combines; apple is inert (no recipe, `use` does nothing).
ISO_NONHOSTILE = Mech(
    gather_yield=("wood", 1),
    recipes=({"in": {"stick": 2}, "out": {"axe": 1}},),
    use_yields={"axe": ("wood", 2)},
    goal_item="wood", tool_item="axe",
    elements=("wood", "stick", "axe", "apple"),
    extra_drops=("stick", "apple"),    # each independent w.p. p; apple inert (no recipe)
)

# non-hostile, TWO inert decoys (apple + pear): same as iso_nonhostile but widens
# the candidate-combine space with a second do-nothing item. Used with budget slack
# (mult 1.2) to test whether a viable brute-force alternative + extra distractors
# pull capable models off the build (R drop).
ISO_NONHOSTILE_PEAR = Mech(
    gather_yield=("wood", 1),
    recipes=({"in": {"stick": 2}, "out": {"axe": 1}},),
    use_yields={"axe": ("wood", 2)},
    goal_item="wood", tool_item="axe",
    elements=("wood", "stick", "axe", "apple", "pear"),
    extra_drops=("stick", "apple", "pear"),   # apple + pear inert decoys; stick the ingredient
)

# recipe-search variant: non-hostile (stick+stick -> axe, sticks free) but the four
# gatherable parts (stick + 3 inert decoys apple/pear/cherry) are relabeled as a
# shared-stem numbered FAMILY ("r1".."r4", random stem + order per episode), so they
# look interchangeable and the agent must SEARCH which combine -- C(4,2)+4 = 10
# candidate pairs, one correct (the stick-pair). Mimics ToolWorld's byproduct-type
# recipe search, isolating build-discovery cost from hostility. Tests whether a costly
# build-search (vs a trivial obvious combine) makes capable models decline it -> R inversion.
ISO_RECIPE = Mech(
    gather_yield=("wood", 1),
    recipes=({"in": {"stick": 2}, "out": {"axe": 1}},),
    use_yields={"axe": ("wood", 2)},
    goal_item="wood", tool_item="axe",
    elements=("wood", "stick", "axe", "apple", "pear", "cherry"),
    extra_drops=("stick", "apple", "pear", "cherry"),   # each independent w.p. p; 3 inert
    family_items=("stick", "apple", "pear", "cherry"),  # relabeled as a uniform part family
)

VARIANTS = {"base": BASE, "iso_apples3": ISO_APPLES3, "iso_nonhostile": ISO_NONHOSTILE,
            "iso_nonhostile_pear": ISO_NONHOSTILE_PEAR, "iso_recipe": ISO_RECIPE}


def _familize(labels: dict, mech: "Mech", relabel_seed: int) -> dict:
    """Relabel mech.family_items as a shared-stem numbered family '{stem}{i}'
    (deterministic in relabel_seed): a random stem letter + a random index order, so
    the parts are a priori indistinguishable (mimics ToolWorld's obfuscated byproduct
    types). Non-family elements (wood, axe) keep their assign() labels; the stem avoids
    their tokens and the 2-char '{stem}{i}' labels never collide with them."""
    rng = random.Random(relabel_seed * 99991 + 17)
    labels = dict(labels)
    nonfam = {labels[e] for e in mech.elements if e not in mech.family_items}
    alph = list("abcdefghijkmnpqrstuvwxyz")          # drop l/o (look like 1/0)
    rng.shuffle(alph)
    stem = next(c for c in alph if c not in nonfam)
    order = list(mech.family_items)
    rng.shuffle(order)
    for i, it in enumerate(order, 1):
        labels[it] = f"{stem}{i}"
    return labels


def _tool_recipe_in(labels: dict, mech: "Mech") -> dict:
    """Token-keyed input multiset of the recipe whose output is the tool (axe).
    Used by the recipe-agnostic held-ingredients instrumentation."""
    for r in mech.recipes:
        if mech.tool_item in r["out"]:
            return {labels[k]: c for k, c in r["in"].items()}
    return {}


def _build_world(labels: dict, n: int, gather_prob: float = GATHER_PROB,
                 mech: "Mech" = BASE) -> dict:
    """Assemble a world dict from a latent->token label map + a mechanics spec.
    Everything State touches is a token string (or scalars), so replay needs only
    these labels + the recorded gather_prob + the mech (defaults to BASE)."""
    return {
        "n": n,
        "gather_prob": gather_prob,                        # per-episode p
        "labels": labels,                                  # latent -> token
        "tok2latent": {v: k for k, v in labels.items()},
        "goal_tok": labels[mech.goal_item],
        "tool_tok": labels[mech.tool_item],
        "gather_tok": labels[mech.gather_yield[0]],
        "gather_n": mech.gather_yield[1],
        # inert decoy tokens gather also drops, one-of-k (empty for BASE)
        "decoy_toks": [labels[d] for d in mech.decoy_items],
        # tokens each dropped by an INDEPENDENT Bernoulli(p) draw (empty for BASE)
        "extra_toks": [labels[e] for e in mech.extra_drops],
        # token-keyed input multiset of the tool recipe (held-ingredients check)
        "tool_recipe_in": _tool_recipe_in(labels, mech),
        # token-keyed copies of the recipe table + use yields
        "recipes": [{"in": {labels[k]: c for k, c in r["in"].items()},
                     "out": {labels[k]: c for k, c in r["out"].items()}}
                    for r in mech.recipes],
        "use_yields": {labels[k]: (labels[y], c) for k, (y, c) in mech.use_yields.items()},
    }


def make_world(relabel_seed: int, n: int, gather_prob: float = GATHER_PROB,
               obfuscate: bool = True, mech: "Mech" = BASE) -> dict:
    """Per-episode world: relabel the latent roles uniformly at random. With
    obfuscate=False the latent roles keep their real English names (wood/stick/
    axe) -- an ablation isolating how much of the discovery difficulty is the
    obfuscation itself versus the hidden recipe/payoff structure. `mech` selects
    the game variant (default BASE = the module-global mechanics)."""
    labels = (assign(list(mech.elements), seed=relabel_seed) if obfuscate
              else {e: e for e in mech.elements})
    if obfuscate and mech.family_items:
        labels = _familize(labels, mech, relabel_seed)
    return _build_world(labels, n, gather_prob, mech)


def world_from_labels(labels: dict, n: int, gather_prob: float = GATHER_PROB,
                      mech: "Mech" = BASE) -> dict:
    """Reconstruct a world from the labels recorded in a result/JSONL row,
    bypassing assign(). make_world derives its label strings from the GLOBAL,
    mutable lomekwi.obfuscation.DEFAULT_SCHEME, so a replay that re-derives them
    only matches if that global happens to hold the same scheme the run used.
    Reconstructing from the recorded labels (+ recorded gather_prob) removes that
    coupling: State reads only these, so replay is byte-exact regardless of the
    ambient scheme (mirrors toolworld_v2.world_from_labels). `mech` defaults to
    BASE; variant replays should pass the matching mech (trailing arg keeps the
    positional replay_woodworld call valid)."""
    return _build_world(dict(labels), n, gather_prob, mech)


def _fmt(multiset: dict) -> str:
    """'2 X, 4 Y' rendering of a token->count multiset."""
    return ", ".join(f"{c} {tok}" for tok, c in multiset.items())


@dataclass
class State:
    world: dict
    gather_rng: random.Random
    hint: bool = True
    inv: dict = field(default_factory=dict)        # token -> count held
    turn: int = 0
    # instrumentation
    gather_count: int = 0
    gather_success: int = 0
    craft_attempts: int = 0
    craft_success: int = 0
    use_count: int = 0
    use_axe_count: int = 0
    built_axe: bool = False
    build_turn: int | None = None
    wood_from_gather: int = 0
    wood_from_use: int = 0
    held_ingredients: bool = False                 # ever held the tool recipe's inputs

    # --- helpers ---
    def held(self, tok: str) -> int:
        return self.inv.get(tok, 0)

    def _check_held_ingredients(self):
        """Sticky flag: set once inventory satisfies the tool (axe) recipe's input
        multiset. Recipe-agnostic (reads world['tool_recipe_in']) so it works for
        any variant -- this is the C = P(hold ingredients) acquisition signal."""
        if self.held_ingredients:
            return
        need = self.world["tool_recipe_in"]
        if need and all(self.held(t) >= c for t, c in need.items()):
            self.held_ingredients = True

    def _add(self, tok: str, k: int):
        self.inv[tok] = self.inv.get(tok, 0) + k

    def _consume(self, multiset: dict):
        for tok, c in multiset.items():
            self.inv[tok] = self.inv.get(tok, 0) - c
            if self.inv[tok] <= 0:
                self.inv.pop(tok, None)

    def goal_wood(self) -> int:
        return self.inv.get(self.world["goal_tok"], 0)

    def solved(self) -> bool:
        return self.goal_wood() >= self.world["n"]

    # --- actions ---
    def gather(self) -> str:
        """Stochastic raw-material draw. gather_rng is consumed EXACTLY ONCE per
        call (both branches), and is the ONLY RNG consumer in State -- this is
        what makes stochastic replay byte-exact."""
        self.gather_count += 1
        hit = self.gather_rng.random() < self.world["gather_prob"]
        if hit:
            tok, k = self.world["gather_tok"], self.world["gather_n"]
            self._add(tok, k)
            self.gather_success += 1
            self.wood_from_gather += k
            msg = (f"You gather. You find {k} {tok}. You now hold "
                   f"{self.held(tok)} {tok}.")
        else:
            msg = "You gather, but find nothing this time."
        # inert decoy drop: an INDEPENDENT prob-p draw, gated on decoy_toks so the
        # BASE game (no decoys) keeps the single-RNG-draw invariant and byte-exact
        # replay. One uniformly-chosen decoy; decoys have no recipe/use effect.
        decoys = self.world["decoy_toks"]
        if decoys and self.gather_rng.random() < self.world["gather_prob"]:
            d = self.gather_rng.choice(decoys)
            self._add(d, 1)
            msg += f" You also find 1 {d}. You now hold {self.held(d)} {d}."
        # extra drops: each an INDEPENDENT prob-p draw (a free side-ingredient and/or
        # an inert decoy). Also gated so BASE/decoy-only variants keep their RNG-draw
        # counts and byte-exact replay.
        for tok in self.world["extra_toks"]:
            if self.gather_rng.random() < self.world["gather_prob"]:
                self._add(tok, 1)
                msg += f" You also find 1 {tok}. You now hold {self.held(tok)} {tok}."
        return msg

    def craft(self, toks: list[str]) -> str:
        """Combine EXACTLY two held items. Every recipe takes two inputs, so
        combining is binary: the agent can only ever pair two items, never dump a
        pile. The provided pair of tokens must EQUAL a recipe's input multiset AND
        be held in full (multiset equality, so an unmatched pair never fires). On
        success the obs echoes the exact consumed/produced multisets, so recipe
        counts are learnable from a successful craft; failures are opaque
        ('nothing happens') to keep discovery honest. A non-binary attempt is
        rejected with an explicit message (it still counts as a craft attempt)."""
        self.craft_attempts += 1
        if len(toks) != 2:
            return (f"You can only combine two items at a time (you tried "
                    f"{len(toks)}). Combine exactly two.")
        provided = Counter(toks)
        for r in self.world["recipes"]:
            need = Counter(r["in"])
            if provided == need and all(self.held(t) >= c for t, c in need.items()):
                self._consume(dict(need))
                for t, c in r["out"].items():
                    self._add(t, c)
                self.craft_success += 1
                made_tool = self.world["tool_tok"] in r["out"]
                if made_tool and not self.built_axe:
                    self.built_axe = True
                    self.build_turn = self.turn
                tail = " It persists." if made_tool else ""
                # dynamic hint (cryptic, ToolWorld-style sensory nudge): if the
                # product is itself an ingredient in another recipe (e.g. sticks),
                # describe it as faintly reactive near a DIFFERENT kind -- counters
                # the "I lost progress" misread and the stick-on-stick dead end,
                # without naming the recipe.
                combinable = [o for o in r["out"]
                              if any(o in r2["in"] for r2 in self.world["recipes"])]
                if self.hint and combinable and not made_tool:
                    ct = combinable[0]
                    tail += (f" The {ct} feels faintly active, and the sensation "
                             f"shifts when held alongside other items.")
                return (f"You combine {_fmt(dict(need))} into {_fmt(r['out'])} "
                        f"(new).{tail}")
        return f"You try to combine {', '.join(toks)}: nothing happens."

    def use(self, item: str) -> str:
        """Use (operate/apply) a held item. The tool (axe) yields wood and is NOT
        consumed; anything else does nothing. Never consumes."""
        self.use_count += 1
        if item in self.world["use_yields"]:
            if self.held(item) < 1:
                return f"You don't have a {item}."
            yield_tok, k = self.world["use_yields"][item]
            self._add(yield_tok, k)
            self.use_axe_count += 1
            self.wood_from_use += k
            return (f"You use the {item}. It yields {k} {yield_tok}. (the {item} "
                    f"remains with you.) You now hold {self.held(yield_tok)} "
                    f"{yield_tok}.")
        return f"You use {item}: nothing happens."

    def initial_obs(self, budget: int | None) -> str:
        w = self.world
        s = (
            f"You are in a sealed workshop.\n\n"
            f"GOAL: accumulate {w['n']} {w['goal_tok']} (you currently hold none), "
            f"using as few actions as possible.\n\n"
            f"You may act with these verbs (one action per turn, on its own line):\n"
            f"  gather                - search for raw material\n"
            f"  combine <a> <b>       - combine exactly two items you hold\n"
            f"  use <item>            - operate or apply an item you hold\n"
            f"You hold nothing yet. Things you obtain persist unless stated otherwise.")
        if budget is not None:
            s += (f"\n\nYou have a STRICT BUDGET of {budget} actions total. If you "
                  f"do not hold {w['n']} {w['goal_tok']} within {budget} actions, "
                  f"you fail. Spend them wisely.")
        if self.hint:
            s += "\n\nHint: some items transform when the right ones are combined."
        return s


ACT = re.compile(r"^\s*(?:>?\s*)?(gather|combine|craft|use)\b[\s:]*(.*)\s*$", re.IGNORECASE)


def _clean(t: str) -> str:
    return re.sub(r"^(a|an|the)\s+", "", t.strip().lower())


def parse(line: str):
    m = ACT.match(line.strip())
    if not m:
        return None
    v, rest = m.group(1).lower(), m.group(2).strip()
    if v == "gather":
        return ("gather",)
    # tokenize rest: numbers are counts for the following name, names are items
    rest = re.sub(r"[,+]|\bwith\b|\band\b", " ", rest, flags=re.IGNORECASE)
    raw = [x for x in rest.split() if x]
    if v == "use":
        names = [_clean(x) for x in raw if not x.isdigit()]
        return ("use", names[0]) if names else None
    if v in ("craft", "combine"):
        items: list[str] = []
        pending = 1
        for tok in raw:
            if tok.isdigit():
                pending = int(tok)
            else:
                name = _clean(tok)
                if name:
                    items.extend([name] * pending)
                pending = 1
        return ("combine", items) if len(items) >= 2 else None
    return None


def extract(text: str):
    for ln in text.splitlines():
        a = parse(ln)
        if a:
            return a
    return None


SYS = ("You are an agent in a workshop and must accumulate a target item using as "
       "few actions as possible. End each turn with exactly one action on its own "
       "line: gather, combine <a> <b> (exactly two items), or use <item>. Think "
       "briefly, then act. Minimize total actions.")


async def run(model: str, n: int = 16, relabel_seed: int = 0, gather_seed: int = 0,
              hint: bool = True, max_turns: int = 120, budget: int | None = None,
              no_progress_window: int | None = None, stop_on_build: bool = False,
              gather_prob: float = GATHER_PROB, obfuscate: bool = True,
              mech: "Mech" = BASE,
              world_override: dict | None = None, forced_prefix: list | None = None):
    """One woodworld episode. Mirrors toolworld_v2.run.

    gather_seed seeds the stochastic gather (analog of toolworld drop_seed); the
    seeding formula matches toolworld so replay reconstructs identical draws.
    gather_prob is the per-episode success probability p (region sweeps vary it;
    the caller is responsible for budget = budget_for(n, p)).
    budget: strict cap on total actions (the agent is told the remaining count).
    no_progress_window/stop_on_build/world_override/forced_prefix mirror toolworld.
    """
    load_dotenv()
    client = RawChat()
    world = (world_override if world_override is not None
             else make_world(relabel_seed, n, gather_prob, obfuscate, mech))
    s = State(world=world, gather_rng=random.Random(gather_seed * 6151 + 1), hint=hint)

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

    goal_tok = world["goal_tok"]

    def dispatch(act):
        if act[0] == "gather":
            obs = s.gather()
        elif act[0] in ("craft", "combine"):
            obs = s.craft(act[1])
        else:
            obs = s.use(act[1])
        s._check_held_ingredients()    # recipe-agnostic acquisition signal (C)
        return obs

    # --- forced prefix: replay actions with NO API calls (efficiency sweeps) ---
    if forced_prefix:
        for action, atext in forced_prefix:
            s.turn = len(trace)
            obs = dispatch(action)
            trace.append({"turn": len(trace), "action": list(action), "obs": obs,
                          "agent_text": atext, "usage": None})
            opened = s.goal_wood()
            left = f", {budget - len(trace)} actions left" if budget is not None else ""
            msgs += [{"role": "assistant", "content": atext},
                     {"role": "user", "content": obs +
                      f"\n\n[{opened}/{n} {goal_tok}{left}] What next?"}]

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
                     {"role": "user", "content": "No parseable action. Use "
                      "gather / combine <a> <b> / use <item>, one action on its own line."}]
            continue
        noop = 0

        prev_types = len(s.inv)
        prev_wood = s.goal_wood()
        prev_built = s.built_axe
        prev_use_axe = s.use_axe_count
        obs = dispatch(act)
        if s.use_axe_count > prev_use_axe and t_star is None:
            t_star = len(trace)            # first turn the tool is operated

        done = s.solved()
        wood = s.goal_wood()
        print(f"  [t{t+1}] {act} -> {obs[:74]} [{wood}/{n}]", flush=True)
        trace.append({"turn": t, "action": list(act), "obs": obs, "agent_text": text,
                      "usage": client.last_usage})

        progressed = (len(s.inv) > prev_types or wood > prev_wood
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
                      (f"\n\nYou hold {wood} {goal_tok}. Done." if done
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
                  f"\n\n[{wood}/{n} {goal_tok}{left}] What next?"}]

    stopped_reason = stopped_reason or "max_turns"

    # action efficiency vs oracle (lazy import to avoid a circular dependency).
    # brute_expected MUST use the episode's actual p -- with the default p=0.8 the
    # grind baseline is wrong off the default cell (it understates grind cost at low
    # p), making solved low-p episodes look absurdly inefficient.
    action_efficiency = None
    if s.solved():
        from scripts.validate_woodworld import brute_expected, oracle_min
        be, om = brute_expected(n, world["gather_prob"]), oracle_min(n)
        if be > om:
            action_efficiency = (be - len(trace)) / (be - om)

    result = {
        "model": model, "task": "woodworld", "n": n, "N": n,
        "relabel_seed": relabel_seed, "gather_seed": gather_seed, "hint": hint,
        "obfuscate": obfuscate, "gather_prob": world["gather_prob"],
        "labels": world["labels"],
        "solved": s.solved(), "total_actions": len(trace), "budget": budget,
        "built_axe": s.built_axe, "build_turn": s.build_turn, "t_star": t_star,
        "held_ingredients": s.held_ingredients,
        "gather_count": s.gather_count, "gather_success": s.gather_success,
        "craft_attempts": s.craft_attempts, "craft_success": s.craft_success,
        "use_count": s.use_count, "use_axe_count": s.use_axe_count,
        "wood_by_source": {"gather": s.wood_from_gather, "use": s.wood_from_use},
        "final_wood": s.goal_wood(),
        "noop_total": noop_total, "refusals": refusals, "unparsed": unparsed,
        "stopped_reason": stopped_reason, "action_efficiency": action_efficiency,
        "usage": usage_tot,
    }
    print(f"\n  RESULT {model} N={n} seed=({relabel_seed},{gather_seed}): "
          f"solved={result['solved']} actions={len(trace)}/{budget} "
          f"built_axe={s.built_axe} build_turn={s.build_turn} t*={t_star} "
          f"axe_uses={s.use_axe_count} reason={stopped_reason}", flush=True)
    return result, trace


async def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-4-5-20251001"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    hint = "--no-hint" not in sys.argv
    from scripts.validate_woodworld import budget_for
    budget = budget_for(n)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / f"woodworld_{model.replace('/', '_')}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    result, trace = await run(model, n=n, relabel_seed=0, gather_seed=0, hint=hint,
                              budget=budget, max_turns=300)
    (out / "result.json").write_text(json.dumps({"result": result, "trace": trace},
                                                indent=2))
    print(f"\nSaved: {out/'result.json'}")


if __name__ == "__main__":
    asyncio.run(main())
