"""Family kit for the tool-amortization benchmark.

A registry of distinct, parameterized OPERATION families. Each family:
  - OWNS its exact-integer `reference` -> the gold is defined by the operation we state, so there
    is NO hidden-units / formula ambiguity (the failure mode A0 found in raw CREATOR items).
  - has a `sampler(rng, m)` with PER-ARG roles: `scale` args are resampled large (the magnitude
    knob that cracks hand-arithmetic), `count`/`exponent` args stay in a controlled modest range
    (so the procedure is well-defined and the gold doesn't blow up).
  - carries several surface-varied `covers` (cover stories) that all encode the SAME operation, so
    recognizing that two problems share a procedure takes real recognition, not template-matching.

Heterogeneity is the point: products/dot-products are hard BY MAGNITUDE; LCG/modpow are hard BY
STRUCTURE (iterate K steps / repeated squaring, no closed form). No single script solves the set,
so the optimal "build one tool and reuse it" strategy is NOT handed to the model.

Members are emitted in the same dict shape the A0 harness consumes (family, magnitude, keys,
inputs, vals_order, gold, question), so a0_oracle_gap can grade them unchanged.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.family_kit   # self-test
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

_INT_LO, _INT_HI = 137, 9973  # base resample range at magnitude m=1 (matches dataset.py)


def _scale(rng: random.Random, m: int) -> int:
    """A large integer, size scaled by integer magnitude m. The hand-difficulty knob."""
    return rng.randint(_INT_LO * m, _INT_HI * m)


@dataclass
class Family:
    name: str
    arg_names: list[str]                       # signature order; UPPERCASE (also the prompt keys)
    sampler: Callable[[random.Random, int], dict]   # rng, m -> {ARG: value}
    reference: Callable[[dict], int]           # values-dict -> exact integer gold
    covers: list[str]                          # str.format templates over fields()
    fields: Callable[[dict], dict]             # values-dict -> format kwargs (renders lists nicely)
    hard_by: str                               # "magnitude" | "structure" (documentation)

    def make_member(self, rng: random.Random, m: int) -> dict:
        vals = self.sampler(rng, m)
        gold = self.reference(vals)
        assert isinstance(gold, int), f"{self.name}: reference returned non-int {gold!r}"
        cover = rng.choice(self.covers)
        return {
            "family": self.name,
            "magnitude": m,
            "keys": list(self.arg_names),
            "inputs": {k: vals[k] for k in self.arg_names},
            "vals_order": [vals[k] for k in self.arg_names],
            "gold": gold,
            "question": cover.format(**self.fields(vals)),
        }


# --------------------------------------------------------------------- 1. k-factor product
def _product3_sampler(rng, m):
    return {"A": _scale(rng, m), "B": _scale(rng, m), "C": _scale(rng, m)}


PRODUCT3 = Family(
    name="product3",
    arg_names=["A", "B", "C"],
    sampler=_product3_sampler,
    reference=lambda v: v["A"] * v["B"] * v["C"],
    hard_by="magnitude",
    fields=lambda v: v,
    covers=[
        "A warehouse has {A} aisles. Each aisle holds {B} racks, and each rack stores {C} "
        "cartons. How many cartons are in the warehouse in total?",
        "A data center runs {A} clusters; each cluster has {B} nodes; and each node runs {C} "
        "worker processes. What is the total number of worker processes?",
        "A farm plants {A} fields, each field has {B} rows, and each row contains {C} plants. "
        "How many plants are there altogether?",
        "A factory operates {A} lines for {B} shifts each, producing {C} units per shift. What "
        "is the total number of units produced?",
    ],
)


# --------------------------------------------------------------------- 2. weighted sum (dot product)
def _dot_sampler(rng, m):
    n = rng.randint(3, 6)
    return {"QUANTITIES": [_scale(rng, m) for _ in range(n)],
            "PRICES": [_scale(rng, m) for _ in range(n)]}


def _dot_fields(v):
    qs = ", ".join(str(x) for x in v["QUANTITIES"])
    ps = ", ".join(str(x) for x in v["PRICES"])
    return {"QTYS": f"[{qs}]", "PRICES": f"[{ps}]"}


WEIGHTED_SUM = Family(
    name="weighted_sum",
    arg_names=["QUANTITIES", "PRICES"],
    sampler=_dot_sampler,
    reference=lambda v: sum(q * p for q, p in zip(v["QUANTITIES"], v["PRICES"])),
    hard_by="magnitude",
    fields=_dot_fields,
    covers=[
        "A shipment has several product lines. The quantities per line are {QTYS} and the matching "
        "unit prices are {PRICES}. What is the total value, i.e. the sum over lines of quantity "
        "times unit price?",
        "A portfolio holds these share counts: {QTYS}, at these per-share prices: {PRICES}. What "
        "is the total portfolio value (sum of shares times price across holdings)?",
        "Employees logged these hours: {QTYS}, paid at these hourly rates: {PRICES}. What is the "
        "total wage bill (sum over employees of hours times rate)?",
    ],
)


# --------------------------------------------------------------------- 3. iterated LCG
def _lcg_sampler(rng, m):
    return {"SEED": _scale(rng, m), "A": _scale(rng, m), "B": _scale(rng, m),
            "M": _scale(rng, m), "K": rng.randint(6, 14)}


def _lcg_ref(v):
    x = v["SEED"] % v["M"]
    for _ in range(v["K"]):
        x = (v["A"] * x + v["B"]) % v["M"]
    return x


LCG = Family(
    name="lcg",
    arg_names=["SEED", "A", "B", "M", "K"],
    sampler=_lcg_sampler,
    reference=_lcg_ref,
    hard_by="structure",
    fields=lambda v: v,
    covers=[
        "A register starts at X = {SEED}. It is updated {K} times; each update replaces X with "
        "(({A} * X) + {B}) mod {M}. Report the final value of X.",
        "A pseudorandom counter begins at {SEED}. For {K} rounds, set X = ({A} * X + {B}) mod "
        "{M}. What is X after the final round?",
        "Initialize s = {SEED}. Repeat the following {K} times: s = ({A} * s + {B}) % {M}. Give "
        "the resulting value of s.",
    ],
)


# --------------------------------------------------------------------- 4. modular power
def _modpow_sampler(rng, m):
    return {"BASE": _scale(rng, m), "EXP": rng.randint(12, 40), "MOD": _scale(rng, m)}


MODPOW = Family(
    name="modpow",
    arg_names=["BASE", "EXP", "MOD"],
    sampler=_modpow_sampler,
    reference=lambda v: pow(v["BASE"], v["EXP"], v["MOD"]),
    hard_by="structure",
    fields=lambda v: v,
    covers=[
        "Compute {BASE} raised to the power {EXP}, modulo {MOD} (the remainder when {BASE}^{EXP} "
        "is divided by {MOD}).",
        "A key-exchange step needs g^x mod p with g = {BASE}, x = {EXP}, and p = {MOD}. What is "
        "the result?",
        "Find the remainder when {BASE} to the {EXP}th power is divided by {MOD}.",
    ],
)


# ===================================================================== one-off pool
# Distinct owned procedures used as GENUINE one-offs (each appears once per seed; no two share a
# tool). All magnitude-sensitive so 'easy' (low m -> hand-feasible) vs 'hard' (high m -> not) is a
# knob. Kept disjoint in KIND from the recurring families so a recurring tool never serves a one-off.
def _list_sampler(lo: int, hi: int):
    def s(rng, m):
        return {"VALUES": [_scale(rng, m) for _ in range(rng.randint(lo, hi))]}
    return s


def _vals_field(v):
    return {"VALS": "[" + ", ".join(str(x) for x in v["VALUES"]) + "]"}


def _scalar_sampler(*names):
    def s(rng, m):
        return {n: _scale(rng, m) for n in names}
    return s


LIST_SUM = Family("list_sum", ["VALUES"], _list_sampler(4, 8),
                  reference=lambda v: sum(v["VALUES"]), covers=[
                      "The meter logged these readings: {VALS}. What is their sum?",
                      "Add up the following amounts: {VALS}. What is the total?"],
                  fields=_vals_field, hard_by="magnitude")

ALT_SUM = Family("alt_sum", ["VALUES"], _list_sampler(4, 8),
                 reference=lambda v: sum(x if i % 2 == 0 else -x for i, x in enumerate(v["VALUES"])),
                 covers=["Given the sequence {VALS}, compute the alternating sum (first minus "
                         "second plus third, minus fourth, and so on).",
                         "Evaluate v1 - v2 + v3 - v4 ... for these values: {VALS}."],
                 fields=_vals_field, hard_by="magnitude")

SUM_SQ = Family("sum_of_squares", ["VALUES"], _list_sampler(3, 6),
                reference=lambda v: sum(x * x for x in v["VALUES"]), covers=[
                    "For the values {VALS}, compute the sum of their squares.",
                    "Square each of these and add the results: {VALS}."],
                fields=_vals_field, hard_by="magnitude")

SUM_CUBE = Family("sum_of_cubes", ["VALUES"], _list_sampler(3, 5),
                  reference=lambda v: sum(x ** 3 for x in v["VALUES"]), covers=[
                      "For the values {VALS}, compute the sum of their cubes.",
                      "Cube each of these and add the results: {VALS}."],
                  fields=_vals_field, hard_by="magnitude")

DIFF_PROD = Family("diff_of_products", ["A", "B", "C", "D"], _scalar_sampler("A", "B", "C", "D"),
                   reference=lambda v: v["A"] * v["B"] - v["C"] * v["D"], covers=[
                       "Compute {A} times {B}, then subtract {C} times {D}. What is the result?",
                       "What is ({A} * {B}) - ({C} * {D})?"],
                   fields=lambda v: v, hard_by="magnitude")

TWO_STAGE = Family("two_stage", ["A", "B", "C"], _scalar_sampler("A", "B", "C"),
                   reference=lambda v: (v["A"] + v["B"]) * v["C"], covers=[
                       "Add {A} and {B}, then multiply the result by {C}. What do you get?",
                       "Compute ({A} + {B}) * {C}."],
                   fields=lambda v: v, hard_by="magnitude")

COMBINED_BILL = Family("combined_bill", ["RATE", "QTY", "FEE"], _scalar_sampler("RATE", "QTY", "FEE"),
                       reference=lambda v: v["RATE"] * v["QTY"] + v["FEE"], covers=[
                           "A service charges {RATE} per unit for {QTY} units, plus a flat fee of "
                           "{FEE}. What is the total cost?",
                           "Compute {RATE} * {QTY} + {FEE}."],
                       fields=lambda v: v, hard_by="magnitude")

QUAD = Family("quad_eval", ["A", "B", "C", "X"], _scalar_sampler("A", "B", "C", "X"),
              reference=lambda v: v["A"] * v["X"] * v["X"] + v["B"] * v["X"] + v["C"], covers=[
                  "Evaluate the polynomial {A}*x^2 + {B}*x + {C} at x = {X}.",
                  "For x = {X}, compute {A}*x*x + {B}*x + {C}."],
              fields=lambda v: v, hard_by="magnitude")

CUBIC = Family("cubic_eval", ["A", "B", "C", "D", "X"], _scalar_sampler("A", "B", "C", "D", "X"),
               reference=lambda v: (v["A"] * v["X"] ** 3 + v["B"] * v["X"] ** 2
                                    + v["C"] * v["X"] + v["D"]), covers=[
                   "Evaluate the polynomial {A}*x^3 + {B}*x^2 + {C}*x + {D} at x = {X}.",
                   "For x = {X}, compute {A}*x*x*x + {B}*x*x + {C}*x + {D}."],
               fields=lambda v: v, hard_by="magnitude")

SUM_4TH = Family("sum_of_fourth_powers", ["VALUES"], _list_sampler(3, 5),
                 reference=lambda v: sum(x ** 4 for x in v["VALUES"]), covers=[
                     "For the values {VALS}, compute the sum of their fourth powers.",
                     "Raise each of these to the fourth power and add the results: {VALS}."],
                 fields=_vals_field, hard_by="magnitude")


FAMILIES: dict[str, Family] = {f.name: f for f in (PRODUCT3, WEIGHTED_SUM, LCG, MODPOW)}
ONE_OFF_POOL: dict[str, Family] = {f.name: f for f in (
    LIST_SUM, ALT_SUM, SUM_SQ, SUM_CUBE, DIFF_PROD, TWO_STAGE, COMBINED_BILL, QUAD,
    CUBIC, SUM_4TH)}
ALL_FAMILIES: dict[str, Family] = {**FAMILIES, **ONE_OFF_POOL}

# Hand-solvability regimes, MEASURED by A0 for Haiku (runs/a0_oneoff_pool_haiku). Used to make the
# one-off difficulty knob op-aware instead of uniform-magnitude (m=1 is NOT hand-easy for
# squaring/cubing). NOTE Haiku-calibrated: stronger models fail these later -> need bigger m.
EASY_ONEOFFS = ["list_sum", "alt_sum", "combined_bill", "two_stage"]   # a_hand~1.0 up to m=100
HARD_ONEOFFS = ["sum_of_squares", "sum_of_cubes", "quad_eval",         # a_hand~0 at m=100
                "cubic_eval", "sum_of_fourth_powers"]                  # (both a_hand 0.0 @ m=10&100)
HARD_ONEOFF_MAGNITUDE = 100   # magnitude at which the HARD_ONEOFFS are hand-infeasible for Haiku


# --------------------------------------------------------------------- self-test (no model calls)
def _flat_nums(inputs: dict) -> list:
    out = []
    for x in inputs.values():
        out.extend(x if isinstance(x, list) else [x])
    return out


def _selftest():
    rng = random.Random(0)
    print(f"recurring families: {list(FAMILIES)}\none-off pool: {list(ONE_OFF_POOL)}\n")
    for tag, reg in (("RECURRING", FAMILIES), ("ONE-OFF", ONE_OFF_POOL)):
        print(f"================ {tag} ================")
        for fam in reg.values():
            for m in (1, 100):
                mem = fam.make_member(rng, m)
                g = mem["gold"]
                assert isinstance(g, int), f"{fam.name}: non-int gold {g!r}"
                present = all(str(x) in mem["question"] for x in _flat_nums(mem["inputs"]))
                assert present, f"{fam.name}: rendered numbers missing from question"
                if m == 100:
                    print(f"  {fam.name:<18} m={m:>3} gold={g} ({len(str(abs(g)))}d)  "
                          f"Q: {mem['question'][:60]}")
    print("\nself-test OK")


if __name__ == "__main__":
    _selftest()
