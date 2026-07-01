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
from math import gcd
from typing import Callable

_INT_LO, _INT_HI = 137, 9973  # base resample range at magnitude m=1 (matches dataset.py)


def _scale(rng: random.Random, m: int) -> int:
    """A large integer, size scaled by integer magnitude m. The hand-difficulty knob."""
    return rng.randint(_INT_LO * m, _INT_HI * m)


# ------------------------------------------------------------------ per-model difficulty profile
# "moderate a_hand" is model-specific (stronger models need harder settings to sit in the band).
# The FAMILY SET is identical across models; only the difficulty knobs shift. Samplers read the
# module-global _PROFILE. Callers set it once, from the (single) model, BEFORE generating problems
# (a0_oracle_gap builds instances shared across models, so run one model per grid when profiles
# differ). Any family with no branch for the active profile falls back to the "haiku" tuning.
_PROFILES = ("haiku", "sonnet", "opus")
_PROFILE = "haiku"


def set_profile(name: str) -> None:
    global _PROFILE
    key = (name or "").lower()
    _PROFILE = key if key in _PROFILES else "haiku"


def profile() -> str:
    return _PROFILE


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
    # n=3-4 (was 3-6): fewer big products keeps a_hand in the moderate band for Haiku (n=3-6 @ m=1
    # was 0.33 -- too hard). Magnitude m still cracks it for stronger models.
    n = rng.randint(3, 4)
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
    # MID regime for Haiku (a<=30, 3-4 digit M, K 6-12). Stronger models need bigger M (harder mod
    # per step) and more steps (K) so error accumulates.
    ahi, mlo, mhi, klo, khi = {"sonnet": (300, 10000, 80000, 16, 26),
                               "opus": (900, 50000, 400000, 22, 32)}.get(_PROFILE, (30, 300, 1200, 6, 12))
    return {"SEED": rng.randint(50, 300), "A": rng.randint(2, ahi) * m, "B": rng.randint(1, ahi) * m,
            "M": rng.randint(mlo, mhi) * m, "K": rng.randint(klo, khi)}


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

# internal m*2: two 5-6 digit products then a subtraction -> moderate for Haiku (plain m=1 was 1.00,
# too easy: two 4-digit products is trivial). Still m-responsive for the A0 grid.
def _diffprod_sampler(rng, m):
    return {k: _scale(rng, m * 5) for k in ("A", "B", "C", "D")}


DIFF_PROD = Family("diff_of_products", ["A", "B", "C", "D"], _diffprod_sampler,
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

# X kept to 2 digits so X^2 stays ~4-digit and the A*X^2 term is ~8-digit -> moderate for Haiku
# (full-scale X @ m=1 gave a_hand ~0.17, too near hand-infeasible). A/B/C carry the magnitude knob.
def _quad_sampler(rng, m):
    return {"A": _scale(rng, m), "B": _scale(rng, m), "C": _scale(rng, m),
            "X": rng.randint(100, 999)}


QUAD = Family("quad_eval", ["A", "B", "C", "X"], _quad_sampler,
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


# ===================================================================== EXPANDED REGISTRY (v2)
# Added to grow the vocabulary of DISTINCT tool-identities so N~80 streams have many independent
# build-decisions (see docs/online-tool-investment-plan.md). Each is owned + exact-integer. A
# handful share a GENERALIZATION CLUSTER (one tool serves them all) -> stream_builder enforces at
# most one class per cluster across BOTH roles, so no script is ever reusable across two classes.

# ---- new recurring-eligible (worth a tool) --------------------------------------------------
def _horner_sampler(rng, m):
    deg = rng.randint(4, 5)
    return {"COEFFS": [_scale(rng, m) for _ in range(deg + 1)], "X": _scale(rng, m)}


def _horner_ref(v):
    r = 0
    for c in reversed(v["COEFFS"]):
        r = r * v["X"] + c
    return r


HORNER_POLY = Family("horner_poly", ["COEFFS", "X"], _horner_sampler, reference=_horner_ref,
                     covers=[
                         "Evaluate the polynomial with coefficients {COEFFS} (constant term first) "
                         "at x = {X}: compute c0 + c1*x + c2*x^2 + ... .",
                         "A polynomial has coefficients {COEFFS} in increasing order of degree. "
                         "What is its value at x = {X}?"],
                     fields=lambda v: {"COEFFS": "[" + ", ".join(map(str, v["COEFFS"])) + "]",
                                       "X": v["X"]}, hard_by="magnitude")


def _matvec_sampler(rng, m):
    # fixed 2x2 with 3-digit entries (full _scale 4-digit @ m=1 gave a_hand 0.20, too hard): 4 smaller
    # products keeps it moderate for Haiku. Entry size scales with m for stronger models.
    r, c = 2, 2
    ent = lambda: rng.randint(50 * m, 1500 * m)
    return {"MATRIX": [[ent() for _ in range(c)] for _ in range(r)],
            "VECTOR": [ent() for _ in range(c)]}


def _matvec_ref(v):
    vec = v["VECTOR"]
    return sum(sum(row[j] * vec[j] for j in range(len(vec))) for row in v["MATRIX"])


def _matvec_fields(v):
    mat = "[" + ", ".join("[" + ", ".join(map(str, row)) + "]" for row in v["MATRIX"]) + "]"
    return {"MATRIX": mat, "VECTOR": "[" + ", ".join(map(str, v["VECTOR"])) + "]"}


MATVEC_SUM = Family("matvec_sum", ["MATRIX", "VECTOR"], _matvec_sampler, reference=_matvec_ref,
                    covers=[
                        "Multiply the matrix {MATRIX} by the vector {VECTOR}, then report the sum "
                        "of the entries of the resulting vector.",
                        "For matrix M = {MATRIX} and vector v = {VECTOR}, compute M*v and give the "
                        "total of all components of the result."],
                    fields=_matvec_fields, hard_by="magnitude")


def _pairwise_ref(v):
    xs = v["VALUES"]
    return sum(xs[i] * xs[j] for i in range(len(xs)) for j in range(i + 1, len(xs)))


# 3-4 values -> 3-6 pairwise products (was 4-6 values = 6-15 products, a_hand 0.00 @ m=1, too hard).
PAIRWISE_PROD_SUM = Family("pairwise_prod_sum", ["VALUES"], _list_sampler(3, 4),
                           reference=_pairwise_ref, covers=[
                               "For the values {VALS}, compute the sum of the products of every "
                               "distinct pair (i<j).",
                               "Given {VALS}, add up x_i * x_j over all pairs with i < j."],
                           fields=_vals_field, hard_by="magnitude")


def _fibmod_sampler(rng, m):
    return {"X0": _scale(rng, m), "X1": _scale(rng, m), "M": _scale(rng, m), "K": rng.randint(8, 16)}


def _fibmod_ref(v):
    a, b, mod = v["X0"] % v["M"], v["X1"] % v["M"], v["M"]
    for _ in range(v["K"]):
        a, b = b, (a + b) % mod
    return a


FIB_MOD = Family("fib_mod", ["X0", "X1", "M", "K"], _fibmod_sampler, reference=_fibmod_ref,
                 covers=[
                     "A sequence starts x0 = {X0}, x1 = {X1}. Each later term is "
                     "(previous + one-before-that) mod {M}. Report x_{K} (advance {K} steps).",
                     "Let a = {X0}, b = {X1}. Repeat {K} times: (a, b) <- (b, (a + b) mod {M}). "
                     "Give the final value of a."],
                 fields=lambda v: v, hard_by="structure")


def _collatz_sampler(rng, m):
    return {"N": _scale(rng, m), "K": rng.randint(10, 25)}


def _collatz_ref(v):
    n = v["N"]
    for _ in range(v["K"]):
        n = n // 2 if n % 2 == 0 else 3 * n + 1
    return n


COLLATZ_STEPS = Family("collatz_steps", ["N", "K"], _collatz_sampler, reference=_collatz_ref,
                       covers=[
                           "Start with n = {N}. Apply the Collatz step {K} times: if n is even "
                           "n <- n/2, else n <- 3n+1. Report n after {K} steps.",
                           "Run {K} iterations from {N}, where each even value halves and each odd "
                           "value maps to 3n+1. What value do you end on?"],
                       fields=lambda v: v, hard_by="structure")


def _digitsq_sampler(rng, m):
    return {"N": _scale(rng, m), "K": rng.randint(5, 10)}


def _digitsq_ref(v):
    n = v["N"]
    for _ in range(v["K"]):
        n = sum(int(d) ** 2 for d in str(n))
    return n


DIGITSQ_ITER = Family("digitsq_iter", ["N", "K"], _digitsq_sampler, reference=_digitsq_ref,
                      covers=[
                          "Start with n = {N}. Repeat {K} times: replace n with the sum of the "
                          "squares of its decimal digits. Report the final n.",
                          "From {N}, apply {K} times the map n -> (sum of squared digits of n). "
                          "What is the result?"],
                      fields=lambda v: v, hard_by="structure")


def _gcd_sampler(rng, m):
    # Haiku moderate at 3-4 values x 4-digit multiples. Stronger models: bigger cofactors (harder
    # Euclid steps) and more values in the chain.
    gmult, nlo, nhi = {"sonnet": (60, 5, 7), "opus": (100, 5, 6)}.get(_PROFILE, (1, 3, 4))
    g = rng.randint(2, 999)
    return {"VALUES": [g * _scale(rng, m * gmult) for _ in range(rng.randint(nlo, nhi))]}


def _gcd_ref(v):
    r = 0
    for x in v["VALUES"]:
        r = gcd(r, x)
    return r


EUCLID_GCD_CHAIN = Family("euclid_gcd_chain", ["VALUES"], _gcd_sampler, reference=_gcd_ref,
                          covers=[
                              "Compute the greatest common divisor of all of these numbers: {VALS}.",
                              "What is the GCD of the list {VALS}?"],
                          fields=_vals_field, hard_by="structure")


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def _factmod_sampler(rng, m):
    # MOD must be PRIME and > N, else N! mod MOD is almost always 0 (composite moduli are swamped by
    # the factorial's factors) -> degenerate golds. A prime > N guarantees a genuine nonzero result.
    # Stronger models: more multiply-mod steps (bigger N) and a bigger prime modulus.
    (nlo, nhi), modmult = {"sonnet": ((30, 60), 5), "opus": ((45, 80), 20)}.get(_PROFILE, ((15, 40), 1))
    n = rng.randint(nlo, nhi)
    lo, hi = _INT_LO * m * modmult, _INT_HI * m * modmult
    while True:
        cand = rng.randint(max(lo, n + 1), hi)
        if _is_prime(cand):
            return {"N": n, "MOD": cand}


def _factmod_ref(v):
    r = 1
    for i in range(2, v["N"] + 1):
        r = (r * i) % v["MOD"]
    return r


FACTORIAL_MOD = Family("factorial_mod", ["N", "MOD"], _factmod_sampler, reference=_factmod_ref,
                       covers=[
                           "Compute {N} factorial modulo {MOD} (the remainder when {N}! is divided "
                           "by {MOD}).",
                           "What is ({N}!) mod {MOD}?"],
                       fields=lambda v: v, hard_by="structure")


# ---- new one-off-eligible (distinct clusters) -----------------------------------------------
ALT_WEIGHTED_SUM = Family("alt_weighted_sum", ["QUANTITIES", "PRICES"], _dot_sampler, reference=(
    lambda v: sum((1 if i % 2 == 0 else -1) * q * p
                  for i, (q, p) in enumerate(zip(v["QUANTITIES"], v["PRICES"])))), covers=[
        "Given quantities {QTYS} and prices {PRICES}, compute the ALTERNATING weighted sum: "
        "q1*p1 - q2*p2 + q3*p3 - ... .",
        "For lists {QTYS} and {PRICES}, add q_i*p_i with alternating signs (first +, then -)."],
    fields=_dot_fields, hard_by="magnitude")


def _geom_sampler(rng, m):
    # A at m*3 and R>=4: the final A*(series-sum) product is big enough to be error-prone by hand
    # (plain m=1 with small R factored to a trivial single multiply -> a_hand 1.00, too easy).
    return {"A": _scale(rng, m * 3), "R": rng.randint(4, 9), "K": rng.randint(6, 12)}


GEOMETRIC_PARTIAL = Family("geometric_partial", ["A", "R", "K"], _geom_sampler, reference=(
    lambda v: sum(v["A"] * v["R"] ** i for i in range(v["K"] + 1))), covers=[
        "Sum the geometric series A + A*R + A*R^2 + ... + A*R^{K} with A = {A}, R = {R}, K = {K}.",
        "Compute the total of {A} times R^i for i = 0 to {K}, where R = {R}."],
    fields=lambda v: v, hard_by="magnitude")


# operands at m*2: the two chained big multiplications push a_hand off the easy end (plain m=1 gave
# 0.85). Still m-responsive.
def _nested_sampler(rng, m):
    return {k: _scale(rng, m * 2) for k in ("A", "B", "C", "D", "E")}


NESTED_TWO_STAGE = Family("nested_two_stage", ["A", "B", "C", "D", "E"],
                          _nested_sampler, reference=(
    lambda v: ((v["A"] + v["B"]) * v["C"] - v["D"]) * v["E"]), covers=[
        "Compute ((({A} + {B}) * {C}) - {D}) * {E}.",
        "Add {A} and {B}, multiply by {C}, subtract {D}, then multiply the result by {E}."],
    fields=lambda v: v, hard_by="magnitude")


def _base_sampler(rng, m):
    return {"N": _scale(rng, m), "BASE": rng.randint(3, 16)}


def _base_ref(v):
    n, b, s = v["N"], v["BASE"], 0
    while n > 0:
        s += n % b
        n //= b
    return s


BASE_CONVERT_DIGITSUM = Family("base_convert_digitsum", ["N", "BASE"], _base_sampler,
                               reference=_base_ref, covers=[
                                   "Convert {N} to base {BASE} and report the sum of its digits in "
                                   "that base.",
                                   "Write {N} in base {BASE}; what is the total of the base-{BASE} "
                                   "digits?"],
                               fields=lambda v: v, hard_by="structure")


def _reset_sampler(rng, m):
    # 12-16 big values (was 6-10, a_hand 1.00): a long running sum with a reset-on-exceed rule
    # accumulates errors over the extra additions and reset decisions.
    vals = [_scale(rng, m) for _ in range(rng.randint(12, 16))]
    return {"VALUES": vals, "T": int(2.2 * (sum(vals) / len(vals)))}


def _reset_ref(v):
    s = 0
    for x in v["VALUES"]:
        s += x
        if s > v["T"]:
            s = 0
    return s


RUNNING_RESET_ACCUM = Family("running_reset_accum", ["VALUES", "T"], _reset_sampler,
                             reference=_reset_ref, covers=[
                                 "Scan the list {VALS} left to right keeping a running total; after "
                                 "adding each value, if the total exceeds {T} reset it to 0. Report "
                                 "the final total.",
                                 "Accumulate {VALS} in order; whenever the running sum goes above "
                                 "{T}, it drops back to 0. What is the ending sum?"],
                             fields=lambda v: {"VALS": "[" + ", ".join(map(str, v["VALUES"])) + "]",
                                               "T": v["T"]}, hard_by="structure")


def _checksum_sampler(rng, m):
    return {"VALUES": [_scale(rng, m) for _ in range(rng.randint(4, 7))], "M": _scale(rng, m)}


def _checksum_ref(v):
    return sum((i + 1) * x for i, x in enumerate(v["VALUES"])) % v["M"]


WEIGHTED_CHECKSUM = Family("weighted_checksum", ["VALUES", "M"], _checksum_sampler,
                           reference=_checksum_ref, covers=[
                               "For the list {VALS}, compute the position-weighted sum "
                               "(1*first + 2*second + 3*third + ...), then take it modulo {M}.",
                               "Weight each value in {VALS} by its 1-based index, sum, and report "
                               "the result mod {M}."],
                           fields=lambda v: {"VALS": "[" + ", ".join(map(str, v["VALUES"])) + "]",
                                             "M": v["M"]}, hard_by="structure")


def _cfrac_sampler(rng, m):
    # 3-4 SMALL (2-3 digit) terms for Haiku (the a*p multiply must stay bounded). Stronger models:
    # more terms and larger terms so the growing p*a product is harder.
    tlo, thi, nlo, nhi = {"sonnet": (200, 900, 6, 7), "opus": (400, 2000, 7, 9)}.get(
        _PROFILE, (11, 120, 3, 4))
    return {"TERMS": [rng.randint(tlo * m, thi * m) for _ in range(rng.randint(nlo, nhi))]}


def _cfrac_ref(v):
    p_prev, p = 1, v["TERMS"][0]
    for a in v["TERMS"][1:]:
        p_prev, p = p, a * p + p_prev
    return p


CONTINUED_FRAC = Family("continued_frac", ["TERMS"], _cfrac_sampler, reference=_cfrac_ref, covers=[
    "For the continued fraction with terms {VALS}, compute the numerator of its value (use "
    "p_i = a_i * p_(i-1) + p_(i-2), with p_(-1)=1, p_0 = a_0).",
    "Given continued-fraction terms {VALS}, apply p <- a*p + p_prev across the terms and report "
    "the final numerator p."],
    fields=lambda v: {"VALS": "[" + ", ".join(map(str, v["TERMS"])) + "]"}, hard_by="structure")


def _luhn_ref(v):
    total = 0
    for i, d in enumerate(reversed([int(c) for c in str(v["N"])])):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total


# 13-16 digit number (was a 4-digit _scale, a_hand 0.83): the alternating double / cast-out-nine /
# sum over many digits is error-prone at length. Not magnitude-composable (digit procedure).
def _luhn_sampler(rng, m):
    # digit count is the knob (more digits -> longer alternating double/cast-out/sum). Stronger
    # models need longer numbers.
    dlo, dhi = {"sonnet": (32, 40), "opus": (44, 54)}.get(_PROFILE, (13, 16))
    ndig = rng.randint(dlo, dhi)
    return {"N": rng.randint(10 ** (ndig - 1), 10 ** ndig - 1)}


LUHN_SUM = Family("luhn_sum", ["N"], _luhn_sampler, reference=_luhn_ref, covers=[
    "Apply the Luhn procedure to {N}: reading digits from the right, double every second digit "
    "(subtracting 9 if the result exceeds 9), then sum all the resulting digits. Report the sum.",
    "Compute the Luhn digit-sum of {N} (double every 2nd digit from the right, cast out nines, "
    "add everything up)."], fields=lambda v: v, hard_by="structure")


# ---- new a_hand~=1 "hand-possible but tedious" one-offs, each a DISTINCT computational KIND -----
# (no big multiplication, no modular arithmetic -> stay hand-solvable; core step logic differs so no
#  single script serves two of them: nested-comparison vs sort-select vs tally vs running-state).
def _inversions_ref(v):
    xs = v["VALUES"]
    return sum(1 for i in range(len(xs)) for j in range(i + 1, len(xs)) if xs[i] > xs[j])


def _inversions_sampler(rng, m):
    # list length is the knob (# pairs to compare grows ~n^2). Stronger models need longer lists.
    lo, hi = {"sonnet": (15, 20), "opus": (24, 30)}.get(_PROFILE, (9, 11))
    return {"VALUES": [_scale(rng, m) for _ in range(rng.randint(lo, hi))]}


COUNT_INVERSIONS = Family("count_inversions", ["VALUES"], _inversions_sampler,
                          reference=_inversions_ref, covers=[
                              "In the sequence {VALS}, count the number of inversions: pairs of "
                              "positions (i, j) with i before j but the earlier value LARGER than "
                              "the later one.",
                              "How many out-of-order pairs are in {VALS}? (count pairs where an "
                              "earlier entry exceeds a later entry)"],
                          fields=_vals_field, hard_by="structure")


def _median_sampler(rng, m):
    n = rng.choice([5, 7])
    return {"VALUES": [_scale(rng, m) for _ in range(n)]}


def _median_ref(v):
    xs = sorted(v["VALUES"])
    return xs[len(xs) // 2]


MEDIAN_OF_LIST = Family("median_of_list", ["VALUES"], _median_sampler, reference=_median_ref,
                        covers=[
                            "Sort the values {VALS} and report the median (the middle value).",
                            "What is the median of {VALS}? (order them and take the middle one)"],
                        fields=_vals_field, hard_by="structure")


def _mode_sampler(rng, m):
    pool = [_scale(rng, m) for _ in range(rng.randint(3, 4))]
    return {"VALUES": [rng.choice(pool) for _ in range(rng.randint(7, 9))]}


def _mode_ref(v):
    xs = v["VALUES"]
    return max(xs.count(x) for x in set(xs))


MODE_COUNT = Family("mode_count", ["VALUES"], _mode_sampler, reference=_mode_ref, covers=[
    "In the list {VALS}, how many times does the most frequently occurring value appear?",
    "Find the value that occurs most often in {VALS}; report how many times it appears."],
    fields=_vals_field, hard_by="structure")


def _walk_sampler(rng, m):
    # 12-16 signed 5-digit steps (was 6-10 at m*1, a_hand 1.00): a long running signed sum with a
    # max-abs to track is error-prone at this length/scale.
    return {"STEPS": [rng.choice([-1, 1]) * _scale(rng, m * 3) for _ in range(rng.randint(12, 16))]}


def _walk_ref(v):
    pos, mx = 0, 0
    for s in v["STEPS"]:
        pos += s
        mx = max(mx, abs(pos))
    return mx


NUMBER_LINE_WALK = Family("number_line_walk", ["STEPS"], _walk_sampler, reference=_walk_ref, covers=[
    "Starting at position 0 on a number line, apply these moves in order: {STEPS}. Report the "
    "greatest distance from 0 (absolute position) reached at any point during the walk.",
    "A token starts at 0 and moves by each of {STEPS} in turn. What is the farthest it ever gets "
    "from the origin (maximum absolute position)?"],
    fields=lambda v: {"STEPS": "[" + ", ".join(map(str, v["STEPS"])) + "]"}, hard_by="structure")


# ---- more a_hand~=1 "long-but-easy" families (distinct per-step logic; a script for one CANNOT
#      compute another, and none matches collatz/digitsq/the scan/tally/sort kinds already present) --
def _revadd_sampler(rng, m):
    # more rounds (6-9): n grows to 8-12 digits, so each 'add the digit-reversal' is a long-integer
    # addition and errors compound across rounds.
    return {"N": _scale(rng, m), "K": rng.randint(6, 9)}


def _revadd_ref(v):
    n = v["N"]
    for _ in range(v["K"]):
        n = n + int(str(n)[::-1])
    return n


DIGIT_REVERSE_ADD = Family("digit_reverse_add", ["N", "K"], _revadd_sampler, reference=_revadd_ref,
                           covers=[
                               "Start with n = {N}. Repeat {K} times: add to n the number formed "
                               "by reversing its decimal digits. Report the final n.",
                               "From {N}, apply {K} rounds of 'reverse the digits and add to the "
                               "original'. What value do you end with?"],
                           fields=lambda v: v, hard_by="structure")


def _kaprekar_sampler(rng, m):
    # 6-digit start (was 4-digit): each step's 'largest-arrangement minus smallest-arrangement' is a
    # 6-digit subtraction, fiddly to do by hand across 4-7 rounds.
    return {"N": _scale(rng, m * 100), "K": rng.randint(4, 7)}


def _kaprekar_ref(v):
    n = v["N"]
    for _ in range(v["K"]):
        ds = sorted(str(n))
        n = int("".join(ds[::-1])) - int("".join(ds))
    return n


KAPREKAR_ROUTINE = Family("kaprekar_routine", ["N", "K"], _kaprekar_sampler, reference=_kaprekar_ref,
                          covers=[
                              "Start with n = {N}. Repeat {K} times: rearrange n's digits into the "
                              "largest and smallest numbers possible and replace n with (largest "
                              "minus smallest). Report the final n.",
                              "From {N}, run {K} steps of Kaprekar's routine (biggest-digit-"
                              "arrangement minus smallest). What is the result?"],
                          fields=lambda v: v, hard_by="structure")


def _las_sampler(rng, m):
    # 2-digit start, 2-3 rounds for Haiku (look-and-say grows explosively). Stronger models: +1 round
    # (each extra round roughly doubles the string length to track).
    klo, khi = {"sonnet": (3, 4), "opus": (4, 5)}.get(_PROFILE, (2, 3))
    return {"N": rng.randint(10, 99), "K": rng.randint(klo, khi)}


def _las_ref(v):
    s = str(v["N"])
    for _ in range(v["K"]):
        out, i = [], 0
        while i < len(s):
            j = i
            while j < len(s) and s[j] == s[i]:
                j += 1
            out.append(str(j - i))
            out.append(s[i])
            i = j
        s = "".join(out)
    return int(s)


LOOK_AND_SAY = Family("look_and_say", ["N", "K"], _las_sampler, reference=_las_ref, covers=[
    "Start with the digit string {N}. Apply the 'look-and-say' step {K} times: read off each run "
    "of equal consecutive digits as (count)(digit), left to right. Report the resulting number.",
    "From {N}, perform {K} look-and-say rounds (describe the digit runs as count-then-digit). What "
    "number results?"], fields=lambda v: v, hard_by="structure")


def _maxrun_ref(v):
    xs = v["VALUES"]
    best = cur = 1
    for i in range(1, len(xs)):
        cur = cur + 1 if xs[i] > xs[i - 1] else 1
        best = max(best, cur)
    return best


MAX_RUN_LENGTH = Family("max_run_length", ["VALUES"], _list_sampler(7, 10), reference=_maxrun_ref,
                        covers=[
                            "In the sequence {VALS}, find the length of the longest run of "
                            "consecutive strictly-increasing values.",
                            "What is the longest stretch of consecutive entries in {VALS} that keep "
                            "strictly increasing?"],
                        fields=_vals_field, hard_by="structure")


# ---- HIGH-K test family: difficulty from STEP COUNT (many trivial steps, bounded small state) ----
# each step is tiny arithmetic (numbers < ~120) so a_hand SHOULD stay high, but K~60 steps make
# hand-solving token-EXPENSIVE (the model must emit every step) while a tool call is O(1). Tests
# whether building becomes token-favorable at a_hand~=1 (see docs/online-tool-investment-plan.md).
def _lcgsmall_sampler(rng, m):
    return {"SEED": rng.randint(10, 60), "A": rng.randint(3, 12), "B": rng.randint(3, 12),
            "M": rng.randint(80, 120), "K": rng.randint(50, 65)}


def _lcgsmall_ref(v):
    x = v["SEED"] % v["M"]
    for _ in range(v["K"]):
        x = (v["A"] * x + v["B"]) % v["M"]
    return x


LCG_SMALL = Family("lcg_small", ["SEED", "A", "B", "M", "K"], _lcgsmall_sampler,
                   reference=_lcgsmall_ref, covers=[
                       "A counter starts at X = {SEED}. Repeat {K} times: replace X with "
                       "({A}*X + {B}) mod {M}. Report the final X.",
                       "Start with x = {SEED}. Apply x = ({A}*x + {B}) % {M} exactly {K} times. "
                       "What is the resulting x?"],
                   fields=lambda v: v, hard_by="structure")


# ---- DISTINCT-PRIMITIVE one-off families to reach 6+6. Division and modulo of big integers are NOT
#      composable from any product/dot/poly tool (unlike (A+B)(C+D)... which is just product3 of sums),
#      so these are genuine single-use traps even when product3/dot are recurring. Calibrated via A0.
def _intdiv_sampler(rng, m):
    n = rng.randint(3, 4)
    # numerator scale / divisor range per profile (Sonnet aced 6-digit/3-digit -> 8-digit/5-digit)
    # Sonnet does long division flawlessly at 8-digit/5-digit; crank hard (~10-digit / 6-digit ->
    # many quotient digits per term). Keep in the grid even if it may still not crack.
    mult, blo, bhi = {"sonnet": (1000000, 100003, 999983), "opus": (10000000, 1000003, 9999991)}.get(
        _PROFILE, (50, 101, 997))
    return {"A": [_scale(rng, m * mult) for _ in range(n)], "B": [rng.randint(blo, bhi) for _ in range(n)]}


def _twolist_fields(v):
    return {"A": "[" + ", ".join(map(str, v["A"])) + "]",
            "B": "[" + ", ".join(map(str, v["B"])) + "]"}


INT_DIV_SUM = Family("int_div_sum", ["A", "B"], _intdiv_sampler,
                     reference=lambda v: sum(a // b for a, b in zip(v["A"], v["B"])),
                     covers=[
                         "For A = {A} and B = {B}, compute the sum over positions of the integer "
                         "quotient floor(A_i / B_i) (whole-number division, discarding remainders).",
                         "Divide each A_i by the matching B_i keeping only the whole-number part, "
                         "then add those quotients. A = {A}, B = {B}. What is the total?"],
                     fields=_twolist_fields, hard_by="magnitude")


def _modpair_sampler(rng, m):
    n = rng.randint(3, 4)
    # crank hard for Sonnet (~10-digit values mod 6-digit -> full long division for each remainder).
    mult, blo, bhi = {"sonnet": (1000000, 100003, 999983), "opus": (10000000, 1000003, 9999991)}.get(
        _PROFILE, (50, 1009, 9973))
    return {"A": [_scale(rng, m * mult) for _ in range(n)], "B": [rng.randint(blo, bhi) for _ in range(n)]}


MOD_PAIR_SUM = Family("mod_pair_sum", ["A", "B"], _modpair_sampler,
                      reference=lambda v: sum(a % b for a, b in zip(v["A"], v["B"])),
                      covers=[
                          "For A = {A} and B = {B}, compute the sum over positions of (A_i mod B_i), "
                          "the remainder when A_i is divided by B_i.",
                          "Take each A_i modulo the matching B_i (the remainder after division) and "
                          "add the remainders. A = {A}, B = {B}. What is the total?"],
                      fields=_twolist_fields, hard_by="magnitude")


FAMILIES: dict[str, Family] = {f.name: f for f in (
    PRODUCT3, WEIGHTED_SUM, LCG, MODPOW, LCG_SMALL,
    HORNER_POLY, MATVEC_SUM, PAIRWISE_PROD_SUM,
    FIB_MOD, COLLATZ_STEPS, DIGITSQ_ITER, EUCLID_GCD_CHAIN, FACTORIAL_MOD)}
ONE_OFF_POOL: dict[str, Family] = {f.name: f for f in (
    LIST_SUM, ALT_SUM, SUM_SQ, SUM_CUBE, DIFF_PROD, TWO_STAGE, COMBINED_BILL, QUAD,
    CUBIC, SUM_4TH,
    ALT_WEIGHTED_SUM, GEOMETRIC_PARTIAL, NESTED_TWO_STAGE, BASE_CONVERT_DIGITSUM,
    RUNNING_RESET_ACCUM, WEIGHTED_CHECKSUM, CONTINUED_FRAC, LUHN_SUM,
    COUNT_INVERSIONS, MEDIAN_OF_LIST, MODE_COUNT, NUMBER_LINE_WALK,
    DIGIT_REVERSE_ADD, KAPREKAR_ROUTINE, LOOK_AND_SAY, MAX_RUN_LENGTH,
    INT_DIV_SUM, MOD_PAIR_SUM)}
ALL_FAMILIES: dict[str, Family] = {**FAMILIES, **ONE_OFF_POOL}

# Generalization clusters: procedures a SINGLE reasonable tool would serve. stream_builder allows
# at most one class per cluster across recurring+one-off, so no script transfers between classes.
# Anything not listed is its own singleton cluster (its name).
GEN_CLUSTER: dict[str, str] = {
    "horner_poly": "polyeval", "quad_eval": "polyeval", "cubic_eval": "polyeval",
    "weighted_sum": "dot", "alt_weighted_sum": "dot",
    "list_sum": "listreduce", "alt_sum": "listreduce", "sum_of_squares": "listreduce",
    "sum_of_cubes": "listreduce", "sum_of_fourth_powers": "listreduce",
}


def cluster_of(name: str) -> str:
    """Generalization cluster of a family (defaults to the family name = its own singleton)."""
    return GEN_CLUSTER.get(name, name)

# Hand-solvability regimes, MEASURED by A0 for Haiku (runs/a0_oneoff_pool_haiku). Used to make the
# one-off difficulty knob op-aware instead of uniform-magnitude (m=1 is NOT hand-easy for
# squaring/cubing). NOTE Haiku-calibrated: stronger models fail these later -> need bigger m.
EASY_ONEOFFS = ["list_sum", "alt_sum", "combined_bill", "two_stage"]   # a_hand~1.0 up to m=100
# cluster-diverse hard one-offs (each a distinct gen_cluster so many can coexist in one stream).
# The first two (sum_of_squares/quad_eval) are Haiku-calibrated a_hand~0 @ m=100; the rest are new
# and PROVISIONAL pending A0 (runs/a0_kit_v2_haiku) — prune any that turn out hand-feasible.
HARD_ONEOFFS = ["sum_of_squares", "quad_eval", "diff_of_products", "geometric_partial",
                "nested_two_stage", "alt_weighted_sum", "base_convert_digitsum",
                "running_reset_accum", "weighted_checksum", "continued_frac"]
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
