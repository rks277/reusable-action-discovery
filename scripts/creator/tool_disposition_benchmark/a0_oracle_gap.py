"""A0 — oracle-gap calibration for the tool-amortization benchmark.

Question: on integer-exact CREATOR families, does BUILDING buy accuracy? i.e. is there a
magnitude band where solving BY HAND fails (acc_hand low) but a WRITTEN-AND-EXECUTED tool
succeeds (acc_script ~ 1)? A finite ski-rental break-even m* = C/(h-r) only exists if
acc_script > acc_hand, so this gate must pass before the amortization layer is worth building.

Design (kills the selection confound that made the old effScript/effHand uncomparable):
  - pick a few self-evident, INTEGER-EXACT CREATOR families (no pi / div / float)
  - per family x magnitude, generate K resampled instances with EXACT-INTEGER golds
  - run TWO conditions on the SAME instances:
      forced-hand : model solves mentally, emits `ANSWER: <int>`            -> acc_hand
      forced-build: model emits ONLY a `def solve(...)`; WE execute it       -> acc_script
  - grade by EXACT-INTEGER equality (Python big ints; no sig-fig float fragility)
  - log per-instance tokens (the m* denominator, for later)

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.a0_oracle_gap \
      --models haiku --magnitudes 1 10 100 1000 --k 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.creator_batch import _render, _spans
from scripts.creator.creator_exec import _run, extract_code
from scripts.creator.creator_heldout import parse_inputs
from scripts.creator.tool_disposition_benchmark.dataset import _load_items, render_concrete

CLAUDE = {"haiku": "claude-haiku-4-5-20251001",
          "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}

# Shortlist: self-evident procedure, integer-exact, SCALAR args that are all the same flavour so a
# blanket integer up-scale is semantically valid (multi-factor products). Accumulator/modular
# families (Fibonacci_Sum, pow_mod, recursive_sequence) have STRUCTURAL args — a loop count, a rate,
# an exponent — that blanket scaling corrupts; they need a per-arg resample spec and are deferred.
# pool/adhesive/concrete are MULTIPLICATIVE (3-4 factor products -> 10-28 digit golds, hand-hard);
# total_manufacturing_cost is ADDITIVE (6-term sum -> small gold) = a negative control where building
# should NOT buy accuracy. (construction_cost/electricity_consumption have float formulas -> skipped.)
SHORTLIST = ["pool_water_amount", "adhesive_mixer_power_consumption",
             "concrete_volume", "total_manufacturing_cost"]

# integer-magnitude resample range (dataset.py uses 137..9973 at m=1.0; we scale by integer m)
_INT_LO, _INT_HI = 137, 9973


# --------------------------------------------------------------------- exact-integer helpers
_NUM = re.compile(r"[-+]?\d[\d,_]*\.?\d*(?:[eE][-+]?\d+)?")


def _last_number_str(text: str | None) -> str | None:
    """Last numeric token in text (the reference prints the answer inside prose like
    '... is 1747191750000 liters.'); strips thousands separators."""
    if not text:
        return None
    ms = _NUM.findall(text)
    return ms[-1].replace(",", "").replace("_", "") if ms else None


def _stdout_int(text: str | None):
    """Exact Python int from the last number printed in `text`, or None. Uses Decimal so huge ints
    (full or sci-notation) stay EXACT — never routes through float."""
    s = _last_number_str(text)
    if s is None:
        return None
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    return int(d) if d == d.to_integral_value() else None


def _tool_name(item: dict) -> str:
    m = re.search(r"def\s+(\w+)\s*\(", item.get("tool", ""))
    return m.group(1) if m else ""


def _ref_gold_int(item: dict, vals_lower: dict):
    """Run the reference tool+solution with init values overridden by `vals_lower` (lowercase
    names, as parse_inputs yields); return the EXACT integer gold, or None."""
    sol = extract_code(item["solution"])
    for name, newv in vals_lower.items():
        sol = re.sub(rf"(?m)^(\s*{re.escape(name)}\s*=\s*)[-+]?\d[\d_]*\.?\d*",
                     rf"\g<1>{newv!r}", sol, count=1)
    stdout, err = _run(extract_code(item["tool"]) + "\n\n" + sol)
    return None if err else _stdout_int(stdout)


def _resample_int(value, rng: random.Random, m: int) -> int:
    """A larger same-flavour INTEGER value, size scaled by integer magnitude m."""
    lo, hi = _INT_LO * m, _INT_HI * m
    base = abs(int(value)) or 1
    # keep it in the family's neighbourhood but scaled up; floats are coerced to int
    return rng.randint(max(2, lo), max(lo + 1, hi))


# --------------------------------------------------------------------- instance generation
def make_instances(item: dict, m: int, k: int, seed0: int = 0) -> list[dict]:
    """k integer-input variants of one family at magnitude m, each with an exact-integer gold."""
    inputs = parse_inputs(item["solution"])           # [(name_lower, value)]
    if not inputs:
        return []
    spans = _spans(item["question"], inputs)
    if spans is None:
        return []
    template = _render(item["question"], spans, {n: n.upper() for n, _ in inputs})
    out, seen = [], set()
    rng = random.Random(hash((_tool_name(item), m, seed0)) & 0xFFFFFFFF)
    for _ in range(k * 40):
        if len(out) >= k:
            break
        vals_lower = {n: _resample_int(v, rng, m) for n, v in inputs}
        key = tuple(vals_lower.values())
        if key in seen:
            continue
        seen.add(key)
        gold = _ref_gold_int(item, vals_lower)
        if gold is None:
            continue
        up = {n.upper(): vals_lower[n] for n, _ in inputs}
        out.append({
            "family": _tool_name(item),
            "magnitude": m,
            "keys": [n.upper() for n, _ in inputs],
            "inputs": up,                              # uppercase keys (for the prompt)
            "vals_order": [up[n.upper()] for n, _ in inputs],
            "gold": gold,
            "question": render_concrete(template, up),
        })
    return out


# --------------------------------------------------------------------- prompts
def _hand_sys() -> str:
    return ("You solve a numeric problem BY HAND. Work it out and end with a single line "
            "`ANSWER: <integer>`. The answer is an exact integer. You have no tools.")


def _hand_user(inst: dict) -> str:
    keys = ", ".join(inst["keys"])
    return (f"{inst['question']}\n\nThe named values are: {inst['inputs']!r} (keys: [{keys}]).\n"
            "Give the EXACT integer answer, ending with `ANSWER: <integer>`.")


def _build_sys() -> str:
    return ("You write a Python function to solve a numeric problem. Output ONLY a ```python code "
            "block defining `def solve(...)` that RETURNS the exact integer answer. Do NOT compute "
            "the answer yourself and do NOT print anything — just define the function.")


def _build_user(inst: dict) -> str:
    sig = ", ".join(inst["keys"])
    return (f"{inst['question']}\n\nWrite `def solve({sig})` taking those named values as arguments "
            f"(here they are {inst['inputs']!r}) and returning the exact integer answer. "
            "Output only the code block.")


# --------------------------------------------------------------------- execution / grading
def _exec_solve(code: str, vals_order: list) -> object:
    """Append a positional call to the model's solve and run it in the sandbox; return canon int."""
    body = extract_code(code or "")
    if "def solve" not in body:
        return None
    harness = body + f"\n\nprint(solve(*{vals_order!r}))"
    stdout, err = _run(harness)
    return None if err else _stdout_int(stdout)


async def _call(client, model, system, user, max_tokens):
    """One single-turn call; returns (text, tokens). Reads last_usage right after the await
    (sync, so no concurrent coroutine can overwrite it before we read)."""
    text = await client.chat(model, system, [{"role": "user", "content": user}],
                             max_tokens=max_tokens)
    u = client.last_usage or {}
    # spent tokens = uncached input + output (matches the project's cost convention)
    tokens = int(u.get("input_tokens", 0) or 0) + int(u.get("output_tokens", 0) or 0)
    return text or "", tokens


async def grade_hand(client, model, inst, max_tokens):
    text, tok = await _call(client, model, _hand_sys(), _hand_user(inst), max_tokens)
    m = re.split(r"ANSWER:", text)
    ans = _stdout_int(m[-1]) if len(m) > 1 else None
    return {"condition": "hand", "answer": ans, "correct": ans is not None and ans == inst["gold"],
            "tokens": tok}


async def grade_build(client, model, inst, max_tokens):
    text, tok = await _call(client, model, _build_sys(), _build_user(inst), max_tokens)
    ans = _exec_solve(text, inst["vals_order"])
    return {"condition": "build", "answer": ans, "correct": ans is not None and ans == inst["gold"],
            "tokens": tok}


# --------------------------------------------------------------------- driver
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["haiku"])
    ap.add_argument("--source", choices=["kit", "cc"], default="kit",
                    help="kit = owned family_kit families (unambiguous); cc = raw CREATOR items")
    ap.add_argument("--families", nargs="+", default=None,
                    help="default: all kit families, or the CC SHORTLIST when --source cc")
    ap.add_argument("--magnitudes", type=int, nargs="+", default=[1, 10, 100, 1000])
    ap.add_argument("--k", type=int, default=4, help="instances per (family, magnitude)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    load_dotenv()

    # build instances up front (deterministic; no model calls)
    instances = []
    if args.source == "kit":
        from scripts.creator.tool_disposition_benchmark.family_kit import ALL_FAMILIES as FAMILIES
        fams = args.families or list(FAMILIES)
        for fam in fams:
            f = FAMILIES.get(fam)
            if f is None:
                print(f"WARNING: kit family not found: {fam}")
                continue
            for m in args.magnitudes:
                rng = random.Random(hash((fam, m)) & 0xFFFFFFFF)
                instances.extend(f.make_member(rng, m) for _ in range(args.k))
    else:
        fams = args.families or SHORTLIST
        items_by_name = {_tool_name(it): it for _, it in _load_items()}
        missing = [f for f in fams if f not in items_by_name]
        if missing:
            print(f"WARNING: families not found in CC.jsonl: {missing}")
        for fam in fams:
            it = items_by_name.get(fam)
            if it is None:
                continue
            for m in args.magnitudes:
                insts = make_instances(it, m, args.k)
                if len(insts) < args.k:
                    print(f"  note: {fam} @ m={m} produced only {len(insts)}/{args.k} instances")
                instances.extend(insts)
    print(f"built {len(instances)} instances [source={args.source}] over {len(fams)} families x "
          f"{len(args.magnitudes)} magnitudes\n", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out or f"runs/a0_oracle_gap_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    client = RawChat()
    sem = asyncio.Semaphore(args.concurrency)

    async def one(model, inst, cond_fn):
        async with sem:
            try:
                return await cond_fn(client, model, inst, args.max_tokens)
            except Exception as e:
                return {"condition": cond_fn.__name__, "answer": None, "correct": False,
                        "tokens": 0, "error": str(e)[:120]}

    results = []
    for mkey in args.models:
        model = CLAUDE.get(mkey, mkey)
        tasks = []
        for inst in instances:
            tasks.append((inst, "hand", one(model, inst, grade_hand)))
            tasks.append((inst, "build", one(model, inst, grade_build)))
        done = await asyncio.gather(*(t[2] for t in tasks))
        for (inst, _, _), res in zip(tasks, done):
            rec = {"model": mkey, "family": inst["family"], "magnitude": inst["magnitude"],
                   "gold": inst["gold"], "inputs": inst["inputs"], **res}
            results.append(rec)
        _report(mkey, [r for r in results if r["model"] == mkey])

    (out_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
    print(f"\nwrote {out_dir}/results.jsonl ({len(results)} records)")


def _report(model: str, recs: list):
    def acc(rs):
        rs = [r for r in rs if "error" not in r or r.get("answer") is not None or True]
        return (sum(r["correct"] for r in rs) / len(rs)) if rs else float("nan")
    mags = sorted({r["magnitude"] for r in recs})
    fams = sorted({r["family"] for r in recs})
    print(f"=== {model} : acc_hand -> acc_script  (gap) ===")
    hdr = f"{'family':<24}" + "".join(f"{('m='+str(m)):>20}" for m in mags)
    print(hdr); print("-" * len(hdr))
    for fam in fams:
        cells = []
        for m in mags:
            h = acc([r for r in recs if r["family"] == fam and r["magnitude"] == m
                     and r["condition"] == "hand"])
            b = acc([r for r in recs if r["family"] == fam and r["magnitude"] == m
                     and r["condition"] == "build"])
            cells.append(f"{h:>5.2f}->{b:<5.2f}({b-h:+.2f})")
        print(f"{fam:<24}" + "".join(f"{c:>20}" for c in cells))
    print(f"{'POOLED':<24}" + "".join(
        f"{(lambda h,b: f'{h:>5.2f}->{b:<5.2f}({b-h:+.2f})')(acc([r for r in recs if r['magnitude']==m and r['condition']=='hand']), acc([r for r in recs if r['magnitude']==m and r['condition']=='build'])):>20}"
        for m in mags))
    print()


if __name__ == "__main__":
    asyncio.run(main())
