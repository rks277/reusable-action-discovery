"""Build the tool-disposition problem set: N DISTINCT CREATOR problems, each with one
hard-resampled input row, its reference gold, and a required-significant-figures `d`.

Each problem's question is rendered CONCRETE (the resampled values substituted back into the
template) and annotated with the precision requirement. We keep only problems whose gold
genuinely needs all `d` figures (grading.sigfigs_meaningful) so the precision knob bites.

`magnitude` scales the size of the resampled inputs (the real driver of by-hand difficulty):
m=1.0 reproduces creator_eval_hard.hard_resample (3-4 digit ints); smaller m → smaller numbers
that are feasible to grind by hand, so the sig-figs requirement becomes the binding constraint.
Use calibrate.py to pick the m (and D) giving ~50% by-hand solve rate.

  python -m scripts.creator.tool_disposition_benchmark.dataset --n 30 --sig-figs 6 --magnitude 1.0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from scripts.creator.creator_batch import _render, _spans
from scripts.creator.creator_heldout import gold_for, parse_inputs
from scripts.creator.tool_disposition_benchmark.grading import sigfigs_meaningful

DATA = Path("external/CC.jsonl")
DATASET_DIR = Path("scripts/creator/tool_disposition_benchmark/datasets")

# Reference (m=1.0) resampling ranges, from creator_eval_hard.hard_resample.
_INT_LO, _INT_HI = 137, 9973
_FMULT_LO, _FMULT_HI = 50.0, 500.0


def _resample(value, rng: random.Random, m: float):
    """A same-flavour but harder value, with size scaled by magnitude `m` (m=1.0 == hard_resample).
    Ints stay ints (keeps range()/count formulas valid); floats stay multi-decimal."""
    if isinstance(value, int):
        lo = max(2, round(_INT_LO * m))
        hi = max(lo + 1, round(_INT_HI * m))
        return rng.randint(lo, hi)
    base = abs(value) or 1.0
    lo, hi = max(1.0, _FMULT_LO * m), max(2.0, _FMULT_HI * m)
    return round(rng.uniform(lo * base, hi * base) + rng.random(), 3)


def _make_instance(item: dict, seed: int, magnitude: float, sig_figs: int) -> dict | None:
    """One concrete problem instance: template + a magnitude-scaled input row whose gold is finite
    and genuinely needs all `sig_figs` figures. Returns None if no valid row is found."""
    inputs = parse_inputs(item["solution"])
    if len(inputs) < 1:
        return None
    spans = _spans(item["question"], inputs)
    if spans is None:
        return None
    template = _render(item["question"], spans, {n: n.upper() for n, _ in inputs})
    rng = random.Random(seed * 100003 + 17)
    seen = set()
    for _ in range(60):
        vals = {n: _resample(v, rng, magnitude) for n, v in inputs}
        key = tuple(vals[n] for n, _ in inputs)
        if key in seen:
            continue
        seen.add(key)
        gold = gold_for(item, vals)
        if gold is None or gold != gold or abs(gold) == float("inf"):
            continue
        if not sigfigs_meaningful(gold, sig_figs):
            continue
        keys = [n.upper() for n, _ in inputs]
        return {"keys": keys, "template": template,
                "inputs": {n.upper(): vals[n] for n, _ in inputs}, "gold": float(gold)}
    return None


def _fmt_num(v) -> str:
    """Clean human string for a value substituted into a question (ints stay ints)."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer()):
        return str(int(v))
    return f"{v:g}"


def render_concrete(template: str, inputs: dict) -> str:
    """Substitute UPPERCASE placeholders in the template with their values. Longest keys
    first so e.g. BASE1 is replaced before BASE (avoids partial-token corruption)."""
    q = template
    for k in sorted(inputs, key=len, reverse=True):
        q = q.replace(k, _fmt_num(inputs[k]))
    return q


def _load_items() -> list[tuple[int, dict]]:
    return [(i, json.loads(l))
            for i, l in enumerate(DATA.read_text().splitlines()) if l.strip()]


def build_dataset(n: int, sig_figs: int, seed: int, magnitude: float = 1.0) -> list[dict]:
    """Collect n distinct feasible problems with magnitude-scaled inputs and golds that need all
    `sig_figs` figures."""
    problems: list[dict] = []
    for item_idx, item in _load_items():
        if len(problems) >= n:
            break
        try:
            inst = _make_instance(item, seed * 1009 + item_idx, magnitude, sig_figs)
        except Exception:
            inst = None
        if inst is None:
            continue
        problems.append({
            "idx": len(problems),
            "item_idx": item_idx,
            "keys": inst["keys"],
            "template": inst["template"],
            "inputs": inst["inputs"],
            "gold": inst["gold"],
            "sig_figs": sig_figs,
            "magnitude": magnitude,
            "question": render_concrete(inst["template"], inst["inputs"]),
        })
    return problems


def _mtag(magnitude: float) -> str:
    return f"{magnitude:g}".replace(".", "p")


def dataset_path(n: int, sig_figs: int, seed: int, magnitude: float = 1.0) -> Path:
    return DATASET_DIR / f"toold_N{n}_seed{seed}_D{sig_figs}_m{_mtag(magnitude)}.json"


def load_or_build(n: int, sig_figs: int, seed: int, magnitude: float = 1.0) -> list[dict]:
    p = dataset_path(n, sig_figs, seed, magnitude)
    if p.exists():
        return json.loads(p.read_text())
    problems = build_dataset(n, sig_figs, seed, magnitude)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(problems, indent=2))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--sig-figs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--magnitude", type=float, default=1.0,
                    help="scales resampled input size; <1 makes by-hand arithmetic easier")
    args = ap.parse_args()
    problems = build_dataset(args.n, args.sig_figs, args.seed, args.magnitude)
    p = dataset_path(args.n, args.sig_figs, args.seed, args.magnitude)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(problems, indent=2))
    print(f"built {len(problems)}/{args.n} problems @ {args.sig_figs} sig figs, "
          f"magnitude {args.magnitude} -> {p}")
    for pr in problems[:3]:
        print(f"\n[{pr['idx']}] item={pr['item_idx']} gold={pr['gold']!r}\n{pr['question']}")


if __name__ == "__main__":
    main()
