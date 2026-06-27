"""Generate CREATOR-style numerical variations of the curated GSM-Hard seed problems.

Mirrors scripts/creator/tool_disposition_benchmark/dataset.py (which does this for the CC.jsonl
CREATOR set) but adapted to GSM-Hard's native format: each item is {input, code, target} where
`code` is a self-contained `def solution(): ... return result` (literal assignments at the top),
versus CREATOR's `tool`+`solution` that print their answer.

Per source problem we emit 1 EXAMPLE + 20 TEST instances (mimicking CREATOR's current 1-shown +
20-held-out structure). Every instance is fully-specified (all numbers shown in the question
text) and its gold is recomputed by re-executing the reference `code`. Numbers are resampled at a
tunable magnitude `m` (dataset.py's _resample; m=1.0 == the tool-disposition reference scale).

We carry the curated no-negative/nonsensical filter onto the generated variants: every gold must
be positive and finite, and if the seed's ORIGINAL answer was an integer (a discrete-count
quantity such as bolts/cars/people) the variant gold must also be integer — so we never emit a
fractional count. Fractional money/length/weight answers stay allowed (their seed gold was
fractional). See [[disposition-bench-math-too-easy]] for why GSM-Hard fits this benchmark.

  python -m scripts.creator.tool_disposition_benchmark.gsmhard_variations --seed 0
  python -m scripts.creator.tool_disposition_benchmark.gsmhard_variations --all --magnitude 1.0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

from scripts.creator.creator_ablation import _ASSIGN
from scripts.creator.creator_exec import _parse_answer, _run
from scripts.creator.creator_heldout import _fmt
from scripts.creator.tool_disposition_benchmark.dataset import (
    _resample, render_concrete)
from scripts.creator.tool_disposition_benchmark.make_gsmhard_seeds import (
    SRC as SEED_SRC, is_clean)
from scripts.creator.tool_disposition_benchmark.make_gsmhard_seeds import (
    stream as seed_stream)

DATASET_DIR = Path("scripts/creator/tool_disposition_benchmark/datasets")
N_TEST = 20          # held-out test cases per problem
N_EXAMPLE = 1        # shown example per problem
MAX_TRIES = 4000     # resampling attempts to fill example+tests for one problem

# A whole number literal in prose: optional commas as thousands separators, optional decimal.
# Leading `$`/digit-adjacency is handled by the lookbehind; a trailing sentence `.`/`,` is fine
# (it isn't followed by a digit) so "$5601592." matches, unlike CREATOR's _standalone_count.
_NUM_TOK = re.compile(r"(?<![\d.])\d[\d,]*(?:\.\d+)?(?![\d])")


def parse_inputs_code(code: str) -> list[tuple[str, float]]:
    """Numeric `name = value` literal assignments from a GSM-Hard `solution()` body."""
    out = []
    for name, raw in _ASSIGN.findall(code):
        v = float(raw.replace("_", ""))
        out.append((name, int(v) if v.is_integer() else v))
    return out


def _num_tokens(text: str) -> list[tuple[int, int, float]]:
    """All numeric literals in `text` as (start, end, value); commas stripped before parsing."""
    out = []
    for m in _NUM_TOK.finditer(text):
        tok = m.group().rstrip(".,")           # drop a trailing sentence period / comma
        val = float(tok.replace(",", ""))
        out.append((m.start(), m.start() + len(tok), int(val) if val.is_integer() else val))
    return out


def _eq(a: float, b: float) -> bool:
    return a == b if isinstance(a, int) and isinstance(b, int) else abs(a - b) < 1e-9


def resolve_spans(text: str, inputs: list[tuple[str, float]]):
    """Vary only inputs that appear EXACTLY ONCE as a number literal in the text and whose value
    is unique among the inputs (so the occurrence unambiguously belongs to that input). All other
    inputs are held fixed at their original value. Returns (varyable_spans_sorted, varyable_names)
    where spans are [(start, end, name)]; varyable_names may be empty (caller rejects)."""
    toks = _num_tokens(text)
    val_counts: dict = {}
    for _, _, tv in toks:
        val_counts[tv] = val_counts.get(tv, 0) + 1
    input_val_counts: dict = {}
    for _, v in inputs:
        input_val_counts[v] = input_val_counts.get(v, 0) + 1

    spans, names = [], set()
    for name, val in inputs:
        if input_val_counts[val] != 1:           # two inputs share this value -> ambiguous
            continue
        hits = [(s, e) for s, e, tv in toks if _eq(tv, val)]
        if len(hits) != 1:                        # missing or appears multiple times -> fix it
            continue
        spans.append((hits[0][0], hits[0][1], name))
        names.add(name)
    spans.sort()
    return spans, names


def _render_spans(text: str, spans, repl_by_name: dict[str, str]) -> str:
    """Splice placeholder/values into `text` at the resolved spans (position-based, one pass)."""
    out, last = [], 0
    for s, e, name in spans:
        out.append(text[last:s])
        out.append(repl_by_name[name])
        last = e
    out.append(text[last:])
    return "".join(out)


def gold_for_code(code: str, values: dict[str, float]) -> float | None:
    """Run the reference `code` with its init assignments overridden by `values`; return the
    value of `solution()` (None if it errors / yields no number).

    The reference `code` is the DATASET's own GSM-Hard solution (trusted, straight-line
    arithmetic — no loops, imports, or I/O), so we exec it IN-PROCESS for speed (~100x vs a
    sandboxed subprocess per candidate). A sandboxed subprocess is the fallback if in-process
    exec raises, so anything unexpected still degrades safely rather than crashing the build."""
    sol = code
    for name, newv in values.items():
        sol = re.sub(rf"(?m)^(\s*{re.escape(name)}\s*=\s*)[-+]?\d[\d_]*\.?\d*",
                     rf"\g<1>{_fmt(newv)}", sol, count=1)
    try:
        ns: dict = {}
        exec(sol, ns)
        result = ns["solution"]()
        return float(result)
    except Exception:
        stdout, err = _run(sol + '\nprint("ANSWER:", solution())')
        return None if err else _parse_answer(stdout)


def _ok(gold, require_int: bool) -> bool:
    """Carry the curated filter: positive, finite, and integer when the quantity is discrete."""
    if gold is None or not math.isfinite(gold) or gold <= 0:
        return False
    if require_int and not float(gold).is_integer():
        return False
    return True


def make_variations(item: dict, seed: int, magnitude: float) -> dict | str:
    """1 example + N_TEST test instances for one GSM-Hard item. Returns the problem dict, or a
    short string reason on failure ('no_inputs' / 'spans' / 'insufficient:<k>').

    Instances are deduped by GOLD so all N_EXAMPLE+N_TEST answers are DISTINCT: this both
    blocks hardcoding (memorizing one answer fails the other tests) and auto-rejects degenerate
    problems whose varied input doesn't change the answer (e.g. 'hours each painter worked' is
    3/8 day x 21 days regardless of the painter count -> only one possible gold)."""
    inputs = parse_inputs_code(item["code"])
    if len(inputs) < 1:
        return "no_inputs"
    spans, varyable = resolve_spans(item["input"], inputs)
    if not varyable:
        return "no_varyable"
    varied = [(n, v) for n, v in inputs if n in varyable]
    template = _render_spans(item["input"], spans, {n: n.upper() for n in varyable})
    require_int = float(item["target"]).is_integer()

    rng = random.Random(seed * 100003 + 17)
    instances, seen_inputs, seen_gold = [], set(), set()
    need = N_EXAMPLE + N_TEST
    for _ in range(MAX_TRIES):
        if len(instances) >= need:
            break
        vals = {n: _resample(v, rng, magnitude) for n, v in varied}   # fixed inputs keep code value
        key = tuple(vals[n] for n, _ in varied)
        if key in seen_inputs:
            continue
        seen_inputs.add(key)
        gold = gold_for_code(item["code"], vals)
        if not _ok(gold, require_int) or float(gold) in seen_gold:
            continue
        seen_gold.add(float(gold))
        up = {n.upper(): vals[n] for n, _ in varied}
        instances.append({
            "inputs": up,
            "gold": float(gold),
            "question": render_concrete(template, up),
        })
    if len(instances) < need:
        return f"insufficient:{len(instances)}"
    return {
        "keys": [n.upper() for n, _ in varied],
        "template": template,
        "require_int": require_int,
        "source_target": item["target"],
        "n_inputs": len(inputs),
        "n_varied": len(varied),
        "example": instances[0],
        "tests": instances[1:need],
    }


def _try(item: dict, rng_seed: int, magnitude: float):
    try:
        return make_variations(item, rng_seed, magnitude)
    except Exception as e:
        return f"error:{type(e).__name__}"


def _backfill_pool(seed: int, used_inputs: set[str]) -> list[dict]:
    """Clean GSM-Hard items from this seed's shuffled stream, beyond the curated 20, that are
    safe to add without manual review: positive INTEGER original answer (excludes negatives and
    ALL fractional answers, so no nonsensical fractional-count can sneak in). Skips items already
    in the curated seed file (by question text)."""
    clean = [d for d in (json.loads(l) for l in SEED_SRC.open()) if is_clean(d)]
    pool = []
    for d in seed_stream(clean, seed):
        t = d["target"]
        if d["input"] in used_inputs or not float(t).is_integer():
            continue
        pool.append(d)
    return pool


def build_seed(seed: int, magnitude: float, target_n: int = 20) -> tuple[list[dict], dict, int]:
    """Variations for the curated gsmhard_N20_seed{seed}.jsonl, then BACKFILL from the seed's
    clean stream until `target_n` problems have valid variations. Returns (problems, failures,
    n_backfilled)."""
    src = DATASET_DIR / f"gsmhard_N20_seed{seed}.jsonl"
    items = [json.loads(l) for l in src.open()]
    problems, failed = [], {}
    for i, item in enumerate(items):
        v = _try(item, seed * 1009 + i, magnitude)
        if isinstance(v, str):
            failed[i] = v
            continue
        problems.append({"idx": len(problems), "src_idx": i, "backfill": False, **v})

    n_backfilled = 0
    if len(problems) < target_n:
        used = {it["input"] for it in items}
        for j, item in enumerate(_backfill_pool(seed, used)):
            if len(problems) >= target_n:
                break
            v = _try(item, seed * 100003 + 5000 + j, magnitude)
            if isinstance(v, str):
                continue
            used.add(item["input"])
            problems.append({"idx": len(problems), "src_idx": f"bf{j}", "backfill": True, **v})
            n_backfilled += 1
    return problems, failed, n_backfilled


def _mtag(m: float) -> str:
    return f"{m:g}".replace(".", "p")


def out_path(seed: int, magnitude: float) -> Path:
    return DATASET_DIR / f"gsmhard_var_seed{seed}_m{_mtag(magnitude)}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="build all 4 seeds")
    ap.add_argument("--magnitude", type=float, default=1.0,
                    help="resampled-number size (dataset.py scale; raise for bigger numbers)")
    args = ap.parse_args()

    seeds = [0, 1, 2, 3] if args.all else [args.seed]
    for s in seeds:
        problems, failed, n_bf = build_seed(s, args.magnitude)
        p = out_path(s, args.magnitude)
        p.write_text(json.dumps(problems, indent=2))
        n_var = sum(1 + len(pr["tests"]) for pr in problems)
        print(f"seed {s}: {len(problems)} problems ({n_bf} backfilled) x {1 + N_TEST} "
              f"= {n_var} instances -> {p}", flush=True)
        if failed:
            print(f"         curated-failed (replaced by backfill): {failed}", flush=True)


if __name__ == "__main__":
    main()
