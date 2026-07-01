"""Stream builder for the tool-amortization benchmark.

Assembles ONE persistent session (a "stream") of problems drawn from `family_kit`, where the reuse
structure is CONTROLLED and KNOWN to us but LATENT to the model. A "class" is a recurring family
(size >= 2 members, all the same procedure but different inputs and a random cover story); a
"distractor" is a one-off (size-1 class). The model sees only the rendered problems, one at a time;
the hidden per-slot labels are what the ski-rental scorer reads.

Knobs (the experiment conditions):
  - classes:  [(family_name, size), ...]   size>=2 recurs; size==1 is a one-off distractor
  - magnitude m:  difficulty dial (A0: Haiku m~10 = genuine rent-vs-buy; m>=100 degenerate)
  - arrival:  how members are ordered -> controls how much the model knows at each build decision:
        'blocked'     each class's members contiguous (repeats immediately follow) - extreme front
        'interleaved' round-robin across classes
        'spread'      each class's members spread evenly across the whole stream
        'back'        first member of each class early, the rest deferred to the end (max gap -
                      the hardest ONLINE gamble: build early without seeing that it recurs)
        'random'      shuffled
  - announce: whether the class structure/horizon is revealed to the model (awareness condition).
              (Recorded here; the session runner/prompt consumes it. The stream is identical.)

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.stream_builder   # self-test
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from scripts.creator.tool_disposition_benchmark.family_kit import (
    ALL_FAMILIES, EASY_ONEOFFS, HARD_ONEOFFS, HARD_ONEOFF_MAGNITUDE, cluster_of)

# fields the MODEL sees (the runner presents only these); everything else is a hidden label.
PROBLEM_FIELDS = ("family", "magnitude", "keys", "inputs", "vals_order", "gold", "question")


@dataclass
class StreamSpec:
    """recurring: (family, size>=2) classes worth a tool. n_one_offs: count of GENUINE singletons,
    each a DISTINCT procedure from the one-off pool (disjoint from the recurring families). Their
    difficulty is a knob: 'easy' -> low magnitude (hand-feasible, building is wasteful), 'hard' ->
    the recurring magnitude (hand-infeasible, building is tempting but still single-use)."""
    recurring: list[tuple[str, int]]
    n_one_offs: int = 0
    one_offs: list[str] | None = None       # explicit one-off procedures (else drawn from the pool)
    one_off_difficulty: str = "hard"        # 'easy' | 'hard' (selects op pool + magnitude, A0-calibrated)
    magnitude: int = 10                     # recurring-class magnitude
    one_off_magnitude: int | None = None    # override the difficulty-derived one-off magnitude
    arrival: str = "spread"
    announce: bool = False
    seed: int = 0
    note: str = ""

    def oo_pool_and_magnitude(self) -> tuple[list[str], int]:
        """A0-calibrated: 'easy' -> ops Haiku hand-solves (at the recurring magnitude, so building
        is wasteful); 'hard' -> ops that are hand-infeasible at HARD_ONEOFF_MAGNITUDE."""
        if self.one_off_difficulty == "easy":
            return EASY_ONEOFFS, (self.one_off_magnitude or self.magnitude)
        return HARD_ONEOFFS, (self.one_off_magnitude or HARD_ONEOFF_MAGNITUDE)


def _order(class_sizes: list[int], arrival: str, rng: random.Random) -> list[int]:
    """Return a list of class-ids of length sum(sizes): which class occupies each slot, in order."""
    ids: list[int] = []
    if arrival == "blocked":
        for c, s in enumerate(class_sizes):
            ids += [c] * s
    elif arrival == "random":
        for c, s in enumerate(class_sizes):
            ids += [c] * s
        rng.shuffle(ids)
    elif arrival == "interleaved":
        remaining = {c: s for c, s in enumerate(class_sizes)}
        while remaining:
            for c in list(remaining):
                ids.append(c)
                remaining[c] -= 1
                if remaining[c] == 0:
                    del remaining[c]
    elif arrival == "spread":
        keyed = []
        for c, s in enumerate(class_sizes):
            for j in range(s):
                keyed.append(((j + 0.5) / s, rng.random(), c))   # even fractional positions + jitter
        keyed.sort()
        ids = [c for _, _, c in keyed]
    elif arrival == "back":
        firsts, rests = [], []
        for c, s in enumerate(class_sizes):
            firsts.append(c)
            rests += [c] * (s - 1)
        rng.shuffle(firsts)
        rng.shuffle(rests)
        ids = firsts + rests
    elif arrival == "oneoff_per_block":
        # random, EXCEPT exactly one (distinct) one-off is guaranteed in each successive block, the
        # block width scaled so the one-offs spread evenly across the WHOLE stream (BLOCK≈N/#oneoffs)
        # — so the "one-off or recurring-first-member?" decision recurs throughout, every seed.
        oneoff = [c for c, s in enumerate(class_sizes) if s == 1]
        BLOCK = max(1, sum(class_sizes) // max(1, len(oneoff)))
        recur = [c for c, s in enumerate(class_sizes) if s >= 2 for _ in range(s)]
        rng.shuffle(recur)
        N = sum(class_sizes)
        ids = [None] * N
        for i, cid in enumerate(oneoff):
            lo, hi = i * BLOCK, min(i * BLOCK + BLOCK, N)
            empty = [j for j in range(lo, hi) if ids[j] is None] or \
                    [j for j in range(N) if ids[j] is None]
            ids[rng.choice(empty)] = cid
        ri = iter(recur)
        ids = [x if x is not None else next(ri) for x in ids]
    elif arrival == "random_oneoff_early":
        # fully random order, EXCEPT guarantee >=1 one-off within the first 6 slots -- so the eager-
        # vs-wait decision is forced early (while the write budget is still free) without otherwise
        # constraining the natural interleaving of recurring members and one-offs.
        for c, s in enumerate(class_sizes):
            ids += [c] * s
        rng.shuffle(ids)
        oneoff = {c for c, s in enumerate(class_sizes) if s == 1}
        head = min(6, len(ids))
        if oneoff and not any(ids[i] in oneoff for i in range(head)):
            src = next(i for i, c in enumerate(ids) if c in oneoff)   # first one-off past the head
            dst = rng.randrange(head)
            ids[src], ids[dst] = ids[dst], ids[src]
    elif arrival == "firsts_spread":
        # The FIRST sighting of every class (recurring AND one-off) is anchored across the opening
        # ~half of the stream, shuffled, with >=1 one-off FORCED into slot 0 -- so build decisions
        # recur throughout while the budget is still available, and recurring-vs-one-off is
        # indistinguishable at first sight (only later recurrence reveals it). Recurring remainders
        # then fill the gaps, each placed AFTER its class's first sighting.
        N = sum(class_sizes); n = len(class_sizes)
        firsts = list(range(n)); rng.shuffle(firsts)
        oo = [c for c in firsts if class_sizes[c] == 1]
        if oo:                                        # guarantee an early one-off
            firsts.remove(oo[0]); firsts.insert(0, oo[0])
        span = max(n, N // 2)                          # spread first-sightings over the opening half
        raw = [round(i * (span - 1) / max(1, n - 1)) for i in range(n)] if n > 1 else [0]
        anchors, taken = [], set()                     # de-collide to distinct ascending slots
        for a in raw:
            while a in taken:
                a += 1
            taken.add(a); anchors.append(a)
        ids = [None] * N
        first_slot = {}
        for c, a in zip(firsts, anchors):
            ids[a] = c; first_slot[c] = a
        empties = [j for j in range(N) if ids[j] is None]
        rem = [c for c, s in enumerate(class_sizes) for _ in range(s - 1)]
        rng.shuffle(rem)
        rem.sort(key=lambda c: first_slot[c], reverse=True)   # place late-anchored classes first
        for c in rem:                                  # RANDOM empty slot after the first sighting
            cands = [j for j in empties if j > first_slot[c]] or empties
            j = rng.choice(cands)
            empties.remove(j); ids[j] = c
        assert all(x is not None for x in ids), "firsts_spread left an empty slot"
    else:
        raise ValueError(f"unknown arrival: {arrival}")
    return ids


def _resolve_classes(spec: StreamSpec, rng: random.Random) -> list[tuple[str, int, int]]:
    """(family, size, magnitude) per class: recurring at spec.magnitude, then one-off singletons at
    oo_magnitude. GENERALIZATION-CLUSTER CONSTRAINT: no two classes (across recurring AND one-off)
    may share a gen_cluster — otherwise a single tool would serve both and they would not be
    independent build-decisions / genuine single-use traps. One-offs are either taken verbatim from
    spec.one_offs or drawn from the difficulty pool, in both cases respecting the constraint."""
    for fam, _ in spec.recurring:
        if fam not in ALL_FAMILIES:
            raise ValueError(f"unknown family: {fam}")
    # recurring classes must themselves be cluster-distinct (else two recurring share one tool)
    recur_clusters: dict[str, str] = {}
    for fam, _ in spec.recurring:
        c = cluster_of(fam)
        if c in recur_clusters:
            raise ValueError(f"recurring {fam} shares gen_cluster '{c}' with "
                             f"{recur_clusters[c]} -> one tool would serve both")
        recur_clusters[c] = fam
    classes = [(fam, size, spec.magnitude) for fam, size in spec.recurring]
    recurring_fams = {f for f, _ in spec.recurring}
    used_clusters = set(recur_clusters)

    _, oo_mag = spec.oo_pool_and_magnitude()

    if spec.one_offs is not None:                       # explicit one-off list
        for fam in spec.one_offs:
            if fam not in ALL_FAMILIES:
                raise ValueError(f"unknown one-off family: {fam}")
            if fam in recurring_fams:
                raise ValueError(f"one-off {fam} is also a recurring class")
            c = cluster_of(fam)
            if c in used_clusters:
                raise ValueError(f"one-off {fam} shares gen_cluster '{c}' with an existing class "
                                 f"-> the one-off would be solvable by that class's tool")
            used_clusters.add(c)
            classes.append((fam, 1, oo_mag))
        return classes

    # else: draw n_one_offs distinct-cluster procedures from the difficulty pool
    pool_names, _ = spec.oo_pool_and_magnitude()
    pool = [f for f in pool_names if f not in recurring_fams and cluster_of(f) not in used_clusters]
    picks: list[str] = []
    for fam in rng.sample(pool, len(pool)):             # greedy over a shuffled pool, cluster-unique
        if len(picks) >= spec.n_one_offs:
            break
        c = cluster_of(fam)
        if c in used_clusters:
            continue
        used_clusters.add(c)
        picks.append(fam)
    if len(picks) < spec.n_one_offs:
        raise ValueError(f"n_one_offs={spec.n_one_offs} exceeds {len(picks)} cluster-distinct "
                         f"{spec.one_off_difficulty} one-offs available given the recurring set; "
                         f"add procedures in fresh gen_clusters or reduce n_one_offs")
    for fam in picks:
        classes.append((fam, 1, oo_mag))
    return classes


def build_stream(spec: StreamSpec) -> list[dict]:
    """Return an ordered list of slot dicts: each carries the PROBLEM_FIELDS plus hidden labels
    (slot_index, class_id, family, class_size, class_position, members_remaining_after,
    is_recurring)."""
    rng = random.Random(spec.seed)
    classes = _resolve_classes(spec, rng)
    class_sizes = [s for _, s, _ in classes]

    # instantiate each class's members up front (distinct inputs + random cover; per-class magnitude)
    members: dict[int, list[dict]] = {}
    for cid, (fam, size, mag) in enumerate(classes):
        members[cid] = [ALL_FAMILIES[fam].make_member(rng, mag) for _ in range(size)]

    order = _order(class_sizes, spec.arrival, rng)
    seen: dict[int, int] = {c: 0 for c in range(len(classes))}
    slots: list[dict] = []
    for slot_index, cid in enumerate(order):
        pos = seen[cid]                       # 0-based member index within its class
        seen[cid] += 1
        mem = members[cid][pos]
        size = class_sizes[cid]
        slots.append({
            **mem,
            "slot_index": slot_index,
            "class_id": cid,
            "class_size": size,
            "class_position": pos + 1,         # 1-based
            "members_remaining_after": size - (pos + 1),
            "is_recurring": size >= 2,
        })
    return slots


def problems_only(slots: list[dict]) -> list[dict]:
    """The view the model is allowed to see (labels stripped)."""
    return [{k: s[k] for k in PROBLEM_FIELDS} for s in slots]


def write_stream(slots: list[dict], out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "stream.json").write_text(json.dumps(slots, indent=2))          # full (with labels)
    (p / "problems.json").write_text(json.dumps(problems_only(slots), indent=2))  # model view
    return p


# --------------------------------------------------------------------- self-test (no model calls)
def _selftest():
    from collections import Counter
    spec = StreamSpec(recurring=[("product3", 4), ("lcg", 3)], n_one_offs=3,
                      one_off_difficulty="hard", magnitude=10, arrival="spread", seed=0)
    slots = build_stream(spec)

    # invariants: each class appears `size` times with 1..size positions
    cnt = Counter(s["class_id"] for s in slots)
    for cid in cnt:
        sz = next(s["class_size"] for s in slots if s["class_id"] == cid)
        assert cnt[cid] == sz
        positions = sorted(s["class_position"] for s in slots if s["class_id"] == cid)
        assert positions == list(range(1, sz + 1))
    recurring_fams = {s["family"] for s in slots if s["is_recurring"]}
    oneoff_fams = {s["family"] for s in slots if not s["is_recurring"]}
    assert recurring_fams.isdisjoint(oneoff_fams), (recurring_fams, oneoff_fams)
    assert len(oneoff_fams) == spec.n_one_offs, oneoff_fams   # each one-off a DISTINCT procedure

    print(f"stream: {len(slots)} slots  | recurring={sorted(recurring_fams)}  "
          f"one-offs={sorted(oneoff_fams)}\n")
    print(f"{'slot':>4} {'family':<16}{'pos':>4}/{'sz':<3}{'rem':>4}{'recur':>7}  question")
    for s in slots:
        print(f"{s['slot_index']:>4} {s['family']:<16}{s['class_position']:>4}/{s['class_size']:<3}"
              f"{s['members_remaining_after']:>4}{str(s['is_recurring']):>7}  {s['question'][:46]}")

    # easy vs hard one-off knob: op-aware pools + magnitudes (A0-calibrated)
    easy = build_stream(StreamSpec(recurring=[("product3", 3)], n_one_offs=3,
                                   one_off_difficulty="easy", magnitude=10, seed=0))
    hard = build_stream(StreamSpec(recurring=[("product3", 3)], n_one_offs=3,
                                   one_off_difficulty="hard", magnitude=10, seed=0))
    ef = sorted({s["family"] for s in easy if not s["is_recurring"]})
    hf = sorted({s["family"] for s in hard if not s["is_recurring"]})
    em = max(s["magnitude"] for s in easy if not s["is_recurring"])
    hm = max(s["magnitude"] for s in hard if not s["is_recurring"])
    print(f"\neasy one-offs: {ef} @ m={em}\nhard one-offs: {hf} @ m={hm}")
    assert set(ef) <= set(EASY_ONEOFFS) and em == 10
    assert set(hf) <= set(HARD_ONEOFFS) and hm == 100
    print("self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args, _ = ap.parse_known_args()
    _selftest()
