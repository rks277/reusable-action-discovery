"""Print a sampled trail of the model's own outputs through an episode.

For ONE episode per value of n in a run, print the model's text output at the
first turn and every Nth turn after that (turn 0, 10, 20, ... by default). Lets
you eyeball how the model's reasoning/voice evolves over a run without scrolling
the whole transcript -- e.g. does it stay on-task, start hallucinating objects,
or give up -- across the build-margin (n) axis.

"One run per n": by default the first episode found for each n. Pass --rep R to
pin the same relabeling seed across every n (so you read the "same" world
blueprint at each scale); falls back to the first episode if that rep is absent.

Usage:
    python -m scripts.print_model_outputs runs/nt_sweep_XXXX
    python -m scripts.print_model_outputs runs/nt_sweep_XXXX --stride 5 --rep 0
    python -m scripts.print_model_outputs runs/a/episodes.jsonl --truncate 200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from scripts.analyze_episode_stats import Episode, load_episodes, _Palette, _short


def sampled_turns(n_outputs: int, stride: int) -> list[int]:
    """Turn indices to show: 0, stride, 2*stride, ... within range."""
    return list(range(0, n_outputs, stride)) if n_outputs else []


def pick_one_per_n(eps: list[Episode], rep: int | None) -> list[Episode]:
    """One episode per n: the requested relabel_seed if present, else the first
    episode encountered for that n. Returned sorted by n."""
    by_n: dict[int, list[Episode]] = {}
    for ep in eps:
        by_n.setdefault(ep.n, []).append(ep)
    chosen: list[Episode] = []
    for n in sorted(by_n, key=lambda v: v if v is not None else 0):
        group = by_n[n]
        pick = None
        if rep is not None:
            pick = next((e for e in group
                         if e.row.get("relabel_seed") == rep), None)
        chosen.append(pick or group[0])
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="episodes.jsonl file or run directory")
    ap.add_argument("--stride", type=int, default=10,
                    help="print turn 0 then every Nth turn (default: 10)")
    ap.add_argument("--rep", type=int, default=None,
                    help="pin this relabel_seed across all n (default: first episode)")
    ap.add_argument("--truncate", type=int, default=0,
                    help="cap each printed output to this many chars (0 = full)")
    args = ap.parse_args()

    eps = load_episodes([args.path])
    if not eps:
        print("No episodes loaded.")
        return

    c = _Palette(sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)
    for ep in pick_one_per_n(eps, args.rep):
        texts = ep.row.get("agent_texts", [])
        actions = ep.row.get("actions", [])
        turns = sampled_turns(len(texts), args.stride)
        head = (f"{c('bold', _short(ep.model))}  "
                f"n={c('bold', str(ep.n))}  T={ep.n_types}  "
                f"rep={ep.row.get('relabel_seed')}  "
                f"solved={ep.solved}  turns={ep.total_actions}")
        print("\n" + c("bold", "━━━━ ") + head + c("bold", " ━━━━"))
        print(c("dim", f"showing turn 0 + every {args.stride}th "
                        f"({len(turns)} of {len(texts)} outputs)"))
        for i in turns:
            text = texts[i] or "(no output / API error)"
            if args.truncate and len(text) > args.truncate:
                text = text[:args.truncate].rstrip() + " …"
            act = actions[i] if i < len(actions) else None
            print(c("cyan", f"\n[turn {i}]") +
                  (c("dim", f"  → {tuple(act)}") if act else ""))
            print(text)


if __name__ == "__main__":
    main()
