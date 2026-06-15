"""LLM harness for the Treasure Hunt beach game (beach/).

Same shape as run_toolworld_llm.py / toolworld_v2.py: we wrap an environment in
a text loop, drive an LLM with RawChat, parse exactly one action per turn, feed
back the observation, and record instrumentation. Here the environment is the
existing beach.Game (move / interact / use_shovel / use_map) rather than a
bespoke State -- we reuse the game's own rules verbatim so the LLM plays the
identical world a human would.

WHY this is a reusable-action-discovery probe: the beach is a brute-vs-tool
trade-off, exactly like the vault. Brute = dig cells until you hit treasure, but
the shovel only survives `shovel_durability` digs, so blind digging across the
grid is hopeless. The TOOL is the map: collect `papers_needed` paper scraps
(found by `interact`-ing on the right rocks), they auto-combine into a map, and
`use map` reveals the treasure coordinate -- after which a single walk + dig
wins. The map is the reusable shortcut the agent must DISCOVER and choose to
build instead of grinding digs. Instrumentation below tracks whether/when it
did.

We never spell out the strategy; the agent discovers affordances by acting. It
starts on a bare cell with no rock or scrap in view, so even the existence of
scraps must be discovered by moving onto a rock and searching it.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# beach is a package living at <repo>/beach/beach; put its parent on the path so
# `import beach.*` resolves (mirrors how `python -m beach` works from beach/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "beach"))
from beach.config import validate                  # noqa: E402
from beach.game import Game                          # noqa: E402
from beach.world import World                        # noqa: E402

from lomekwi.raw_chat import RawChat                 # noqa: E402


def make_game(grid_size: int = 5, seed: int = 12345, papers_needed: int = 4,
              total_rocks: int = 12, shovel_durability: int = 15, hint: bool = True,
              max_turns: int | None = None) -> Game:
    """Build a fresh Game from parameters, routed through the game's own
    validate() so we honour the same invariants config.json does."""
    cfg = validate({
        "grid_size": grid_size, "seed": seed, "papers_needed": papers_needed,
        "total_rocks": total_rocks, "shovel_durability": shovel_durability,
        "hint": hint, "max_turns": max_turns,
    })
    return Game(cfg, World.generate(cfg))


# --- action parsing --------------------------------------------------------
# The agent emits free text; we lift exactly one action from it (first parseable
# line), tolerating bullets/numbers/markdown and natural synonyms.

_DIRS = {"n": "N", "north": "N", "e": "E", "east": "E",
         "s": "S", "south": "S", "w": "W", "west": "W"}


def parse(line: str):
    t = line.strip().lower()
    t = re.sub(r"^[>\-\*•\d\.\)\(\s]+", "", t)  # strip bullets/numbering/quotes
    toks = [x.strip("*_\"'`.!()") for x in re.split(r"[\s:,;]+", t)]
    toks = [x for x in toks if x]
    if not toks:
        return None
    head = toks[0]
    # bare direction, e.g. "n" / "north"
    if head in _DIRS:
        return ("move", _DIRS[head])
    if head == "move":
        for x in toks[1:]:
            if x in _DIRS:
                return ("move", _DIRS[x])
        return None
    if head in ("interact", "examine", "search", "look-under"):
        return ("interact", None)
    if head == "dig":
        return ("use", "shovel")
    if head in ("use", "read", "consult"):
        rest = toks[1:]
        if any(x.startswith("shovel") or x == "dig" for x in rest):
            return ("use", "shovel")
        if any(x.startswith("map") for x in rest):
            return ("use", "map")
        return None
    return None


def extract(text: str):
    for ln in text.splitlines():
        a = parse(ln)
        if a:
            return a
    return None


def status_text(game: Game) -> str:
    """One-line situational readout: what's at the agent's feet plus inventory.
    Rocks are reported but never pre-disclosed -- the agent must search to learn
    what (if anything) a rock hides."""
    rock = game.world.rock_at(game.pos)
    if rock is None:
        line = "There is no rock here."
    elif rock.examined:
        line = "There is a rock here (already searched)."
    else:
        line = "There is a rock here."
    return (f"Location: {game.pos}. {line} "
            f"Inventory: {game.inventory_summary()}.")


def initial_obs(game: Game) -> str:
    n = game.config.grid_size
    return (
        f"You are on an {n}x{n} beach. Cells run from (0,0) to ({n-1},{n-1}); "
        f"North increases y, East increases x. You start at (0,0).\n\n"
        f"GOAL: find the buried treasure and dig it up, using as few actions as "
        f"possible.\n\n"
        f"You carry a shovel that can dig {game.durability} times before it "
        f"breaks. If it breaks before you uncover the treasure, you lose.\n\n"
        f"You may act with these verbs (one action per turn, on its own line):\n"
        f"  move <N|E|S|W>   step one cell\n"
        f"  interact         search whatever is at your feet\n"
        f"  use <item>       use the specified item\n\n"
        f"Things you obtain persist. {status_text(game)}"
    )


SYS = ("You are an agent exploring a beach to find buried treasure, using as few "
       "actions as possible. Inspect your surroundings and act deliberately. End "
       "each turn with exactly one action on its own line: move <N|E|S|W>, "
       "interact, use <item>. Think briefly, then act.")


async def run(model: str, grid_size: int = 5, seed: int = 12345,
              papers_needed: int = 4, total_rocks: int = 12,
              shovel_durability: int = 15, hint: bool = True,
              max_turns: int = 120, budget: int | None = None):
    """budget: optional strict cap on actions, announced to the agent each turn.
    Like toolworld, this makes building the map an economic choice (spend scarce
    actions hunting scraps vs. grinding digs) without touching game mechanics."""
    load_dotenv()
    client = RawChat()
    game = make_game(grid_size, seed, papers_needed, total_rocks,
                     shovel_durability, hint)
    intro = initial_obs(game)
    if budget is not None:
        intro += (f"\n\nYou have a STRICT BUDGET of {budget} actions total. If you "
                  f"have not dug up the treasure within {budget} actions, you fail.")
    msgs = [{"role": "user", "content": intro + "\n\nWhat do you do?"}]

    trace = []
    noop = 0
    noop_total = 0
    refusals = 0
    unparsed = []
    usage_tot = {k: 0 for k in RawChat.USAGE_FIELDS}
    usage_tot["calls"] = 0
    build_turn = None        # turn the scraps formed a map
    used_map = False         # did the agent ever read the map?
    read_treasure_turn = None
    digs = 0
    stopped_reason = None
    won = False

    for t in range(max_turns):
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
            unparsed.append({"turn": t, "api_error": api_err, "text": text,
                             "debug": dbg})
            print(f"  [t{t+1}] NO-OP{' (API-ERR)' if api_err else ''}: "
                  f"{(api_err or text)[:80]!r}", flush=True)
            if noop >= 4:
                stopped_reason = "noop"
                break
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "No parseable action. Use "
                      "move/interact/use, one action on its own line."}]
            continue
        noop = 0

        had_map = game.has_map
        if act[0] == "move":
            result = game.move(act[1])
        elif act[0] == "interact":
            result = game.interact()
        elif act == ("use", "shovel"):
            digs += 1
            result = game.use_shovel()
        else:  # ("use", "map")
            used_map = True
            result = game.use_map()
            if game.has_map:
                read_treasure_turn = t
        obs = result.message
        if game.has_map and not had_map:
            build_turn = t

        done = result.game_over
        won = result.won
        print(f"  [t{t+1}] {act} -> {obs.splitlines()[0][:70]} "
              f"@{game.pos} dur={game.durability} papers={game.papers}"
              f"{' MAP' if game.has_map else ''}", flush=True)
        trace.append({"turn": t, "action": act, "message": obs,
                      "agent_text": text, "usage": client.last_usage})

        out_of_budget = budget is not None and len(trace) >= budget and not done
        if done or out_of_budget:
            stopped_reason = ("won" if won else "lost") if done else "out_of_budget"
            tail = ("" if done else
                    f"\n\nBudget exhausted ({budget} actions). You fail.")
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": obs + tail}]
            break

        if budget is not None:
            tail = f"\n\n[{budget - len(trace)} actions left] What next?"
        else:
            tail = "\n\nWhat next?"
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user",
                  "content": obs + "\n\n" + status_text(game) + tail}]

    stopped_reason = stopped_reason or "max_turns"
    result = {
        "model": model, "grid_size": grid_size, "seed": seed,
        "papers_needed": papers_needed, "total_rocks": total_rocks,
        "shovel_durability": shovel_durability, "hint": hint, "budget": budget,
        "treasure": list(game.world.treasure),
        "won": won, "total_actions": len(trace), "turns": game.turns,
        "durability_left": game.durability, "papers_collected": game.papers,
        "built_map": game.has_map, "build_turn": build_turn,
        "used_map": used_map, "read_treasure_turn": read_treasure_turn,
        "digs": digs, "wasted_digs": digs - (1 if won else 0),
        "noop_total": noop_total, "refusals": refusals, "unparsed": unparsed,
        "usage": usage_tot, "stopped_reason": stopped_reason,
    }
    print(f"\n  RESULT {model} grid={grid_size} seed={seed}: won={won} "
          f"actions={len(trace)} built_map={game.has_map} build_turn={build_turn} "
          f"used_map={used_map} digs={digs} dur_left={game.durability} "
          f"reason={stopped_reason}", flush=True)
    return result, trace


async def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-4-5-20251001"
    grid = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    hint = "--no-hint" not in sys.argv
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / f"beach_{model.replace('/', '_')}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    result, trace = await run(model, grid_size=grid, hint=hint)
    (out / "result.json").write_text(json.dumps({"result": result, "trace": trace},
                                                indent=2))
    print(f"\nSaved: {out/'result.json'}")


if __name__ == "__main__":
    asyncio.run(main())
