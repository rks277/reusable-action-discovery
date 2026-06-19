"""LLM harness for the Treasure Hunt beach game, v2 (coordinate-addressed, no
movement).

Same shape as run_beach_llm.py -- wrap the environment in a text loop, drive an
LLM with RawChat, parse exactly one action per turn, feed back the observation,
record instrumentation -- but the world is beach.game_v2.GameV2 and there is NO
movement. The agent sees the WHOLE grid every turn (which cells are sand, which
are rocks, which rocks it has already searched) and addresses each action by an
explicit coordinate:
  inspect <x> <y>      search whatever is at that cell
  use <item> <x> <y>   use an inventory item on that cell

The grammar deliberately exposes ONE generic `use <item>` verb: the shovel digs,
and the map -- once built -- reveals the treasure coordinate (coords ignored when
reading it). Crucially the prompt never names the map or hints it exists; the
agent starts holding only the shovel and discovers the map purely by acting.

WHY this is the v2 of the reusable-action-discovery probe: removing movement
strips the spatial-search / navigation skill out of the beach so what's left is
the pure ECONOMIC choice -- spend actions inspecting rocks to build the map (the
reusable TOOL: collect `papers_needed` scraps -> map -> read the treasure
coordinate -> one dig wins) vs. brute-digging the sand cells. The grid render
names BOTH sand and rock cells so the observation doesn't bias the agent toward
rocks; we never spell out that rocks hide scraps, that scraps form a map, or even
that a map exists -- those affordances are all discovered by acting.
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
from beach.game_v2 import GameV2                    # noqa: E402
from beach.world import World                        # noqa: E402

from lomekwi.raw_chat import RawChat                 # noqa: E402
from lomekwi.obfuscation import assign               # noqa: E402

# Obfuscated nouns: the items + the shovel. The verbs `inspect` and `use` stay
# plain English; there is NO `dig` verb -- digging is `use <shovel> <x> <y>`.
OBF_KEYS = ["sand", "rock", "treasure", "map", "paper", "shovel"]


def obf_labels(relabel_seed: int) -> dict:
    """Letter-obfuscated label map for OBF_KEYS, drawn from relabel_seed (same
    randomization protocol as toolworld v2 -- a fresh per-episode assignment that
    strips the English semantics of sand/rock/treasure/map/paper)."""
    return assign(OBF_KEYS, seed=relabel_seed, scheme="letter")


def make_game(grid_size: int = 5, seed: int = 12345, papers_needed: int = 4,
              total_rocks: int = 12, shovel_durability: int = 15, hint: bool = True,
              max_turns: int | None = None, rows: int | None = None,
              labels: dict | None = None) -> GameV2:
    """Build a fresh GameV2 from parameters, routed through the game's own
    validate() so we honour the same invariants config.json does. `rows` is the
    grid height (None -> square grid_size x grid_size; 1 -> a 1-D strip). `labels`
    obfuscates the item nouns (None -> plain English)."""
    cfg = validate({
        "grid_size": grid_size, "rows": rows, "seed": seed,
        "papers_needed": papers_needed, "total_rocks": total_rocks,
        "shovel_durability": shovel_durability, "hint": hint, "max_turns": max_turns,
    })
    return GameV2(cfg, World.generate(cfg), labels=labels)


# --- action parsing --------------------------------------------------------
# The agent emits free text; we lift exactly one action from it (first parseable
# line), tolerating bullets/numbers/markdown and natural synonyms. inspect/dig
# carry an (x, y) target; coordinates are the first two integers on the line, so
# "(3,1)", "3 1" and "3,1" all parse.

_INT = re.compile(r"-?\d+")


def _coords(toks_line: str):
    nums = _INT.findall(toks_line)
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]))
    return None


def parse(line: str, labels: dict | None = None):
    """Map a line to a canonical action. Verbs `inspect`/`use` are plain English;
    digging is `use <shovel> <x> <y>` and reading is `use <map>` (shovel/map are
    obfuscated tokens carried in `labels`). A bare `dig <x> <y>` is also accepted
    as a fallback so the agent isn't locked out if it reverts to that word."""
    L = labels or {}
    ml = str(L.get("map", "map")).lower()
    sl = str(L.get("shovel", "shovel")).lower()

    t = line.strip().lower()
    t = re.sub(r"^[>\-\*•\d\.\)\(\s]+", "", t)  # strip bullets/numbering/quotes
    head_toks = [x for x in re.split(r"[\s:,;]+", t.strip("*_\"'`.!()")) if x]
    if not head_toks:
        return None
    head = head_toks[0]

    if head in ("inspect", "search", "examine", "look", "interact"):
        c = _coords(line)
        return ("inspect", c) if c else None
    if head == "dig":                          # undocumented fallback
        c = _coords(line)
        return ("dig", c) if c else None
    if head in ("use", "read", "consult"):
        rest = head_toks[1:]
        # use <shovel> <x> <y> == dig; check shovel before map so coords route right
        if any(x == sl or x.startswith("shovel") for x in rest):
            c = _coords(line)
            return ("dig", c) if c else None
        if any(x == ml or x.startswith("map") for x in rest):
            return ("use", "map")
        return None
    return None


def extract(text: str, labels: dict | None = None):
    for ln in text.splitlines():
        a = parse(ln, labels=labels)
        if a:
            return a
    return None


def grid_state_text(game: GameV2) -> str:
    """Whole-grid readout as a 2D array of cell contents (sand / rock / searched),
    plus inventory. Rendered row by row as a nested list so the agent sees the
    spatial layout directly; naming sand explicitly keeps the render from biasing
    the agent toward rocks; the treasure is never disclosed. The cell at column x,
    row y -- addressed (x, y) by inspect/dig -- is grid[y][x]."""
    w, h = game.config.grid_size, game.config.rows
    L = game.labels
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            rock = game.world.rock_at((x, y))
            if rock is None:
                row.append(L["sand"])
            elif rock.examined:
                row.append("searched")
            else:
                row.append(L["rock"])
        rows.append("[" + ", ".join(row) + "]")
    grid = "[" + ",\n ".join(rows) + "]"
    return (f"Grid (row y from top=0; column x from left=0; cell (x,y) = "
            f"grid[y][x]; 'searched' = a cell you already inspected):\n{grid}\n"
            f"Inventory: {game.inventory_summary()}.")


def initial_obs(game: GameV2, reveal_tool: bool = False) -> str:
    w, h = game.config.grid_size, game.config.rows
    L = game.labels
    reveal = ""
    if reveal_tool:
        reveal = (
            f" Some {L['rock']} cells hide a {L['paper']}: inspect them to collect "
            f"the {L['paper']}s, and once you have all {game.papers_needed} of them "
            f"they form a {L['map']} that reveals exactly where the {L['treasure']} "
            f"is buried (read it with `use {L['map']}`)."
        )
    if game.durability is None:                  # unbreakable shovel -> budget is the cap
        shovel_line = f"You hold a {L['shovel']} to dig with.\n\n"
    else:
        shovel_line = (
            f"You hold a {L['shovel']}, usable {game.durability} times before it "
            f"breaks; if it breaks before you uncover the {L['treasure']}, you "
            f"lose.\n\n"
        )
    return (
        f"You are looking down on a {w}x{h} grid. Each cell is either {L['sand']} "
        f"or a {L['rock']}. Cells are addressed by (x, y): x is the column (0 to "
        f"{w-1}, left to right) and y is the row (0 to {h-1}, top to bottom). You "
        f"can see the whole grid at once.\n\n"
        f"GOAL: find and uncover the buried {L['treasure']}, using as few actions "
        f"as possible.{reveal}\n\n"
        f"{shovel_line}"
        f"You may act with these verbs (one action per turn, on its own line):\n"
        f"  inspect <x> <y>        inspect cell (x, y)\n"
        f"  use <item> <x> <y>     use an item from your inventory on cell (x, y)\n\n"
        f"Things you obtain persist.\n\n{grid_state_text(game)}"
    )


SYS = ("You are an agent looking down on a grid, pursuing the goal stated in the "
       "first message, using as few actions as possible. You can see the whole "
       "grid. Inspect your surroundings and act deliberately. End each turn with "
       "exactly one action on its own line: inspect <x> <y> or use <item> <x> <y>. "
       "Think briefly, then act.")


async def run(model: str, grid_size: int = 5, seed: int = 12345,
              papers_needed: int = 4, total_rocks: int = 12,
              shovel_durability: int = 15, hint: bool = True,
              max_turns: int = 120, budget: int | None = None,
              rows: int | None = None, reveal_tool: bool = False,
              obfuscate: bool = False, relabel_seed: int | None = None):
    """budget: optional strict cap on actions, announced to the agent each turn.
    Like v1, this makes building the map an economic choice (spend scarce actions
    hunting scraps vs. grinding digs) without touching game mechanics. `rows` sets
    the grid height (None -> square; 1 -> a 1-D strip). reveal_tool: if True, the
    intro explicitly discloses the rock -> scrap -> map -> treasure chain (turns
    the task from tool DISCOVERY into tool EXPLOITATION). obfuscate: if True, the
    five item nouns (sand/rock/treasure/map/paper) are letter-obfuscated per
    episode (relabel_seed, default = seed), stripping their English semantics."""
    load_dotenv()
    client = RawChat()
    labels = obf_labels(relabel_seed if relabel_seed is not None else seed) \
        if obfuscate else None
    game = make_game(grid_size, seed, papers_needed, total_rocks,
                     shovel_durability, hint, rows=rows, labels=labels)
    L = game.labels
    intro = initial_obs(game, reveal_tool)
    if budget is not None:
        intro += (f"\n\nYou ALSO have a STRICT BUDGET of {budget} actions total "
                  f"(every action you take costs one). If you have not uncovered "
                  f"the {game.labels['treasure']} within {budget} actions, you fail.")
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

        act = extract(text, labels=L)
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
                      "inspect <x> <y> or use <item> <x> <y>, one action on its "
                      "own line."}]
            continue
        noop = 0

        had_map = game.has_map
        if act[0] == "inspect":
            result = game.inspect(act[1])
        elif act[0] == "dig":
            digs += 1
            result = game.dig(act[1])
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
              f"dur={game.durability} papers={game.papers}"
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
                  "content": obs + "\n\n" + grid_state_text(game) + tail}]

    stopped_reason = stopped_reason or "max_turns"
    result = {
        "model": model, "grid_size": grid_size, "rows": game.config.rows,
        "seed": seed,
        "papers_needed": papers_needed, "total_rocks": total_rocks,
        "shovel_durability": shovel_durability, "hint": hint, "budget": budget,
        "reveal_tool": reveal_tool, "obfuscate": obfuscate, "labels": game.labels,
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
    out = Path("runs") / f"beach_v2_{model.replace('/', '_')}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    result, trace = await run(model, grid_size=grid, hint=hint)
    (out / "result.json").write_text(json.dumps({"result": result, "trace": trace},
                                                indent=2))
    print(f"\nSaved: {out/'result.json'}")


if __name__ == "__main__":
    asyncio.run(main())
