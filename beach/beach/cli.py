"""Interactive command-line loop.

This is the only layer that does I/O. It renders the per-turn status, reads
player input (numbered menu *or* typed commands like ``move n`` / ``use
shovel`` / ``interact``), dispatches to :class:`~beach.game.Game`, prints the
result, and detects end states.
"""

from __future__ import annotations

import argparse
import os

from .config import Config, ConfigError, load_config
from .game import DIRECTIONS, ActionResult, Game
from .world import World


def intro_text(config: Config, world: World) -> str:
    return f"Grid {config.grid_size}x{config.grid_size}."


def render_status(game: Game) -> str:
    rock = game.world.rock_at(game.pos)
    if rock is None:
        line = "There is no rock here."
    else:
        line = "There is a rock here."
    return "\n".join([
        f"Location: {game.pos}",
        line,
        f"Inventory: {game.inventory_summary()}",
    ])


def parse_command(raw: str):
    """Parse raw input into ``(action, arg)`` or ``None`` if unrecognized.

    Accepts menu numbers, full words, and bare directions. Returns a tuple like
    ``("move", "N")``, ``("interact", None)``, ``("use", "shovel")``,
    ``("look", None)``, ``("help", None)``, ``("quit", None)``.
    """
    tokens = raw.strip().lower().split()
    if not tokens:
        return None

    head = tokens[0]
    rest = tokens[1:]

    # Bare direction, e.g. "n" or "north".
    dir_aliases = {
        "n": "N", "north": "N",
        "e": "E", "east": "E",
        "s": "S", "south": "S",
        "w": "W", "west": "W",
    }
    if head in dir_aliases:
        return ("move", dir_aliases[head])

    # Numbered menu: 1=move, 2=interact, 3=use.
    menu = {"1": "move", "2": "interact", "3": "use"}
    if head in menu:
        head = menu[head]

    if head == "move":
        if rest and rest[0] in dir_aliases:
            return ("move", dir_aliases[rest[0]])
        return ("move", None)  # direction will be prompted
    if head == "interact":
        return ("interact", None)
    if head == "use":
        if rest:
            return ("use", rest[0])
        return ("use", None)  # item will be prompted
    if head in ("look", "status"):
        return ("look", None)
    if head in ("quit", "exit", "q"):
        return ("quit", None)
    return None


def _prompt(prompt_text: str) -> str:
    try:
        return input(prompt_text)
    except EOFError:
        return "quit"


def dispatch(game: Game, action: str, arg) -> "ActionResult | None":
    """Run one command against the game. Returns an ActionResult, or None for
    the meta command (look) that doesn't change game state."""
    if action == "move":
        direction = arg
        while direction is None:
            d = _prompt("  Which direction (N/E/S/W)? ").strip().lower()
            mapping = {"n": "N", "e": "E", "s": "S", "w": "W"}
            if d in mapping:
                direction = mapping[d]
            elif d in ("quit", "q", "exit"):
                return ActionResult("Heading home.", game_over=True)
            else:
                print("  Please enter N, E, S, or W.")
        return game.move(direction)

    if action == "interact":
        return game.interact()

    if action == "use":
        item = arg
        while item is None:
            choices = "shovel/map" if game.has_map else "shovel"
            item = _prompt(f"  Use what ({choices})? ").strip().lower() or None
        if item.startswith("shovel"):
            return game.use_shovel()
        if item.startswith("map"):
            return game.use_map()
        return ActionResult(f"You don't have a '{item}'.")

    return None


def play(game: Game, *, show_intro: bool = True) -> bool:
    """Run the interactive loop. Returns True if the player won."""
    if show_intro:
        print(intro_text(game.config, game.world))

    while not game.over:
        print(render_status(game))
        raw = _prompt("\n> ")
        parsed = parse_command(raw)
        if parsed is None:
            print("Unrecognized command.")
            continue

        action, arg = parsed
        if action == "look":
            continue  # status reprints at the top of the loop
        if action == "quit":
            print("Goodbye.")
            return False

        result = dispatch(game, action, arg)
        if result is None:
            continue
        print(result.message)
        if result.game_over:
            game.over = True
            return result.won

    return game.won


def build_game(config_path: str) -> Game:
    config = load_config(config_path)
    world = World.generate(config)
    return Game(config, world)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Treasure Hunt — a CLI treasure game.")
    default_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="path to the JSON config file (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        game = build_game(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}")
        return 2

    try:
        play(game)
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")
        return 0
    return 0
