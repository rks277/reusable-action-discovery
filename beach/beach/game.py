"""Core game state and rules.

``Game`` holds all mutable state and exposes pure-ish action methods
(``move``, ``interact``, ``use_shovel``, ``use_map``) that mutate state and
return an :class:`ActionResult`. The methods do no I/O, so they can be unit
tested directly; the CLI layer is responsible for printing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .config import Config
from .world import Coord, World


# Direction -> (dx, dy). North increases y; coords are (x, y).
DIRECTIONS = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
}


@dataclass
class ActionResult:
    """Outcome of a single action.

    - ``message``: human-readable description of what happened.
    - ``turn_consumed``: whether this counted as a turn (blocked moves do not).
    - ``game_over``: whether the game has ended.
    - ``won``: True only on a winning end; False with ``game_over`` means a loss.
    """

    message: str
    turn_consumed: bool = False
    game_over: bool = False
    won: bool = False


class Game:
    """Mutable game state plus rules."""

    def __init__(self, config: Config, world: World):
        self.config = config
        self.world = world
        self.pos: Coord = (0, 0)
        self.durability = config.shovel_durability
        self.papers = 0
        self.has_map = False
        self.turns = 0
        self.over = False
        self.won = False

    # --- derived state -----------------------------------------------------

    @property
    def papers_needed(self) -> int:
        return self.config.papers_needed

    def inventory_summary(self) -> str:
        items = [f"shovel (durability {self.durability})"]
        if self.has_map:
            items.append("map")
        elif self.papers > 0:
            items.append(f"paper x{self.papers}")
        return ", ".join(items)

    def has_rock_here(self) -> bool:
        return self.world.rock_at(self.pos) is not None

    # --- turn accounting ---------------------------------------------------

    def _consume_turn(self) -> "str | None":
        """Increment the turn counter; return a loss message if the cap is hit."""
        self.turns += 1
        if self.config.max_turns is not None and self.turns >= self.config.max_turns:
            self.over = True
            return "Out of turns. You lose."
        return None

    # --- actions -----------------------------------------------------------

    def move(self, direction: str) -> ActionResult:
        direction = direction.upper()
        if direction not in DIRECTIONS:
            return ActionResult(f"Unknown direction: {direction!r}")
        if self.over:
            return ActionResult("The game is already over.")

        dx, dy = DIRECTIONS[direction]
        target: Coord = (self.pos[0] + dx, self.pos[1] + dy)
        if not self.world.in_bounds(target):
            return ActionResult(f"Can't move {direction}; edge of the map.")

        self.pos = target
        msg = f"Moved to {self.pos}."
        loss = self._consume_turn()
        if loss:
            return ActionResult(f"{msg}\n{loss}", turn_consumed=True, game_over=True)
        return ActionResult(msg, turn_consumed=True)

    def interact(self) -> ActionResult:
        if self.over:
            return ActionResult("The game is already over.")

        rock = self.world.rock_at(self.pos)
        if rock is None:
            msg = "Nothing happens."
            loss = self._consume_turn()
            if loss:
                return ActionResult(f"{msg}\n{loss}", turn_consumed=True, game_over=True)
            return ActionResult(msg, turn_consumed=True)

        if rock.examined:
            msg = "Nothing under the rock."
            loss = self._consume_turn()
            if loss:
                return ActionResult(f"{msg}\n{loss}", turn_consumed=True, game_over=True)
            return ActionResult(msg, turn_consumed=True)

        # First time examining this rock.
        self.world.mark_examined(self.pos)
        if not rock.has_paper:
            msg = "Nothing under the rock."
            loss = self._consume_turn()
            if loss:
                return ActionResult(f"{msg}\n{loss}", turn_consumed=True, game_over=True)
            return ActionResult(msg, turn_consumed=True)

        # Found a piece of paper.
        self.papers += 1
        lines = ["You find a piece of paper."]
        if not self.has_map and self.papers >= self.papers_needed:
            self.has_map = True
            if self.config.hint:
                lines.append("The pieces form a map.")
        msg = "\n".join(lines)
        loss = self._consume_turn()
        if loss:
            return ActionResult(f"{msg}\n{loss}", turn_consumed=True, game_over=True)
        return ActionResult(msg, turn_consumed=True)

    def use_shovel(self) -> ActionResult:
        if self.over:
            return ActionResult("The game is already over.")
        if self.durability <= 0:
            # Defensive: shouldn't happen since hitting 0 ends the game.
            return ActionResult("Your shovel is broken; you can't dig.")

        self.durability -= 1

        if self.world.is_treasure(self.pos):
            self.over = True
            self.won = True
            self._consume_turn()
            return ActionResult(
                "You dig and find the treasure. You win!",
                turn_consumed=True,
                game_over=True,
                won=True,
            )

        msg = "You dig but find nothing."
        if self.durability <= 0:
            self.over = True
            return ActionResult(
                f"{msg}\nYour shovel breaks. You lose.",
                turn_consumed=True,
                game_over=True,
            )

        loss = self._consume_turn()
        if loss:
            return ActionResult(f"{msg}\n{loss}", turn_consumed=True, game_over=True)
        return ActionResult(msg, turn_consumed=True)

    def use_map(self) -> ActionResult:
        if self.over:
            return ActionResult("The game is already over.")
        if not self.has_map:
            return ActionResult("You don't have a map yet.")

        # Reading the map is free — it doesn't consume a turn.
        return ActionResult(f"The treasure is at {self.world.treasure}.")
