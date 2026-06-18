"""Coordinate-addressed game state and rules (beach v2 -- no movement).

``GameV2`` is the position-free analog of :class:`beach.game.Game`. There is no
agent position and no ``move``: the agent sees the whole grid up front and
addresses every action by an explicit ``(x, y)`` coordinate
(``inspect`` / ``dig`` / ``use_map``). The world rules are otherwise identical to
v1 -- it reuses :class:`beach.world.World` and :class:`beach.config.Config`
verbatim -- so a given seed is the SAME world a v1 episode would see, just
played without navigation.

Like ``Game``, the action methods mutate state and return an
:class:`beach.game.ActionResult`; they do no I/O, so they can be unit tested
directly.
"""

from __future__ import annotations

from .config import Config
from .game import ActionResult
from .world import Coord, World


class GameV2:
    """Mutable game state plus rules, addressed by coordinate (no position)."""

    def __init__(self, config: Config, world: World):
        self.config = config
        self.world = world
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

    # --- turn accounting ---------------------------------------------------

    def _consume_turn(self) -> "str | None":
        """Increment the turn counter; return a loss message if the cap is hit."""
        self.turns += 1
        if self.config.max_turns is not None and self.turns >= self.config.max_turns:
            self.over = True
            return "Out of turns. You lose."
        return None

    # --- actions -----------------------------------------------------------

    def inspect(self, coord: Coord) -> ActionResult:
        """Search whatever is at ``coord`` -- the coordinate-addressed analog of
        ``Game.interact``."""
        if self.over:
            return ActionResult("The game is already over.")
        if not self.world.in_bounds(coord):
            return ActionResult(f"{coord} is off the map.")

        rock = self.world.rock_at(coord)
        if rock is None:
            return self._with_turn(f"Nothing but sand at {coord}.")

        if rock.examined:
            return self._with_turn(f"Nothing under the rock at {coord}.")

        # First time examining this rock.
        self.world.mark_examined(coord)
        if not rock.has_paper:
            return self._with_turn(f"Nothing under the rock at {coord}.")

        # Found a piece of paper.
        self.papers += 1
        lines = [f"You find a piece of paper under the rock at {coord}."]
        if not self.has_map and self.papers >= self.papers_needed:
            self.has_map = True
            if self.config.hint:
                lines.append("The pieces form a map.")
        return self._with_turn("\n".join(lines))

    def dig(self, coord: Coord) -> ActionResult:
        """Dig at ``coord`` -- the coordinate-addressed analog of
        ``Game.use_shovel``. Bounds-checked first so an off-map dig wastes no
        shovel charge."""
        if self.over:
            return ActionResult("The game is already over.")
        if not self.world.in_bounds(coord):
            return ActionResult(f"{coord} is off the map.")
        if self.durability <= 0:
            # Defensive: shouldn't happen since hitting 0 ends the game.
            return ActionResult("Your shovel is broken; you can't dig.")

        self.durability -= 1

        if self.world.is_treasure(coord):
            self.over = True
            self.won = True
            self._consume_turn()
            return ActionResult(
                f"You dig at {coord} and find the treasure. You win!",
                turn_consumed=True,
                game_over=True,
                won=True,
            )

        msg = f"You dig at {coord} but find nothing."
        if self.durability <= 0:
            self.over = True
            return ActionResult(
                f"{msg}\nYour shovel breaks. You lose.",
                turn_consumed=True,
                game_over=True,
            )
        return self._with_turn(msg)

    def use_map(self) -> ActionResult:
        if self.over:
            return ActionResult("The game is already over.")
        if not self.has_map:
            return ActionResult("You don't have a map yet.")

        # Reading the map is free -- it doesn't consume a turn.
        return ActionResult(f"The treasure is at {self.world.treasure}.")

    # --- helpers -----------------------------------------------------------

    def _with_turn(self, msg: str) -> ActionResult:
        """Consume a turn and wrap ``msg``, folding in a turn-cap loss if hit."""
        loss = self._consume_turn()
        if loss:
            return ActionResult(f"{msg}\n{loss}", turn_consumed=True, game_over=True)
        return ActionResult(msg, turn_consumed=True)
