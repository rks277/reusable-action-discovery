"""The game world: grid, treasure, and rock/paper placement.

Placement is fully determined by the config seed, so the same seed always
produces the same map. The ``World`` owns the static layout plus a small bit of
mutable state (which rocks have been examined).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .config import Config


Coord = Tuple[int, int]


@dataclass
class Rock:
    """A rock on the map. ``has_paper`` is fixed; ``examined`` changes in play."""

    has_paper: bool
    examined: bool = False


@dataclass
class World:
    """Static map layout plus mutable rock-examined state.

    Coordinates are ``(x, y)`` with ``0 <= x, y < grid_size``.
    """

    grid_size: int
    treasure: Coord
    distinctive: Coord
    rocks: Dict[Coord, Rock] = field(default_factory=dict)

    @classmethod
    def generate(cls, config: Config) -> "World":
        """Build a world from ``config`` using a seeded RNG.

        The start cell (0, 0) always holds a paper rock — that's the player's
        hint. The treasure goes anywhere else (never on a rock), then the
        remaining rocks are placed and the first P rocks (starting with (0, 0))
        carry paper.
        """
        rng = random.Random(config.seed)
        n = config.grid_size
        start = (0, 0)

        all_cells = [(x, y) for x in range(n) for y in range(n)]

        treasure = rng.choice([c for c in all_cells if c != start])

        candidates = [c for c in all_cells if c != treasure and c != start]
        rock_cells = [start] + rng.sample(candidates, config.total_rocks - 1)

        rocks: Dict[Coord, Rock] = {}
        for i, cell in enumerate(rock_cells):
            rocks[cell] = Rock(has_paper=(i < config.papers_needed))

        return cls(
            grid_size=n,
            treasure=treasure,
            distinctive=start,
            rocks=rocks,
        )

    # --- queries -----------------------------------------------------------

    def in_bounds(self, coord: Coord) -> bool:
        x, y = coord
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def rock_at(self, coord: Coord) -> Optional[Rock]:
        return self.rocks.get(coord)

    def is_paper(self, coord: Coord) -> bool:
        rock = self.rocks.get(coord)
        return rock is not None and rock.has_paper

    def is_treasure(self, coord: Coord) -> bool:
        return coord == self.treasure

    # --- mutation ----------------------------------------------------------

    def mark_examined(self, coord: Coord) -> None:
        rock = self.rocks.get(coord)
        if rock is not None:
            rock.examined = True
