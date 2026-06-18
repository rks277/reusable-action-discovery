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

    Coordinates are ``(x, y)`` with ``0 <= x < grid_size`` (width) and
    ``0 <= y < rows`` (height). Square when ``rows == grid_size``; a 1-D strip
    when ``rows == 1``.
    """

    grid_size: int
    rows: int
    treasure: Coord
    rocks: Dict[Coord, Rock] = field(default_factory=dict)

    @classmethod
    def generate(cls, config: Config) -> "World":
        """Build a world from ``config`` using a seeded RNG.

        The treasure goes on a random non-start cell (so the player can't win on
        turn 0). Rocks are then placed uniformly over every remaining cell — the
        start cell (0, 0) included, with no special treatment — so whether the
        player begins on sand or on a searchable rock just falls out of the same
        draw as every other cell. Paper goes to a random P-sized subset of the
        rocks, so a start rock is no likelier than any other to hide a scrap.
        """
        rng = random.Random(config.seed)
        w, h = config.grid_size, config.rows
        start = (0, 0)

        all_cells = [(x, y) for y in range(h) for x in range(w)]

        treasure = rng.choice([c for c in all_cells if c != start])

        candidates = [c for c in all_cells if c != treasure]
        rock_cells = rng.sample(candidates, config.total_rocks)

        paper_cells = set(rng.sample(rock_cells, config.papers_needed))
        rocks: Dict[Coord, Rock] = {
            cell: Rock(has_paper=(cell in paper_cells)) for cell in rock_cells
        }

        return cls(
            grid_size=w,
            rows=h,
            treasure=treasure,
            rocks=rocks,
        )

    # --- queries -----------------------------------------------------------

    def in_bounds(self, coord: Coord) -> bool:
        x, y = coord
        return 0 <= x < self.grid_size and 0 <= y < self.rows

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
