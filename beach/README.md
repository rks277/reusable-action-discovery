# Treasure Hunt

A text-only, command-line treasure hunting game. No graphics, no ASCII art —
just prompts and text. Built with the Python standard library only.

## Running

From this directory:

```sh
python game.py            # or: python -m beach
python game.py --config myconfig.json
```

Requires Python 3.8+.

## How to play

You start at `(0, 0)` on an `n x n` beach with a shovel. Somewhere a treasure is
buried (its location is fixed by the config seed).

- **Move** one cell North/East/South/West.
- **Interact** to look under whatever is at your feet. Some rocks hide a scrap of
  an old map; most hide nothing. Other cells hide nothing at all.
- Collect `papers_needed` scraps and they automatically combine into a **map**.
- **Use the map** to reveal the treasure's coordinates.
- Walk to the treasure and **use the shovel** there to dig it up and win.

Every dig wears the shovel down by 1. If the shovel breaks (durability hits 0)
before you find the chest, you lose. Reading the map is free; blocked moves
(into the map edge) don't cost a turn.

You always start at `(0, 0)` standing on a rock that has a scrap under it — your
first scrap is one `interact` away.

### Controls

Type a word or a menu number; bare directions also work:

```
move <N|E|S|W>   move one cell   (e.g. "move n", "n", "1 n")
interact         look at your feet
use <shovel|map> use an item     (e.g. "use shovel", "use map")
look             reprint status
quit             give up and exit
```

## Configuration (`config.json`)

| Key                 | Type        | Meaning                                                                 |
| ------------------- | ----------- | ----------------------------------------------------------------------- |
| `grid_size`         | int ≥ 1     | Side length `n` of the `n x n` grid (coords `0..n-1`).                   |
| `seed`              | int         | Single seed driving treasure + rock + paper placement (reproducible).   |
| `papers_needed`     | int ≥ 1     | Scraps (P) needed to form a map. Exactly P paper-bearing rocks exist.   |
| `total_rocks`       | int ≥ P     | Total rocks on the map, including the P paper rocks.                     |
| `shovel_durability` | int ≥ 1     | Number of digs before the shovel breaks.                                |
| `hint`              | bool        | When true, prints a one-line notice the moment the scraps form a map.   |
| `max_turns`         | int ≥ 1 / null | Optional cap on successful actions; `null` = unlimited.              |

Validation runs on load: `total_rocks >= papers_needed` and the grid must hold
all rocks plus a rock-free treasure cell (`total_rocks + 1 <= grid_size²`).

## Tests

```sh
python -m unittest discover -s tests
```
