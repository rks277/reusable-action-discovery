"""Configuration loading and validation for Treasure Hunt.

The whole game is driven by a small JSON config so that runs are reproducible
(seeded) and tunable. ``load_config`` reads the file, fills in defaults for any
omitted keys, and validates the values, raising ``ConfigError`` with a clear
message on anything invalid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


DEFAULTS = {
    "grid_size": 8,
    "seed": 12345,
    "papers_needed": 3,
    "total_rocks": 12,
    "shovel_durability": 5,
    "hint": True,
    "max_turns": None,
}


class ConfigError(ValueError):
    """Raised when a config file is missing required values or has bad ones."""


@dataclass(frozen=True)
class Config:
    """Validated, immutable game configuration.

    Attributes mirror the JSON keys:

    - ``grid_size``: side length ``n`` of the ``n x n`` grid (coords ``0..n-1``).
    - ``seed``: single integer seed driving treasure + rock + paper placement.
    - ``papers_needed`` (P): number of paper-bearing rocks; collecting all P
      forms exactly one map.
    - ``total_rocks``: total rocks on the map, including the P paper rocks.
    - ``shovel_durability``: number of times the shovel can be used before it
      breaks (0 durability without a win = game over).
    - ``hint``: when True, notify the player the moment a map auto-forms.
    - ``max_turns``: optional cap on successful actions (``None`` = unlimited).
    """

    grid_size: int
    seed: int
    papers_needed: int
    total_rocks: int
    shovel_durability: int
    hint: bool
    max_turns: "int | None"


def _require_int(value, name: str) -> int:
    # bool is a subclass of int; reject it so True/False can't sneak in as numbers.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{name}' must be an integer, got {value!r}")
    return value


def validate(data: dict) -> Config:
    """Validate a raw config dict (defaults already merged) into a ``Config``."""
    grid_size = _require_int(data["grid_size"], "grid_size")
    seed = _require_int(data["seed"], "seed")
    papers_needed = _require_int(data["papers_needed"], "papers_needed")
    total_rocks = _require_int(data["total_rocks"], "total_rocks")
    shovel_durability = _require_int(data["shovel_durability"], "shovel_durability")

    hint = data["hint"]
    if not isinstance(hint, bool):
        raise ConfigError(f"'hint' must be a boolean, got {hint!r}")

    max_turns = data["max_turns"]
    if max_turns is not None:
        max_turns = _require_int(max_turns, "max_turns")
        if max_turns < 1:
            raise ConfigError("'max_turns' must be >= 1 (or null for unlimited)")

    if grid_size < 1:
        raise ConfigError("'grid_size' must be >= 1")
    if papers_needed < 1:
        raise ConfigError("'papers_needed' must be >= 1")
    if shovel_durability < 1:
        raise ConfigError("'shovel_durability' must be >= 1")
    if total_rocks < papers_needed:
        raise ConfigError(
            f"'total_rocks' ({total_rocks}) must be >= 'papers_needed' "
            f"({papers_needed})"
        )

    # Need room for every rock plus a treasure cell that carries no rock.
    capacity = grid_size * grid_size
    if total_rocks + 1 > capacity:
        raise ConfigError(
            f"grid is too small: {total_rocks} rocks + 1 treasure cell need "
            f"{total_rocks + 1} cells but the {grid_size}x{grid_size} grid only "
            f"has {capacity}"
        )

    return Config(
        grid_size=grid_size,
        seed=seed,
        papers_needed=papers_needed,
        total_rocks=total_rocks,
        shovel_durability=shovel_durability,
        hint=hint,
        max_turns=max_turns,
    )


def load_config(path: str) -> Config:
    """Load and validate a config JSON file at ``path``.

    Missing keys fall back to ``DEFAULTS``; unknown keys are rejected so typos
    don't silently do nothing.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("config file must contain a JSON object")

    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ConfigError(f"unknown config keys: {', '.join(sorted(unknown))}")

    merged = {**DEFAULTS, **raw}
    return validate(merged)
