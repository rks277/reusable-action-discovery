"""Entry point so the game runs with ``python -m beach``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
