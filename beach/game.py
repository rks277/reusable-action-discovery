#!/usr/bin/env python3
"""Convenience shim so the game can be launched with ``python game.py``.

Equivalent to ``python -m beach``. All real logic lives in the ``beach``
package next to this file.
"""

import sys

from beach.cli import main

if __name__ == "__main__":
    sys.exit(main())
