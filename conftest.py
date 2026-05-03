"""Repo-level pytest conftest.

Adds the repo root to ``sys.path`` so test modules can import the
``bench.harness`` package without requiring a project install. The file
itself is intentionally empty otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
