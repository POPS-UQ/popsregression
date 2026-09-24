"""Make the example helpers (``comparisons``, ``example_*``) importable in tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
