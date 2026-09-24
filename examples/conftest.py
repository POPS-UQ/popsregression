"""Make the example helpers (``comparisons``, ``example_*``) importable in tests."""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Plotting helpers need the optional ``examples`` extra; skip them when it is
# not installed (the tests themselves never import them).
collect_ignore = []
if importlib.util.find_spec("matplotlib") is None:
    collect_ignore.append("comparisons/plotting.py")
