import sys
from pathlib import Path

# Make the `num9` package importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
