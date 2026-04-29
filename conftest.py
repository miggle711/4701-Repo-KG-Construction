import sys
from pathlib import Path

# Ensure run.py at repo root is importable from e2e tests
sys.path.insert(0, str(Path(__file__).parent))
