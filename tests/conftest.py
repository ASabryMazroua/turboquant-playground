"""Ensure the repo root is importable so ``import turbo_kv`` works under pytest."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
