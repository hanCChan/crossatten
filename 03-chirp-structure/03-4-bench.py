#!/usr/bin/env python3
"""Benchmark all eight arms without concurrent GPU training."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from xa.bench import benchmark

if __name__ == "__main__":
    benchmark(json.loads((HERE / "arms.json").read_text())["arms"], HERE / "summary")
