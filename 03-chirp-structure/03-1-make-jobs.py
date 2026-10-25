#!/usr/bin/env python3
"""Write smoke, exploration, or confirmation tasks using the frozen protocol."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from xa.jobs import make_jobs

if __name__ == "__main__":
    make_jobs(HERE, sys.argv[1])
