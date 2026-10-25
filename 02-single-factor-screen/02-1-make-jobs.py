#!/usr/bin/env python3
"""02-1  Write the job list.   python 02-1-make-jobs.py smoke|explore|confirm"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from xa.jobs import make_jobs  # noqa: E402

if __name__ == "__main__":
    make_jobs(HERE, sys.argv[1])
