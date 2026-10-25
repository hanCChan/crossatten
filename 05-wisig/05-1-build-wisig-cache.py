#!/usr/bin/env python3
"""Convert existing compact pickles, stop at any mismatch with published training counts."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from xa.data import sha256_file
from xa.wisig import DATA, build_dataset, link_official_layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=["ManyRx", "ManySig"], default=["ManyRx", "ManySig"])
    args = ap.parse_args()
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    for name in args.datasets:
        result = build_dataset(name)
        result["source_sha256"] = sha256_file(DATA / f"{name}.pkl")
        archive = (HERE.parent / "../gai/data/wisig-manyrx-kaggle/ManyRx.kaggle.v1.zip"
                   if name == "ManyRx" else DATA / "ManySig.official.zip")
        result["archive_sha256"] = sha256_file(archive)
        result["source"] = ("user's existing Kaggle v1 copy" if name == "ManyRx" else
                            "UCLA official WiSig page -> Google Drive file 1szuns8MhcYocdbipK9t9TM9MLgEMklxk")
        (out / f"05-1-{name}-manifest.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(dict(dataset=name, checks=result["checks"], counts_match=result["counts_match"])), flush=True)
        if not result["counts_match"]:
            raise SystemExit(f"{name}: published count mismatch; STOP before accuracy evaluation")
        link_official_layout(name)


if __name__ == "__main__":
    main()
