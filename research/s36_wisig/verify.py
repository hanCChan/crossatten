"""Independent QA for S36 source partitions and main numerical claims."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_npz(path, digest):
    assert sha(path) == digest
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def matched(x, y):
    assert np.allclose(x, y, rtol=0, atol=1e-10)


def verify(root, s33, s34):
    ad, e = root/"AD", root/"E"
    summary, physical = read(ad/"SUMMARY.json"), read(e/"SUMMARY.json")
    neural, preamble, hist, old34 = (read(p/"RECEIPT.json") for p in (ad,e,s33,s34))
    actual, dual, shuffled, oracles = {m: [] for m in ("HStem","A1")}, [], [], []
    component = {m: [] for m in ("raw","shared","residual","joint")}
    checks = 0
    for view in ("V0","V1"):
        current = read_npz(ad/f"PRED_{view}.npz", neural["predictions"][view])
        previous = read_npz(s33/f"AUDIT_{view}.npz", hist["packets"][view])
        earlier = read_npz(s34/f"PRED_{view}.npz", old34["predictions"][view])
        blocks = read_npz(e/f"PRED_{view}.npz", preamble["predictions"][view])
        for field in ("y","rx","day","ids"):
            assert np.array_equal(current[field], blocks[field])
            assert np.array_equal(current[field], previous["source_selection_"+field])
            assert np.array_equal(current[field], earlier[field])
        y, rx = (current[k] for k in ("y","rx"))
        for seed in (0,1):
            prefix = f"{seed}_HStem_"
            for tap in ("token","mix1"):
                assert np.array_equal(current[prefix+tap+"_linear"], earlier[prefix+tap+"_mean_ridge"])
                assert np.array_equal(current[prefix+tap+"_linear"], current[prefix+tap+"_linear_logits"].argmax(1))
                checks += 1
            for arch in ("HStem","A1"):
                prefix = f"{seed}_{arch}_"
                prior = f"{seed}_HStem_source_selection_logits" if arch == "HStem" else f"{seed}_A1_logits"
                assert np.array_equal(current[prefix+"native"], previous[prior])
                native = current[prefix+"native"].argmax(1)==y
                after = current[prefix+"early_fusion"].argmax(1)==y
                early = current[prefix+"token_linear"]==y
                actual[arch].extend(100*(after.astype(float)-native)[rx==r].mean() for r in np.unique(rx))
                if arch=="A1":
                    capacity = current[prefix+"deep_double"].argmax(1)==y
                    placebo = current[prefix+"shuffle_fusion"].argmax(1)==y
                    dual.extend(100*(after.astype(float)-capacity)[rx==r].mean() for r in np.unique(rx))
                    shuffled.extend(100*(after.astype(float)-placebo)[rx==r].mean() for r in np.unique(rx))
                    oracles.extend(100*(native | early)[rx==r].mean() for r in np.unique(rx))
        for key in component:
            component[key].extend(100*(blocks[key+"_ridge"]==y)[rx==r].mean() for r in np.unique(rx))
    for arch in actual:
        matched(np.mean(actual[arch]), summary["fusion"][arch]["vs_original"]["mean_pp"])
    matched(np.mean(dual), summary["fusion"]["A1"]["vs_capacity"]["mean_pp"])
    matched(np.mean(shuffled), summary["fusion"]["A1"]["vs_shuffled"]["mean_pp"])
    matched(np.mean(oracles), summary["transitions"]["A1"]["native_head"]["rates"]["all"]["oracle_pct"])
    for key, values in component.items():
        matched(np.mean(values), physical["readout"][key]["tx_linear"])
    assert sha(ad/"CALIBRATION.json") == neural["calibrations"]
    for row in read(ad/"CALIBRATION.json"):
        assert row["alpha"] == (0.,.125,.25,.5,1.)[int(np.argmin(row["fusion_ce"]))]
        for side in ("deep","second"):
            fit = row[side]
            assert fit["scale"] == (.5,1.,2.,4.,8.,16.,32.)[int(np.argmin(fit["calibration_ce"]))]
    checked_heads = 0
    for name, digest in neural["nonlinear"].items():
        assert sha(ad/name) == digest
        d = read(ad/name)
        chosen = np.argmin(np.asarray(d["ce_history"]), axis=0)+1
        assert np.array_equal(chosen, list(d["selected_epochs"].values()))
        checked_heads += len(chosen)
    assert summary["decision"]["verdict"] == ("CANDIDATE_SEPARATE_CONFIRMATION"
          if all(summary["decision"]["gate"].values()) else "STOP_SOURCE_GATE")
    qa = dict(status="PASS", historical_readout_exact=checks, frozen_original_heads=8,
              calibration_entries=len(read(ad/"CALIBRATION.json")), nonlinear_epoch_choices=checked_heads,
              preamble_tx_endpoints=len(component), matched_primary_recounts=5, backbone_fits=0)
    (root/"QA.json").write_text(json.dumps(qa,indent=2)+"\n",encoding="utf-8")
    print(qa)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("output","s33","s34"):
        parser.add_argument(name,type=Path)
    a = parser.parse_args()
    verify(a.output,a.s33,a.s34)
