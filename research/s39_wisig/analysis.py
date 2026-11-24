"""Summarize the three frozen or equal-budget ManyRx arms on identical packets."""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"s29_wisig"))
from source import read, sha, write
from summarize import NAMES, check_access, csv_write, group_scores, packet_scores, pair_summary

VIEWS=("V0","V1")
ARMS=("A0","A1","extra")


def execute(root):
    out=root/"D"
    definition=read(out/"EXECUTION.json")
    assert definition["protocol"]==sha(HERE/"PROTOCOL.zh-CN.md")
    assert definition["engine"]==sha(HERE/"compute_control.py")
    assert definition["audit_hashes"]=={k:sha(root/(k+".json")) for k in "ABC"}
    seal=read(out/"FIT_SEAL.json")
    assert seal["fits"]==8 and seal["updates"]==37600 and len(seal["records"])==4
    role_checks=check_access(out,[("fit",v,r) for v in VIEWS for r in ("fit_train","checkpoint_val")]
                            +[("new_prediction",v,"source_selection") for v in VIEWS])
    summaries=[]
    for entry in seal["records"]:
        view,seed=entry["view"],entry["seed"]
        trial=out/f"{view}_{seed}"
        assert sha(trial/"down/C1_selected.pt")==entry["weights_sha"]
        assert entry["down"]["identical_downstream_order_to_a1"]
        assert entry["down"]["updates"]==5080
        assert entry["first"]["updates"]==4320 and entry["first"]["observations"]==20736*40
        assert all(h["exposures"]==20736 for h in read(trial/"first/HISTORY.json"))
        summaries.append(dict(view=view,seed=seed,epoch_pretrain=entry["first"]["selected_epoch"],
            epoch_downstream=entry["down"]["best_epochs"]["C1"],source_pretrain_rows=entry["first"]["unique_fit_packets"],
            pretrain_seconds=entry["first"]["seconds"],downstream_seconds=entry["down"]["costs"]["total_fit_loop_seconds"]))
    prediction_seal=read(out/"PREDICTION_SEAL.json")
    metrics={arm:np.empty((2,2,6,3)) for arm in ARMS}
    names={}
    rx_rows,tx_rows=[],[]
    for index,view in enumerate(VIEWS):
        file=out/f"PRED_{view}.npz"
        assert sha(file)==prediction_seal[view]
        with np.load(file,allow_pickle=False) as z:
            y,rx=z["y"],z["rx"]
            rx_codes=np.unique(rx)
            assert len(rx_codes)==6
            names[view]=[NAMES[c] for c in rx_codes]
            for seed in (0,1):
                for arm in ARMS:
                    p,ce=packet_scores(z[f"{seed}_{arm}"],y)
                    assert np.isfinite(ce).all()
                    for ri,r in enumerate(rx_codes):
                        mask=rx==r
                        scored=group_scores(y[mask],p[mask],ce[mask])
                        metrics[arm][index,seed,ri]=list(scored.values())
                        key=dict(view=view,seed=seed,arm=arm,rx=NAMES[r])
                        rx_rows.append(dict(**key,rows=int(mask.sum()),**scored))
                        for tx in range(10):
                            cell=mask&(y==tx)
                            tx_rows.append(dict(**key,tx=tx,rows=int(cell.sum()),wrong=int((p[cell]!=y[cell]).sum())))
    assert not set(names["V0"]) & set(names["V1"])
    scores={}
    for arm,values in metrics.items():
        scores[arm]=dict(acc=float(values[...,0].mean()),macro_f1=float(values[...,1].mean()),
            ce=float(values[...,2].mean()),worst_rx=float(values[...,0].mean(1).min()),
            view_accuracy=values[...,0].mean((1,2)).tolist())
    pairs={}
    for left,right in (("A1","A0"),("extra","A0"),("A1","extra"),("extra","A1")):
        delta=metrics[left]-metrics[right]
        pairs[f"{left}_minus_{right}"]=dict(**pair_summary(delta[...,0]),macro_f1_pp=float(delta[...,1].mean()))
    primary=pairs["extra_minus_A1"]
    gates=dict(clean_ge_1pp=primary["mean_pp"]>=1,both_views_positive=min(primary["view_pp"])>0.,
               interval_above_zero=primary["interval95_pp"][0]>0,
               worst_rx_ge_minus_1pp=primary["worst_rx_pp"]>=-1,
               macro_f1_nondecreasing=primary["macro_f1_pp"]>=0.)
    a1=pairs["A1_minus_extra"]
    supports_a1_material_advantage=min(a1["view_pp"])>0 and a1["interval95_pp"][0]>0
    old=read(root/"C.json")["effect"]
    assert np.isclose(pairs["A1_minus_A0"]["mean_pp"],old["mean_pp"],rtol=0.,atol=1e-11)
    csv_write(out/"RX_METRICS.csv",rx_rows)
    csv_write(out/"TX_RX_METRICS.csv",tx_rows)
    write(out/"SUMMARY.json",dict(metrics=scores,pairs=pairs,gates=gates,
         a1_advantage_over_matched_target_exposure=supports_a1_material_advantage,
         disposition="CANDIDATE_INDEPENDENT_CONFIRMATION" if all(gates.values()) else "NO_NEW_CANDIDATE",
         receiver_names=names,all_fours=summaries,role_payload_checks=role_checks,
         new_fits=seal["fits"],updates=seal["updates"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    execute(parser.parse_args().output)
