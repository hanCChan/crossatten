"""Bounded source-only periodic-coordinate feasibility; never trains a model."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
C_STAGE = ROOT / "research/standard_method_competition_c"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def require_committed(path):
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    blob = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:" + rel])
    if blob.replace(b"\r\n", b"\n") != Path(path).read_bytes().replace(b"\r\n", b"\n"):
        raise ValueError("Uncommitted audit prerequisite: " + rel)


def register():
    """Reads protocol documents only, not scientific arrays or weights."""
    original = C_STAGE / "PRE_REGISTRATION.json"
    plan_path = C_STAGE / "SOURCE_PLAN.json"
    reg, plan = read(original), read(plan_path)
    common = sorted(set.intersection(*(set(v["fit_rx"]) for v in plan["views"])))
    if len(common) != 6 or set(common) & set.union(*(set(v["source_selection_rx"]) for v in plan["views"])):
        raise ValueError("Expected six RX trained in both source views")
    files = [HERE / "audit.py", HERE / "README.zh-CN.md", ROOT / "tests/test_periodic_hstem_audit.py"]
    value = dict(schema_version=1, stage="PERIODIC_HSTEM_FEASIBILITY", registered_at=now(),
        base_commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        gradient_fits=0, maximum_packets_per_cell=16, minimum_packets_per_present_cell=8,
        period=20, primary_window=[20, 140], window_offsets=list(range(20)),
        min_adjacent_coherence=0.7, min_neighbor_lag_margin=0.05,
        min_packet_pass_fraction=0.9, min_rx_day_pass_fraction=0.8,
        sample_rule="top16 SHA256(periodic-feasibility-v1|pool_row) in original train_ids per common-fit RX x TX x day",
        audit_rx=common, rx_names=reg["rx_names"], allowed_days=[1, 2],
        input_whitelist={k: reg["source_input_whitelist"][k] for k in ("fit_x", "fit_y", "fit_rx", "fit_day", "train_ids", "val_ids")},
        source_registration_sha256=sha(original), source_plan_sha256=sha(plan_path),
        source_roles_read_for_IQ=["common_fit_train_only"],
        historical_held6_IQ=False, target8_IQ=False, dates3_4_IQ=False,
        checkpoint_val_IQ=False, source_selection_IQ=False, classifier_or_TX_separability_evaluated=False,
        stop_if_failed=True, automatic_window_or_period_search=False,
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    write_new(HERE / "AUDIT_REGISTRATION.json", value)
    print(json.dumps(dict(status="REGISTERED_NO_IQ_READ", audit_rx=common, max_packets=6*10*2*16)))


def coherence(a, b):
    """Normalized complex correlation, phase insensitive; zero energy fails."""
    cross = np.abs(np.sum(a * b.conj(), axis=-1))
    denom = np.sqrt(np.sum(np.abs(a)**2, axis=-1) * np.sum(np.abs(b)**2, axis=-1))
    return np.divide(cross, denom, out=np.zeros_like(cross, dtype=np.float64), where=denom > 1e-20).clip(0, 1)


def lag_coherence(z, lag):
    if not 0 < lag < z.shape[-1]:
        raise ValueError("Invalid correlation lag")
    return coherence(z[:, lag:], z[:, :-lag])


def packet_metrics(iq, reg):
    if iq.ndim != 3 or iq.shape[1:] != (2, 256):
        raise ValueError("Expected IQ [N,2,256]")
    valid = np.isfinite(iq).all(axis=(1, 2))
    z = np.where(np.isfinite(iq[:, 0]), iq[:, 0], 0).astype(np.float64) + 1j * np.where(np.isfinite(iq[:, 1]), iq[:, 1], 0)
    valid &= np.sum(np.abs(z)**2, axis=1) > 1e-20
    period = reg["period"]
    start, stop = reg["primary_window"]
    if period != 20 or (start, stop) != (20, 140) or reg["window_offsets"] != list(range(20)):
        raise ValueError("Only frozen period and window supported")
    pair, margins, rho = [], [], []
    for offset in reg["window_offsets"]:
        segment = z[:, start+offset:stop+offset]
        grid = segment.reshape(len(z), 6, period)
        pair.append(coherence(grid[:, 1:], grid[:, :-1]).min(axis=1))
        r = lag_coherence(segment, period)
        rho.append(r)
        margins.append(r - np.maximum(lag_coherence(segment, period-1), lag_coherence(segment, period+1)))
    pair, margins, rho = np.stack(pair, 1), np.stack(margins, 1), np.stack(rho, 1)
    passes = valid & (pair.min(1) >= reg["min_adjacent_coherence"]) & (margins.min(1) >= reg["min_neighbor_lag_margin"])
    windows = [(0,160), (0,200), (20,140), (40,160), (160,256), (0,256)]
    regions = {f"{a}:{b}":lag_coherence(z[:,a:b],20) for a,b in windows}
    sliding = np.stack([lag_coherence(z[:,a:a+80],20) for a in range(0,177,4)],1)
    return dict(valid=valid, passed=passes, adjacent_min_by_offset=pair, neighbor_margin_by_offset=margins,
                rho20_by_offset=rho, lag_spectrum=np.stack([lag_coherence(z,l) for l in range(1,81)],1),
                sliding80_rho20=sliding, **{"region_"+k:v for k,v in regions.items()})


def select_rows(y, rx, day, train_ids, val_ids, reg):
    if not (y.shape == rx.shape == day.shape == (34668,)):
        raise ValueError("Unexpected source metadata shape")
    if not np.array_equal(np.sort(np.r_[train_ids, val_ids]), np.arange(len(y))):
        raise ValueError("Original train/validation partition changed")
    if set(day) != {1,2} or not set(y) <= set(range(10)):
        raise ValueError("Unexpected day/class labels")
    allowed = [reg["rx_names"].index(r) for r in reg["audit_rx"]]
    groups, selected, missing = [], [], []
    for code in allowed:
        for tx in range(10):
            for date in reg["allowed_days"]:
                rows = train_ids[(rx[train_ids]==code) & (y[train_ids]==tx) & (day[train_ids]==date)]
                if not len(rows):
                    missing.append(dict(rx=reg["rx_names"][code], tx=tx, day=date))
                    continue
                rows = sorted(map(int, rows), key=lambda i:hashlib.sha256(f"periodic-feasibility-v1|{i}".encode()).digest())[:reg["maximum_packets_per_cell"]]
                selected.extend(rows)
                groups.append(dict(rx=reg["rx_names"][code],tx=tx,day=date,rows=len(rows)))
    result = np.asarray(selected, dtype=np.int64)
    if len(np.unique(result)) != len(result) or np.intersect1d(result,val_ids).size or not np.isin(result,train_ids).all():
        raise ValueError("Audit rows overlap validation or duplicate")
    return result, groups, missing


def quantiles(x):
    return {k:float(v) for k,v in zip(("min","p10","median","p90","max"),np.quantile(x,[0,.1,.5,.9,1]))}


def summarize(metrics, ids, y, rx, day, groups, missing, reg):
    passed = metrics["passed"]
    receiver_days = []
    for receiver in reg["audit_rx"]:
        code = reg["rx_names"].index(receiver)
        for date in reg["allowed_days"]:
            mask = (rx[ids]==code) & (day[ids]==date)
            receiver_days.append(dict(rx=receiver,day=date,packets=int(mask.sum()),pass_fraction=float(passed[mask].mean()) if mask.any() else 0.))
    cells=[]
    for group in groups:
        mask=(rx[ids]==reg["rx_names"].index(group["rx"])) & (y[ids]==group["tx"]) & (day[ids]==group["day"])
        cells.append(dict(**group,pass_fraction=float(passed[mask].mean())))
    checks = dict(overall_coverage=float(passed.mean()) >= reg["min_packet_pass_fraction"],
        every_rx_day_coverage=all(r["pass_fraction"]>=reg["min_rx_day_pass_fraction"] for r in receiver_days),
        present_cell_support=bool(groups) and all(g["rows"]>=reg["minimum_packets_per_present_cell"] for g in groups),
        all_rx_days_present=all(r["packets"]>0 for r in receiver_days))
    status="PASS_PERIODIC_COORDINATES_REQUIRE_SEPARATE_TRAINING_REGISTRATION" if all(checks.values()) else "STOP_FIXED_PERIODIC_WINDOW_NOT_RELIABLE"
    return dict(status=status, checks=checks, packets=len(ids), passed_packets=int(passed.sum()),
        pass_fraction=float(passed.mean()), invalid_packets=int((~metrics["valid"]).sum()),
        rho20_primary=quantiles(metrics["rho20_by_offset"][:,0]),
        worst_adjacent_coherence=quantiles(metrics["adjacent_min_by_offset"].min(1)),
        worst_neighbor_lag_margin=quantiles(metrics["neighbor_margin_by_offset"].min(1)),
        rx_day=receiver_days, cells=cells, missing_cells=missing,
        region_rho20={k.removeprefix("region_"):quantiles(v) for k,v in metrics.items() if k.startswith("region_")},
        lag_spectrum_median=np.median(metrics["lag_spectrum"],axis=0).tolist(),
        sliding80_starts=list(range(0,177,4)), sliding80_rho20_median=np.median(metrics["sliding80_rho20"],axis=0).tolist(),
        gradient_fits=0, source_selection_IQ=False, checkpoint_val_IQ=False, target_IQ=False,
        absolute_short_symbol_alignment_verified=False, TX_RX_separation_or_classifier_gain_verified=False)


def run():
    reg_path=HERE / "AUDIT_REGISTRATION.json"
    reg=read(reg_path)
    for p in [reg_path, C_STAGE/"PRE_REGISTRATION.json", C_STAGE/"SOURCE_PLAN.json"]:
        require_committed(p)
    if sha(C_STAGE/"PRE_REGISTRATION.json") != reg["source_registration_sha256"] or sha(C_STAGE/"SOURCE_PLAN.json") != reg["source_plan_sha256"]:
        raise ValueError("Frozen parent protocol bytes changed")
    for relative,digest in reg["code_sha256"].items():
        p=ROOT/relative
        require_committed(p)
        if sha(p)!=digest:
            raise ValueError("Frozen audit code changed: "+relative)
    write_new(HERE/"AUDIT_ATTEMPT.json",dict(started_at=now(),registration_sha256=sha(reg_path),status="ATTEMPT_STARTED_NO_RETRY"))
    log=[]

    def source(name, decode=True):
        item=reg["input_whitelist"][name]
        p=Path(item["path"])
        log.append(dict(time=now(),action="hash_source_bytes",name=name,path=str(p)))
        # Persist before opening, including failure cases.
        (HERE/"ACCESS_LOG.json").write_text(json.dumps(log,indent=2)+"\n",encoding="utf-8",newline="\n")
        if sha(p)!=item["sha256"]:
            raise ValueError("Source hash changed: "+name)
        if decode:
            log.append(dict(time=now(),action="decode_metadata",name=name,path=str(p)))
            return np.load(p,allow_pickle=False)
        return p

    arrays={name:source(name) for name in ("fit_y","fit_rx","fit_day","train_ids","val_ids")}
    y,rx,day=(arrays["fit_"+k] for k in ("y","rx","day"))
    ids,groups,missing=select_rows(y,rx,day,arrays["train_ids"],arrays["val_ids"],reg)
    if len(ids)==0:
        raise ValueError("No eligible source rows")
    iq_path=source("fit_x",decode=False)
    log.append(dict(time=now(),action="mmap_header_and_copy_common_fit_train_rows_only",path=str(iq_path),rows=len(ids),indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
    (HERE/"ACCESS_LOG.json").write_text(json.dumps(log,indent=2)+"\n",encoding="utf-8",newline="\n")
    mmap=np.load(iq_path,mmap_mode="r",allow_pickle=False)
    if mmap.shape!=(34668,2,256) or mmap.dtype!=np.float32:
        raise ValueError("Unexpected source IQ shape or dtype")
    iq=np.array(mmap[ids],copy=True)
    del mmap
    metrics=packet_metrics(iq,reg)
    np.savez_compressed(HERE/"PACKET_METRICS.npz",pool_ids=ids,y=y[ids],rx=rx[ids],day=day[ids],**metrics)
    result=summarize(metrics,ids,y,rx,day,groups,missing,reg)
    result.update(completed_at=now(),audit_registration_sha256=sha(reg_path),packet_metrics_sha256=sha(HERE/"PACKET_METRICS.npz"),
        indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),IQ_copy_sha256=hashlib.sha256(iq.tobytes()).hexdigest(),
        access_log_sha256=sha(HERE/"ACCESS_LOG.json"))
    write_new(HERE/"AUDIT_RESULT.json",result)
    print(json.dumps({k:result[k] for k in ("status","packets","pass_fraction","rho20_primary","worst_adjacent_coherence","checks")},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["register","run"])
    args=parser.parse_args()
    {"register":register,"run":run}[args.command]()
