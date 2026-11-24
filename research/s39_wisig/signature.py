"""Source-validation calibration and two-view transfer of damage signatures."""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"s34_wisig"))
from common import DEVICE, VIEWS, FrozenStudy, featurize, restrict_torch, sha, tensor, torch, write
from preamble import decompose, ratio, remove_line
from extract import fit_reference, physical
from experiment import build_original, predict
from physics import complex_iq
sys.path.insert(0,str(HERE.parent/"s36_wisig"))
from run import qualified_models

MEASURES = ("rank1_residual","phase_residual_rad","repeat_loss","spectral_distance")
TYPES = ("white","phase_walk")
DOSES = (.04,.08,.12,.16,.24)


def perturb(x, kind, dose, seed):
    z = complex_iq(x)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    if kind=="white":
        vector = complex_iq(torch.randn(x.shape,generator=g,device=DEVICE))
        out = z + vector*(dose/ratio(vector,z))[:,None]
    else:
        t = torch.arange(z.shape[1],device=DEVICE,dtype=x.dtype)
        t -= t.mean()
        step = remove_line(torch.randn(z.shape,generator=g,device=DEVICE).cumsum(1),t)
        energy = z.abs().square()
        size = ((energy*step.square()).sum(1)/energy.sum(1)).sqrt()
        step = step/size[:,None]
        def rotate(scale):
            p = step*scale[:,None]
            return z*torch.polar(torch.ones_like(p),p)
        lo = torch.zeros(len(z),device=DEVICE)
        hi = lo + 2*dose
        assert bool((ratio(rotate(hi)-z,z)>dose).all())
        for _ in range(17):
            center = (lo+hi)/2
            above = ratio(rotate(center)-z,z)>dose
            lo,hi = torch.where(above,lo,center), torch.where(above,center,hi)
        out = rotate((lo+hi)/2)
    error = float((ratio(out-z,z)-dose).abs().max())
    assert error < 6.e-6
    return torch.stack((out.real,out.imag),1), error


def group_metadata(data):
    r,y,d = (data[k] for k in ("rx","y","day"))
    return [(int(a),np.flatnonzero((r==a)&(y==b)&(d==c)))
             for a,b,c in np.unique(np.column_stack((r,y,d)),axis=0)]


def profile(data, native_fields, shifted_fields, reference, after):
    truth, rx = data["y"], data["rx"]
    by_rec, cell_counts, natural, synthetic = {},[],[],[]
    for seed,(op,npred) in enumerate(zip(reference,after)):
        easy = op == truth
        hard = ~easy
        new = easy & (npred != truth)
        natural.append(np.bincount(10*truth[hard]+op[hard],minlength=100).reshape(10,10))
        synthetic.append(np.bincount(10*truth[new]+npred[new],minlength=100).reshape(10,10))
        for r,idx in group_metadata(data):
            before_ok,before_bad,new_bad = (idx[m[idx]] for m in (easy,hard,new))
            if min(len(before_ok),len(before_bad),len(new_bad))<8:
                continue
            real = [float(np.median(native_fields[m][before_bad])-np.median(native_fields[m][before_ok]))
                    for m in MEASURES]
            art = [float(np.median(shifted_fields[m][new_bad]-native_fields[m][new_bad])) for m in MEASURES]
            by_rec.setdefault(r,[]).append((real,art))
            cell_counts.append((r,int(truth[idx[0]]),int(data["day"][idx[0]]),seed))
    f0,f1 = np.sum(natural,axis=0), np.sum(synthetic,axis=0)
    eligible = (f0.sum(1)>=20)&(f1.sum(1)>=20)
    distances, null = [], []
    for t in np.flatnonzero(eligible):
        q,p = f0[t]/f0[t].sum(), f1[t]/f1[t].sum()
        uniform = np.full(10,1/9)
        uniform[t] = 0
        distances.append(.5*np.abs(q-p).sum())
        null.append(.5*np.abs(q-uniform).sum())
    averages = np.stack([np.mean(records,axis=0) for records in by_rec.values()]) if by_rec else None
    if averages is None:
        target = response = signed_ratio = None
    else:
        target,response = averages.mean(0)
        signed_ratio = response/np.where(np.abs(target)>1e-8,target,np.nan)
        if not np.isfinite(signed_ratio).all():
            signed_ratio = None
    return dict(real_delta=target.tolist() if target is not None else None,
        synthetic_delta=response.tolist() if response is not None else None,
        ratio=signed_ratio.tolist() if signed_ratio is not None else None,
        shared_cells=len(cell_counts), receivers=len(by_rec), compared_tx=int(eligible.sum()),
        confusion_tv=float(np.mean(distances)) if distances else None,
        uniform_wrong_tv=float(np.mean(null)) if null else None,
        natural_error_matrix=f0.tolist(), new_error_matrix=f1.tolist())


def measure(view,data,models,refs,doses,batch_tag):
    x = tensor(data["x"])
    y,day = (data[i] for i in ("y","day"))
    native = decompose(x,refs,y,day)
    ref = build_original("ManyRx",24101)
    f = featurize(ref,x)
    old = [predict(model,f).argmax(1).cpu().numpy() for _,model in models]
    result = {}
    for kind,values in doses.items():
        result[kind] = {}
        seed = batch_tag+int(view[-1])*100
        for dose in values:
            changed,error = perturb(x,kind,dose,seed)
            issues = decompose(changed,refs,y,day)
            f = featurize(ref,changed)
            after = [predict(model,f).argmax(1).cpu().numpy() for _,model in models]
            result[kind][str(dose)] = dict(**profile(data,native,issues,old,after),max_dose_error=error)
    return result


def select(fits):
    selection = {}
    for k in TYPES:
        cost = {}
        for d in DOSES:
            rows = [fits[v][k][str(d)] for v in VIEWS]
            if all(r["ratio"] is not None for r in rows):
                cost[d] = float(np.mean([np.mean(np.abs(np.asarray(r["ratio"])-1)) for r in rows]))
        if cost:
            selection[k] = min(cost,key=cost.get)
    return selection


@torch.inference_mode()
def execute(s29,s33,out):
    if (out/"B.json").exists():
        raise FileExistsError("S39-B already sealed")
    study = FrozenStudy(s29,s33,out)
    fits, assets, stored = {},{},{}
    for view in VIEWS:
        models = [(s,m) for s,arch,m in qualified_models(study,view,assets) if arch=="A1"]
        fit = study.source.load(view,"fit_train","s39_signature_profile")
        refs = fit_reference(physical(tensor(fit["x"])),fit)
        val = study.source.load(view,"checkpoint_val","s39_signature_calibration")
        fits[view] = measure(view,val,models,refs,{k:DOSES for k in TYPES},39017)
        stored[view] = models,refs
        print(view,"source calibration complete",flush=True)
    picked = select(fits)
    write(out/"B_CALIBRATION.json",dict(source_validation=fits,chosen_doses=picked))
    held = {}
    for view in VIEWS:
        packet = study.source.load(view,"source_selection","s39_signature_readout")
        held[view] = measure(view,packet,*stored[view],{k:[d] for k,d in picked.items()},39117)
        print(view,"held signature complete",flush=True)
    gates = {}
    for k,d in picked.items():
        gate = {}
        for view in VIEWS:
            row = held[view][k][str(d)]
            gate[view] = dict(physical=row["real_delta"] is not None and row["ratio"] is not None and
               all(t>0 and .5<=a<=1.5 for t,a in zip(row["real_delta"],row["ratio"])),
               confusion=row["compared_tx"]>=5 and row["confusion_tv"]<=.40 and
               row["uniform_wrong_tv"]-row["confusion_tv"]>=.10)
        gates[k] = gate
    passed = [k for k, v in gates.items() if all(all(flags.values()) for flags in v.values())]
    write(out/"B.json",dict(selected=picked,held=held,gates=gates,passes=passed,measure_order=MEASURES,
          engine=sha(Path(__file__)),calibration_sha256=sha(out/"B_CALIBRATION.json"), frozen_a1=assets,
          new_fits=0))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for n in ("s29","s33","output"):
        parser.add_argument(n,type=Path)
    a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    restrict_torch()
    execute(a.s29,a.s33,a.output)
