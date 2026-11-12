"""Check identity, selection, all baseline arrays and forbidden data access."""
import csv
import itertools
import json
import shutil
from s22_core import HERE,read,write,sha,now,job_id
from s22_evaluate import sealed,reference_checks
from s22_report import outpath
from s22_models import ARMS,REFERENCES
from s22_data import PROTOCOL,roles

def audit():
    reg=sealed();reference_checks(reg);evidence=HERE/'evidence';evidence.mkdir(exist_ok=True);manifest={}
    for j in reg['jobs']:
        p=HERE/'runs'/job_id(j);m=read(p/'metrics.json');cfg=read(p/'config.json')
        with (p/'history.csv').open(encoding='utf-8') as f:history=list(csv.DictReader(f))
        best=min(history,key=lambda r:float(r['source_val_ce']))
        assert len(history)==m['epochs']==cfg['epochs_budget']==200 and cfg['batch']==64
        assert int(best['epoch'])==m['best_epoch'] and float(best['source_val_ce'])==m['best_source_ce']
        assert m['model_sha256']==sha(p/'model.pt')
        for name in ['config.json','metrics.json','history.csv','pairing.json','evaluation.json']:
            out=evidence/p.name/name;out.parent.mkdir(exist_ok=True);shutil.copyfile(p/name,out);manifest[out.relative_to(HERE).as_posix()]=sha(out)
    count=0
    for arm,fold,seed in itertools.product(REFERENCES,[0,1],[201,202,203]):
        p=outpath(arm,fold,seed);d=read(p/'evaluation.json')
        assert len(d['baseline_prediction_recovery'])==9 and all(v['float32_predictions_bit_exact'] for v in d['baseline_prediction_recovery'])
        count+=len(d['baseline_prediction_recovery']);out=evidence/p.name/'evaluation.json';out.parent.mkdir(exist_ok=True);shutil.copyfile(p/'evaluation.json',out);manifest[out.relative_to(HERE).as_posix()]=sha(out)
    assert count==216
    entries=[json.loads(v) for v in (HERE/'access.jsonl').read_text(encoding='utf-8').splitlines()];seal=read(HERE/'decision_p1_source.json')
    for e in entries:
        assert e['dataset']=='Shen' and e['rx'] not in PROTOCOL['blind_receivers_never_open']
        allowed=roles(e['fold'])['source'] if e['role']=='source' else roles(e['fold'])['dev_val'] if e['role']=='dev_val' else PROTOCOL['confirm_receivers'] if e['role']=='confirm' else []
        assert e['rx'] in allowed
        if e['role']!='source':assert e['time']>=seal['time']
    shutil.copyfile(HERE/'access.jsonl',evidence/'access.jsonl');manifest['evidence/access.jsonl']=sha(evidence/'access.jsonl')
    for rel,digest in manifest.items():assert sha(HERE/rel)==digest
    write(HERE/'FINAL_AUDIT.json',dict(time=now(),status='PASS',formal_fits=12,reused_fits=24,all_epochs_200=True,minimum_source_ce_checkpoints=True,strict_baseline_arrays=216,no_blind_or_wisig_access=True,all_target_access_after_seal=True,artifacts_sha256=manifest))
    print('PASS:12 source-selected fits,24 strict controls,216 baseline RX arrays,access and artifacts',flush=True)

if __name__=='__main__':audit()
