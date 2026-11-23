"""Read-only archived source-metric recovery; differences are disclosed."""
from pathlib import Path
import gc
import numpy as np
import torch
import torch.nn.functional as F
from s27_common import HERE,read,write,validate,now
from s27_data import source
from s27_models import build
from s27_training import setup
import train as canonical


def main():
    registration=validate();rows=[]
    for fold in (0,1):
        training,validation=source(fold);del training;gc.collect()
        x=torch.from_numpy(validation['x']).cuda();y=torch.from_numpy(validation['y']).cuda();rx=torch.from_numpy(validation['rx']).cuda()
        for record in [r for r in registration['references'] if r['fold']==fold]:
            setup(record['data_seed']);model=build(record['arm'],record['data_seed']).cuda().eval()
            model.load_state_dict(torch.load(record['checkpoint'],map_location='cuda',weights_only=True),strict=True)
            logits=canonical.predict(model,x)
            ce=float(torch.stack([F.cross_entropy(logits[rx==r],y[rx==r]) for r in torch.unique(rx)]).mean())
            current=canonical.score(logits,y,validation['rx'])
            old=read(Path(record['checkpoint']).parent/'metrics.json')
            rows.append(dict(id=record['id'],source_ce=ce,archived_ce=old['best_source_ce'],ce_delta=ce-old['best_source_ce'],
                source_accuracy=current['summary']['acc_mean'],archived_accuracy=old['source_validation']['summary']['acc_mean'],
                source_metrics_exact=current==old['source_validation']))
            del model,logits
        del x,y,rx,validation;gc.collect();torch.cuda.empty_cache()
    write(HERE/'SOURCE_RECOVERY.json',dict(status='COMPLETE_READ_ONLY',time=now(),rows=rows,
        all_metrics_exact=all(r['source_metrics_exact'] for r in rows),
        interpretation='Numeric recovery is reported; frozen historical values are never overwritten. S27 primary pairing uses common fresh inference of all arms. No target packet read or parameter choice.'))


if __name__=='__main__':main()
