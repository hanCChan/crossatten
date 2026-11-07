"""Read current job state without allocating CUDA or importing the training stack."""
import csv
from core import HERE, read, now


def main():
    print('S13',now())
    if (HERE/'execution.pid').exists():print('Workflow PID:',(HERE/'execution.pid').read_text().strip())
    for phase in ['smoke','p0','p1','p2','p2_confirmation']:
        p=HERE/f'jobs_{phase}.json'
        if not p.exists():continue
        jobs=read(p);done=0
        for j in jobs:
            name=f"{j['phase']}__{j['dataset']}__{j['arm']}__f{j['fold']}__{j.get('group','all')}__s{j['seed']}"
            out=HERE/'runs'/name
            if (out/'metrics.json').exists():done+=1;continue
            if not (out/'attempt.json').exists():continue
            failed=(out/'failure.json').exists()
            print('FAILED' if failed else 'RUNNING/interrupted',name)
            if (out/'live.json').exists():
                v=read(out/'live.json');print(f"  epoch {v['epoch']}/{v['epochs_budget']}; source CE={v['source_val_ce']:.5f}; elapsed={v['seconds']/60:.1f} min; updated={v['updated_at']}")
        print(f'{phase}: {done}/{len(jobs)}')
    for name in ['decision_p0.json','decision_p2.json','decision_p2_confirmation.json']:
        if (HERE/name).exists():print(name,read(HERE/name)['status'])
    if (HERE/'execution_failure.json').exists():print('Workflow stopped with a recorded technical failure: execution_failure.json')
    if (HERE/'progress.json').exists():print('Last workflow status:',read(HERE/'progress.json')['status'])


if __name__=='__main__':main()
