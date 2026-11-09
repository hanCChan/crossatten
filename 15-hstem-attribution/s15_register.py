from pathlib import Path
from s15_core import HERE,ROOT,S13,read,write,sha,now,git,commit
from s15_models import search,build,receptive_field
import torch

S14=Path('D:/crossatten/worktrees/s14-stem-mixer/14-stem-mixer-attribution')
def main():
    if (HERE/'registration.json').exists():raise ValueError('Registration already exists')
    torch.set_num_threads(2)
    architecture=search();write(HERE/'architecture_search.json',architecture)
    audit={}
    arms=['E1_patch_mha','E2_patch_hstu','A1_mha','A3_rfhstu','C2_lite_mha','C2_lite_hstu']
    for arm in arms:
        model=build(arm,123)
        audit[arm]=dict(params=sum(p.numel() for p in model.parameters()),bn_layers=sum(isinstance(m,torch.nn.BatchNorm1d) for m in model.modules()),**receptive_field(model))
    write(HERE/'architecture_audit.json',audit)
    reused={}
    for arm in arms[:4]:
        stage=S14 if arm.startswith('E') else S13
        for fold in [0,1]:
            for seed in [201,202,203]:
                name=f'p1__Shen__{arm}__f{fold}__all__s{seed}';out=stage/'runs'/name
                metric=read(out/'metrics.json')
                if sha(out/'model.pt')!=metric['model_sha256']:raise ValueError('Reference checkpoint mismatch')
                reused[name]=dict(root=str(stage),model_sha256=metric['model_sha256'],config_sha256=sha(out/'config.json'),target_metrics_sha256=sha(out/'target_metrics.json'))
    code=list(HERE.glob('*.py'))+list((HERE/'tests').glob('*.py'))+list((ROOT/'13-mixer-cfo-mechanisms').glob('*.py'))
    code+=list((ROOT/'lib/xa').glob('*.py'))+list((ROOT/'lib/legacy_rfhstu').glob('*.py'))+[ROOT/'configs/protocol.json',ROOT/'09-phase-invariance/phase_features.py']
    jobs=[dict(phase='p1',dataset='Shen',arm=a,fold=f,seed=s) for f in [0,1] for s in [201,202,203] for a in ['C2_lite_mha','C2_lite_hstu']]
    reg=dict(stage='S15',time=now(),base_git=git('rev-parse','HEAD'),code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(code)},
        architecture_search_sha256=sha(HERE/'architecture_search.json'),architecture_audit_sha256=sha(HERE/'architecture_audit.json'),reused=reused,jobs=jobs,
        data='Unchanged P-H5x160; source160/TX, sourceval40/TX; target only after all12 checkpoint seals; no blind',
        training='Frozen S13 fit function; AdamW lr1e-3 wd5e-4 batch64 AWGN[0,50] warm5/cos200; source RX-equal CE checkpoint; paired seed/data/noise; eval batch128',
        purpose='HStem versus automatically parameter-matched Patch-Lite under both mixers; not proof of unique hierarchical causation',
        primary_contrasts=['HStem-Lite for MHA','HStem-Lite for HSTU','Lite-Patch for MHA','Lite-Patch for HSTU'],
        uncertainty='RX cluster5000 bootstrap conditional on fitted seeds; report6 pair directions, worstRX and macroF1; confirm reporting only',
        no_bn=dict(status='NOT_SCHEDULED',new_fits=0),
        limits=['Grouped low-rank Patch-Lite changes inductive bias, not capacity alone','CNN RF differs from physical/global receptive field','No tuning by dev/confirm results'],
        execution=dict(gpu=0,concurrency=1,automatic_retry=False,new_formal_fits=12))
    write(HERE/'registration.json',reg);write(HERE/'jobs_p1.json',jobs)
    paths=[p for p in HERE.iterdir() if p.is_file()]+list((HERE/'tests').glob('*.py'))
    print('Registration commit:',commit(paths,'S15 preregister automatically parameter-matched Patch-Lite attribution,12 paired fits'))

if __name__=='__main__':main()
