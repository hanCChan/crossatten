"""Current authorization activates only B; prior plan and A/C records immutable."""
from pathlib import Path
import subprocess
import torch
from s27b_common import HERE,ROOT,P,read,write,sha,now


def main():
    assert not (HERE/'registration.json').exists();assert read(HERE/'GATE.json')['status']=='PASS'
    original=ROOT/'27-mechanism-identification/PLAN_B.json';assert sha(original)==sha(HERE/'PLAN_B.original.json')
    plan=read(original);prior=read(ROOT/'27-mechanism-identification/registration.json');refs=prior['references']
    streams=[]
    for r in refs:
        folder=Path(r['checkpoint']).parent
        for n,h in r['artifacts_sha256'].items():assert sha(folder/n)==h
        if r['arm']=='LP':streams.append(dict(fold=r['fold'],seed=r['seed'],**read(folder/'streams.json')))
    assert len(streams)==6
    code=[HERE/'resume.py']+list(HERE.glob('s27b_*.py'))+list((ROOT/'lib').rglob('*.py'))
    code += [ROOT/folder/name for folder,names in {
        '13-mixer-cfo-mechanisms':['train.py','models.py','core.py','data.py','metrics.py','wifi.py'],
        '15-hstem-attribution':['s15_models.py','s15_core.py'],
        '21-hstem-mechanisms':['s21_models.py','s21_core.py'],
        '25-tifs-integration':['geometry_stats.py'],
        '26-long-patch-control':['s26_models.py','s26_common.py'],
        '27-mechanism-identification':['s27_models.py','s27_common.py']}.items() for name in names]
    evidence=[original,HERE/'PLAN_B.original.json',HERE/'GATE.json',ROOT/'27-mechanism-identification/FINAL_AUDIT.json',
        ROOT/'27-mechanism-identification/registration.json',ROOT/'configs/protocol.json',ROOT/'15-hstem-attribution/architecture_search.json']
    write(HERE/'registration.json',dict(stage='S27_B_TOKENIZER_BACKEND_INTERACTION',time=now(),
        base_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        authority='Currentuser explicitlyrequests B12fits basedon636da47 and attachedinstructions. Activates B only in thisnewstage; previous automaticbudget0 and C/A records preserved, not retroactivelyedited.',
        maximum_new_fits=12,jobs=plan['jobs'],original_plan_sha256=sha(original),original_plan=plan,
        references=refs,source_caches=prior['source_caches'],reference_streams=streams,
        training=plan['training'],statistics=plan['statistics'],
        permissions=dict(source='Original5RX/fold160train40val/TX/RX; source only insidefits',
            dev='All12newmodels/configs/history sealed andcommitted beforeanynewDev inference; fourcells commonGPU',
            confirm=False,blind=False,C=False,IQ_only=False,ask_strict=False),
        source_features=dict(packets='All2000sourceval/fold, fixedlast40/TX/RX,all24models',
            taps=['tokens','pre_backend','final'],pool='Mean32tokens to64D; finalclassifierinput64D',
            normalization='PerpacketL2; epsilon1e-12 ratios; identicalS25geometry_stats.py definitions.',
            invalid='Record undefined forzero/nonfinite embeddings; no alternate normalization or layerselection.',
            scope='Descriptive source-selected features, not causal informationflow,MI or physicalRXinvariance.'),
        recovery='UniqueprocessJSONtempfiles andbounded Windowsrename retry, full-epoch optimizer/model/allRNG. No resetseed; no targetdrivenfits.',
        stop='Stop after12B fits/report. NoC,IQ-only,ASKStrict,newheadselection,weakseedreplacement ornewarchitectures.',
        canonical_training_sha256=sha(ROOT/'13-mixer-cfo-mechanisms/train.py'),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in code},
        evidence_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in evidence}))
    print('Registered B12fits; originalplanunchanged, noC. CommitbeforeRF reads.',flush=True)


if __name__=='__main__':main()
