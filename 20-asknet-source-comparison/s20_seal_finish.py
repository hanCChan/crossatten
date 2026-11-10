"""Documented pre-held serialization handoff; never fits or changes a model.

The original registered seal() duplicated fold while expanding baseline metadata.
All four fits succeeded before the TypeError. Preserve original code/registration
and repair only dictionary assembly in an additive, committed handoff.
"""
import json
import subprocess
from s20_core import HERE,ROOT,read,write,sha,now,validate,committed
from s20_run import outdir,olddir,evaluate,decide


def run():
    validate()
    if (HERE/'seal_pilot.json').exists():raise RuntimeError('Seal already exists; no automatic re-publication')
    if (HERE/'held_attempt.json').exists():raise RuntimeError('Held evaluation already attempted; this correction must precede it')
    entries=[json.loads(line) for line in (HERE/'access.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(entries)!=16 or any(r['kind']!='fit_cache' for r in entries):raise ValueError('Unexpected pre-held data access')
    completed={};refs={};metadata={}
    reuse=read(HERE/'REUSE_AUDIT.json')['s19_baselines']
    for fold in range(4):
        ask=outdir(fold,'ASK');metric=read(ask/'metrics.json')
        if metric['status']!='COMPLETE' or read(ask/'attempt.json')['status']!='COMPLETE' or (ask/'failure.json').exists():raise ValueError('Formal fit incomplete')
        if sha(ask/'model.pt')!=metric['model_sha256']:raise ValueError('ASK model changed')
        completed[str(fold)]=dict(model_sha256=metric['model_sha256'],metrics_sha256=sha(ask/'metrics.json'),epochs=metric['epochs'])
        record=reuse[fold];old=olddir(fold);dest=outdir(fold,'B0');dest.mkdir(parents=True,exist_ok=True)
        for name,h in [('config.json',record['config_sha256']),('metrics.json',record['metrics_sha256'])]:
            if sha(old/name)!=h:raise ValueError('Frozen baseline metadata changed')
        source=record['checkpoint']
        if sha(source['path'])!=source['sha256']:raise ValueError('Frozen B0 weights changed')
        # Same fields/values as intended by original seal, assembled once.
        write(dest/'reference.json',{**record,'fold':fold,'arm':'B0','reused':True})
        refs[f'B0_f{fold}']=source
        for name in ['config.json','metrics.json','reference.json']:
            path=dest/name if name=='reference.json' else old/name;metadata[str(path)]=sha(path)
        refs[f'ASK_f{fold}']=dict(fold=fold,arm='ASK',path=str(ask/'model.pt'),sha256=metric['model_sha256'])
        for name in ['config.json','metrics.json']:metadata[str(ask/name)]=sha(ask/name)
    write(HERE/'ENGINEERING_HANDOFF.json',dict(time=now(),status='PRE_HELD_SERIALIZATION_HANDOFF',
          error="TypeError: dict() got multiple values for keyword argument 'fold'",original_location='s20_run.py seal baseline reference assembly',
          original_registration_sha256=sha(HERE/'registration.json'),handoff_code_sha256=sha(__file__),
          original_registered_code_and_protocol_unchanged=True,source_fits_completed=completed,formal_fits_repeated=0,
          held_data_access_before_handoff=0,models_predictions_or_thresholds_changed=False))
    write(HERE/'seal_pilot.json',dict(time=now(),all_checkpoints_complete=True,new_fits=4,reused_B0=4,checkpoints=refs,metadata_sha256=metadata,
          baseline_held_predictions_preexisting=True,new_ask_held_source_read=False,real_target_access=False,
          engineering_handoff_sha256=sha(HERE/'ENGINEERING_HANDOFF.json'),engineering_handoff_code_sha256=sha(__file__)))
    relative=HERE.relative_to(ROOT).as_posix()
    subprocess.run(['git','-C',str(ROOT),'add','--',relative],check=True)
    subprocess.run(['git','-C',str(ROOT),'-c','user.name=Codex','-c','user.email=codex@local.invalid','commit','-m',
                    'S20 retain serialization incident and seal completed models before held-source evaluation','--',relative],check=True)
    committed(HERE/'ENGINEERING_HANDOFF.json');committed(HERE/'seal_pilot.json')
    if sha(__file__)!=read(HERE/'seal_pilot.json')['engineering_handoff_code_sha256']:raise ValueError('Handoff code changed')
    evaluate();decide()


if __name__=='__main__':run()
