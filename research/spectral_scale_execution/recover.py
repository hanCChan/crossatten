"""Restore12 sealed source checkpoints and compare on synthetic CUDA input only."""
from __future__ import annotations
from pathlib import Path
import sys
import traceback

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as H


def main():
    if (HERE/'REFERENCE_RECOVERY.json').exists():raise FileExistsError('Keep original recovery evidence')
    scope=H.read(HERE/'ENGINEERING_SCOPE.json');H.committed(HERE/'ENGINEERING_SCOPE.json');H.committed(HERE/'REFERENCE_INPUTS.json')
    if scope['reference_registry_sha256']!=H.sha(HERE/'REFERENCE_INPUTS.json'):raise ValueError('Reference registry changed')
    H.validate_design()
    H.A.setup(27601)
    generator=torch.Generator(device='cuda').manual_seed(27601)
    x=torch.randn(128,2,256,generator=generator,device='cuda')
    records={};registry=H.read(HERE/'REFERENCE_INPUTS.json')['references']
    for key,group in registry.items():
        view,stream=key.split('/');seed=24101+int(stream[-1])
        for role,item in group.items():
            source=Path(item['origin']);target=(HERE/item['copy_relative']).resolve();target.relative_to((HERE/'artifacts').resolve())
            H.log_access(source,'verify_original_checkpoint_SHA_and_copy',key+'/'+role)
            if H.sha(source)!=item['sha256']:raise ValueError('Original source checkpoint differs from sealed identity')
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise FileExistsError('Recovered copy exists; do not overwrite')
            with target.open('xb') as f:f.write(source.read_bytes())
            if H.sha(target)!=item['sha256']:raise ValueError('Restored checkpoint copy differs')
            state,_=H.recovered_weights(view,stream,role)
            bare={k.removeprefix('backbone.'):v for k,v in state.items()}
            original=H.M.build_original('ManyRx',seed).cuda().eval();original.load_state_dict(bare,strict=True)
            frozen=H.M.build('F0',seed,backbone_state=state).cuda().eval()
            before=H.semantic_hash(frozen.backbone.state_dict())
            with torch.no_grad():
                a,b=original(x),frozen(x)
                difference=float((a-b).abs().max());same_prediction=torch.equal(a.argmax(1),b.argmax(1))
                tokens=frozen.tokens(x)
            if difference>1e-6 or not same_prediction:raise ValueError(f'Original/frozen CUDA parity failed:{key}/{role}, max={difference}')
            branch_checks={}
            if role=='F0':
                for arm in ('F1','F2'):
                    model=H.M.build(arm,seed,26101+int(stream[-1]),backbone_state=state).cuda().train()
                    with torch.no_grad():logits=model(x)
                    branch_checks[arm]=torch.equal(logits,b) and H.semantic_hash(model.backbone.state_dict())==before
                    if not branch_checks[arm]:raise ValueError('alpha0 does not preserve the frozen source P0')
                    del model
            if H.semantic_hash(frozen.backbone.state_dict())!=before:raise ValueError('Restore parity changed frozen weights/BN')
            records[key+'/'+role]=dict(origin_sha256=item['sha256'],recovered_path=item['copy_relative'],copy_sha256=H.sha(target),
                semantic_state_sha256=before,parameter_elements=sum(p.numel() for p in frozen.parameters()),
                max_abs_original_vs_frozen_logit=difference,identical_argmax=same_prediction,
                token_shape=list(tokens.shape),alpha0_branch_bitwise=branch_checks,
                all_backbone_parameters_frozen=all(not p.requires_grad for p in frozen.backbone.parameters()),
                frozen_buffers_unchanged=True,scientific_input_used=False)
            del original,frozen,state
            torch.cuda.empty_cache()
    value=dict(status='PASS_TWELVE_CHECKPOINTS_RESTORED_SYNTHETIC_CUDA_PARITY',references=records,
        environment=dict(torch=str(torch.__version__),numpy=H.np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name()),
        synthetic_input_seed=27601,synthetic_packets_per_reference=128,source_IQ_reads=0,source_prediction_decodes=0,
        scientific_gradient_fits=0,source_checkpoint_val_parity='PENDING_EXPLICIT_EXECUTION_AUTHORIZATION',time=H.now())
    H.write(HERE/'REFERENCE_RECOVERY.json',value)
    print(value['status'],len(records))


if __name__=='__main__':
    try:main()
    except Exception:
        H.write(HERE/'REFERENCE_RECOVERY_FAILURE.json',dict(time=H.now(),traceback=traceback.format_exc(),scientific_fits=0,source_IQ_reads=0))
        raise
