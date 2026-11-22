"""Data-free geometry/capacity audit only. This file contains no training entrypoint."""
import copy
import csv
import shutil
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from s25_common import HERE, digest, now, write
from s21_models import build, geometry


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(25025)
    h0,p0=build('H0',25025).eval(),build('P0',25025).eval()
    target=sum(p.numel() for p in h0.parameters())
    shared=sum(p.numel() for p in p0.parameters())-sum(p.numel() for p in p0.net.embedder.cnn_stem.tokenize.parameters())
    grid=[]
    for groups in [1,2,4,8,16,32]:
        for mid in range(groups,257,groups):
            params=shared+mid*(32//groups)*340+mid+64*mid+64
            grid.append(dict(groups=groups,mid=mid,kernel=340,stride=256,padding=42,params=params,
                relative_parameter_error=abs(params/target-1),linear_output_rank_upper_bound=min(mid,64)))
    best=min(grid,key=lambda r:(r['relative_parameter_error'],r['groups'],r['mid']))
    fullrank=min((r for r in grid if r['linear_output_rank_upper_bound']==64),
                 key=lambda r:(r['relative_parameter_error'],r['groups'],r['mid']))
    recommended=next(r for r in grid if r['groups']==8 and r['mid']==64)
    long=copy.deepcopy(p0)
    long.net.embedder.cnn_stem.tokenize=nn.Sequential(nn.Conv1d(32,80,340,256,42,groups=8),nn.Conv1d(80,64,1))
    nearby=copy.deepcopy(p0)
    nearby.net.embedder.cnn_stem.tokenize=nn.Sequential(nn.Conv1d(32,64,340,256,42,groups=8),nn.Conv1d(64,64,1))
    fullrank_model=copy.deepcopy(p0)
    g,m=fullrank['groups'],fullrank['mid']
    fullrank_model.net.embedder.cnn_stem.tokenize=nn.Sequential(nn.Conv1d(32,m,340,256,42,groups=g),nn.Conv1d(m,64,1))
    audits={}
    for name,model in [('H0',h0),('P0',p0),('LongPatch_same_groups_width',long),
                       ('LongPatch_groups8_width64',nearby),('LongPatch_best_fullrank',fullrank_model)]:
        output=model.net.embedder.cnn_stem(torch.randn(1,5,8192))
        assert output.shape==(1,32,64)
        result=geometry(model)
        with FlopCounterMode(display=False) as counter:model(torch.randn(1,2,8192))
        result['partial_dispatcher_macs']=counter.get_total_flops()/2
        assert result['partial_dispatcher_macs'] > 0, 'Dispatcher counter did not observe operations'
        audits[name]=result
    assert audits['LongPatch_same_groups_width']['cnn_receptive_field']==350
    assert audits['LongPatch_same_groups_width']['first_token_center']==127.5
    with (HERE/'long_patch_candidates.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(grid[0]),lineterminator='\n');writer.writeheader();writer.writerows(grid)
    original=HERE/'LONG_PATCH_AUDIT.original_counter_failed.json'
    if not original.exists() and (HERE/'LONG_PATCH_AUDIT.json').exists():
        shutil.copyfile(HERE/'LONG_PATCH_AUDIT.json',original)
    write(HERE/'LONG_PATCH_AUDIT.json',dict(status='AUDIT_COMPLETE_NO_TRAINING',time=now(),new_fits=0,new_optimizer_steps=0,
        new_waveform_reads=0,closed_data_free_candidate_count=len(grid),one_percent_matches=[v for v in grid if v['relative_parameter_error']<=.01],
        best_parameter_match=best,actual_architecture_checks=audits,
        best_rank64_capable_match=fullrank,
        nearby_rank64_capable_candidates=[r for r in grid if r['linear_output_rank_upper_bound']==64 and r['relative_parameter_error']<=.05],
        proposed_single_future_arm=recommended,
        decision='STOP automatic new fitting. A rank64-capable groups8/width64 long patch is a feasible optional development control at +3.48% parameters. It tests a freely learned long projection, but width, BN and MACs remain confounded; it cannot uniquely isolate hierarchy.',
        reasons=['The within1% candidate has a rank8 bottleneck, unlike the rank64-capable matched baseline; do not use it as a conveniently weak control.',
                 'Preserving groups8 and mid80 gives the correct350 support but161146 parameters (about20.5% more), two total BN versus five in H0, and different MACs.',
                 'Do not infer resource infeasibility from the1% threshold: groups8/mid64 gives138346 (+3.48%) and rank64 capability; groups16/mid112 gives130586 (-2.32%) but changes grouping. Report both, without testing accuracy.',
                 'The proposed groups8/mid64 arm changes internal width relative to P0 and keeps two BN versus five in H0. No strict single-factor match established.'],
        proposed_future_budget_only='At most one reviewed arm x2fold x3seed; no HStem retraining, no automatic execution; requires a separately frozen design and user instruction.',
        counter_scope='Partial dispatcher MACs include supported dense/conv/matmul operations; exclude FFT and unsupported elementwise work.',
        technical_correction=dict(original_preserved=original.name,original_sha256=digest(original),
            reason='Initial inference_mode suppressed dispatcher counting and returned zero for all models. Rerun with no_grad and assert positive counts; candidates, architectures and no-fit decision unchanged.'),
        audit_code_sha256=digest(__file__)))
    print('Long-Patch audit only:',best,'same-width params',audits['LongPatch_same_groups_width']['params'],flush=True)


if __name__=='__main__':main()
