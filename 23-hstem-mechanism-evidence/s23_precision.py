"""Post-failure numerical appendix: same H0 pairs, unchanged tolerance, no fits."""
import torch
from s23_core import HERE,read,write,sha,now,validate,baseline_path
from s23_data import source_val
from s23_math import build,fold_h0,equivalence
from s23_run import setup,predict,score,save_csv

def run():
    validate();result=[]
    def forbidden(*args,**kwargs):raise RuntimeError('S23 forbids optimizer creation')
    torch.optim.Optimizer.__init__=forbidden
    for fold in [0,1]:
        data=source_val(fold);x=torch.as_tensor(data['x'],device='cuda')
        for seed in [201,202,203]:
            path=baseline_path('H0',fold,seed);cfg=read(path/'config.json');setup(cfg['data_seed'])
            model=build('H0',cfg['data_seed']).cuda().eval();model.load_state_dict(torch.load(path/'model.pt',map_location='cuda',weights_only=True),strict=True);folded=fold_h0(model)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=True
            default=predict(model,x);default_fold=predict(folded,x)
            result.append(dict(fold=fold,seed=seed,convolution_tf32=True,**equivalence(default_fold,default)))
            torch.backends.cudnn.allow_tf32=False
            strict=predict(model,x);strict_fold=predict(folded,x);eq=equivalence(strict_fold,strict)
            result.append(dict(fold=fold,seed=seed,convolution_tf32=False,**eq,default_to_strict_max_abs=equivalence(strict,default)['max_abs'],default_to_strict_prediction_changes=int((strict.argmax(1)!=default.argmax(1)).sum())))
            if eq['allclose']:
                out=HERE/'converted'/f'H0_f{fold}_s{seed}_strict_fp32.pt';out.parent.mkdir(exist_ok=True);torch.save({k:v.detach().cpu() for k,v in folded.state_dict().items()},out)
            del model,folded,default,default_fold,strict,strict_fold
        del x;torch.cuda.empty_cache()
    save_csv('precision_equivalence.csv',result)
    strict=[r for r in result if not r['convolution_tf32']];default=[r for r in result if r['convolution_tf32']]
    write(HERE/'PRECISION_DIAGNOSTIC.json',dict(time=now(),scope='Post-failure numerical control;all original failures retained;no tolerance or scientific threshold changes',new_fits=0,atol=1e-4,rtol=1e-4,default_cudnn_tf32_all_pass=all(r['allclose'] for r in default),strict_fp32_all_pass=all(r['allclose'] for r in strict),default_max_abs=max(r['max_abs'] for r in default),strict_max_abs=max(r['max_abs'] for r in strict),strict_prediction_disagreements=sum(r['packets']-r['predictions_equal'] for r in strict),default_to_strict_original_prediction_changes=sum(r['default_to_strict_prediction_changes'] for r in strict),export_requirement='Disable cudnn and matmul TF32 for strict FP32 numerical equivalence;no global exactness claim in default reduced-precision convolution'))
    print('Precision control complete;original tolerance unchanged;zero training',flush=True)

if __name__=='__main__':run()
