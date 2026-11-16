import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
sys.path.insert(0,str(STAGE))
from s24_common import method_input,write_json,sha
from s24_dann import ShenDANN,Reverse

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    (HERE/'local').mkdir(exist_ok=True)
    if args.prepare:
        iq=np.load(STAGE/'local/source_probe.npz')['iq'][:1]
        synthetic=np.random.default_rng(2401).normal(size=(1,2,8192)).astype(np.float32)
        cis=method_input('shen_tmc2024',np.concatenate([iq,synthetic]),'native').astype(np.float32)
        np.save(HERE/'local/parity_input.npy',cis);print('2 fixed native CIS inputs exported.');return
    torch.set_num_threads(4)
    policy=json.loads((HERE/'gate_policy.json').read_text(encoding='utf-8'))['parity']
    meta=json.loads((HERE/'local/tf_reference.json').read_text(encoding='utf-8'))
    data=np.load(HERE/'local/tf_reference.npz')
    model=ShenDANN().cpu().eval()
    convolution=[model.stem]+[layer for b in model.blocks for layer in (b.c1,b.c2,*([b.sc] if b.sc is not None else []))]
    conv_names=sorted([l['name'] for l in meta['layers'] if l['cls']=='Conv2D'],key=lambda s:int(s.split('_')[-1]) if s!='conv2d' else -1)
    layers=dict(zip(conv_names,convolution))
    layers.update({'dense':model.feat,'tx_clf_input':model.tx_input,'tx_classifier':model.classifier,'dense_1':model.rx_input,'rx_classifier':model.rx_classifier})
    mapping={}
    with torch.no_grad():
        for name,layer in layers.items():
            for suffix,param in [('kernel:0',layer.weight),('bias:0',layer.bias)]:
                key=name+'/'+suffix;a=data['weight.'+key]
                if a.ndim==4:a=a.transpose(3,2,0,1)
                elif a.ndim==2:a=a.T
                param.copy_(torch.from_numpy(a.copy()));mapping[key]=dict(torch_shape=list(param.shape),tf_sha256=sha(data['weight.'+key].tobytes()))
    checks=[]
    def compare(name,a,b,gradient=False):
        a=np.asarray(a);b=np.asarray(b)
        atol=policy['gradient_atol' if gradient else 'forward_atol'];rtol=policy['gradient_rtol' if gradient else 'forward_rtol']
        okay=a.shape==b.shape and np.allclose(a,b,atol=atol,rtol=rtol)
        checks.append(dict(name=name,passed=bool(okay),max_abs=float(np.max(np.abs(a-b))) if a.shape==b.shape else None,atol=atol,rtol=rtol))
    x=torch.from_numpy(data['input'].transpose(0,3,1,2).copy()).requires_grad_()
    captures={}
    hooks=[layer.register_forward_hook(lambda m,i,o,key=name:captures.__setitem__(key,o.detach())) for name,layer in layers.items()]
    h=F.relu(model.stem(F.pad(x,(2,3,2,3))));h=model.blocks(h)
    pooled=F.avg_pool2d(h,2);flat=pooled.permute(0,2,3,1).reshape(2,-1)
    feature=F.normalize(model.feat(flat),p=2,dim=1,eps=1e-12)
    tx=model.classifier(F.relu(model.tx_input(feature)));rx=model.rx_classifier(F.relu(model.rx_input(Reverse.apply(feature))))
    for name in conv_names:
        v=captures[name];v=F.relu(v) if name=='conv2d' else v
        compare('forward.'+name,v.permute(0,2,3,1).numpy(),data['layer.'+name])
    poolname=next(l['name'] for l in meta['layers'] if l['cls']=='AveragePooling2D')
    compare('forward.pool',pooled.detach().permute(0,2,3,1).numpy(),data['layer.'+poolname])
    compare('forward.flatten',flat.detach().numpy(),data['layer.flatten'])
    compare('forward.feature',feature.detach().numpy(),data['layer.feature_layer'])
    compare('forward.tx_probability',tx.softmax(1).detach().numpy(),data['layer.tx_classifier'])
    compare('forward.rx_probability',rx.softmax(1).detach().numpy(),data['layer.rx_classifier'])
    for hook in hooks:hook.remove()
    targets=torch.tensor([0,1])
    for mode in ('tx','rx','joint'):
        model.zero_grad(set_to_none=True);x.grad=None
        feature=model.features(x)
        t=model.classifier(F.relu(model.tx_input(feature)));r=model.rx_classifier(F.relu(model.rx_input(Reverse.apply(feature))))
        lt=F.cross_entropy(t,targets);lr=F.cross_entropy(r,targets)
        loss=lt if mode=='tx' else lr if mode=='rx' else lt+lr
        loss.backward();compare('loss.'+mode,float(loss),meta['losses'][mode])
        compare('gradient.'+mode+'.input',x.grad.permute(0,2,3,1).numpy(),data['grad.'+mode+'.input'],True)
        for name,layer in layers.items():
            for suffix,param in [('kernel:0',layer.weight),('bias:0',layer.bias)]:
                key='grad.'+mode+'.'+name+'/'+suffix
                if key not in data:
                    assert param.grad is None;continue
                a=data[key]
                if a.ndim==4:a=a.transpose(3,2,0,1)
                elif a.ndim==2:a=a.T
                compare(key,param.grad.numpy(),a,True)
    result=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,weight_mapping=mapping,
                policy_sha256=sha((HERE/'gate_policy.json').read_bytes()),tensorflow=meta['tensorflow'],torch=torch.__version__,
                reference_npz_sha256=sha((HERE/'local/tf_reference.npz').read_bytes()),fits=0,optimizer_steps=0)
    write_json(HERE/'DANN_PARITY.json',result)
    print(result['status'],len(checks),'comparisons; failed:',[c for c in checks if not c['passed']])
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
