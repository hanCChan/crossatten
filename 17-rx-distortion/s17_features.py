"""RX-associated descriptors, not identifiable receiver transfer functions."""
import numpy as np
from scipy.stats import spearmanr

def features(x):
    z=x[:,0].astype(np.float64)+1j*x[:,1].astype(np.float64)
    z=z/np.sqrt(np.mean(abs(z)**2,axis=1,keepdims=True)).clip(1e-12)
    spectrum=np.fft.fftshift(np.fft.fft(z,axis=1),axes=1);mag=abs(spectrum)
    db=20*np.log10(mag.clip(1e-6));amp=db.reshape(-1,16,16).mean(2);amp-=amp.mean(1,keepdims=True)
    unit=spectrum/mag.clip(1e-9)
    curvature=unit[:,2:]*unit[:,:-2]*unit[:,1:-1].conj()**2
    valid=(mag[:,2:]>mag.mean(1,keepdims=True)*.1)&(mag[:,:-2]>mag.mean(1,keepdims=True)*.1)&(mag[:,1:-1]>mag.mean(1,keepdims=True)*.1)
    curved=np.pad(curvature*valid,((0,0),(1,1))).reshape(-1,16,16)
    count=np.pad(valid,((0,0),(1,1))).reshape(-1,16,16).sum(2).clip(1)
    circular=curved.sum(2)/count
    centered=z-z.mean(1,keepdims=True);kappa=np.mean(centered**2,axis=1)/np.mean(abs(centered)**2,axis=1).clip(1e-12)
    return dict(amp=amp.astype(np.float32),phase=np.concatenate([circular.real,circular.imag],1).astype(np.float32),
        iq=np.stack([kappa.real,kappa.imag],1).astype(np.float32),dc=np.stack([z.mean(1).real,z.mean(1).imag],1).astype(np.float32))

def cell_profiles(value,data,names):
    cells=[]
    for tx in np.unique(data['y']):
        for rx in np.unique(data['rx']):
            for day in [1,2]:
                mask=(data['y']==tx)&(data['rx']==rx)&(data['day']==day)
                if mask.sum()>=5:cells.append(dict(tx=int(tx),rx=int(rx),day=day,count=int(mask.sum()),v=value[mask].mean(0)))
    raw=np.stack([r['v'] for r in cells]);y=np.array([r['tx'] for r in cells]);rx=np.array([r['rx'] for r in cells]);day=np.array([r['day'] for r in cells])
    residual=raw.copy()
    for tx in np.unique(y):
        for d in [1,2]:
            ids=(y==tx)&(day==d);residual[ids]-=raw[ids].mean(0)
    return dict(raw=raw,residual=residual,tx=y,rx=rx,day=day,cells=cells)

def corr(a,b):
    a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1)
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else 0.

def rx_probe(residual,tx,rx,day):
    scores=[]
    for label in np.unique(tx):
        train=tx!=label;test=~train;labels=np.unique(rx[train])
        scale=residual[train].std(0).clip(1e-6);centers=np.stack([residual[train&(rx==r)].mean(0) for r in labels])
        dist=(((residual[test,None]-centers[None])/scale)**2).mean(2);pred=labels[dist.argmin(1)]
        scores.extend(float((pred[rx[test]==r]==r).mean()) for r in np.unique(rx[test]))
    return float(np.mean(scores))

def analyze(value,data,names,errors,permutations=199):
    c=cell_profiles(value,data,names);v=c['residual'];rx=c['rx'];tx=c['tx'];day=c['day'];labels=np.unique(rx)
    profile=np.stack([v[rx==r].mean(0) for r in labels]);fitted=np.stack([profile[np.flatnonzero(labels==r)[0]] for r in rx])
    energy=float(np.mean(v**2));eta=0. if energy<1e-15 else 1-float(np.mean((v-fitted)**2))/energy
    day_profiles=[np.stack([v[(rx==r)&(day==d)].mean(0) for r in labels]) for d in [1,2]]
    half_profiles=[np.stack([v[(rx==r)&(tx%2==k)].mean(0) for r in labels]) for k in [0,1]]
    actual=rx_probe(v,tx,rx,day);rng=np.random.default_rng(18017);null=[]
    for _ in range(permutations):
        perm=rx.copy()
        for t in np.unique(tx):
            for d in [1,2]:
                ids=np.flatnonzero((tx==t)&(day==d));perm[ids]=rng.permutation(perm[ids])
        null.append(rx_probe(v,tx,perm,day))
    p=(1+sum(n>=actual for n in null))/(1+permutations)
    scale=v.std(0).clip(1e-6);distance=np.sqrt(np.mean((profile/scale)**2,axis=1))
    error=np.array([errors.get(int(r),0) for r in labels]);rho=float(spearmanr(distance,error).statistic) if np.std(error)>0 and np.std(distance)>0 else 0.
    if not np.isfinite(rho):rho=0.
    count=len(labels);chance=1/count
    stable=eta>=.1 and corr(*day_profiles)>=.4 and corr(*half_profiles)>=.4 and actual>=2*chance and p<=.01
    return dict(partial_rx_eta2=eta,cross_day_correlation=corr(*day_profiles),cross_tx_half_correlation=corr(*half_profiles),
        leave_tx_out_rx_accuracy=actual,chance=chance,rx_permutation_p=p,source_val_error_spearman=rho,stable=bool(stable),
        profile_by_rx={names[int(r)]:profile[i].tolist() for i,r in enumerate(labels)},profile_rms=float(np.sqrt(np.mean(profile**2))),
        receiver_distances={names[int(r)]:float(distance[i]) for i,r in enumerate(labels)},complete_cells=len(v),
        cell_count_min=int(min(c['count'] for c in c['cells'])),scope='TX/day centered cell means; RX/channel/hardware still confounded')
