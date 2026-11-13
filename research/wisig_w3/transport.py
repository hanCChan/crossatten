"""Fixed statistical transport controls. No neural gradients or held-TX tuning."""
import numpy as np

METHODS=('identity','shift','diagonal','rank2','rank8','shuffled_rank8')


def pca_basis(fit_centroids,k=8):
    a=np.asarray(fit_centroids,dtype=np.float64)
    center=a.mean(0);_,_,vt=np.linalg.svd(a-center,full_matrices=False)
    if len(vt)<k:raise ValueError('Insufficient fit-TX support for frozen dimension')
    return vt[:k].T.copy()


def cycle_permutation(n,seed):
    rng=np.random.default_rng(seed);p=np.arange(n)
    for i in range(n-1,0,-1):
        j=int(rng.integers(i));p[i],p[j]=p[j],p[i]
    assert np.all(p!=np.arange(n))
    return p


def fit(method,x,y,basis,seed=31013):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if method not in METHODS or x.shape!=y.shape:raise ValueError('Transport schema differs')
    if method=='shuffled_rank8':y=y[cycle_permutation(len(y),seed)]
    center=x.mean(0);shift=(y-x).mean(0)
    result=dict(method=method,center=center,shift=shift,fit_rows=len(x),seed=seed)
    if method=='identity':return result
    if method=='shift':return result
    xc=x-center;residual=y-x-shift
    if method=='diagonal':
        ss=np.sum(xc*xc,axis=0);ridge=max(.1*float(ss.mean()),1e-12)
        result['diagonal']=np.sum(xc*residual,axis=0)/(ss+ridge)
    else:
        k=2 if method=='rank2' else 8;projection=basis[:,:k]
        z=xc@projection;gram=z.T@z
        ridge=max(.1*float(np.trace(gram)/k),1e-12)
        result.update(projection=projection,weight=np.linalg.solve(gram+ridge*np.eye(k),z.T@residual))
    return result


def predict(operator,x):
    x=np.asarray(x,dtype=np.float64);m=operator['method']
    if m=='identity':return x.copy()
    out=x+operator['shift']
    if m=='shift':return out
    if m=='diagonal':return out+(x-operator['center'])*operator['diagonal']
    return out+(x-operator['center'])@operator['projection']@operator['weight']


def transport_error(pred,target):
    return np.sum((pred-target)**2,axis=-1)/(np.sum(target**2,axis=-1)+1e-12)


def separation_ratio(original,changed):
    i,j=np.triu_indices(len(original),1)
    before=np.sum((original[i]-original[j])**2,axis=1)
    after=np.sum((changed[i]-changed[j])**2,axis=1)
    return float(np.median(after/np.maximum(before,1e-12)))


def prototype_metrics(query,prototypes):
    """Held-TX diagnostic with labeled target anchors; NOT deployment accuracy."""
    n,k,d=query.shape;x=query.reshape(n*k,d)
    dist=np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(prototypes*prototypes,axis=1)[None,:]-2*x@prototypes.T,0)
    y=np.repeat(np.arange(n),k);true=dist[np.arange(len(x)),y]
    other=dist.copy();other[np.arange(len(x)),y]=np.inf
    margin=(other.min(1)-true)/(dist.mean(1)+1e-12)
    return (dist.argmin(1)==y).reshape(n,k).mean(1),margin.reshape(n,k).mean(1)
