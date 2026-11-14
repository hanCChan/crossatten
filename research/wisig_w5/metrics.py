"""Metrics and fixed conditional RX intervals; no estimator selection."""
import numpy as np


def metrics(logits,y):
    logits=np.asarray(logits,dtype=np.float64);y=np.asarray(y);n=len(y)
    if not n:return dict(packets=0,accuracy_pct=None,macro_F1_pct=None,CE=None)
    pred=logits.argmax(1);conf=np.zeros((10,10),dtype=np.int64);np.add.at(conf,(y,pred),1)
    tp=np.diag(conf);den=conf.sum(0)+conf.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(10,dtype=float),where=den>0)
    shifted=logits-logits.max(1,keepdims=True)
    ce=np.log(np.exp(shifted).sum(1))-shifted[np.arange(n),y]
    return dict(packets=n,accuracy_pct=float(100*np.mean(pred==y)),macro_F1_pct=float(100*f1.mean()),CE=float(ce.mean()))


def rx_interval(per_RX_delta):
    d=np.asarray(per_RX_delta,dtype=np.float64);rng=np.random.default_rng(43101)
    samples=rng.integers(0,len(d),size=(10000,len(d)))
    return dict(mean_pp=float(d.mean()),CI95_pp=np.quantile(d[samples].mean(1),[.025,.975]).tolist(),
        RX_clusters=len(d),replicates=10000,seed=43101,
        condition='Fixed three trained seeds and six repeatedly developed internal source RX; not independent target or fresh-RX uncertainty')
