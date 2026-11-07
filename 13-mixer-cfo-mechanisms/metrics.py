"""Metrics over the model's declared TX classes, including missing classes as zero."""
import numpy as np


def rx_metrics(logits,y):
    logits=np.asarray(logits);y=np.asarray(y,dtype=np.int64)
    if logits.ndim!=2 or len(logits)!=len(y) or not len(y) or not np.isfinite(logits).all():
        raise ValueError('Invalid logits/labels')
    k=logits.shape[1]
    if np.any(y<0) or np.any(y>=k):raise ValueError('Label outside declared TX classes')
    pred=logits.argmax(1)
    cm=np.bincount(y*k+pred,minlength=k*k).reshape(k,k)
    tp=np.diag(cm).astype(float)
    denominator=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*tp,denominator,out=np.zeros(k),where=denominator>0)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(k),where=cm.sum(1)>0)
    return dict(acc=float((pred==y).mean()),macro_f1=float(f1.mean()),recall=recall.round(4).tolist(),confusion=cm.tolist())
