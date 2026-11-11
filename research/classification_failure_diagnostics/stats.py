"""Pure array statistics, independent of all historical training modules."""
from __future__ import annotations
import math
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def packet_stats(logits, y):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y)
    assert z.ndim == 2 and z.shape[1] == 10 and y.shape == (len(z),) and len(z)
    assert np.isfinite(z).all() and y.dtype.kind in "iu" and np.all((y>=0)&(y<10))
    logprob = z-logsumexp(z, axis=1, keepdims=True)
    prob = np.exp(logprob)
    other = z.copy()
    other[np.arange(len(y)), y] = -np.inf
    hardest = other.argmax(1)
    margin = z[np.arange(len(y)), y]-other[np.arange(len(y)), hardest]
    scale = z.std(axis=1)
    normalized = np.divide(margin, scale, out=np.zeros_like(margin), where=scale>0)
    pred = z.argmax(1)
    return dict(pred=pred, error=pred!=y, ce=-logprob[np.arange(len(y)), y], margin=margin,
                normalized_margin=normalized, hardest_negative=hardest, confidence=prob.max(1),
                true_probability=prob[np.arange(len(y)), y], entropy=-(prob*logprob).sum(1),
                zero_logit_scale=scale==0)


def wilson(correct, n):
    if not n:
        return None
    z = 1.959963984540054
    p = correct/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0., center-half), min(1., center+half)]


def group_stats(p, ids, y):
    n = int(np.count_nonzero(ids))
    if not n:
        return dict(n=0, errors=0, accuracy=None, ce=None, margin_mean=None,
                    normalized_margin_mean=None, margin_quantiles=None, normalized_margin_quantiles=None,
                    nonpositive_margin_fraction=None, zero_margin_count=0, confidence_mean=None,
                    high_confidence_errors=0, high_confidence_fraction_among_errors=None,
                    entropy_mean=None, hardest_negative_counts=[0]*10, predicted_counts=[0]*10,
                    wrong_predicted_counts=[0]*10, accuracy_wilson_packet_interval=None)
    errors = int(p["error"][ids].sum())
    high_wrong = int(((p["confidence"]>=0.9)&p["error"])[ids].sum())
    return dict(n=n, errors=errors, accuracy=1-errors/n, ce=float(p["ce"][ids].mean()),
                margin_mean=float(p["margin"][ids].mean()), normalized_margin_mean=float(p["normalized_margin"][ids].mean()),
                margin_quantiles=np.quantile(p["margin"][ids], QUANTILES).tolist(),
                normalized_margin_quantiles=np.quantile(p["normalized_margin"][ids], QUANTILES).tolist(),
                nonpositive_margin_fraction=float((p["margin"][ids]<=0).mean()),
                zero_margin_count=int((p["margin"][ids]==0).sum()), confidence_mean=float(p["confidence"][ids].mean()),
                high_confidence_errors=high_wrong, high_confidence_fraction_among_errors=high_wrong/errors if errors else None,
                entropy_mean=float(p["entropy"][ids].mean()),
                hardest_negative_counts=np.bincount(p["hardest_negative"][ids], minlength=10).tolist(),
                predicted_counts=np.bincount(p["pred"][ids], minlength=10).tolist(),
                wrong_predicted_counts=np.bincount(p["pred"][ids & p["error"]], minlength=10).tolist(),
                accuracy_wilson_packet_interval=wilson(n-errors, n))


def macro_f1(y, pred):
    cm = np.bincount(y*10+pred, minlength=100).reshape(10, 10)
    denom = cm.sum(0)+cm.sum(1)
    return float(np.divide(2*np.diag(cm), denom, out=np.zeros(10, dtype=float), where=denom>0).mean()), cm.tolist()


def equal_tail(values, fraction=0.2):
    """Upper-tail mean with fractional last group, equal group weights."""
    values = np.sort(np.asarray(values, dtype=float))[::-1]
    assert len(values) and 0 < fraction <= 1
    mass = fraction*len(values)
    full = int(np.floor(mass))
    total = values[:full].sum()
    if full < len(values):
        total += (mass-full)*values[full]
    return float(total/mass)


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:
        return None
    return float(spearmanr(a, b).statistic)


def paired_stats(a, b, ids):
    n = int(ids.sum())
    if not n:
        return dict(n=0)
    ae, be = a["error"][ids], b["error"][ids]
    both = int((ae & be).sum())
    only_b0 = int((ae & ~be).sum())
    only_ask = int((~ae & be).sum())
    union = both+only_b0+only_ask
    return dict(n=n, both_correct=int((~ae & ~be).sum()), both_wrong=both,
                b0_wrong_ask_correct=only_b0, b0_correct_ask_wrong=only_ask,
                both_wrong_same_prediction=int((ae & be & (a["pred"][ids]==b["pred"][ids])).sum()),
                prediction_disagreements=int((a["pred"][ids]!=b["pred"][ids]).sum()),
                delta_accuracy_ask_minus_b0=(only_b0-only_ask)/n,
                error_jaccard=both/union if union else None,
                ask_error_given_b0_error=both/(both+only_b0) if both+only_b0 else None,
                b0_error_given_ask_error=both/(both+only_ask) if both+only_ask else None,
                diagnostic_oracle_accuracy=1-both/n)


def calibration(p, bins=15):
    edges = np.linspace(0, 1, bins+1)
    idx = np.minimum((p["confidence"]*bins).astype(int), bins-1)
    rows = []
    ece = 0.
    for j in range(bins):
        ids = idx==j
        n = int(ids.sum())
        acc = float((~p["error"][ids]).mean()) if n else None
        conf = float(p["confidence"][ids].mean()) if n else None
        if n:
            ece += n/len(idx)*abs(acc-conf)
        rows.append(dict(low=float(edges[j]), high=float(edges[j+1]), n=n, accuracy=acc, confidence=conf))
    return dict(packet_ece=float(ece), bins=rows, calibration_fitted=False)
