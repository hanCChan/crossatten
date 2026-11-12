"""One prespecified observation descriptor. No classifier, torch or IQ file IO."""
from __future__ import annotations
import numpy as np

FS=25_000_000
TONES=np.array([-24,-20,-16,-12,-8,-4,4,8,12,16,20,24])
SIGNS=np.array([1,-1,1,-1,-1,1,-1,-1,1,1,1,1])
PHASES=np.arange(80)/4
N=np.arange(20,180)


def template(n,shift=0.,control=False):
    signs=np.roll(SIGNS,1) if control else SIGNS
    return np.exp(2j*np.pi*np.asarray(n)[...,None]/80*TONES-2j*np.pi*shift/80*TONES)@((1+1j)*signs)/np.sqrt(24)


def imbalance(z,mu=1.,nu=0.):return mu*z+nu*np.conj(z)


def forward(n,tx=0j,rx=0j,cfo=0.,shift=0.,channel=((0,1+0j),),phase=.3):
    v=sum(g*imbalance(template(np.asarray(n)-delay,shift),nu=tx) for delay,g in channel)
    v=v*np.exp(1j*(2*np.pi*cfo*np.asarray(n)/FS+phase))
    return imbalance(v,nu=rx)


def svd_fit(U,y,max_condition=10.):
    U=np.asarray(U,dtype=np.complex128);y=np.asarray(y,dtype=np.complex128)
    if U.ndim!=2 or y.shape!=(len(U),) or not np.isfinite(U).all() or not np.isfinite(y).all():
        return dict(valid=False,reason='nonfinite_or_shape',theta=None,condition=None,rank=0)
    _,s,_=np.linalg.svd(U,full_matrices=False)
    rank=int((s>s[0]*1e-12).sum()) if len(s) and s[0]>0 else 0
    condition=float(s[0]/s[-1]) if len(s) and s[-1]>s[0]*1e-12 else None
    if rank<U.shape[1] or condition is None or condition>max_condition:
        return dict(valid=False,reason='rank_or_condition',theta=None,condition=condition,rank=rank)
    theta=np.linalg.lstsq(U,y,rcond=1e-12)[0]
    return dict(valid=True,theta=theta,condition=condition,rank=rank,reason=None)


def full_design(n,shift,cfo):
    s=template(n,shift);p=np.exp(2j*np.pi*cfo*np.asarray(n)/FS)
    return np.stack([p*s,p*s.conj(),p.conj()*s.conj(),p.conj()*s],1)


class Estimator:
    def __init__(self):
        s=np.array([template(N,tau) for tau in PHASES])
        c=np.array([template(N,tau,True) for tau in PHASES])
        self.banks={'L':s[...,None],'WL':np.stack([s,s.conj()],-1),'NC':np.stack([s,c],-1)}
        self.pinv={a:np.linalg.pinv(U[:,:80],rcond=1e-12) for a,U in self.banks.items()}
        self.conditions={a:np.linalg.cond(U[:,:80]) for a,U in self.banks.items()}

    def packet(self,z):
        z=np.asarray(z,dtype=np.complex128)
        if z.shape!=(256,) or not np.isfinite(z).all() or np.mean(abs(z[20:100])**2)<1e-20:
            return dict(valid=False,reasons=['invalid_input'],tx_iq_estimate=None,advantage=0.)
        z=z/np.sqrt(np.mean(abs(z[20:100])**2));fit=z[20:100]
        cross=np.vdot(fit[:-20],fit[20:])
        omega=float(np.angle(cross)/20);cfo=omega*FS/(2*np.pi)
        v=z[20:180]*np.exp(-1j*omega*N)
        results={}
        for arm,U in self.banks.items():
            th=np.einsum('pkn,n->pk',self.pinv[arm],v[:80])
            pred=np.einsum('pnk,pk->pn',U,th)
            losses=np.sum(abs(v[None,:80]-pred[:,:80])**2,axis=1)
            # Fixed gauge for exact equivalent residual minima: dominant direct
            # coefficient, then earliest phase. No heldout sample chooses a gauge.
            candidates=np.flatnonzero(losses<=losses.min()+1e-12*np.sum(abs(v[:80])**2))
            best=min(candidates,key=lambda i:(-float(abs(th[i,0])),int(i)))
            den=np.sum(abs(v[80:])**2)
            results[arm]=dict(phase=float(PHASES[best]),condition=float(self.conditions[arm][best]),
                fit_nmse=float(losses[best]/max(np.sum(abs(v[:80])**2),1e-20)),
                held_nmse=float(np.sum(abs(v[80:]-pred[best,80:])**2)/max(den,1e-20)),
                coefficients=[[float(x.real),float(x.imag)] for x in th[best]])
        a,b=[complex(*x) for x in results['WL']['coefficients']]
        q=b/a if abs(a)>1e-12 else None
        reasons=[]
        if abs(cfo)>200_000:reasons.append('CFO_out_of_registered_range')
        if max(r['condition'] for r in results.values())>10:reasons.append('condition')
        if results['WL']['fit_nmse']>.15:reasons.append('fit_template_mismatch')
        if results['WL']['held_nmse']>.15:reasons.append('held_template_mismatch')
        if abs(a)<.1 or q is None or abs(q)>.5:reasons.append('weak_direct_or_large_effective_mirror')
        advantage=(results['NC']['held_nmse']-results['WL']['held_nmse'])/max(results['L']['held_nmse'],1e-12)
        delta=(results['L']['held_nmse']-results['WL']['held_nmse'])/max(results['L']['held_nmse'],1e-12)
        # Channel proxy is descriptive, not a channel/TX/RX separation.
        cycles=z[20:180].reshape(8,20);power=np.mean(abs(np.fft.fft(cycles,axis=1))**2,axis=0)
        active=np.r_[1:7,14:20];channel_cv=float(np.std(power[active])/max(np.mean(power[active]),1e-20))
        return dict(valid=not reasons,reasons=reasons,cfo_hz=cfo,models=results,
            q_effective=None if q is None else [float(q.real),float(q.imag)],q_abs=None if q is None else float(abs(q)),
            advantage=float(advantage) if not reasons else 0.,raw_advantage=float(advantage),delta_L_WL=float(delta),
            channel_proxy_cv=channel_cv,structural_half_period_ambiguity=True,
            gauge='dominant-direct among exact residual ties; reference convention only',tx_iq_estimate=None)
