"""Independent aggregate arithmetic and QR replay of only the same registered rows."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import lstsq
import runner as R

HERE,ROOT=R.HERE,R.ROOT


def main():
    reg=R.validate();checks={}
    def check(name,test):
        checks[name]=bool(test)
        if not test:raise ValueError('Independent QA failed:'+name)
    syn=R.read(HERE/'SYNTHETIC_RESULTS.json')
    check('synthetic_count',syn['packets']==1728)
    check('known_nuisance_recovery',syn['max_error_over_registered_tolerance']<=1)
    for name,digest in reg['history_sha256'].items():check('unchanged_history:'+name,R.sha(ROOT/name)==digest)
    rows=R.read(HERE/'PACKET_DIAGNOSTICS.json')['rows'];result=R.read(HERE/'SOURCE_VALIDITY.json')
    keys=[x['pool_row'] for x in rows];check('unique_fixed_rows',len(keys)==len(set(keys)))
    # Reconstruct the sample, metadata and reference independently of the audit.
    pool={}
    for name in ('fit_y','fit_rx','fit_day','train_ids'):
        item=reg['input_whitelist'][name];path=Path(item['path']);R.access(path,'independent_boundary_metadata:'+name)
        check('input_SHA:'+name,R.sha(path)==item['sha256']);pool[name]=np.load(path,allow_pickle=False)
    import hashlib
    expected=[]
    for rxname in reg['audit_rx']:
        for tx in range(10):
            for day in (1,2):
                candidates=[int(i) for i in pool['train_ids'] if reg['rx_names'][pool['fit_rx'][i]]==rxname and pool['fit_y'][i]==tx and pool['fit_day'][i]==day]
                expected.extend(sorted(candidates,key=lambda i:hashlib.sha256(('M1-A-v1|'+str(i)).encode()).digest())[:16])
    check('fixed_sample_recreated',keys==expected)
    p=Path(reg['input_whitelist']['fit_x']['path']);R.access(p,'independent_hash_same_source_file_bytes')
    check('IQ_SHA',R.sha(p)==reg['input_whitelist']['fit_x']['sha256'])
    raw=np.load(p,mmap_mode='r',allow_pickle=False);R.access(p,'independent_QR_replay_same_registered_rows:'+str(len(keys)))
    tones=np.array([-24,-20,-16,-12,-8,-4,4,8,12,16,20,24]);signs=np.array([1,-1,1,-1,-1,1,-1,-1,1,1,1,1]);n=np.arange(20,180)
    for row in rows:
        i=row['pool_row'];m=row['metrics'];x=raw[i];z=x[0].astype(float)+1j*x[1]
        check('row_scope:'+str(i),i in pool['train_ids'] and row['day'] in (1,2) and row['rx'] in reg['audit_rx'])
        if 'models' not in m:continue
        z=z/np.sqrt(np.mean(abs(z[20:100])**2));fit=z[20:100]
        omega=np.angle(sum(fit[k+20]*fit[k].conjugate() for k in range(60)))/20
        check('CFO:'+str(i),abs(omega*25e6/(2*np.pi)-m['cfo_hz'])<1e-7)
        v=z[20:180]*np.exp(-1j*omega*n)
        for arm,audit in m['models'].items():
            tau=audit['phase']
            s=sum((1+1j)*b*np.exp(2j*np.pi*k*(n-tau)/80) for k,b in zip(tones,signs))/np.sqrt(24)
            control=sum((1+1j)*b*np.exp(2j*np.pi*k*(n-tau)/80) for k,b in zip(tones,np.roll(signs,1)))/np.sqrt(24)
            U=s[:,None] if arm=='L' else np.c_[s,s.conj() if arm=='WL' else control]
            theta=lstsq(U[:80],v[:80],lapack_driver='gelsy')[0]
            logged=np.array([complex(*x) for x in audit['coefficients']])
            check('independent_QR_coefficients:'+str(i)+'/'+arm,np.allclose(theta,logged,rtol=0,atol=1e-9))
            for part,slc in (('fit',slice(0,80)),('held',slice(80,160))):
                loss=np.linalg.norm(v[slc]-U[slc]@theta)**2/np.linalg.norm(v[slc])**2
                check('independent_'+part+'_residual:'+str(i)+'/'+arm,abs(loss-audit[part+'_nmse'])<1e-9)
    ok=sum(x['metrics']['valid'] for x in rows)
    check('overall_valid',result['overall']['valid']==ok and result['overall']['valid_fraction']==ok/len(rows))
    for group in result['per_rx_day']:
        selected=[x for x in rows if x['rx']==group['rx'] and x['day']==group['day']]
        check('RX_day:'+group['rx']+'/'+str(group['day']),group['valid']==sum(x['metrics']['valid'] for x in selected) and group['packets']==len(selected))
    means=np.array([np.mean([x['metrics']['advantage'] for x in rows if x['rx']==rx]) for rx in sorted(reg['audit_rx'])])
    draws=np.random.default_rng(27102).integers(0,6,(10000,6));ci=np.quantile(means[draws].mean(1),[.025,.975])
    check('independent_bootstrap',np.allclose(ci,result['advantage_RX_conditional95'],atol=1e-12,rtol=0))
    check('independent_mean_advantage',abs(means.mean()-result['mean_advantage_RX_equal'])<1e-12)
    gates=dict(overall_valid_at_least90=ok/len(rows)>=.9,every_RX_day_at_least80=all(x['valid_fraction']>=.8 for x in result['per_rx_day']),
        present_cell_support_at_least8=all(x['packets']>=8 for x in result['cells'] if x['packets']),all_RX_days_present=all(x['packets']>0 for x in result['per_rx_day']),
        mean_validation_advantage_at_least2percent=means.mean()>=.02,all_RX_advantage_positive=(means>0).all(),
        both_days_advantage_positive=all(x['advantage_mean']>0 for x in result['per_day'].values()),RX_conditional_interval_lower_positive=ci[0]>0,
        WL_beats_L_on_held_segment=result['linear_to_WL_RX_equal_gain']>0)
    check('all_registered_gates',gates==result['gates'])
    status='M1_A_PASSED_B_REQUIRES_SEPARATE_AUTHORIZATION' if all(gates.values()) else 'STOP_EXPLICIT_MIRROR_ESTIMATOR'
    check('decision',R.read(HERE/'DECISION.json')['status']==status)
    log=[json.loads(x) for x in (HERE/'ACCESS_LOG.jsonl').read_text(encoding='utf-8').splitlines()]
    allowed={str(Path(x['path'])) for x in reg['input_whitelist'].values()}
    for x in log:
        if Path(x['path']).suffix in ('.npy','.npz','.pkl','.pt'):check('payload_path:'+x['action'],x['path'] in allowed)
    check('no_classifier_or_network',R.read(HERE/'DECISION.json')['classifier_fits']==R.read(HERE/'DECISION.json')['neural_updates']==0)
    R.new(HERE/'INDEPENDENT_QA.json',dict(status='PASS',check_count=len(checks),checks=checks,
        QR_replay_rows=len(rows),additional_unique_IQ_rows=0,classifier_fits=0,neural_updates=0,decision=status))
    print(dict(status='PASS',checks=len(checks),decision=status))


if __name__=='__main__':main()
