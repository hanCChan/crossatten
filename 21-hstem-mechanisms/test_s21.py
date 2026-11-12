"""Meaningful invariant checks before the registered training budget is spent."""
import tempfile
from pathlib import Path
from s21_models import verify_architecture
from s21_report import recovery, bn_signal
from s21_core import write, read

def synthetic_report():
    import itertools
    import numpy as np
    import s21_report as reporter
    import s21_evaluate as evaluator
    original_here=reporter.HERE;original_sealed=evaluator.sealed
    try:
        with tempfile.TemporaryDirectory() as folder:
            reporter.HERE=Path(folder);evaluator.sealed=lambda: {}
            (reporter.HERE/'MANUSCRIPT_DRAFT.zh-CN.md').write_text('Synthetic only\n<!-- S21_RESULTS -->\n',encoding='utf-8')
            for arm,fold,seed in itertools.product(['H0','P0','H1','H2'],[0,1],[201,202,203]):
                out=reporter.outpath(arm,fold,seed);out.mkdir(parents=True);metrics={}
                for role in ['dev_val','confirm']:
                    saved={};per={}
                    for rx in ['rx0','rx1']:
                        y=np.repeat(np.arange(10),3);logits=np.eye(10,dtype=np.float32)[y]*3
                        errors={'H0':1,'P0':4,'H1':2,'H2':4}[arm]
                        logits[:errors]=0;logits[np.arange(errors),(y[:errors]+1)%10]=3
                        distance=np.abs(np.arange(10)[:,None]-np.arange(10)[None,:]).astype(np.float32)
                        saved[rx]=logits;saved['y_'+rx]=y;saved['margin_'+rx]=np.where(logits.argmax(1)==y,3.,-3.);saved['distances_'+rx]=distance
                        per[rx]=evaluator.previous.rx_metrics(logits,y)
                    np.savez_compressed(out/f'diagnostics_{role}.npz',**saved);metrics[role]=dict(per_rx=per)
                write(out/'evaluation.json',dict(metrics=metrics))
            reporter.report()
            d=read(reporter.HERE/'decision.json')
            assert d['status']=='COMPLETE_STOP' and d['bn_importance_signal']
            assert abs(d['recovery_ratio']-2/3)<1e-12
            assert (reporter.HERE/'tx_rx_deltas.csv').exists()
    finally:
        reporter.HERE=original_here;evaluator.sealed=original_sealed

def run():
    verify_architecture()
    assert abs(recovery(97,94,95.5)-.5)<1e-12
    assert recovery(94,94,95) is None
    assert recovery(93,94,95) is None
    assert recovery(94.1,94,95) is None
    assert recovery(97,94,93)<0
    assert recovery(97,94,98)>1
    assert bn_signal(dict(mean_delta_pp=-1,negative_pairs=5,fold_delta_pp={'0':-.5,'1':-1.5}))
    assert not bn_signal(dict(mean_delta_pp=-1,negative_pairs=4,fold_delta_pp={'0':-.5,'1':-1.5}))
    assert not bn_signal(dict(mean_delta_pp=-1,negative_pairs=5,fold_delta_pp={'0':0,'1':-2}))
    # JSON must reject nonfinite summaries instead of silently shipping invalid evidence.
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)/'value.json';write(p,dict(value=1));assert read(p)==dict(value=1)
        try:write(p,dict(value=float('nan')))
        except ValueError:pass
        else:raise AssertionError('Nonfinite report accepted')
    synthetic_report()
    print('PASS: architecture, gradients, paired initialization, recovery and BN gates including ties, finite artifacts, full synthetic report/manuscript/CSV generation')

if __name__=='__main__':run()
