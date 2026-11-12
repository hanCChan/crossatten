"""Meaningful invariant checks before the registered training budget is spent."""
import tempfile
from pathlib import Path
from s21_models import verify_architecture
from s21_report import recovery, bn_signal
from s21_core import write, read

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
    print('PASS: parameter count, token geometry, surviving initialization, differentiable exact interpolation, recovery denominator guards, finite artifacts')

if __name__=='__main__':run()
