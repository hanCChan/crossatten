"""Registered one-character parser repair; original frozen files stay intact."""
import argparse
import ast
import ctypes
import importlib.util
import shutil
import sys
import time
from s26_common import HERE,read,write,sha,committed,validate,now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int);args=parser.parse_args()
    validate();record=HERE/'TECHNICAL_FIX_001.json';committed(record);fix=read(record)
    original=HERE/'s26_evaluate.py';corrected=HERE/'technical_evaluate_001.py'
    assert sha(original)==fix['original_sha256'] and sha(corrected)==fix['corrected_sha256']
    assert sha(__file__)==fix['resume_code_sha256']
    # The sole scientific-source difference is deleting the unmatched closing
    # parenthesis on the COSTS write. No original registered byte is changed.
    source=original.read_text(encoding='utf-8')
    assert source.count("Not pooled with older costs.')))")==1
    assert corrected.read_text(encoding='utf-8')==source.replace("Not pooled with older costs.')))","Not pooled with older costs.'))")
    ast.parse(corrected.read_text(encoding='utf-8'))
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if args.wait_pid:
            kernel=ctypes.windll.kernel32
            kernel.OpenProcess.restype=ctypes.c_void_p
            handle=kernel.OpenProcess(0x00100000,False,args.wait_pid)
            if handle:
                print('Waiting for original controller PID',args.wait_pid,'to finish its six source-only fits.',flush=True)
                while kernel.WaitForSingleObject(ctypes.c_void_p(handle),0)==258:time.sleep(5)
                kernel.CloseHandle(ctypes.c_void_p(handle))
            expected=[HERE/'runs'/f"LP_f{j['fold']}_s{j['seed']}"/'metrics.json' for j in read(HERE/'registration.json')['jobs']]
            if not all(p.exists() and read(p)['status']=='COMPLETE' for p in expected):
                write(HERE/'TECHNICAL_FIX_001_RUNTIME.json',dict(time=now(),status='STOP_MISSING_COMPLETED_FITS',
                    explanation='The parser repair cannot restart a failed scientific fit or alter its parameters. Inspect the preserved fit failure separately.'))
                raise RuntimeError('Original controller stopped before all6 fits completed; no score-driven retries')
        if (HERE/'EXECUTION.json').exists() and not (HERE/'EXECUTION.pre_technical_001.json').exists():
            shutil.copyfile(HERE/'EXECUTION.json',HERE/'EXECUTION.pre_technical_001.json')
        specification=importlib.util.spec_from_file_location('s26_evaluate',corrected)
        module=importlib.util.module_from_spec(specification);sys.modules['s26_evaluate']=module;specification.loader.exec_module(module)
        write(HERE/'TECHNICAL_FIX_001_RUNTIME.json',dict(time=now(),status='RUNNING_REGISTERED_REPAIR',
            original_frozen_source_retained=True,new_fits_added=0))
        # The original coordinator reuses completed models/seal and imports the
        # corrected module. Child fit processes never import the evaluation path.
        import s26_execute
        sys.argv=[str(HERE/'s26_execute.py')]
        s26_execute.main()
        write(HERE/'TECHNICAL_FIX_001_RUNTIME.json',dict(time=now(),status='COMPLETE',
            original_frozen_source_retained=True,new_fits_added=0,protocol_changes=False))
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
