"""IO-only recovery after all18 fits sealed; no scientific byte is modified."""
from pathlib import Path
import tempfile
import numpy as np
from ac_common import HERE,read,write,sha,now,validate,committed,identity
import ac_evaluate

def save_predictions(stage,arm,fold,seed,filename,**data):
    assert Path(filename).name==filename
    name=identity(dict(stage=stage,arm=arm,fold=fold,seed=seed));out=HERE/'predictions'/name;out.mkdir(parents=True,exist_ok=True)
    path=out/(filename+'.npz');np.savez_compressed(path,**data);return path

def main():
    global HERE
    validate();receipt=read(HERE/'TECHNICAL_REPAIR_001.json');committed(HERE/'TECHNICAL_REPAIR_001.json');committed(Path(__file__).resolve())
    assert sha(__file__)==receipt['repair_script_sha256']
    for stage,h in receipt['sealed_cohort_sha256'].items():assert sha(HERE/f'seal_{stage}.json')==h
    assert sha(HERE/'ac_evaluate.py')==receipt['original_evaluator_sha256']
    # Use the same writer against a temporary data-free output root.
    original=HERE
    with tempfile.TemporaryDirectory(prefix='ac-save-repair-') as folder:
        HERE=Path(folder)
        try:
            path=save_predictions('A','H0',0,201,'SOURCE',logits=np.zeros((4,10)),y=np.arange(4),rx=np.arange(4),rows=np.arange(4))
            with np.load(path,allow_pickle=False) as z:assert z['rx'].tolist()==list(range(4)) and z['logits'].shape==(4,10)
        finally:HERE=original
    ac_evaluate.save_predictions=save_predictions
    ac_evaluate.main()
    write(HERE/'REPAIR_RESULT_001.json',dict(time=now(),status='COMPLETE_IO_ONLY_NO_REFIT',synthetic_writer_fixture='PASS',scientific_optimizer_steps=0,
        scientific_fits=0,all_original_code_bytes_preserved=True,plans_and_all18_sealed_models_preserved=True,repair_script_sha256=sha(__file__)))
    write(HERE/'STATUS.json',dict(time=now(),status='TRAINING_AND_REGISTERED_EVALUATION_COMPLETE',new_fits=18,report='PENDING_INDEPENDENT_AUDIT',IO_repair='COMPLETE_001'))
    print('All18 sealed models evaluated; IO recovery complete; no training rerun.',flush=True)

if __name__=='__main__':main()
