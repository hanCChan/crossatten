"""Append the user's four-arm/source-metadata request without rewriting initial results."""
from pathlib import Path
from s25_common import HERE, ROOT, digest, now, read, validate, write


def main():
    validate();assert not (HERE/'analysis_extension_001.json').exists()
    refs=read(ROOT/'23-hstem-mechanism-evidence/registration.json')['references'];models=[]
    for arm in ['S1','H1']:
        for fold in (0,1):
            for seed in (201,202,203):
                identity=f'{arm}_f{fold}_s{seed}';r=refs[identity];p=Path(r['path'])
                assert digest(p/'model.pt')==r['sha256']['model.pt']
                config=read(p/'config.json')
                models.append(dict(id=identity,arm=arm,fold=fold,seed=seed,data_seed=config['data_seed'],params=config['params'],
                    checkpoint=str(p/'model.pt'),checkpoint_sha256=r['sha256']['model.pt']))
    code=[HERE/'mechanism_extension.py',HERE/'register_extension.py',ROOT/'22-mechanism-closure/s22_models.py',ROOT/'22-mechanism-closure/s22_core.py']
    evidence=[ROOT/'23-hstem-mechanism-evidence/kernel_bands.csv',ROOT/'22-mechanism-closure/tx_rx_metrics.csv',ROOT/'21-hstem-mechanisms/class_geometry.csv']
    write(HERE/'analysis_extension_001.json',dict(time=now(),status='FROZEN_USER_STEERING_EXTENSION',new_fits=0,
        parent_registration_sha256=digest(HERE/'registration.json'),initial_source_result_sha256=digest(HERE/'SOURCE_GEOMETRY.json'),
        reason='User confirms no new capture; requests A/B/C read-only mechanism evidence with H0/P0/S1/H1. No automatic Long-Patch or other training.',
        prior_exposure='Initial H0/P0 source geometry already observed; expanded comparisons remain exploratory.',additional_models=models,
        source_rule='Same original last40/TX/RX validation rows, all folds/seeds, all three registered layer taps, unchanged per-packet L2 geometry.',
        margins=['True logit minus maximum wrong logit, all samples','Cosine feature/weight margin, classifier bias omitted only for this explicitly separate diagnostic'],
        metadata=dict(fields=['CFO','SNR','label'],role='source_val_only',selected_rows_only=True,waveform_data_field_forbidden=True,
            schema={'CFO':[1,8000],'SNR':[1,8000],'label':[1,8000]},
            associations='Raw Spearman and TXxRX-cell-centered Spearman with raw and cosine margin; no p-values; finite entries only; range<=1e-12 yields NA. CFO absolute; SNR as provided. Never model inputs.'),
        kernels='Reuse S23 high-frequency effective-kernel energy for P0/S1/H1 only; descriptive association across18 fits, confounded by arm; no RF hardware interpretation.',
        legacy_geometry_audit='S21 class_geometry has only Dev/Confirm pairwise class-distance correlation/RMSE; it does not contain the requested source-validation normalized scatter at all registered taps. Reuse existing margins/error CSVs; compute missing source features only.',
        new_data_status='User confirms no new capture: independent validation not run.',
        long_patch_status='Audit/proposal only; no new fit allowed in this stage.',
        code_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in code},evidence_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in evidence}))
    print('Frozen four-arm extension before S1/H1 extraction or source metadata values.')


if __name__=='__main__':main()
