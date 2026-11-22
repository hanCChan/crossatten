"""Freeze a zero-fit exploratory analysis before extracting new source features."""
import subprocess
from pathlib import Path
from s25_common import HERE, ROOT, S24, digest, now, read, write


def main():
    assert not (HERE/'registration.json').exists()
    assert read(HERE/'ENGINEERING_TESTS.json')['status'] == 'PASS'
    references = read(ROOT/'23-hstem-mechanism-evidence/registration.json')['references']
    models = []
    for arm in ['H0', 'P0']:
        for fold in (0, 1):
            for seed in (201, 202, 203):
                identity = f'{arm}_f{fold}_s{seed}'
                reference = references[identity]
                folder = Path(reference['path'])
                for name in ['model.pt', 'config.json', 'metrics.json']:
                    assert digest(folder/name) == reference['sha256'][name]
                config = read(folder/'config.json')
                models.append(dict(id=identity, arm=arm, fold=fold, seed=seed, data_seed=config['data_seed'],
                                   params=config['params'], checkpoint=str(folder/'model.pt'),
                                   checkpoint_sha256=reference['sha256']['model.pt']))
    code = [HERE/name for name in ['s25_common.py','geometry_stats.py','source_geometry.py','register_analysis.py','test_source_geometry.py','per_rx_analysis.py']]
    code += list((ROOT/'lib').rglob('*.py'))
    code += [ROOT/'13-mixer-cfo-mechanisms/models.py',ROOT/'13-mixer-cfo-mechanisms/core.py',
             ROOT/'15-hstem-attribution/s15_models.py',ROOT/'15-hstem-attribution/s15_core.py',
             ROOT/'21-hstem-mechanisms/s21_models.py',ROOT/'21-hstem-mechanisms/s21_core.py']
    evidence = [ROOT/'configs/protocol.json',ROOT/'15-hstem-attribution/architecture_search.json',
                ROOT/'23-hstem-mechanism-evidence/registration.json',ROOT/'23-hstem-mechanism-evidence/PAPER_OVERLAP_REVIEW.zh-CN.md',
                ROOT/'24-lora-external-audit/phase2/budget_update/RESULTS.json',
                ROOT/'24-lora-external-audit/phase2/budget_update/COSTS.json',ROOT/'24-lora-external-audit/DATA_AUDIT.json']
    cache = {r['rx']:r for r in read(S24/'DATA_AUDIT.json')['files']}
    obj=dict(stage='S25_TIFS_INTEGRATION', time=now(), base_git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             new_fit_budget=0, models=models, source_caches=cache,
             prior_exposure='All S15-S24 Dev results and previous source-validation diagnostics already seen; this is exploratory, not independent preregistered confirmation.',
             source_rule='Only original source-val last40/TX/RX; five source RX/fold; all 2fold x3seed xH0/P0; no noise, adaptation or optimization.',
             primary_geometry='Final 64D classifier input, per-packet L2 normalized; balanced between-TX / TX-conditional RX centroid / within-cell scatter decomposition.',
             secondary_geometry=['Mean tokenizer output','Mean pre-MHA output'], epsilon=1e-12,
             report='All six paired differences, raw scatter terms and ratios; no hypothesis selection, p-values or new generalization claims.',
             per_rx='Existing saved S24 confusion matrices only; all RX/TX/seed. Pluto contribution is a post-hoc descriptive decomposition, no exclusions or changes to primary tables.',
             costs='H0/P0 f0seed201; 1/128 batch; FP32 TF32-off; 3 warmup15 samples; synthetic input seed25025; separate benchmark session.',
             independent_data='No qualified new capture identified yet; metadata audit and closed evaluation gate only until provenance and enrolled-TX identity are available.',
             new_dev_waveform_reads=0, confirm_blind_forbidden=True, ask_strict_ib_still_deferred=True,
             code_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in code},
             evidence_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in evidence})
    write(HERE/'registration.json',obj)
    print('Frozen zero-fit analysis registration; commit before any source feature IO.')


if __name__=='__main__':
    main()
