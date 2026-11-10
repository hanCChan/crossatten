"""Seal the fixed D0 before reading learned features or intervention outcomes."""
from s18_core import HERE, ROOT, HISTORY, now, read, sha, write, git
from s18_data import locations
from s18_decide import gates


def register():
    path = HERE/'registration.json'
    if path.exists(): raise RuntimeError('S18 already registered')
    if list(HERE.glob('D0_fold*.json')) or (HERE/'access.jsonl').exists():
        raise RuntimeError('Scientific reads/results precede registration')
    tests = read(HERE/'ENGINEERING_TESTS.json')
    if tests['status'] != 'PASS': raise ValueError('Synthetic checks must pass first')
    code = (list(HERE.glob('*.py'))+list((ROOT/'lib').rglob('*.py'))+
            [ROOT/'13-mixer-cfo-mechanisms'/p for p in ['core.py', 'models.py', 'wifi.py', 'metrics.py']]+
            [ROOT/'17-rx-distortion'/p for p in ['s17_core.py', 's17_models.py']]+[ROOT/'09-phase-invariance/phase_features.py'])
    history = [HISTORY/p for p in ['15-hstem-attribution/decision.json', '16-rf-qwen3/STATUS.json',
                '17-rx-distortion/decision_d0.json', '17b-phase-intervention/decision_d1.json',
                '17b-phase-intervention/final_evidence/MANIFEST.json', '10-tifs-validation/final_comparisons.csv',
                '10-tifs-validation/receiver_cluster_intervals.csv']]
    for fold in range(4):
        cache, run = locations(fold)
        history.append(cache/'fit_manifest.json')
        history.extend(run/p for p in ['config.json', 'metrics.json'])
    result = dict(time=now(), stage='S18-D0', base_commit=git('rev-parse', 'HEAD'),
                  code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(code))},
                  historical_sha256={p.relative_to(HISTORY).as_posix():sha(p) for p in history},
                  design_sha256={p.name:sha(p) for p in [HERE/'README.zh-CN.md', HERE/'PROTOCOL_AUDIT.zh-CN.md', HERE/'ENGINEERING_TESTS.json']},
                  primary_layer='tokens', diagnostic_layers=['stem1', 'stem_mid', 'tokens', 'mixer'],
                  rank=3, alpha=.5, ridge_relative=1e-3, estimation_cap_per_cell=48,
                  random_bases=3, random_basis_seed='18200+10*fold+draw', split_seed=16001, baseline_seed=18101,
                  data_scope='ManyRx fit18 source RX only dates1-2; training70% directions, validation30% evaluation',
                  evaluation_limit='Existing source validation was used for historical checkpoint selection; not untouched confirmation',
                  gates=gates(), decision_unit='RX identity; repeated folds averaged per RX before bootstrap',
                  formal_fits_d0=0, automatic_training_on_failure=False, real_target_access=False, held_source_access_d0=False,
                  pilot_budget=dict(new_fits_if_B0_reusable=16, new_fits_otherwise=20),
                  no_parameter_search=True, no_automatic_scientific_retry=True)
    write(path, result)


if __name__ == '__main__': register()
