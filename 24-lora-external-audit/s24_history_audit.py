"""Read-only compatibility audit of source manifests and frozen H0 artifacts."""
import json
from pathlib import Path
from s24_common import HERE, ROOT, sha, write_json


def main():
    reference = Path('D:/crossatten/stage_1/13-mixer-cfo-mechanisms')
    data = json.loads((HERE/'DATA_AUDIT.json').read_text(encoding='utf-8'))
    comparisons = []
    for item in data['files']:
        p = reference / 'cache/shen' / item['rx'] / 'manifest.json'
        old = json.loads(p.read_text(encoding='utf-8'))
        comparisons.append(dict(rx=item['rx'], reference_manifest=str(p), sha256=sha(p.read_bytes()),
                                rows_equal=old['rows_sha256']==item['selected_rows_sha256'],
                                iq_equal=old['x_sha256']==item['selected_iq_float32_sha256']))
    reg = json.loads((ROOT/'23-hstem-mechanism-evidence/registration.json').read_text(encoding='utf-8'))
    h0 = []
    for key, value in reg['references'].items():
        if not key.startswith('H0_'): continue
        path = Path(value['path'])
        artifacts = {name:dict(sha256=sha((path/name).read_bytes()), expected=expected)
                     for name, expected in value['sha256'].items()}
        passed = all(v['sha256']==v['expected'] for v in artifacts.values())
        config = json.loads((path/'config.json').read_text(encoding='utf-8'))
        h0.append(dict(key=key, path=str(path), passed=passed, artifacts=artifacts,
                       job=config['job'], data_seed=config['data_seed'], params=config['params'],
                       checkpoint_rule=config['checkpoint']))
    result = dict(source_manifest_comparisons=comparisons, frozen_hstem_checkpoints=h0,
                  passed=all(v['rows_equal'] and v['iq_equal'] for v in comparisons) and len(h0)==6 and all(v['passed'] for v in h0),
                  scope='Compatibility and frozen artifact hashes only; no target logits, target metrics or new model inference.')
    write_json(HERE/'HISTORY_AUDIT.json',result)
    assert result['passed'], 'historical identity/checkpoint mismatch'
    print('10 source IQ/row manifests and 6 frozen H0 artifact sets verified unchanged.')


if __name__ == '__main__': main()
