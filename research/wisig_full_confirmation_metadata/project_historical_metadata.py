"""Project receiver identities from existing text, and ZIP central-directory names only."""
from pathlib import Path
import csv
import hashlib
import json
import re
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKSPACE = Path('D:/crossatten')

def main():
    metadata = json.loads((HERE / 'CANDIDATE_METADATA.json').read_text(encoding='utf-8'))
    candidates = [c['rx'] for c in metadata['candidates']]
    path = ROOT / '10-tifs-validation/final_receiver_metrics.csv'
    # Identity-only projection. No metric field is retained, computed or displayed.
    with path.open(encoding='utf-8-sig', newline='') as handle:
        identity = {row['rx'] for row in csv.DictReader(handle) if row['dataset'] == 'ManyRx'}
    assert identity == set(metadata['manyrx32'])
    archive_records = []
    for name in ['multiple_receiver_train.zip', 'multiple_receiver_test.zip', 'receiver_drift_dataset.zip']:
        archive = WORKSPACE / name
        if not archive.exists():
            archive_records.append({'archive': name, 'status': 'ABSENT'})
            continue
        with zipfile.ZipFile(archive) as zf:
            entries = zf.namelist()  # No ZipFile.open/read/extract; no payload decoding.
        hits = {r: [n for n in entries if re.search(r'(?<![0-9-])' + re.escape(r) + r'(?![0-9-])', n)] for r in candidates}
        archive_records.append({'archive': name, 'entry_count': len(entries), 'candidate_identity_name_matches': hits,
                                'interpretation': 'Path-name evidence only; cannot prove absence of embedded receiver data or historical use.'})
    result = {'status': 'NO_UNSEEN_CLAIM_FROM_LOCAL_ABSENCE',
              'historical_manyrx_identity_projection': {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rx_count': len(identity), 'rx': sorted(identity), 'candidate_overlap': sorted(identity & set(candidates))},
              'archive_central_directory_only': archive_records,
              'not_proven': ['No use in unarchived/deleted experiments', 'No use in other workspaces/machines/accounts', 'No inspection of candidate IQ, labels beyond metadata or model outputs in prior human sessions'],
              'historical_exposure_status_for_all_nine': 'UNKNOWN', 'confirmation_access': 'CLOSED'}
    (HERE / 'HISTORICAL_EXPOSURE.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'historical_manyrx_rx': len(identity), 'candidate_overlap': sorted(identity & set(candidates)), 'archive_entries': {r['archive']:r.get('entry_count') for r in archive_records}, 'historical_exposure':'UNKNOWN'}, indent=2))

if __name__ == '__main__':
    main()
