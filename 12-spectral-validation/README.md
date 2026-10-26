# S12 Spectral Validation

Started 2026-10-26 from S11 `db126640`, on `exp/s12-spectral-validation`.
Training authorized by the user on 2026-10-26: diagnostics, smoke, pilot,
then independent confirmation only if the frozen source gate passes.
Historical results are accepted as correct. Do not find, download, load, hash,
or replay old checkpoints. S00-S11 and shared libraries remain unchanged.

## Scope

- S12-A fixes F2 smoke statistics and stratified smoke coverage.
- F2 always estimates its prior from the complete legal 70% fit partition,
  using at least 10 packets/cell and 3 receivers/TX. Validation/held data cannot
  enter prior estimation. Smoke reduces only network samples and epochs.
- Smoke covers observed TX/RX/day cells; its accuracy is engineering-only.
- Raw TX scatter retention remains descriptive, with no standalone veto.
  Scale-free scatter ratio, unit-feature geometry and worst RX are complementary
  diagnostics, not proof of identity-information preservation.
- Uniform, frequency-shuffled and within-RX/day TX-permuted weights are frozen
  source diagnostic controls, not additional trained candidates.
- S12-B reserves 16 candidate fits: 2 datasets x 4 folds x first group x
  seed8101 x F1/F2. Reuse eight historical Hard metric/prediction records only.
- Independent confirmation reserves 32 winner fits **plus 32 fresh Hard fits**.
  Seeds12801/12802 are new; historical-seed controls cannot be called paired.
  Confirmation requires a separate committed source decision and job matrix.
- Target access is disabled. No retroactive change to S11's stop verdict,
  new normalization/backbone search, or target-guided threshold changes.

## Execution

Preparation and CPU synthetic regression tests do not launch training or read IQ.
The registration and code must be committed before any execution is accepted.

```bash
PY=/data/miniforge/envs/env-3.12/bin/python
$PY 12-spectral-validation/register.py
$PY -m unittest discover -s 12-spectral-validation -p 'test*.py' -v
$PY 12-spectral-validation/launch.py --phase pilot
```

After separate training approval and local preregistration commit, use
`launch.py --phase smoke --index 0 --execute` for one registered job.
Every attempt is exclusive; no automatic retries. Results: `RESULTS.md`.
`execute.py` waits for the complete bounded source workflow and commits
source evidence before conditional confirmation; it never pushes or reads targets.

## Outcome

4/4 smoke and 16/16 pilot fits completed without failure.
Both candidates failed the frozen joint source gate; confirmation did not run.
See `RESULTS.md` and `PILOT_SOURCE_RESULTS.md`. Do not rerun the finished
workflow or reinterpret ManySig gains as authorization to bypass the gate.
