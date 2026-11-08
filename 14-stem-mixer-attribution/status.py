"""Fast read-only status for both stages; no torch/CUDA import."""
from s14_core import HERE,S13,read,now

print('Time:',now())
if (S13/'progress.json').exists():print('S13:',read(S13/'progress.json'))
if (HERE/'STATUS.json').exists():print('S14:',read(HERE/'STATUS.json'))
for p in (HERE/'runs').glob('*/live.json'):
    if not (p.parent/'metrics.json').exists():print('Active:',read(p))
