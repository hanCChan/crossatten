"""Synthetic paired source-RX gate; no benchmark data access."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "07-wifi-osu/07-3-source-select.py"
spec = importlib.util.spec_from_file_location("s07_select", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(a, b):
    return dict(acc=(a + b) / 2, entry=dict(per_rx=dict(A=dict(acc=a), B=dict(acc=b))))


controls = [row(.8, .8), row(.8, .8)]
assert module.supported(row(.82, .83), row(.82, .84), controls)
assert not module.supported(row(.82, .83), row(.79, .8), controls)
assert not module.supported(row(.90, .77), row(.90, .77), controls)
assert not module.supported(row(.805, .805), row(.805, .805), controls)
print("PASS paired source-only improvement and worst-RX combination gates")
