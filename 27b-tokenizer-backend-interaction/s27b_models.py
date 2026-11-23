"""Use byte-frozen S27 B architecture, without redefining its readout."""
import hashlib
from s27b_common import ROOT
from s27_models import build,TokenMLP,state_hash,geometry,shared_hash

def surviving_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        if '.encoder.blocks.' in k:continue
        h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
