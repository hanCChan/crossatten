"""Read the frozen WiSig cohort and weights."""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for relative in ("s29_wisig", "s33_wisig"):
    sys.path.insert(0, str(HERE.parent / relative))
from experiment import DEVICE, FeatureModel, featurize, tensor, torch
from extract import centers
from source import Source, read, sha, write

VIEWS = ("V0", "V1")
ROLES = ("fit_train", "checkpoint_val", "source_selection")
ARCHES = ("HStem", "LongPatch")
LAYERS = ("token", "mix1", "mix2", "pre_pool")


def restrict_torch():
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


class FrozenStudy:
    def __init__(self, s29, s33, out):
        self.s29 = s29 / "raw_control"
        self.s33 = s33
        self.source = Source(out)
        self.records = read(self.s29 / "FIT_SEAL.json")["results"]
        self.audit = read(s33 / "RECEIPT.json")
        assert len(self.records) == 8 and read(s33 / "QA.json")["status"] == "PASS"

    def data(self, view, roles, stage):
        data = {role: self.source.load(view, role, stage) for role in roles}
        archive = self.s33 / f"AUDIT_{view}.npz"
        assert sha(archive) == self.audit["packets"][view]
        with np.load(archive, allow_pickle=False) as z:
            stored = {k: z[k] for k in z.files}
        held = data["source_selection"]
        assert all(np.array_equal(stored["source_selection_" + k], held[k])
                   for k in ("ids", "y", "rx", "day"))
        return data, stored

    def models(self, view):
        for record in self.records:
            v, seed, architecture, arm = record["job"]
            if v != view:
                continue
            checkpoint = self.s29 / record["selected_path"]
            assert arm == "raw" and sha(checkpoint) == record["selected_sha256"]
            model = FeatureModel(architecture, 24101 + seed)
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu")["model"])
            yield seed, architecture, model.to(DEVICE).eval()
