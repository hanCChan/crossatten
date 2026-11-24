"""Read and check only the frozen C fit18 cache and source-role indices."""

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
C = HERE.parent / "standard_method_competition_c"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


class Source:
    def __init__(self, output):
        self.log = output / "ACCESS.jsonl"
        self.registration = read(C / "PRE_REGISTRATION.json")
        self.names = self.registration["rx_names"]
        self.views = {view["id"]: view for view in read(C / "SOURCE_PLAN.json")["views"]}
        self.meta = {name: np.load(self.bound_path(name), allow_pickle=False)
                     for name in ("fit_y", "fit_rx", "fit_day", "train_ids", "val_ids")}
        self.x = np.load(self.bound_path("fit_x"), mmap_mode="r", allow_pickle=False)
        self.y, self.rx, self.day = (self.meta["fit_" + name] for name in ("y", "rx", "day"))
        self.nr = len(self.y)
        assert self.x.shape == (34668, 2, 256) and self.x.dtype == np.float32
        assert self.y.shape == self.rx.shape == self.day.shape == (34668,)
        assert set(self.day) == {1, 2} and set(self.y) == set(range(10))
        assert np.array_equal(np.sort(np.r_[self.meta["train_ids"], self.meta["val_ids"]]),
                              np.arange(self.nr))
        assert {self.names[r] for r in set(self.rx)} == set(self.registration["fit_rx"])
        assert not set(self.registration["fit_rx"]) & set(
            self.registration["forbidden_target_rx"] + self.registration["forbidden_historical_held_rx"])

    def record(self, **event):
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")

    def bound_path(self, name):
        item = self.registration["source_input_whitelist"][name]
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError(f"Source checksum differs for {name}")
        self.record(action="verified_file_bytes", name=name, path=str(path), sha256=item["sha256"])
        return path

    def ids(self, view, role):
        config = self.views[view]
        held = np.isin(self.rx, [self.names.index(r) for r in config["source_selection_rx"]])
        if role == "source_selection":
            ids = np.flatnonzero(held).astype(np.int64)
        else:
            ids = self.meta[{"fit_train": "train_ids", "checkpoint_val": "val_ids"}[role]]
            ids = ids[~held[ids]]
        spec = config[role]
        assert ids.dtype.name == spec["dtype"] and len(ids) == spec["rows"]
        assert array_sha(ids) == spec["indices_array_sha256"]
        assert {self.names[r] for r in set(self.rx[ids])} == set(spec["rx"])
        return ids

    def load(self, view, role, stage):
        if stage == "fit" and role not in ("fit_train", "checkpoint_val"):
            raise ValueError("Training access restricted to fit receivers")
        if stage == "new_prediction" and not (self.log.parent / "FIT_SEAL.json").is_file():
            raise ValueError("New weights must be sealed before held prediction")
        ids = self.ids(view, role)
        x = self.x[ids]
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite source IQ")
        self.record(action="copy_role_iq", stage=stage, view=view, role=role, rows=len(ids),
                    rx=sorted({self.names[r] for r in set(self.rx[ids])}), indices_sha256=array_sha(ids))
        return dict(x=x, y=self.y[ids], rx=self.rx[ids], day=self.day[ids], ids=ids)
