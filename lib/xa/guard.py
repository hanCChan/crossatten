"""Data-access guard: refuses blind receivers, gates confirmation receivers, logs every access."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import ROOT

PROTOCOL_PATH = ROOT / "configs" / "protocol.json"
ACCESS_LOG = ROOT / "logs" / "data_access.jsonl"


def load_protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


_P = load_protocol()
BLIND = tuple(_P["blind_receivers_never_open"])
DEV = tuple(_P["dev_receivers"])
CONFIRM = tuple(_P["confirm_receivers"])


class AccessDenied(RuntimeError):
    pass


def _contains_blind(name: str) -> bool:
    # 'b200_mini_2' must not match 'b200_2' check accidentally the other way round; test every blind token.
    stem = Path(name).name
    for b in BLIND:
        i = stem.find(b)
        while i >= 0:
            after = stem[i + len(b): i + len(b) + 1]
            if not after.isdigit():  # avoid n210_2 matching n210_20 style names
                return True
            i = stem.find(b, i + 1)
    return False


def check_raw_path(path: str | Path) -> Path:
    p = Path(path)
    if _contains_blind(p.name):
        raise AccessDenied(f"blind receiver file refused: {p.name}")
    return p


def log_access(rx: str, purpose: str, who: str) -> None:
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(t=time.strftime("%Y-%m-%dT%H:%M:%S"), rx=rx, purpose=purpose, who=who, pid=os.getpid())
    with ACCESS_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def check_rx(rx: str, purpose: str, who: str, decision_file: str | None = None) -> None:
    """purpose in {'cache_build','source','dev_val','confirm','audit'}."""
    if rx in BLIND or _contains_blind(rx):
        raise AccessDenied(f"blind receiver refused: {rx}")
    if rx not in DEV:
        raise AccessDenied(f"unknown receiver: {rx}")
    if rx in CONFIRM and purpose not in ("cache_build", "confirm"):
        raise AccessDenied(f"confirmation receiver {rx} requested for purpose={purpose}")
    if purpose == "confirm":
        if rx not in CONFIRM and rx not in DEV:
            raise AccessDenied(rx)
        if decision_file is None or not Path(decision_file).is_file():
            raise AccessDenied("confirm access requires an existing frozen decision file")
    log_access(rx, purpose, who)
