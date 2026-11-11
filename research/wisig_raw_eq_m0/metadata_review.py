"""Bounded text/schema and filename inventory; never decodes scientific payloads.

Stdlib only. It does not import the project's scientific loaders. ZIP inspection
uses infolist() only; there is no read(), open(), extract() or NumPy/pickle import.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = Path("D:/crossatten")

TEXT = [
    REPO / "research/classification_failure_diagnostics/D4_FRONTEND_AUDIT.zh-CN.md",
    REPO / "research/classification_failure_diagnostics/INPUT_REGISTRY.json",
    REPO / "05-wisig/README.md",
    REPO / "05-wisig/05-1-build-wisig-cache.py",
    REPO / "05-wisig/out/05-1-ManyRx-manifest.json",
    REPO / "lib/xa/wisig.py",
    REPO / "13-mixer-cfo-mechanisms/data.py",
    REPO / "13-mixer-cfo-mechanisms/pickle_index.py",
    REPO / "17-rx-distortion/s17_data.py",
    WORKSPACE / "worktrees/s17-rx-distortion/17b-phase-intervention/cache/ManyRx_f2_first_s16001/fit_manifest.json",
    WORKSPACE / "stage_1/data/wisig/DOWNLOAD_MANIFEST.json",
]
DIRECTORIES = [
    WORKSPACE / "stage_1/data/wisig",
    WORKSPACE / "stage_1/13-mixer-cfo-mechanisms/cache/wisig/ManyRx",
    WORKSPACE / "worktrees/s17-rx-distortion/17b-phase-intervention/cache/ManyRx_f2_first_s16001",
]
ARCHIVE = WORKSPACE / "stage_1/data/wisig/ManyRx.kaggle.v1.zip"


def schema(value, depth=0):
    if isinstance(value, dict):
        return {key: schema(item, depth + 1) for key, item in value.items()} if depth < 2 else {"type": "dict", "keys": sorted(value)}
    if isinstance(value, list):
        # Union of object fields, without opening referenced files or reporting
        # archived scientific metric values.
        fields = sorted({key for item in value if isinstance(item, dict) for key in item})
        return {"type": "list", "length": len(value), "object_fields": fields}
    return {"type": type(value).__name__}


def main():
    result = {"scope": "text_schema_and_filename_metadata_only", "text_files": [], "directory_inventories": [], "archive_central_directory": {}, "scientific_payload_decodes": 0, "scientific_forward_passes": 0, "scientific_fits": 0}
    for path in TEXT:
        item = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            raw = path.read_bytes()
            # Only files explicitly enumerated in TEXT can enter this branch.
            body = raw.decode("utf-8-sig")
            item.update(size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            if path.suffix.lower() == ".json":
                obj = json.loads(body)
                item["schema"] = schema(obj)
                if path.name == "05-1-ManyRx-manifest.json":
                    fields = ("dataset", "source", "source_sha256", "archive_sha256", "non_equalized_index", "first_packets_per_cell", "tx_list", "capture_dates")
                    item["saved_provenance"] = {key: obj[key] for key in fields if key in obj}
            item["identity_related_text_lines"] = [
                {"line": i, "text": line}
                for i, line in enumerate(body.splitlines(), 1)
                if re.search(r"packet.?id|packet.?map|pair.?map|pairing_evidence|No immutable packet|跨版本包 ID|逐包对应尚未", line, re.I)
            ]
        result["text_files"].append(item)
    for base in DIRECTORIES:
        names = sorted(str(path.relative_to(base)).replace("\\", "/") for path in base.rglob("*") if path.is_file()) if base.is_dir() else []
        result["directory_inventories"].append({"path": str(base), "exists": base.is_dir(), "filenames_only": names, "json_sidecar_filenames": [name for name in names if name.lower().endswith((".json", ".jsonl"))], "mapping_named_files": [name for name in names if re.search(r"pair|mapping|packet.?id|timestamp|offset", name, re.I)]})
    if ARCHIVE.is_file():
        with zipfile.ZipFile(ARCHIVE) as archive:
            entries = [{"name": member.filename, "uncompressed_bytes": member.file_size, "compressed_bytes": member.compress_size, "crc32": f"{member.CRC:08x}"} for member in archive.infolist()]
        result["archive_central_directory"] = {"path": str(ARCHIVE), "entries": entries, "member_payloads_read": 0}
    else:
        result["archive_central_directory"] = {"path": str(ARCHIVE), "exists": False, "member_payloads_read": 0}
    out = HERE / "LOCAL_METADATA_REVIEW.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "text_files": len(result["text_files"]), "directories": [{"path": item["path"], "files": len(item["filenames_only"]), "json_sidecars": item["json_sidecar_filenames"], "mapping_named_files": item["mapping_named_files"]} for item in result["directory_inventories"]], "archive": result["archive_central_directory"], "scientific_payload_decodes": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
