"""Evaluate a matching ASKNet checkpoint on the separate public drift captures."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
from statistics import mean

import torch

from audit_upstream import ASK_COMMIT, ASK_LOG, ASK_MODEL, ask_log, pinned_reader


def source_namespace(read, file):
    namespace = {}
    exec(compile(read(file), file, "exec"), namespace)
    return namespace


@torch.inference_mode()
def run(repo, checkpoint, drift_root, origin):
    read, hashes = pinned_reader(repo, ASK_COMMIT)
    candidates, _, reference = ask_log(read(ASK_LOG))
    model_code = source_namespace(read, ASK_MODEL)
    loader = source_namespace(read, "Lora/utils/load_data_h5.py")
    with checkpoint.open("rb") as file:
        checkpoint_sha = file_digest(file, "sha256").hexdigest()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cutoff = state["ask.cutoff.selection_logits"]
    masks = state["ask.cutoff.masks"]
    expected = model_code["LearnableCutoff"](8192, candidates).state_dict()["masks"]
    if cutoff.shape != (len(candidates),) or not torch.equal(masks, expected):
        raise ValueError("Checkpoint cutoff masks differ from the public log configuration")
    with torch.device("meta"):
        model = model_code["ResCISNet"](num_classes=10, candidates=candidates)
    model.load_state_dict(state, strict=True, assign=True)
    selected = candidates[int(cutoff.argmax())]
    if origin == "author_release" and (
        selected != reference["rtl_6"]["selected_k"] or
        abs(float(cutoff.softmax(0).max()) - 0.1399) > 0.00005
    ):
        raise ValueError("Released checkpoint selection statistics differ from the public log")
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.cuda().eval()
    allowed = ("n210_1", "rtl_6")
    rows, source_files = {}, {}
    for rx in allowed:
        by_day = {}
        for day in range(1, 5):
            source = drift_root / "Test" / f"{rx}_day{day}_wireless_test.h5"
            with source.open("rb") as file:
                source_files[source.name] = file_digest(file, "sha256").hexdigest()
            x, y = loader["prepare_test_by_day"](
                str(drift_root), [rx], [day], last_k_classes=10,
                per_class=800, seq_len=8192,
            )[day]
            if x.shape != (2000, 2, 8192) or len(y) != 2000:
                raise ValueError(f"Unexpected receiver-drift support in {source.name}")
            n_correct = sum(
                int((model(inputs.cuda()).argmax(1).cpu() == labels).sum())
                for inputs, labels in zip(torch.from_numpy(x).split(512), torch.from_numpy(y).split(512))
            )
            match = reference[rx]["day_accuracy"][day]
            by_day[day] = dict(n_correct=n_correct, n=len(y), accuracy=n_correct / len(y),
                               author_accuracy=match, delta_pp=100 * (n_correct / len(y) - match))
            print(f"{rx} day {day}: {n_correct}/{len(y)} (author {match:.4f})", flush=True)
        rows[rx] = dict(days=by_day, mean=mean(r["accuracy"] for r in by_day.values()),
                        mean_days_2_to_4=mean(by_day[i]["accuracy"] for i in (2, 3, 4)))
    return dict(checkpoint=checkpoint.name, checkpoint_sha256=checkpoint_sha,
                checkpoint_origin=origin, cutoff_candidates=candidates, selected_k=selected,
                source_commit=ASK_COMMIT, source_code_sha256=hashes, test_files_sha256=source_files,
                torch_version=torch.__version__, device=torch.cuda.get_device_name(),
                protocol="receiver-drift N210-1 day 1 checkpoint; 2000 packets per test receiver/day",
                training_fits=0, model_results=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--drift", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--origin", choices=("author_release", "local_rerun"), required=True)
    args = parser.parse_args()
    evidence = run(args.repo, args.checkpoint, args.drift, args.origin)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
