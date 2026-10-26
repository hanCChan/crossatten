import importlib.util
import torch
import pytest
from training import HERE, paired_model, s08, read_cells, state_sha, n_params


def test_exact_shared_initialization_and_capacity():
    for dataset in ("ManySig", "ManyRx"):
        models = {f: paired_model(dataset, f, 77) for f in ("frequency", "null", "first", "centered", "curvature")}
        assert len({value[1] for value in models.values()}) == 1
        state = models["frequency"][0].state_dict()
        for feature, (model, _) in models.items():
            for name, value in state.items():
                other = model.state_dict()[name]
                if name == "frequency.stem.0.weight":
                    other = other[:, :3]
                torch.testing.assert_close(value, other, rtol=0, atol=0)
        assert n_params(models["first"][0]) - n_params(models["frequency"][0]) == 448
        assert n_params(models["first"][0]) == n_params(models["null"][0])


def test_real_feature_and_model_parity_with_s08():
    x = torch.randn(4, 2, 256)
    model, _ = paired_model("ManySig", "first", 19)
    previous = s08.build_model("ManySig", "phase")
    previous.load_state_dict(model.state_dict())
    model.eval()
    previous.eval()
    torch.testing.assert_close(model(x), previous(x), rtol=0, atol=0)


def test_null_and_frequency_shared_function():
    x = torch.randn(4, 2, 256)
    freq, _ = paired_model("ManySig", "frequency", 19)
    null, _ = paired_model("ManySig", "null", 19)
    freq.eval()
    null.eval()
    torch.testing.assert_close(freq(x), null(x), rtol=1e-5, atol=1e-6)


def test_source_guard_rejects_target_and_dates():
    job = dict(dataset="ManySig", fold=0, phase="source", id="unit")
    with pytest.raises(ValueError):
        read_cells(job, ["1-1"], (1, 2), "fit")
    with pytest.raises(ValueError):
        read_cells(job, ["2-19"], (3, 4), "held")


def test_gate_requires_paired_groups_and_seed_evidence():
    spec = importlib.util.spec_from_file_location("selector", HERE / "09-3-select.py")
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    def rows(acc):
        return [dict(job=dict(group=g, seed=s),
                     entry=dict(epoch=5, per_rx={g: dict(acc=acc)},
                                summary=dict(acc_mean=acc)))
                for g in ("first", "last") for s in (1, 2)]
    assert selector.gate(rows(.9), rows(.8))["supported"]
    assert not selector.gate(rows(.79), rows(.8))["supported"]
    with pytest.raises(ValueError):
        selector.gate(rows(.9)[:-1], rows(.8))
