"""Only the registered reliability function changes from S09."""
import torch
import torch.nn.functional as F
from stage import s09
from xa.explore_models import frequency_views
from xa.frontends import rms_normalize


def phase_view(base, kind="soft", tau=.05):
    if base.ndim != 3 or base.shape[1] != 3 or base.shape[-1] < 3:
        raise ValueError("expected [N,3,L>=3]")
    if tau != .05:
        raise ValueError("S10 tau is frozen to 0.05")
    if kind == "hard":
        return s09.phase_features_from_base(base, "first")
    if kind not in ("soft", "unweighted"):
        raise ValueError(kind)
    z = torch.complex(base[:, 0], base[:, 1])
    a = z.abs()
    p = z[:, 1:] * z[:, :-1].conj()
    p = p / (a[:, 1:] * a[:, :-1]).clamp_min(1e-6)
    if kind == "soft":
        r = a.square() / (a.square() + tau ** 2)
        p = p * r[:, 1:] * r[:, :-1]
    extra = F.pad(torch.stack((p.real, p.imag), 1), (1, 0))
    return torch.cat((base, extra), 1)


class ReliabilityRF(s09.PhaseRF):
    def __init__(self, classes, feature):
        super().__init__(classes, "first")
        self.feature = feature
        if feature == "no_hstu":
            self.encoder.blocks = torch.nn.ModuleList()

    def tokens(self, x):
        kind = "soft" if self.feature == "no_hstu" else self.feature
        return self.frequency(phase_view(frequency_views(rms_normalize(x)), kind))


def paired_model(dataset, feature, seed):
    canonical, common_sha = s09.paired_model(dataset, "first", seed)
    if feature == "hard":
        return canonical, common_sha
    with torch.random.fork_rng(devices=[]):
        model = ReliabilityRF(10 if dataset == "ManyRx" else 6, feature)
    state = canonical.state_dict()
    if feature == "no_hstu":
        state = {k: v for k, v in state.items() if not k.startswith("encoder.blocks.")}
    model.load_state_dict(state)
    return model, common_sha
