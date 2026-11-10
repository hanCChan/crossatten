"""S19 synthetic CPU model checks; never read scientific inputs/weights."""
from pathlib import Path
import inspect
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parent.parent
for path in [ROOT/'lib', ROOT/'13-mixer-cfo-mechanisms', ROOT/'17-rx-distortion']:
    sys.path.insert(0, str(path))

from s19_models import (ALPHA, SpectralModulator, band_masks, build_model,
                        features_from_spectrum, spectrum_from_iq)
from xa.frontends import rms_normalize


class SpectralModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        cls.x = torch.randn(4, 2, 256)

    def test_masks_partition_smooth_periodic_boundary(self):
        masks = band_masks()
        self.assertEqual(tuple(masks.shape), (8, 256))
        self.assertTrue(torch.all(masks >= 0))
        torch.testing.assert_close(masks.sum(0), torch.ones(256), atol=0, rtol=0)
        torch.testing.assert_close(masks.sum(1), torch.full((8,), 32.))
        self.assertLess(float((masks[:, 1:]-masks[:, :-1]).abs().max()), .4)
        self.assertGreater(float(masks[7, 0]), 0)
        self.assertGreater(float(masks[0, -1]), 0)
        with self.assertRaises(ValueError): band_masks(length=255)
        with self.assertRaises(ValueError): band_masks(transition=20)

    def test_identity_features_logits_and_common_backbone(self):
        base = build_model('B0', 'ManyRx', 18101).eval()
        with torch.no_grad():
            clean_features = base.features(self.x)
            clean_logits = base(self.x)
            for arm in ['B1', 'B2', 'B3', 'B4']:
                model = build_model(arm, 'ManyRx', 18101).eval()
                for key, tensor in base.state_dict().items():
                    self.assertTrue(torch.equal(tensor, model.state_dict()[key]), key)
                self.assertTrue(torch.equal(model.features(self.x), clean_features), arm)
                self.assertTrue(torch.equal(model(self.x), clean_logits), arm)
                self.assertEqual(tuple(model.last_evidence_logits.shape), (4, 8))

    def test_parameter_matching_and_b3_b4_initial_state(self):
        models = {arm:build_model(arm, 'ManyRx', 12) for arm in ['B0','B1','B2','B3','B4']}
        counts = {arm:sum(p.numel() for p in model.parameters()) for arm,model in models.items()}
        self.assertEqual(counts['B0'], 82426)
        self.assertEqual(counts['B1']-counts['B0'], 8)
        for arm in ['B2', 'B3', 'B4']:
            self.assertEqual(counts[arm]-counts['B0'], 272)
        self.assertEqual(set(models['B3'].state_dict()), set(models['B4'].state_dict()))
        for key,value in models['B3'].state_dict().items():
            self.assertTrue(torch.equal(value, models['B4'].state_dict()[key]))
        for arm in models:
            self.assertEqual(list(inspect.signature(models[arm].forward).parameters), ['x'])

    def test_rng_fork_and_checkpoint_round_trip(self):
        build_model('B0', 'ManyRx', 5)
        expected_rng = torch.get_rng_state().clone()
        model = build_model('B4', 'ManyRx', 5).eval()
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        replacement = build_model('B4', 'ManyRx', 6).eval()
        replacement.load_state_dict(model.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.equal(model(self.x), replacement(self.x)))

    def test_bound_and_all_channels_rebuilt_after_nonidentity_filter(self):
        model = build_model('B1', 'ManyRx', 9).eval()
        with torch.no_grad():
            model.spectral_modulator.theta.copy_(torch.tensor([-100., -3., -1., 0., 1., 3., 20., 100.]))
            s = spectrum_from_iq(self.x)
            changed, _ = model.spectral_modulator(s)
            gain = model.spectral_modulator.last_gain
            self.assertGreaterEqual(float(gain.min()), 1-ALPHA-1e-7)
            self.assertLessEqual(float(gain.max()), 1+ALPHA+1e-7)
            expected = features_from_spectrum(changed, reference_spectrum=s)
            self.assertTrue(torch.equal(expected, model.features(self.x)))
            clean = build_model('B0', 'ManyRx', 9).features(self.x)
            self.assertGreater(float((expected[:, :3]-clean[:, :3]).abs().max()), .01)
            # Positive real filtering retains phases away from the validity
            # threshold; phase channels are rebuilt, not independently scaled.
            jointly_valid = ((expected[:, 3:].square().sum(1) > 0)
                             & (clean[:, 3:].square().sum(1) > 0))
            valid = jointly_valid[:, None].expand(-1, 2, -1)
            torch.testing.assert_close(expected[:, 3:][valid], clean[:, 3:][valid], atol=1e-6, rtol=1e-6)

    def test_phase_validity_is_recomputed_from_modified_complex_spectrum(self):
        spectrum = torch.ones(1, 256, dtype=torch.complex64)
        spectrum[:, 99] = .051+.0j
        clean = features_from_spectrum(spectrum)
        gain = torch.ones(1, 256)
        gain[:, 99] = .8
        changed = features_from_spectrum(spectrum*gain, reference_spectrum=spectrum)
        self.assertGreater(float(clean[:, 3:, 99:101].abs().sum()), 0)
        self.assertEqual(float(changed[:, 3:, 99:101].abs().sum()), 0)

    def test_parseval_rms_matches_physical_iq_filtering(self):
        model = build_model('B1', 'ManyRx', 3).eval()
        base = build_model('B0', 'ManyRx', 3).eval()
        with torch.no_grad():
            model.spectral_modulator.theta.copy_(torch.arange(8).float()/5-1)
            s = spectrum_from_iq(self.x)
            _, _ = model.spectral_modulator(s)
            gain = model.spectral_modulator.last_gain
            raw_s = torch.fft.fftshift(torch.fft.fft(torch.complex(self.x[:,0], self.x[:,1])), dim=-1)
            filtered = torch.fft.ifft(torch.fft.ifftshift(raw_s*gain, dim=-1))
            iq = torch.stack((filtered.real, filtered.imag), 1)
            torch.testing.assert_close(model.features(self.x), base.features(iq), atol=3e-6, rtol=3e-6)

    def test_evidence_and_ce_share_a_differentiable_gate(self):
        model = build_model('B4', 'ManyRx', 11)
        labels = torch.tensor([0,1,2,3])
        logits = model(self.x)
        target = torch.nn.functional.one_hot(torch.tensor([0,2,4,6]), num_classes=8).float()
        ce = torch.nn.functional.cross_entropy(logits, labels)
        evidence = -(target*model.last_evidence_logits.log_softmax(-1)).sum(-1).mean()
        (ce+.1*evidence).backward()
        grad = model.spectral_modulator.gate[-1].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0)
        self.assertIsNotNone(model.classifier.weight.grad)
        with torch.no_grad():
            probability = model.last_evidence_logits.softmax(-1)
            torch.testing.assert_close(probability.sum(-1), torch.ones(4))


if __name__ == '__main__': unittest.main()
