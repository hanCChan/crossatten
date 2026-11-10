"""Synthetic-only scientific checks for source counterfactual evidence."""
from pathlib import Path
import json
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s19_core  # Establish the frozen model import paths, without data reads.
from s19_evidence import aggregate_contributions, counterfactuals, collect_evidence, margins


class DummyTeacher(torch.nn.Module):
    def forward(self, x):
        from xa.frontends import rms_normalize
        x = rms_normalize(x)
        z = torch.complex(x[:, 0], x[:, 1])
        s = torch.fft.fftshift(torch.fft.fft(z), dim=-1)
        power = s.abs().square()
        a = power[:, :128].mean(-1)/256
        b = power[:, 128:].mean(-1)/256
        return torch.stack([a, b], dim=1)


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cell_then_day_then_rx_quantile(self):
        rows, ys, rs, ds = [], [], [], []
        for t, r, d, value, repeat in [(0,0,1,[1.,2.],9), (0,0,2,[3.,4.],1),
                                      (0,1,1,[5.,6.],2), (0,2,1,[9.,10.],1), (0,2,2,[11.,12.],3),
                                      (1,0,1,[0.,0.],1), (1,1,2,[0.,0.],1)]:
            for _ in range(repeat):
                rows.append(value); ys.append(t); rs.append(r); ds.append(d)
        a = aggregate_contributions(rows, ys, rs, ds)
        np.testing.assert_allclose(a['q'][0], [3.5,4.5], atol=1e-12)
        np.testing.assert_allclose(a['bar_q'][1], [.5,.5])
        self.assertEqual(a['rx_support_per_tx'], [3,2])
        self.assertIsNone(a['cell_means'][0][1][1])
        json.dumps(a, allow_nan=False)

    def test_packet_replication_does_not_reweight_cells(self):
        grid = np.array([(t,r,d) for t in range(2) for r in range(3) for d in [1,2]])
        drops = np.arange(len(grid)*8, dtype=float).reshape(len(grid),8)
        a = aggregate_contributions(drops, *grid.T)
        repeats = np.ones(len(grid), int); repeats[0] = 20
        ids = np.repeat(np.arange(len(grid)), repeats)
        b = aggregate_contributions(drops[ids], *grid[ids].T)
        np.testing.assert_array_equal(a['q'], b['q'])
        np.testing.assert_array_equal(a['bar_q'], b['bar_q'])

    def test_invalid_contributions_and_support(self):
        with self.assertRaises(ValueError): aggregate_contributions([[-1.],[2.]], [0,0], [0,1], [1,1])
        with self.assertRaises(ValueError): aggregate_contributions([[1.],[2.]], [0,0], [0,0], [1,2])
        with self.assertRaises(ValueError): aggregate_contributions([[np.nan],[2.]], [0,0], [0,1], [1,1])

    def test_energy_matched_controls_and_spectrum_mask(self):
        torch.manual_seed(19)
        x = torch.randn(4,2,256, dtype=torch.float64)
        cf = counterfactuals(x)
        self.assertEqual(cf['masked'].shape, (4,8,2,256))
        clean = torch.complex(cf['clean'][:,0],cf['clean'][:,1])
        for key in ['masked','amplitude','phase']:
            time = torch.complex(cf[key][:,:,0],cf[key][:,:,1])
            evm = ((time-clean[:,None]).abs().square().sum(-1)/clean.abs().square().sum(-1)[:,None]).sqrt()
            torch.testing.assert_close(evm, cf['evm'], rtol=1e-10, atol=1e-12)
        spectrum = torch.fft.fftshift(torch.fft.fft(clean),dim=-1)
        actual = torch.fft.fftshift(torch.fft.fft(torch.complex(cf['masked'][:,:,0],cf['masked'][:,:,1])),dim=-1)
        torch.testing.assert_close(actual, spectrum[:,None]*(1-.5*cf['masks'][None]), rtol=1e-10, atol=1e-11)
        self.assertTrue(torch.all((cf['retained_energy'] >= .25) & (cf['retained_energy'] <= 1)))
        self.assertTrue(torch.all((cf['residual_impulse_tail_fraction'] >= 0) & (cf['residual_impulse_tail_fraction'] <= 1)))

    def test_margin_definition_and_role_guard(self):
        lg = torch.tensor([[3.,1.,2.],[1.,2.,4.]])
        torch.testing.assert_close(margins(lg, [0,1]), torch.tensor([1.,-2.]))
        with self.assertRaises(ValueError): collect_evidence(DummyTeacher(), None, None, None, None, device='cpu', role='val')
        with self.assertRaises(ValueError): collect_evidence(DummyTeacher(), np.ones((2,2,256)), [0,0], [0,1], [3,3], device='cpu')
        with self.assertRaises(ValueError): collect_evidence(DummyTeacher(), np.ones((34,2,256)), [0]*34, [0]*17+[1]*17, [1]*34, device='cpu')

    def test_collect_logits_reconstruct_evidence_and_model_mode(self):
        rng = np.random.default_rng(20)
        grid = np.array([(t,r,d) for t in range(2) for r in range(3) for d in [1,2]])
        x = rng.normal(size=(len(grid),2,256)).astype(np.float32)
        model = DummyTeacher().train()
        arrays, summary = collect_evidence(model,x,*grid.T,device='cpu',batch=5)
        self.assertTrue(model.training)
        self.assertEqual(arrays['masked_logits'].shape, (len(grid),8,2))
        clean = margins(torch.from_numpy(arrays['clean_logits']),grid[:,0]).numpy()
        masked = np.stack([margins(torch.from_numpy(arrays['masked_logits'][:,j]),grid[:,0]).numpy() for j in range(8)],axis=1)
        np.testing.assert_allclose(arrays['signed_delta'],clean[:,None]-masked,atol=1e-7)
        np.testing.assert_allclose(arrays['drop'],np.maximum(clean[:,None]-masked,0),atol=1e-7)
        # Packet RMS makes matched positive global amplitude almost invariant;
        # epsilon prevents a claim of exact scale invariance.
        self.assertLess(abs(summary['controls']['amplitude_signed_mean']),1e-5)
        self.assertLess(summary['artifacts']['amplitude_evm_max_error'],1e-6)
        self.assertLess(summary['artifacts']['phase_evm_max_error'],1e-6)
        np.testing.assert_allclose(np.asarray(summary['bar_q']).sum(1),1)
        json.dumps(summary,allow_nan=False)


if __name__ == '__main__': unittest.main()
