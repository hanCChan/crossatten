import unittest,math
import numpy as np
import torch
from s17b_core import HERE
from s17b_augment import apply
from s17b_features import phase_descriptor
from s17_augment import bounded_kernel,fir,as_complex
from s17_features import features
from s17_models import build
from wifi import phase
from xa.explore_models import frequency_views
from xa.frontends import rms_normalize

class PhaseEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_exact_per_packet_evm_controls_and_zero_dose(self):
        x=torch.randn(12,2,256)
        def run(mode,tau=1):return apply(x,tau,torch.Generator().manual_seed(12),torch.Generator().manual_seed(13),mode)
        energies=[(as_complex(run(mode))-as_complex(x)).abs().square().mean(1) for mode in ['physical','noise','global']]
        for e in energies[1:]:torch.testing.assert_close(e,energies[0],rtol=1e-4,atol=2e-8)
        for mode in ['physical','noise','global']:torch.testing.assert_close(run(mode,0),x,rtol=1e-6,atol=1e-6)
    def test_fir_linear_boundary_and_invertible_response(self):
        x=torch.randn(4,2,256);torch.testing.assert_close(fir(x),x,rtol=1e-6,atol=1e-6)
        h=bounded_kernel(4,'cpu',phase=torch.randn(4,16)*5);H=torch.fft.fft(h,n=4096).abs()
        self.assertGreaterEqual(float(H.min()),.75-1e-6);self.assertLessEqual(float(H.max()),1.25+1e-6)
        edge=torch.zeros(1,2,256);edge[0,0,-1]=1
        self.assertLess(float(fir(edge,phase=torch.randn(1,16))[:,:,:230].abs().max()),1e-6)
    def test_correct_feature_invariance_scope(self):
        x=torch.randn(3,2,256);z=as_complex(x);r=z*np.exp(.7j);rot=torch.stack([r.real,r.imag],1)
        a=phase.phase_features_from_base(frequency_views(rms_normalize(x)),'first')
        b=phase.phase_features_from_base(frequency_views(rms_normalize(rot)),'first')
        torch.testing.assert_close(a[:,3:],b[:,3:],rtol=2e-5,atol=2e-5)
        self.assertGreater(float((a[:,:2]-b[:,:2]).abs().max()),.1)
        m=build('ManyRx',18101).eval()
        with torch.no_grad():torch.testing.assert_close(m(x),m(2*x),rtol=2e-5,atol=2e-5)
        self.assertEqual(sum(p.numel() for p in m.parameters()),82426)
    def test_fixed_support_matches_original_clean_descriptor(self):
        x=np.random.default_rng(4).normal(size=(5,2,256)).astype(np.float32)
        np.testing.assert_allclose(phase_descriptor(x,x),features(x)['phase'],rtol=0,atol=0)
        z=x[:,0]+1j*x[:,1];r=z*np.exp(1j);rot=np.stack([r.real,r.imag],1)
        np.testing.assert_allclose(phase_descriptor(rot,x),phase_descriptor(x,x),rtol=2e-5,atol=2e-5)
    def test_same_initial_state_across_arms(self):
        a=build('ManyRx',18101);b=build('ManyRx',18101)
        for name,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
    def test_gpu_transform_and_model_backward(self):
        self.assertTrue(torch.cuda.is_available());m=build('ManyRx',18101).cuda();x=torch.randn(4,2,256,device='cuda')
        for mode in ['physical','noise','global']:
            m.zero_grad(set_to_none=True);y=apply(x,1,torch.Generator(device='cuda').manual_seed(1),torch.Generator(device='cuda').manual_seed(2),mode)
            loss=torch.nn.functional.cross_entropy(m(y),torch.arange(4,device='cuda'));loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)));self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None))

if __name__=='__main__':unittest.main(verbosity=2)
