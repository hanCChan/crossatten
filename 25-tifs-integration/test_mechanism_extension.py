import unittest
from unittest.mock import patch
import numpy as np
from mechanism_extension import centered, rho, source_metadata
from s25_common import HERE, P, write


class ExtensionTests(unittest.TestCase):
    def test_metadata_forbidden_before_hdf5(self):
        with patch('h5py.File',side_effect=AssertionError('No HDF5 IO')):
            for fold in (0,1):
                for rx in P['confirm_receivers']+P['blind_receivers_never_open']+P['folds'][fold]['dev_val']:
                    with self.assertRaises(PermissionError):source_metadata(fold,rx,np.array([],dtype=int),np.array([],dtype=int))

    def test_group_confounding_and_undefined_within_cell(self):
        tx=np.zeros(4,dtype=int);rx=np.array([0,0,1,1])
        x=np.array([1.,1.,10.,10.]);y=4*x
        self.assertAlmostEqual(rho(x,y),1.)
        self.assertIsNone(rho(centered(x,tx,rx),centered(y,tx,rx)))

    def test_finite_only_and_negative_direction(self):
        self.assertAlmostEqual(rho([1,2,3,np.nan],[5,4,3,0]),-1.)
        self.assertIsNone(rho([1,1,1],[1,2,3]))


if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExtensionTests))
    write(HERE/'EXTENSION_TESTS.json',dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,new_waveform_reads=0,new_optimizer_steps=0))
    raise SystemExit(0 if r.wasSuccessful() else 1)
