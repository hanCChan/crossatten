"""Post-outcome regression evidence; does not modify the frozen candidate code."""
import unittest
import warnings
import numpy as np
from spectral import estimate


class Diagnostics(unittest.TestCase):
    def test_singleton_smoke_is_rejected_before_training(self):
        rng=np.random.default_rng(11)
        y=np.repeat(np.arange(3),6)
        rx=np.tile(np.repeat(np.arange(3),2),3)
        day=np.tile([1,2],9)
        values=rng.normal(size=(len(y),5,256))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            with self.assertRaisesRegex(ValueError,"nonfinite prior"):
                estimate(values,y,rx,day,min_cell=1)

    def test_raw_scatter_decreases_under_invertible_scaling(self):
        centers=np.array([[0.,1.],[1.,0.],[2.,3.]])
        scatter=lambda x: np.var(x,axis=0).mean()
        gain=.5
        self.assertAlmostEqual(scatter(gain*centers)/scatter(centers),gain**2)
        np.testing.assert_array_equal((gain*centers)/gain,centers)


if __name__=="__main__":
    unittest.main()
