import unittest
import numpy as np
from transport import pca_basis,fit,predict,transport_error,separation_ratio,prototype_metrics


class TransportTests(unittest.TestCase):
    def test_shared_shift_transfers_to_unseen_TX(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(80,16));b=rng.normal(size=16)
        op=fit('shift',x[:40],x[:40]+b,pca_basis(x[:40]))
        np.testing.assert_allclose(predict(op,x[40:]),x[40:]+b,atol=1e-14)

    def test_low_rank_predictable_change_and_shuffled_negative(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(160,16));p=pca_basis(x[:80]);w=rng.normal(size=(8,16))*.3
        y=x+x@p@w+2
        good=predict(fit('rank8',x[:80],y[:80],p),x[80:])
        null=predict(fit('shuffled_rank8',x[:80],y[:80],p),x[80:])
        base=x[80:]+(y[:80]-x[:80]).mean(0)
        self.assertLess(transport_error(good,y[80:]).mean(),.2*transport_error(base,y[80:]).mean())
        self.assertLess(transport_error(good,y[80:]).mean(),transport_error(null,y[80:]).mean())

    def test_TX_specific_random_change_is_not_automatically_fixed(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(160,16));y=x+rng.normal(size=x.shape)*2
        p=pca_basis(x[:80]);good=predict(fit('rank8',x[:80],y[:80],p),x[80:])
        base=x[80:]+(y[:80]-x[:80]).mean(0)
        self.assertGreater(transport_error(good,y[80:]).mean(),.8*transport_error(base,y[80:]).mean())

    def test_no_held_TX_in_fit_function(self):
        rng=np.random.default_rng(6);x=rng.normal(size=(80,16));y=x*.8
        p=pca_basis(x[:40]);a=fit('rank8',x[:40],y[:40],p)
        x[40:]=1e8;b=fit('rank8',x[:40],y[:40],p)
        np.testing.assert_array_equal(a['weight'],b['weight'])

    def test_geometry_and_diagnostic_labels(self):
        x=np.eye(12);self.assertAlmostEqual(separation_ratio(x,x),1)
        self.assertLess(separation_ratio(x,x*.1),.011)
        acc,margin=prototype_metrics(np.repeat(x[:,None,:],3,axis=1),x)
        np.testing.assert_equal(acc,1);self.assertTrue((margin>0).all())


if __name__=='__main__':unittest.main()
