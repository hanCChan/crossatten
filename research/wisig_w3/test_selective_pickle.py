import io
import pickle
import tempfile
from pathlib import Path
import unittest
import numpy as np
from selective_pickle import index,ArrayRef,read_rows,Span


class Tracked(io.BytesIO):
    def __init__(self,data):super().__init__(data);self.reads=[]
    def read(self,n=-1):
        start=self.tell();v=super().read(n);self.reads.append((start,self.tell()));return v
    def readline(self,n=-1):
        start=self.tell();v=super().readline(n);self.reads.append((start,self.tell()));return v


class IndexTests(unittest.TestCase):
    def fixture(self,protocol):
        rng=np.random.default_rng(31001)
        arrays=[rng.normal(size=(5,256,2)),np.empty((0,256,2)),rng.normal(size=(7,256,2)).astype('float32')]
        source=dict(tx_list=['aux-a','aux-b'],rx_list=['source','forbidden'],data=arrays)
        stream=Tracked(pickle.dumps(source,protocol=protocol));obj,spans=index(stream)
        return source,obj,stream,spans

    def test_protocols_no_waveform_bytes_read(self):
        for protocol in (3,4,5):
            with self.subTest(protocol=protocol):
                source,obj,stream,spans=self.fixture(protocol)
                self.assertEqual(obj['tx_list'],source['tx_list'])
                for original,ref in zip(source['data'],obj['data']):
                    self.assertIsInstance(ref,ArrayRef);self.assertEqual(ref.shape,original.shape)
                    self.assertEqual(ref.payload.length,original.nbytes)
                    for begin,end in stream.reads:
                        self.assertTrue(end<=ref.payload.offset or begin>=ref.payload.offset+ref.payload.length or ref.payload.length==0)

    def test_selected_rows_exact_for_all_protocols(self):
        for protocol in (3,4,5):
            source,obj,stream,_=self.fixture(protocol);ref=obj['data'][0]
            rec=dict(key='allowed',order=ref.order,shape=list(ref.shape),dtype=ref.dtype.str,offset=ref.payload.offset,rows=[1,4])
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'fixture.pkl';p.write_bytes(stream.getvalue())
                actual=read_rows(p,rec,{'allowed'})
                np.testing.assert_array_equal(actual,source['data'][0][1:4])
                rec['key']='forbidden'
                with self.assertRaises(PermissionError):read_rows(p,rec,{'allowed'})

    def test_arbitrary_globals_refused(self):
        with self.assertRaises(pickle.UnpicklingError):index(io.BytesIO(pickle.dumps(eval)))

    def test_array_shape_size_mismatch_rejected(self):
        a=ArrayRef()
        with self.assertRaises(ValueError):a.set((32,256,2),np.dtype('float64'),'C',Span(0,1))
        with self.assertRaises(ValueError):a.set((1,),np.dtype(object),'C',Span(0,8))


if __name__=='__main__':unittest.main()
