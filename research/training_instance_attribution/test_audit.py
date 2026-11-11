import unittest
import numpy as np
from audit import array_sha, pool_keys, overlap, panels, trace, actual_seed


class AuditTests(unittest.TestCase):
    def test_raw_index_dtype_is_preserved(self):
        self.assertNotEqual(array_sha(np.arange(5,dtype=np.int32)),array_sha(np.arange(5,dtype=np.int64)))

    def test_pool_positions_are_not_packet_identity(self):
        def make(rx):
            meta=dict(fit_y=np.array([0,1]),fit_rx=np.array([0,0]),fit_day=np.array([1,2]))
            keys,labels=pool_keys(meta,[rx],{(rx,1):rx+"-sha1",(rx,2):rx+"-sha2"})
            return dict(fold=0,keys=keys,labels=labels,train_ids=np.array([0]),val_ids=np.array([1]),fit_rx=[rx])
        a,b=make("A"),make("B");b["fold"]=1
        r=overlap(a,b)
        self.assertTrue(r["integer_train_index_vectors_equal"])
        self.assertEqual(r["common_pool_rows"],0)
        self.assertEqual(r["global_train_row_jaccard"],0)

    def test_same_input_split_contingency(self):
        keys=[("A",1,"sha",i) for i in range(4)]
        a=dict(fold=0,keys=keys,labels=dict.fromkeys(keys,0),train_ids=np.array([0,1]),val_ids=np.array([2,3]),fit_rx=["A"])
        b=dict(a,fold=1,train_ids=np.array([0,2]),val_ids=np.array([1,3]))
        r=overlap(a,b)
        self.assertEqual([r[k] for k in ("both_train","a_train_b_val","a_val_b_train","both_val")],[1,1,1,1])
        self.assertEqual(r["membership_disagreement_fraction"],.5)

    def test_lr_recording_times(self):
        rows=[dict(epoch=i+1,lr=lr,source_val_ce=ce) for i,(lr,ce) in enumerate([(1e-3,.9),(1e-4,.8),(1e-4,.85)])]
        cfg=dict(initial_state_sha256="0"*64,data_seed=actual_seed(0))
        metrics=dict(epochs=3,best_epoch=2,model_sha256="1"*64)
        r=trace("B0",0,cfg,metrics,rows,257)
        self.assertEqual(r["lr_used_by_epoch"],[1e-3,1e-3,1e-4])
        self.assertEqual(r["selected_updates"],6)

    def test_balanced_panel_design(self):
        names=[f"r{i:02d}" for i in range(18)];keys=[];y=[];day=[];tr=[];va=[]
        for rx in names:
            for t in range(10):
                for d in (1,2):
                    for j in range(3):
                        i=len(y);keys.append((rx,d,rx+str(d),t*3+j));y.append(t);day.append(d)
                        (tr if j<2 else va).append(i)
        pool=dict(fit_rx=names,keys=keys,fit_y=np.array(y),fit_day=np.array(day),train_ids=np.array(tr),val_ids=np.array(va),held_rx=["held"],forbidden_target_rx=["target"])
        result=panels(pool)
        self.assertEqual([r["train_packets"] for r in result["arms"]],[480,480])
        self.assertEqual([r["validation_packets"] for r in result["arms"]],[120,120])
        self.assertEqual(result["arms"][0]["validation_indices_sha256"],result["arms"][1]["validation_indices_sha256"])
        self.assertTrue(result["train_class_and_day_sequences_equal"])


if __name__=="__main__": unittest.main()
