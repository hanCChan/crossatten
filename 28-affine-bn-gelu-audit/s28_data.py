"""Only historical legal Source validation is readable during this audit."""
import hashlib
import numpy as np
from s28_core import HERE,CACHE,P,read,event,validate
from xa.data import source_split

def authorize(rx,fold,role):
    if fold not in (0,1) or role!='audit_source_val' or rx not in P['folds'][fold]['source']:
        raise PermissionError('This no-fit audit allows only historical Source validation')
    if rx in P['confirm_receivers']+P['blind_receivers_never_open']:raise PermissionError('Forbidden RX')

def c_role_permitted(rx,fold,role):
    plan=read(HERE/'PLAN_C.original.json');split=plan['splits'][fold]
    if role in ['fit','selection']:return rx in split['training_receivers']
    if role=='held_report':return False # Requires all twelve new C models sealed and a separate activation.
    return False

def source_validation(fold):
    registration=validate();parts=[]
    for ri,rx in enumerate(P['folds'][fold]['source']):
        authorize(rx,fold,'audit_source_val');folder=CACHE/rx;expected=registration['source_caches'][rx]
        x=np.load(folder/'x.npy',mmap_mode='r');y=np.load(folder/'y.npy');rows=np.load(folder/'rows.npy')
        for array,key in [(x,'selected_iq_float32_sha256'),(y,'selected_labels_sha256'),(rows,'selected_rows_sha256')]:
            assert hashlib.sha256(array.tobytes()).hexdigest()==expected[key],(rx,key)
        _,ids=source_split(y,rows)
        assert len(ids)==400 and np.array_equal(np.bincount(y[ids],minlength=10),np.full(10,40))
        parts.append(dict(x=np.array(x[ids]),y=y[ids],rows=rows[ids],rx=np.full(400,ri)))
        event(role='audit_source_val',fold=fold,rx=rx,cache_packets_hashed=2000,validation_packets_returned=400,
            selected_rows_sha256=hashlib.sha256(rows[ids].tobytes()).hexdigest(),optimizer_access=False)
    return {k:np.concatenate([p[k] for p in parts]) for k in ['x','y','rows','rx']}
