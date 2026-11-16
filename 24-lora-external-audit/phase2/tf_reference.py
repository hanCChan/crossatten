"""Run with isolated TensorFlow 2.15 environment. No fitting or device data loading."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys,json,hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf

HERE=Path(__file__).resolve().parent
STAGE=HERE.parent

def main():
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(2401)
    repo='gxhen/receiverAgnosticRFFI';path='deep_learning_models.py'
    source=(STAGE/'_upstream'/repo/path).read_bytes()
    manifest=json.loads((STAGE/'UPSTREAM.json').read_text(encoding='utf-8'))
    expected=next(f['sha256'] for r in manifest['repos'] if r['repo']==repo for f in r['files'] if f['path']==path)
    assert hashlib.sha256(source).hexdigest()==expected
    ns={};exec(compile(source,'official_tf_deep_learning_models.py','exec'),ns)
    model=ns['adv_net'](None,10,5)
    inp=np.load(HERE/'local/parity_input.npy')
    values=dict(input=inp)
    metadata=dict(tensorflow=tf.__version__,layers=[],weights=[],losses={},source_sha256=expected)
    probe=tf.keras.Model(model.inputs,[l.output for l in model.layers])
    outputs=probe(inp,training=False)
    for layer,out in zip(model.layers,outputs):
        if isinstance(out,(list,tuple)):continue
        values['layer.'+layer.name]=out.numpy()
        metadata['layers'].append(dict(name=layer.name,cls=type(layer).__name__,shape=list(out.shape)))
    for v in model.trainable_weights:
        values['weight.'+v.name]=v.numpy();metadata['weights'].append(dict(name=v.name,shape=list(v.shape)))
    labels_tx=tf.constant([0,1],dtype=tf.int64);labels_rx=tf.constant([0,1],dtype=tf.int64)
    for mode in ('tx','rx','joint'):
        x=tf.Variable(inp)
        with tf.GradientTape() as tape:
            tx,rx=model(x,training=False)
            lt=tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels_tx,tx))
            lr=tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels_rx,rx))
            loss=lt if mode=='tx' else lr if mode=='rx' else lt+lr
        grads=tape.gradient(loss,[x]+model.trainable_weights)
        metadata['losses'][mode]=float(loss.numpy())
        values['grad.'+mode+'.input']=grads[0].numpy()
        for var,g in zip(model.trainable_weights,grads[1:]):
            if g is not None: values['grad.'+mode+'.'+var.name]=g.numpy()
    np.savez(HERE/'local/tf_reference.npz',**values)
    (HERE/'local/tf_reference.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8',newline='\n')
    print('Original TF weights, every layer, and three gradient objectives exported; no optimizer steps.')

if __name__=='__main__':main()
