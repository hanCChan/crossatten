"""Same D0 phase descriptor, holding the clean magnitude support fixed in D1."""
import numpy as np

def phase_descriptor(x,support_from=None):
    def spectrum(a):
        z=a[:,0].astype(np.float64)+1j*a[:,1].astype(np.float64)
        z=z/np.sqrt(np.mean(abs(z)**2,axis=1,keepdims=True)).clip(1e-12)
        f=np.fft.fftshift(np.fft.fft(z,axis=1),axes=1);return f,abs(f)
    f,mag=spectrum(x);ref=mag if support_from is None else spectrum(support_from)[1]
    unit=f/mag.clip(1e-9);curvature=unit[:,2:]*unit[:,:-2]*unit[:,1:-1].conj()**2
    threshold=ref.mean(1,keepdims=True)*.1
    valid=(ref[:,2:]>threshold)&(ref[:,:-2]>threshold)&(ref[:,1:-1]>threshold)
    curved=np.pad(curvature*valid,((0,0),(1,1))).reshape(-1,16,16)
    count=np.pad(valid,((0,0),(1,1))).reshape(-1,16,16).sum(2).clip(1)
    result=curved.sum(2)/count
    return np.concatenate([result.real,result.imag],1).astype(np.float32)
