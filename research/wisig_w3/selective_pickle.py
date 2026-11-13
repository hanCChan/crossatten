"""Index numeric NumPy arrays without reading their payloads or executing globals.

Only protocol3/4/5 ndarray representations are admitted. FRAME is a no-op here:
prefetching a whole frame would otherwise read an unapproved waveform buffer.
"""
from dataclasses import dataclass
import math
import pickle
import struct
import numpy as np


@dataclass
class Span:
    offset:int
    length:int


class ArrayRef:
    def __setstate__(self,state):
        if not isinstance(state,tuple) or len(state)!=5:raise ValueError('Unregistered ndarray state')
        version,shape,dtype,fortran,payload=state
        self.set(shape,dtype,'F' if fortran else 'C',payload)

    def set(self,shape,dtype,order,payload):
        dtype=np.dtype(dtype)
        if dtype.hasobject or dtype.kind not in 'fciuUbS' or dtype.itemsize>4096:raise ValueError('Non-numeric/object waveform array not admitted')
        if not isinstance(shape,tuple) or len(shape)>5 or any(not isinstance(i,int) or i<0 for i in shape):raise ValueError('Invalid array shape')
        expected=math.prod(shape)*dtype.itemsize
        if not isinstance(payload,Span) or payload.length!=expected:raise ValueError('Array payload size/shape differs')
        self.shape,self.dtype,self.order,self.payload=shape,dtype,order,payload


def reconstruct(*args):return ArrayRef()


def frombuffer(payload,dtype,shape,order):
    a=ArrayRef();a.set(shape,dtype,order,payload);return a


class IndexUnpickler(pickle._Unpickler):
    dispatch=pickle._Unpickler.dispatch.copy()
    def __init__(self,stream):
        self.stream=stream;self.skipped=[]
        super().__init__(stream,encoding='ASCII',errors='strict')

    def find_class(self,module,name):
        allowed={('numpy','ndarray'):ArrayRef,('numpy','dtype'):np.dtype,
            ('numpy.core.multiarray','_reconstruct'):reconstruct,('numpy._core.multiarray','_reconstruct'):reconstruct,
            ('numpy.core.numeric','_frombuffer'):frombuffer,('numpy._core.numeric','_frombuffer'):frombuffer}
        if (module,name) not in allowed:raise pickle.UnpicklingError('Unregistered global: '+module+'.'+name)
        return allowed[module,name]

    def frame(self):
        n=struct.unpack('<Q',self.read(8))[0]
        if n>10_000_000_000:raise ValueError('Invalid frame size')
        # Do not call _Unframer.load_frame: leave every subsequent read on file.

    def span(self,n):
        if n>9_000_000_000:raise ValueError('Unexpected byte-buffer size')
        p=Span(self.stream.tell(),n);self.stream.seek(n,1)
        self.skipped.append(p);self.append(p)

    def binbytes(self):self.span(struct.unpack('<I',self.read(4))[0])
    def binbytes8(self):self.span(struct.unpack('<Q',self.read(8))[0])
    def shortbytes(self):self.span(self.read(1)[0])


IndexUnpickler.dispatch[pickle.FRAME[0]]=IndexUnpickler.frame
IndexUnpickler.dispatch[pickle.BINBYTES[0]]=IndexUnpickler.binbytes
IndexUnpickler.dispatch[pickle.BINBYTES8[0]]=IndexUnpickler.binbytes8
IndexUnpickler.dispatch[pickle.BYTEARRAY8[0]]=IndexUnpickler.binbytes8
IndexUnpickler.dispatch[pickle.SHORT_BINBYTES[0]]=IndexUnpickler.shortbytes


def index(stream):
    u=IndexUnpickler(stream);data=u.load();return data,u.skipped


def read_rows(path,record,allowed):
    if record['key'] not in allowed:raise PermissionError('Cell is not in the committed source manifest')
    if record['order']!='C' or record['shape'][1:]!=[256,2]:raise ValueError('Unsupported waveform layout')
    dt=np.dtype(record['dtype']);start,stop=record['rows']
    if not 0<=start<stop<=record['shape'][0]:raise ValueError('Row admission differs')
    size=(stop-start)*512*dt.itemsize
    with open(path,'rb') as f:
        f.seek(record['offset']+start*512*dt.itemsize);payload=f.read(size)
    if len(payload)!=size:raise ValueError('Truncated source cell')
    return np.frombuffer(payload,dtype=dt).reshape(stop-start,256,2).copy()
