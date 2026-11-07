"""Index compact NumPy pickles without loading IQ payloads from forbidden cells.

The Python pickle VM parses metadata; large bytes opcodes become seekable spans.
Only explicitly authorized cell arrays are materialized later. Constructors are
restricted to NumPy array/dtype building blocks; this is not a generic unpickler.
"""
from dataclasses import dataclass
from pathlib import Path
import pickle
import struct
import numpy as np


@dataclass
class Span:
    path: str
    offset: int
    size: int

    def read(self):
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            b = f.read(self.size)
        if len(b) != self.size:
            raise EOFError("truncated array payload")
        return b


class Array:
    def __init__(self):
        self.state = None

    def __setstate__(self, state):
        self.state = state

    def get(self):
        _, shape, dtype, fortran, raw = self.state
        if dtype.hasobject:
            return np.array(raw, dtype=object).reshape(shape)
        data = raw.read() if isinstance(raw, Span) else raw
        return np.frombuffer(data, dtype=dtype).reshape(shape, order="F" if fortran else "C")

    def __getitem__(self, key):
        if self.state[2].hasobject:
            return self.get()[key]
        raise ValueError("IQ payloads must be explicitly materialized after role validation")


def reconstruct(*args):
    return Array()


class IndexUnpickler(pickle._Unpickler):
    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, f):
        super().__init__(f)
        self.path = str(Path(f.name).resolve())
        self.file = f

    def find_class(self, module, name):
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "_reconstruct":
            return reconstruct
        if module == "numpy" and name == "ndarray":
            return Array
        if module == "numpy" and name == "dtype":
            return np.dtype
        raise pickle.UnpicklingError(f"Unapproved pickle constructor: {module}.{name}")

    def blob(self, width):
        size = int.from_bytes(self.read(width), "little", signed=False)
        # Large NumPy blobs are emitted outside pickle frames. Small byte
        # objects remain real bytes for ndarray reconstruction metadata.
        if size < 4096:
            self.append(self.read(size))
            return
        if self._unframer.current_frame is not None:
            current = self._unframer.current_frame
            remaining = len(current.getbuffer()) - current.tell()
            if remaining:
                raise pickle.UnpicklingError("Unexpected in-frame large array; refuse to load unknown cells")
        offset = self.file.tell()
        self.file.seek(size, 1)
        self.append(Span(self.path, offset, size))

    def binbytes(self):
        self.blob(4)

    def binbytes8(self):
        self.blob(8)

    dispatch[pickle.BINBYTES[0]] = binbytes
    dispatch[pickle.BINBYTES8[0]] = binbytes8


def index(path):
    with open(path, "rb") as f:
        return IndexUnpickler(f).load()
