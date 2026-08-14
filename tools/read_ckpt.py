"""Read a torch .pth checkpoint into numpy arrays without importing torch.

A .pth is a zip holding one pickle (data.pkl) plus raw storage blobs under
data/<key>. The pickle references those blobs through persistent ids. This
unpickler resolves them into numpy views so the tensors can be inspected on a
machine with no torch installed.
"""
import io, pickle, zipfile
import numpy as np

_DTYPES = {
    "FloatStorage": np.dtype("<f4"), "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),  "BFloat16Storage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),  "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"), "CharStorage": np.dtype("<i1"),
    "ByteStorage": np.dtype("|u1"),  "BoolStorage": np.dtype("|b1"),
    "float32": np.dtype("<f4"), "float64": np.dtype("<f8"),
    "float16": np.dtype("<f2"), "int64": np.dtype("<i8"),
    "int32": np.dtype("<i4"), "int16": np.dtype("<i2"),
    "int8": np.dtype("<i1"), "uint8": np.dtype("|u1"), "bool": np.dtype("|b1"),
}


class _ODict(dict):
    """dict subclass, so the pickle BUILD opcode has a __dict__ to write into."""
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.update(state)


class _Obj:
    """Stands in for any torch class we do not need to reconstruct faithfully."""
    def __init__(self, *a, **k): pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


class _Stub:
    def __init__(self, name): self.__name__ = name
    def __call__(self, *a, **k): return _Obj()


def _rebuild_tensor(storage, offset, size, stride, *rest):
    buf, dtype = storage
    n = int(np.prod(size)) if len(size) else 1
    flat = np.frombuffer(buf, dtype=dtype, count=n, offset=offset * dtype.itemsize)
    if not size:
        return flat.reshape(())
    # strides in the pickle are element counts; convert to bytes for numpy
    try:
        return np.lib.stride_tricks.as_strided(
            flat, shape=tuple(size), strides=tuple(s * dtype.itemsize for s in stride)).copy()
    except Exception:
        return flat.reshape(tuple(size)).copy()


def load(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.startswith("torch") or mod.startswith("numpy.core.multiarray"):
                if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                    return _rebuild_tensor
                if name == "_rebuild_parameter":
                    return lambda data, rg, hooks: data
                if name == "OrderedDict":
                    return _ODict
                return _Stub(name)
            if mod == "collections" and name == "OrderedDict":
                return _ODict
            try:
                return super().find_class(mod, name)
            except Exception:
                return _Stub(name)

        def persistent_load(self, pid):
            # ('storage', storage_type, key, location, numel)
            _, stype, key, _loc, _numel = pid
            name = getattr(stype, "__name__", str(stype)).split(".")[-1]
            dtype = _DTYPES.get(name, np.dtype("<f4"))
            raw = z.read(f"{root}/data/{key}")
            return (raw, dtype)

    return U(io.BytesIO(z.read(f"{root}/data.pkl"))).load()


def walk(obj, prefix=""):
    """Yield (dotted_key, ndarray) for every array anywhere in the object."""
    if isinstance(obj, np.ndarray):
        yield prefix, obj
    elif isinstance(obj, _Obj):
        yield from walk(obj.__dict__, prefix)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk(v, f"{prefix}[{i}]")


if __name__ == "__main__":
    import sys
    obj = load(sys.argv[1])
    if isinstance(obj, dict):
        print("top-level keys:", list(obj.keys())[:20])
    n = 0
    for k, a in walk(obj):
        print(f"  {k:<60} {str(a.shape):<20} {a.dtype}")
        n += 1
    print(f"{n} arrays")
