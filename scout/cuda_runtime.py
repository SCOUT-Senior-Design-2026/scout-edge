"""Small CUDA Runtime adapter for the TensorRT detector on JetPack.

Host arrays are ordinary NumPy allocations. CUDA may stage pageable transfers;
all transfers are synchronized before the output is read.
"""
import ctypes
import ctypes.util

import numpy as np

_runtime = None


def _library():
    global _runtime
    if _runtime is None:
        candidates = [ctypes.util.find_library('cudart'), '/usr/local/cuda/lib64/libcudart.so']
        for candidate in candidates:
            if not candidate:
                continue
            try:
                _runtime = ctypes.CDLL(candidate)
                break
            except OSError:
                pass
        if _runtime is None:
            raise RuntimeError('CUDA runtime library not found')
        definitions = {
            'cudaGetErrorString': ([ctypes.c_int], ctypes.c_char_p),
            'cudaSetDevice': ([ctypes.c_int], ctypes.c_int),
            'cudaMalloc': ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int),
            'cudaFree': ([ctypes.c_void_p], ctypes.c_int),
            'cudaStreamCreate': ([ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            'cudaStreamDestroy': ([ctypes.c_void_p], ctypes.c_int),
            'cudaStreamSynchronize': ([ctypes.c_void_p], ctypes.c_int),
            'cudaMemcpyAsync': ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                 ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
            'cudaEventCreate': ([ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            'cudaEventRecord': ([ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
            'cudaEventElapsedTime': ([ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                      ctypes.c_void_p], ctypes.c_int),
            'cudaEventDestroy': ([ctypes.c_void_p], ctypes.c_int),
        }
        for name, (argtypes, restype) in definitions.items():
            function = getattr(_runtime, name)
            function.argtypes, function.restype = argtypes, restype
    return _runtime


def _check(code):
    if code:
        message = _library().cudaGetErrorString(code)
        raise RuntimeError(f'CUDA error {code}: {message.decode() if message else "unknown"}')


def init():
    _check(_library().cudaSetDevice(0))


class _Context:
    def __init__(self, device):
        self.device = device

    def push(self):
        _check(_library().cudaSetDevice(self.device))

    def pop(self):
        pass

    def detach(self):
        pass


class Device:
    def __init__(self, device):
        self.device = device

    def retain_primary_context(self):
        return _Context(self.device)


class Allocation:
    def __init__(self, nbytes):
        self.ptr = ctypes.c_void_p()
        _check(_library().cudaMalloc(ctypes.byref(self.ptr), nbytes))

    def __int__(self):
        return self.ptr.value or 0

    def free(self):
        if self.ptr.value:
            _check(_library().cudaFree(self.ptr))
            self.ptr = ctypes.c_void_p()


class Stream:
    def __init__(self):
        self.ptr = ctypes.c_void_p()
        _check(_library().cudaStreamCreate(ctypes.byref(self.ptr)))

    @property
    def handle(self):
        return self.ptr.value or 0

    def synchronize(self):
        _check(_library().cudaStreamSynchronize(self.ptr))

    def close(self):
        if self.ptr.value:
            _check(_library().cudaStreamDestroy(self.ptr))
            self.ptr = ctypes.c_void_p()


class Event:
    def __init__(self):
        self.ptr = ctypes.c_void_p()
        _check(_library().cudaEventCreate(ctypes.byref(self.ptr)))

    def record(self, stream):
        _check(_library().cudaEventRecord(self.ptr, stream.ptr))

    def time_till(self, other):
        elapsed = ctypes.c_float()
        _check(_library().cudaEventElapsedTime(ctypes.byref(elapsed), self.ptr, other.ptr))
        return elapsed.value

    def close(self):
        if self.ptr.value:
            _check(_library().cudaEventDestroy(self.ptr))
            self.ptr = ctypes.c_void_p()


def pagelocked_empty(size, dtype):
    # PyCUDA uses pinned memory here; this fallback uses pageable memory.
    return np.empty(size, dtype=dtype)


def mem_alloc(nbytes):
    return Allocation(nbytes)


def memcpy_htod_async(device, host, stream):
    _check(_library().cudaMemcpyAsync(device.ptr, ctypes.c_void_p(host.ctypes.data),
                                     host.nbytes, 1, stream.ptr))


def memcpy_dtoh_async(host, device, stream):
    _check(_library().cudaMemcpyAsync(ctypes.c_void_p(host.ctypes.data), device.ptr,
                                     host.nbytes, 2, stream.ptr))
