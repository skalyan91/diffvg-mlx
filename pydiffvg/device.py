import mlx.core as mx

# The diffvg core runs on the CPU (with its own thread pool);
# the GPU code path is CUDA-only and is not built.
use_gpu = False
device = mx.cpu

def set_use_gpu(v):
    if v:
        raise NotImplementedError('diffvg-mlx only supports rendering on the CPU.')

def get_use_gpu():
    return use_gpu

def set_device(d):
    if d != mx.cpu:
        raise NotImplementedError('diffvg-mlx only supports rendering on the CPU.')

def get_device():
    return device
