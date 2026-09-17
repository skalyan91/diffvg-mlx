import mlx.core as mx

from . import gpu_backend

# Rendering backend. The C++ core always builds the scene (BVHs, CDFs) on the
# CPU; with use_gpu the rasterisation runs in GPU kernels
# (pydiffvg/render_metal.py, dispatched through pydiffvg/gpu_backend.py to
# either Apple Metal or NVIDIA CUDA), otherwise in the C++ core's thread pool.
# On the GPU, forward and backward passes of colour output, prefiltered
# colour output and signed distance output (also at eval_positions), and
# RenderFunction.render_grad, run in GPU kernels. Colour output at
# eval_positions (unsupported by the CPU core as well) always uses the CPU
# core; see render_mlx._gpu_mode for the dispatch.
# The GPU is the default whenever a GPU backend is available.
use_gpu = gpu_backend.is_available()
device = mx.gpu if use_gpu else mx.cpu

def _is_gpu(d):
    if isinstance(d, mx.Device):
        return d.type == mx.DeviceType.gpu
    return d == mx.DeviceType.gpu

def set_use_gpu(v):
    global use_gpu, device
    if v:
        if not gpu_backend.is_available():
            raise RuntimeError('pydiffvg.set_use_gpu(True): no GPU backend is available on this machine (neither mx.metal.is_available() nor mx.cuda.is_available()).')
        use_gpu = True
        device = mx.gpu
    else:
        use_gpu = False
        device = mx.cpu

def get_use_gpu():
    return use_gpu

def set_device(d):
    set_use_gpu(_is_gpu(d))

def get_device():
    return device

def set_gpu_backend(backend):
    """
        Selects the GPU backend used for the diffvg kernels: 'auto' (detect;
        the default), 'metal' (Apple) or 'cuda' (NVIDIA). See
        pydiffvg/gpu_backend.py.
    """
    gpu_backend.set_gpu_backend(backend)

def get_gpu_backend():
    """ The active GPU backend: 'metal', 'cuda', or None when there is none. """
    return gpu_backend.get_gpu_backend()
