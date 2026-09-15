import mlx.core as mx

# Rendering backend. The C++ core always builds the scene (BVHs, CDFs) on the
# CPU; with use_gpu the rasterisation runs in Metal kernels
# (pydiffvg/render_metal.py), otherwise in the C++ core's thread pool.
# On the GPU, forward and backward passes of colour output, prefiltered
# colour output and signed distance output (also at eval_positions), and
# RenderFunction.render_grad, run in Metal kernels. Colour output at
# eval_positions (unsupported by the CPU core as well) always uses the CPU
# core; see render_mlx._gpu_mode for the dispatch.
# The GPU is the default whenever Metal is available.
use_gpu = mx.metal.is_available()
device = mx.gpu if use_gpu else mx.cpu

def _is_gpu(d):
    if isinstance(d, mx.Device):
        return d.type == mx.DeviceType.gpu
    return d == mx.DeviceType.gpu

def set_use_gpu(v):
    global use_gpu, device
    if v:
        if not mx.metal.is_available():
            raise RuntimeError('pydiffvg.set_use_gpu(True): the Metal backend is not available on this machine (mx.metal.is_available() is False).')
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
