"""
    GPU backend selection and kernel construction for diffvg.

    One place that knows which GPU backend MLX is running on -- Apple Metal
    ('metal') or NVIDIA CUDA ('cuda') -- loads the matching shared kernel
    sources, and builds mx.fast.metal_kernel / mx.fast.cuda_kernel objects with
    the arguments each backend needs.

    ---------------------------------------------------------------------
    Kernel bodies: one text, two backends
    ---------------------------------------------------------------------
    The *shared* device code (SceneView, geometry, colour, backward, distance
    gradients, prefiltering) exists twice, once per backend, because the two
    languages differ throughout:

        pydiffvg/metal/*.metal      pydiffvg/cuda/*.cu

    Both define the same names and signatures, in the same concatenation order
    (common, geometry, color, backward, distance_grad, prefilter); see
    pydiffvg/cuda/README-porting.md for the translation rules.

    The *kernel bodies*, by contrast, are written ONCE (as Python strings in
    render_metal.py and render_metal_stage3.py) in a small backend-neutral
    dialect: ordinary Metal/CUDA-common C, plus the five macros below, which
    this module defines per backend and appends to the header. Everything that
    actually differs between Metal and CUDA in a kernel body -- the thread
    index, the bounds check, atomics, the float3 swizzle and the name of ceil
    -- is confined to those macros, so there is no second copy of the
    rasteriser logic to keep in sync.

        DVG_THREAD_INDEX;       declares `idx` (Metal: thread_position_in_grid.x;
                                CUDA: cooperative_groups thread_rank PLUS the
                                mandatory `if (idx >= n) return;` guard).
                                Expanded in Python, not by the preprocessor --
                                see _THREAD_INDEX below for why.
        DVG_ADD(buf, i, v)      atomic add into an output buffer
        DVG_STORE(buf, i, v)    plain store into an output buffer
        DVG_XYZ(v)              float4 -> float3 swizzle (`v.xyz` / `v.xyz()`)
        DVG_CEIL(x)             ceil (Metal) / ceilf (CUDA)

    On Metal every macro expands to exactly the text the kernels used before
    this layer existed, so the Metal path is unchanged in behaviour.

    ---------------------------------------------------------------------
    Grid and block size
    ---------------------------------------------------------------------
    Metal dispatches threads exactly (dispatchThreads), so a Metal kernel never
    sees an out-of-range thread. MLX's CUDA path launches whole blocks and
    rounds the requested thread count up (measured: grid=10 with a 4-thread
    block launches 12 threads), so every CUDA kernel body MUST bounds-check.
    Callers pass the same `grid=(n, 1, 1)` thread count on both backends;
    DVG_THREAD_INDEX supplies the guard, reading n from an extra `dvg_params`
    int32 buffer that this module appends to the CUDA input list (invisible to
    the caller, and not bound at all on Metal, whose ~30-binding budget is
    tight).

    Block size: Metal uses 256 threads. CUDA defaults to 128 because the
    kernels hold large per-thread arrays (Fragment[256] in the colour scan,
    BlendState in the backward blend), which live in CUDA local memory and
    limit occupancy; see set_block_size().

    MEASURED, and important: CUDA reserves the local-memory pool for the
    device's full resident-thread capacity (~196k threads on an RTX 4090), NOT
    for the launch grid. So when local memory is the binding constraint, a
    smaller block does not help -- on a pod with ~490 MiB of free GPU memory a
    6 KB/thread kernel failed to launch at blocks of 128, 64 AND 32, while a
    1.5 KB/thread kernel launched fine. The lever that works is
    set_fragment_capacity(), which lowers the per-thread array sizes.
"""
import os
import mlx.core as mx

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

BACKENDS = ('metal', 'cuda')

# Per-backend source location and file extension. The base names used by
# render_metal.py / render_metal_stage3.py are extension-free.
_SOURCE = {
    'metal': ('metal', '.metal'),
    'cuda': ('cuda', '.cu'),
}

# Environment override, mostly for testing: PYDIFFVG_GPU_BACKEND=metal|cuda|auto
_ENV_VAR = 'PYDIFFVG_GPU_BACKEND'

_requested = os.environ.get(_ENV_VAR, 'auto') or 'auto'
_resolved = None            # cached detection result ('metal' / 'cuda' / None)
_resolved_valid = False

_headers = {}               # (backend, files, constants) -> header source
_kernels = {}               # (backend, key) -> built kernel
_params = {}                # (backend, n) -> mx.array dvg_params buffer


# ---------------------------------------------------------------------------
# Detection

def _detect():
    """
        The GPU backend MLX was built with, or None.

        mlx.core exposes both mx.metal and mx.cuda submodules on every
        platform; only is_available() distinguishes them (measured: on macOS
        metal True / cuda False, on the CUDA wheel metal False / cuda True).
    """
    for name in BACKENDS:
        mod = getattr(mx, name, None)
        if mod is None:
            continue
        try:
            if mod.is_available():
                return name
        except Exception:
            pass
    # Older wheels may lack is_available() on one of the submodules; fall back
    # to whether MLX has a GPU device at all, and assume Metal on Darwin.
    try:
        if mx.default_device().type == mx.DeviceType.gpu:
            import sys
            return 'metal' if sys.platform == 'darwin' else 'cuda'
    except Exception:
        pass
    return None


def get_gpu_backend():
    """ The active GPU backend: 'metal', 'cuda', or None when there is none. """
    global _resolved, _resolved_valid
    if _requested != 'auto':
        return _requested
    if not _resolved_valid:
        _resolved = _detect()
        _resolved_valid = True
    return _resolved


def set_gpu_backend(backend):
    """
        Selects the GPU backend: 'auto' (detect; the default), 'metal' or
        'cuda'. An explicit backend that MLX cannot provide raises
        RuntimeError. Clears the source and kernel caches.
    """
    global _requested, _resolved_valid
    if backend not in ('auto',) + BACKENDS:
        raise ValueError("set_gpu_backend: expected 'auto', 'metal' or 'cuda', got %r" % (backend,))
    if backend != 'auto':
        mod = getattr(mx, backend, None)
        ok = False
        try:
            ok = mod is not None and mod.is_available()
        except Exception:
            ok = False
        if not ok:
            raise RuntimeError(
                "set_gpu_backend(%r): this MLX build has no %s backend "
                "(mx.%s.is_available() is False)." % (backend, backend, backend))
    _requested = backend
    _resolved_valid = False
    reset_kernel_cache()


def is_available():
    """ Whether any GPU backend is available. """
    return get_gpu_backend() is not None


def require_backend():
    b = get_gpu_backend()
    if b is None:
        raise RuntimeError('pydiffvg: no GPU backend available (neither Metal nor CUDA).')
    return b


def source_dir(backend = None):
    """ Directory holding the shared kernel sources of a backend. """
    backend = backend or require_backend()
    return os.path.join(_THIS_DIR, _SOURCE[backend][0])


def source_ext(backend = None):
    backend = backend or require_backend()
    return _SOURCE[backend][1]


# ---------------------------------------------------------------------------
# Backend-neutral kernel-body dialect

_METAL_PRELUDE = """
// ---- backend prelude (metal) ----
#define DVG_ADD(buf, i, v)   atomic_fetch_add_explicit(&(buf)[(i)], (v), memory_order_relaxed)
#define DVG_STORE(buf, i, v) atomic_store_explicit(&(buf)[(i)], (v), memory_order_relaxed)
#define DVG_XYZ(v) ((v).xyz)
#define DVG_CEIL(x) ceil(x)
"""

_CUDA_PRELUDE = """
// ---- backend prelude (cuda) ----
#define DVG_ADD(buf, i, v)   atomicAdd(&(buf)[(i)], (v))
#define DVG_STORE(buf, i, v) ((buf)[(i)] = (v))
#define DVG_XYZ(v) ((v).xyz())
#define DVG_CEIL(x) ceilf(x)
"""

_PRELUDE = {'metal': _METAL_PRELUDE, 'cuda': _CUDA_PRELUDE}

# DVG_THREAD_INDEX is expanded in PYTHON, not by the C preprocessor, because
# MLX decides which attribute parameters (thread_position_in_grid and friends)
# to put in the generated Metal signature by searching the *source text* for
# their names. Behind a #define the name is invisible to that search, the
# parameter is omitted, and the kernel fails to compile. Substituting here
# keeps the literal in the text MLX sees while the bodies stay backend-neutral.
#
# On CUDA this also carries the mandatory bounds check: MLX launches whole
# blocks and rounds the thread count up (measured: grid=10 with a 4-thread
# block launches 12 threads), so the padded threads must return early.
# dvg_params[0] is the exact count, appended to the inputs by Kernel.__call__.
_THREAD_INDEX = {
    'metal': 'uint idx = thread_position_in_grid.x;',
    'cuda': ('unsigned int idx = (unsigned int) cooperative_groups::this_grid().thread_rank();\n'
             '    if (idx >= (unsigned int) dvg_params[0]) return;'),
}

_THREAD_INDEX_TOKEN = 'DVG_THREAD_INDEX;'


def expand_source(source, backend = None):
    """ Body text with the Python-level tokens expanded for a backend. """
    backend = backend or require_backend()
    return source.replace(_THREAD_INDEX_TOKEN, _THREAD_INDEX[backend])

# Name of the extra CUDA-only params buffer (see DVG_THREAD_INDEX).
PARAMS_INPUT = 'dvg_params'
PARAMS_SIZE = 8


# Preprocessor defines emitted AHEAD of the shared sources, per backend.
# Only CUDA uses them: the .cu sources guard their capacity constants with
# #ifndef, the .metal sources do not (Metal has no such pressure -- the arrays
# are thread-private registers/stack there, not a device-wide local pool).
_defines = {}

# The per-thread arrays these size, and why they matter on CUDA: they live in
# LOCAL memory, and CUDA reserves the local pool for the device's full resident
# -thread capacity (~196k threads on a 4090), independent of the launch grid.
# Measured on the pod with ~490 MiB free: 1.5 KB/thread launches, 6 KB/thread
# (= Fragment[256]) fails with `out of memory` at every block size. Lowering
# the capacity is therefore the effective mitigation, not a smaller block.
_CAPACITY_DEFINES = (
    ('max_hit_shapes', 'DIFFVG_MAX_HIT_SHAPES'),
    ('pf_max_fragments', 'DIFFVG_PF_MAX_FRAGMENTS'),
    ('scene_bvh_stack', 'DIFFVG_MAX_SCENE_BVH_STACK'),
)


def set_fragment_capacity(max_hit_shapes = None, pf_max_fragments = None,
                          scene_bvh_stack = None, backend = None):
    """
        Lowers the per-thread fragment/stack capacities on CUDA, trading the
        maximum number of overlapping shapes per sample for occupancy (and, on
        a busy GPU, for the ability to launch at all).

        Defaults are 256 / 64 / 64. A scene whose samples never hit more than
        `max_hit_shapes` shapes is unaffected; beyond it, extra fragments are
        dropped, exactly as the CPU core asserts above 256.

        Metal is left alone: pass backend='cuda' explicitly, or call this while
        CUDA is active. Clears the source and kernel caches.
    """
    backend = backend or require_backend()
    if backend != 'cuda':
        raise NotImplementedError(
            'set_fragment_capacity is CUDA-only; the .metal sources do not guard '
            'their capacity constants (and Metal does not have the local-memory '
            'pressure that motivates lowering them).')
    requested = {'max_hit_shapes': max_hit_shapes,
                 'pf_max_fragments': pf_max_fragments,
                 'scene_bvh_stack': scene_bvh_stack}
    d = _defines.setdefault(backend, {})
    for key, macro in _CAPACITY_DEFINES:
        value = requested[key]
        if value is not None:
            if int(value) < 1:
                raise ValueError('set_fragment_capacity: %s must be >= 1' % key)
            d[macro] = int(value)
    reset_kernel_cache()


def get_fragment_capacity(backend = None):
    """ The capacity overrides in force, as a {macro: value} dict. """
    backend = backend or require_backend()
    return dict(_defines.get(backend, {}))


def constants_source(constants, backend = None):
    """
        Compile-time int constants for a kernel header, in the spelling each
        backend needs ('constant int x = 1;' / 'static constexpr int x = 1;';
        a namespace-scope `static const` read from CUDA device code is an
        nvrtc hazard, hence constexpr).
    """
    backend = backend or require_backend()
    kw = 'constant int' if backend == 'metal' else 'static constexpr int'
    return ''.join('%s %s = %d;\n' % (kw, name, int(value)) for name, value in constants)


def header(files, constants = (), backend = None):
    """
        The concatenated shared sources of `files` (extension-free base names,
        in the order they must appear), followed by the backend prelude and
        any compile-time constants. Cached per backend.
    """
    backend = backend or require_backend()
    files = tuple(files)
    constants = tuple(constants)
    defs = tuple(sorted(_defines.get(backend, {}).items()))
    key = (backend, files, constants, defs)
    h = _headers.get(key)
    if h is None:
        d, ext = _SOURCE[backend]
        parts = []
        if defs:
            # Ahead of the sources, whose capacity constants are #ifndef-guarded.
            parts.append('// ---- capacity overrides ----\n' +
                         '\n'.join('#define %s %d' % (m, v) for m, v in defs))
        for name in files:
            fname = name + ext
            with open(os.path.join(_THIS_DIR, d, fname), 'r') as f:
                parts.append('// ---- %s ----\n' % fname)
                parts.append(f.read())
        parts.append(_PRELUDE[backend])
        if constants:
            parts.append(constants_source(constants, backend))
        h = _headers[key] = '\n'.join(parts)
    return h


# ---------------------------------------------------------------------------
# Block size

_BLOCK_SIZE = {'metal': 256, 'cuda': 128}
_KERNEL_BLOCK_SIZE = {}     # (backend, kernel name) -> block size


def set_block_size(size, kernel = None, backend = None):
    """
        Threadgroup / CUDA block size, globally or for one kernel by name
        (e.g. set_block_size(64, 'render_edge')). Smaller CUDA blocks are the
        first mitigation for the local-memory pressure of the fragment arrays.
    """
    backend = backend or require_backend()
    size = int(size)
    if size < 1:
        raise ValueError('set_block_size: size must be >= 1')
    if kernel is None:
        _BLOCK_SIZE[backend] = size
    else:
        _KERNEL_BLOCK_SIZE[(backend, kernel)] = size


def block_size(kernel = None, backend = None):
    backend = backend or require_backend()
    return _KERNEL_BLOCK_SIZE.get((backend, kernel), _BLOCK_SIZE[backend])


def threadgroup(n, kernel = None, backend = None):
    """ Launch block for a 1-D dispatch of n threads. """
    return (min(block_size(kernel, backend), max(1, int(n))), 1, 1)


# ---------------------------------------------------------------------------
# Kernel construction

def _params_array(n, backend):
    key = (backend, int(n))
    a = _params.get(key)
    if a is None:
        if len(_params) > 64:
            _params.clear()
        vals = [0] * PARAMS_SIZE
        vals[0] = int(n)
        a = _params[key] = mx.array(vals, dtype = mx.int32)
    return a


class Kernel:
    """
        A backend-neutral kernel. Built through gpu_backend.kernel(); called
        with the same arguments on both backends:

            k(inputs = [...], grid = (n, 1, 1), output_shapes = [...],
              output_dtypes = [...], init_value = 0)

        `grid` is a thread count on both backends. On CUDA the extra
        `dvg_params` input carrying that count is appended here, so callers
        never see it.
    """
    __slots__ = ('backend', 'name', 'fn', 'default_threadgroup')

    def __init__(self, backend, name, fn):
        self.backend = backend
        self.name = name
        self.fn = fn

    def __call__(self, inputs, grid, output_shapes, output_dtypes, init_value = None,
                 threadgroup = None, template = None, verbose = False):
        n = int(grid[0]) * int(grid[1] if len(grid) > 1 else 1) * int(grid[2] if len(grid) > 2 else 1)
        if self.backend == 'cuda':
            inputs = list(inputs) + [_params_array(n, self.backend)]
        if threadgroup is None:
            threadgroup = globals()['threadgroup'](n, self.name, self.backend)
        kwargs = {}
        if init_value is not None:
            kwargs['init_value'] = init_value
        if template is not None:
            kwargs['template'] = template
        if verbose:
            kwargs['verbose'] = True
        return self.fn(inputs = inputs, grid = grid, threadgroup = threadgroup,
                       output_shapes = output_shapes, output_dtypes = output_dtypes, **kwargs)


def kernel(name, input_names, output_names, source, header_files, constants = (),
           shared_memory = 0, ensure_row_contiguous = True, backend = None):
    """
        Builds (and caches) a kernel from a backend-neutral body.

        name          kernel name; 'diffvg_' is prefixed for the compiler.
        input_names   as MLX expects; on CUDA `dvg_params` is appended here.
        source        the body, in the dialect documented at the top of this
                      module (DVG_THREAD_INDEX / DVG_ADD / DVG_STORE /
                      DVG_XYZ / DVG_CEIL).
        header_files  extension-free base names of the shared sources.
        constants     (name, int) pairs compiled into the header.
        shared_memory dynamic shared memory, CUDA only (ignored on Metal).

        Per-backend arguments, and why:
          Metal  atomic_outputs=True    -- every output is device atomic<T>,
                                           which is what DVG_ADD/DVG_STORE emit
                 compile_options math_mode 'fast'
                                        -- mandatory: the AGX code generator
                                           aborts on these kernels in the other
                                           modes (see DESIGN.md)
          CUDA   neither exists on mx.fast.cuda_kernel; nvrtc defaults apply
                 (IEEE division and sqrt, fmad on), which sit closer to the
                 C++ CPU reference than Metal's fast math does.
    """
    backend = backend or require_backend()
    key = (backend, name)
    k = _kernels.get(key)
    if k is not None:
        return k
    hdr = header(header_files, constants, backend)
    source = expand_source(source, backend)
    if backend == 'metal':
        fn = mx.fast.metal_kernel(
            name = 'diffvg_' + name,
            input_names = list(input_names),
            output_names = list(output_names),
            source = source,
            header = hdr,
            ensure_row_contiguous = ensure_row_contiguous,
            atomic_outputs = True,
            compile_options = {'math_mode': 'fast'})
    else:
        fn = mx.fast.cuda_kernel(
            name = 'diffvg_' + name,
            input_names = list(input_names) + [PARAMS_INPUT],
            output_names = list(output_names),
            source = source,
            header = hdr,
            ensure_row_contiguous = ensure_row_contiguous,
            shared_memory = shared_memory)
    k = _kernels[key] = Kernel(backend, name, fn)
    return k


def reset_kernel_cache():
    """ Forget the loaded sources and built kernels (after editing them). """
    _headers.clear()
    _kernels.clear()
    _params.clear()
