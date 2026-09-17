"""
    GPU scene construction, on either GPU backend (Metal or CUDA).

    Builds the flat scene pools of pydiffvg/metal/common.metal (identical
    layout to diffvg.Scene.export_flat()) directly from the serialized scene
    arguments (render_mlx.serialize_scene), without pybind objects or the C++
    Scene:

    - topology_from_args(args) -> SceneTopology: everything that depends only
      on the scene structure (shape/colour types, point and stop counts, group
      membership, flags, filter type): all int pools except BVH children, every
      float-pool offset, the gather permutation that assembles the float pool,
      BVH leaf payloads and the gradient index map. Host side (numpy), cached
      (LRU) by structure.
    - build_pools(topology, args, ...) -> dict with ip / ints / floats: float
      parameters are copied with one bytes join per category (unified memory),
      then path lengths and CDFs, sample CDFs, canvas_to_shape inverses, shape
      and segment bounding boxes are computed on the GPU (five small kernels
      and MLX ops) and all BVHs are built on the GPU (scene_gpu_bvh).

    The five kernels have one body each, written in the backend-neutral
    dialect of gpu_backend.py and compiled for whichever backend is active;
    their helper functions live in pydiffvg/metal/scene.metal and
    pydiffvg/cuda/scene.cu. scene_gpu_bvh is MLX ops only, so it is
    backend-independent already.

    Bit parity with the C++ core: the kernels avoid contraction (Metal: safe
    math and `#pragma METAL fp contract(off)'; CUDA: the __fmul_rn / __fadd_rn
    intrinsics, since nvrtc offers no such pragma) and use explicit fma()
    exactly where clang contracts a*b+c in the C++ sources (-ffp-contract=on:
    matrix.h inverse and xform_pt, the Ramanujan ellipse circumference), and
    the running sums (path length, path CDF, sample CDF) are sequential
    float32 loops in C++ order. sqrt is correctly rounded on both backends
    (Metal needs the correction in sg_sqrt, CUDA's sqrtf already is).

    Everything is lazy: no call in the per-frame path evaluates an array, so
    the pools are one MLX graph that the rasterisation kernels consume.
    The pools can be assembled on either stream: set_scene_build_device('auto'
    | 'cpu' | 'gpu'), where 'auto' decides from a structural size measure
    (scene_build_work) without any sync. The five helper kernels only run on
    the GPU, so the CPU stream uses MLX-op equivalents that produce
    bit-identical pools (see below). 'auto' currently keeps every scene on the
    GPU stream: measured, the CPU stream never made a frame faster, because
    the pools feed GPU kernels and building them on the other stream
    serialises the two (see SCENE_BUILD_CPU_MAX_WORK for the numbers and for
    re-enabling it).
"""
import operator
from collections import OrderedDict

import numpy as np
import mlx.core as mx
import diffvg

from . import gpu_backend as gb
from . import scene_gpu_bvh as sgb

__all__ = ['SceneTopology', 'topology_from_args', 'topology_signature', 'build_pools',
           'build_pools_from_sources', 'sources_from_args', 'grad_index_map', 'clear_topology_cache',
           'set_scene_build_device', 'get_scene_build_device', 'scene_build_work',
           'supports_backend']

# ip slots (pydiffvg/metal/common.metal)
IP_W, IP_H, IP_NSX, IP_NSY = 0, 1, 2, 3
IP_CANVAS_W, IP_CANVAS_H = 4, 5
IP_NUM_SHAPES, IP_NUM_GROUPS, IP_NUM_TOTAL_SHAPES, IP_FILTER_TYPE = 6, 7, 8, 9
IP_USE_PREFILTERING, IP_HAS_BACKGROUND, IP_NUM_EVAL, IP_SEED_LO, IP_SEED_HI = 10, 11, 12, 13, 14
IP_SCENE_BVH_BASE, IP_SHAPES_I_OFF, IP_GROUPS_I_OFF, IP_BVH_I_OFF = 15, 16, 17, 18
IP_SAMPLE_SHAPE_ID_OFF, IP_SAMPLE_GROUP_ID_OFF, IP_BVH_F_OFF = 19, 20, 21
IP_SAMPLE_CDF_OFF, IP_SAMPLE_PMF_OFF, IP_FILTER_RADIUS_OFF = 22, 23, 24
IP_NUM_FLOATS, IP_NUM_INTS, IP_NUM_BVH_NODES, IP_USE_EVAL_POSITIONS = 25, 26, 27, 28
IP_SIZE = 64
MIN_POOL = 64
_MIN_BUFFER = 8  # metal_kernel inputs with < 8 elements arrive as `constant`

SHAPE_CIRCLE, SHAPE_ELLIPSE, SHAPE_PATH, SHAPE_RECT = 0, 1, 2, 3

_SC = diffvg.ShapeType.circle
_SE = diffvg.ShapeType.ellipse
_SP = diffvg.ShapeType.path
_SR = diffvg.ShapeType.rect
_CC = diffvg.ColorType.constant
_CL = diffvg.ColorType.linear_gradient
_CR = diffvg.ColorType.radial_gradient


def _enum_int(e):
    return int(getattr(e, 'value', e))


# ---------------------------------------------------------------------------
# Small host helpers

def _ranges(starts, lengths):
    """ Concatenation of arange(s, s + n) for (s, n) pairs (int64). """
    starts = np.asarray(starts, np.int64)
    lengths = np.asarray(lengths, np.int64)
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    ends = np.cumsum(lengths)
    return np.repeat(starts - ends + lengths, lengths) + np.arange(total, dtype = np.int64)


def _pick(args, pos):
    n = len(pos)
    if n == 0:
        return ()
    if n == 1:
        return (args[pos[0]],)
    return operator.itemgetter(*pos)(args)


_MV_FORMAT = operator.attrgetter('format')
_MV_TOBYTES = operator.methodcaller('tobytes')
_BUFFER_FORMAT = {np.dtype(np.float32): 'f', np.dtype(np.int32): 'i'}


def _as_np(a, np_dtype):
    """ One object as a flat numpy array of np_dtype (casts once: float64/16, bfloat16, int64 ...). """
    if isinstance(a, mx.array):
        target = mx.float32 if np_dtype == np.float32 else mx.int32
        if a.dtype != target:
            a = a.astype(target, stream = mx.cpu)   # float64 exists only on the CPU stream
    return np.asarray(a, dtype = np_dtype).reshape(-1)


def _join(objs, count, np_dtype):
    """
        Flat numpy array (C order) of `count` elements from a sequence of
        arrays / scalars. Always reads the current contents (MLX in-place
        updates keep both the Python object and, through buffer donation, the
        data address, so no cheap change fingerprint exists). Fast path: one
        memoryview per array (evaluates lazy arrays), buffer format check,
        tobytes (C order, also for non-contiguous views) and one join, about
        1 us per array. Other dtypes and Python scalars are cast once.
    """
    np_dtype = np.dtype(np_dtype)
    if len(objs) == 0:
        return np.zeros(0, np_dtype)
    try:
        mvs = list(map(memoryview, objs))
        fmts = set(map(_MV_FORMAT, mvs))
        if len(fmts) == 1 and fmts.pop() == _BUFFER_FORMAT[np_dtype]:
            try:
                # memoryview objects reject non-contiguous layouts here (unlike
                # joining the mx.arrays directly, which silently ignores strides)
                b = b''.join(mvs)
            except (TypeError, BufferError):
                b = b''.join(map(_MV_TOBYTES, mvs))
            if len(b) == np_dtype.itemsize * count:
                return np.frombuffer(b, np_dtype)
    except (TypeError, ValueError, BufferError):
        pass
    out = np.concatenate([_as_np(a, np_dtype) for a in objs])
    if out.size != count:
        raise ValueError('scene argument sizes do not match the topology (%d vs %d)' % (out.size, count))
    return out


def _pad(a, n = _MIN_BUFFER):
    a = a.reshape(-1)
    if a.size >= n:
        return a
    return mx.concatenate([a, mx.zeros((n - a.size,), dtype = a.dtype)])


def _seed_bits(seed):
    s = int(seed) & 0xFFFFFFFFFFFFFFFF
    lo = int(np.array([s & 0xFFFFFFFF], dtype = np.uint32).view(np.int32)[0])
    hi = int(np.array([s >> 32], dtype = np.uint32).view(np.int32)[0])
    return lo, hi


# ---------------------------------------------------------------------------
# GPU kernels (safe math: no contraction / reassociation, explicit fma)
#
# Each body is written once, in the backend-neutral dialect of gpu_backend.py:
# `DVG_THREAD_INDEX;` declares `idx` (on CUDA together with the mandatory
# bounds check). The helpers the bodies call -- sg_sqrt, sg_dist, sg_xf,
# sg_fma, sg_mul, sg_add, sg_inf and the 2-vector sg_f2 -- are defined once per
# backend in pydiffvg/metal/scene.metal and pydiffvg/cuda/scene.cu with
# identical rounding; read those files for why the arithmetic is spelled the
# way it is (contraction off / __fmul_rn, corrected sqrt / plain sqrtf).
_HEADER_FILES = ('scene',)

# Per path (thread): shapes_bbox, compute_shape_length, build_path_cdfs and the
# path BVH leaf boxes / radii of compute_bounding_boxes, in C++ order.
#   pinfo[5p..]: first point, first segment, num_base_points, num_points, first thickness (-1)
#   seg_f[7e..]: pmf, cdf, box(4), radius; path_f[5p..]: length, bbox(4)
_PATHS_SOURCE = """
    DVG_THREAD_INDEX;
    int p = (int) idx;
    int po = pinfo[5 * p];
    int so = pinfo[5 * p + 1];
    int nb = pinfo[5 * p + 2];
    int npt = pinfo[5 * p + 3];
    int to = pinfo[5 * p + 4];
    float r = sw[p];
    float inf = sg_inf();
    float b0x = inf, b0y = inf, b1x = -inf, b1y = -inf;
    if (npt > 0) {
        b0x = pts[2 * po]; b0y = pts[2 * po + 1];
        b1x = b0x; b1y = b0y;
    }
    for (int k = 1; k < npt; k++) {
        float x = pts[2 * (po + k)];
        float y = pts[2 * (po + k) + 1];
        b0x = x < b0x ? x : b0x;
        b0y = y < b0y ? y : b0y;
        b1x = x > b1x ? x : b1x;
        b1y = y > b1y ? y : b1y;
    }
    float t1 = 1.0f / 3.0f;
    float tt1 = 1.0f - t1;
    float c10 = tt1 * tt1 * tt1, c11 = 3.0f * tt1 * tt1 * t1, c12 = 3.0f * tt1 * t1 * t1, c13 = t1 * t1 * t1;
    float t2 = 2.0f / 3.0f;
    float tt2 = 1.0f - t2;
    float c20 = tt2 * tt2 * tt2, c21 = 3.0f * tt2 * tt2 * t2, c22 = 3.0f * tt2 * t2 * t2, c23 = t2 * t2 * t2;
    float length = 0.0f;
    int pid = 0;
    for (int j = 0; j < nb; j++) {
        int c = ctrl[so + j];
        int ids[4];
        for (int q = 0; q <= c; q++) {
            ids[q] = pid + q;
        }
        ids[c + 1] = (pid + c + 1) % npt;
        pid += c + 1;
        sg_f2 P[4];
        for (int q = 0; q <= c + 1; q++) {
            P[q] = sg_f2(pts[2 * (po + ids[q])], pts[2 * (po + ids[q]) + 1]);
        }
        float x0 = inf, y0 = inf, x1 = -inf, y1 = -inf;
        for (int q = 0; q <= c + 1; q++) {
            x0 = P[q].x < x0 ? P[q].x : x0;
            y0 = P[q].y < y0 ? P[q].y : y0;
            x1 = P[q].x > x1 ? P[q].x : x1;
            y1 = P[q].y > y1 ? P[q].y : y1;
        }
        float rr = r;
        if (to >= 0) {
            rr = th[to + ids[0]];
            for (int q = 1; q <= c + 1; q++) {
                float v = th[to + ids[q]];
                rr = rr > v ? rr : v;
            }
        }
        float d = 0.0f;
        if (c == 0) {
            d = sg_dist(P[1], P[0]);
        } else if (c == 1) {
            sg_f2 v1 = sg_f2(0.25f * P[0].x, 0.25f * P[0].y) + sg_f2(0.5f * P[1].x, 0.5f * P[1].y);
            v1 = v1 + sg_f2(0.25f * P[2].x, 0.25f * P[2].y);
            d = sg_dist(v1, P[0]) + sg_dist(v1, P[2]);
        } else {
            sg_f2 v1 = (c10 * P[0] + c11 * P[1]) + c12 * P[2];
            v1 = v1 + c13 * P[3];
            sg_f2 v2 = (c20 * P[0] + c21 * P[1]) + c22 * P[2];
            v2 = v2 + c23 * P[3];
            d = (sg_dist(v1, P[0]) + sg_dist(v1, v2)) + sg_dist(v2, P[3]);
        }
        int e = 7 * (so + j);
        seg_f[e] = d;
        seg_f[e + 2] = x0;
        seg_f[e + 3] = y0;
        seg_f[e + 4] = x1;
        seg_f[e + 5] = y1;
        seg_f[e + 6] = rr;
        length += d;
    }
    float path_length = 0.0f + length;
    path_f[5 * p] = path_length;
    path_f[5 * p + 1] = b0x;
    path_f[5 * p + 2] = b0y;
    path_f[5 * p + 3] = b1x;
    path_f[5 * p + 4] = b1y;
    if (path_length > 0.0f) {
        float inv_length = 1.0f / path_length;
        float cdf = 0.0f;
        for (int j = 0; j < nb; j++) {
            int e = 7 * (so + j);
            float d = sg_mul(seg_f[e], inv_length);
            seg_f[e] = d;
            cdf = j == 0 ? d : d + cdf;
            seg_f[e + 1] = cdf;
        }
    } else {
        for (int j = 0; j < nb; j++) {
            int e = 7 * (so + j);
            seg_f[e] = 1.0f / float(nb);
            seg_f[e + 1] = float(j + 1) / float(nb);
        }
    }
"""

# Sequential float32 running sum (build_shape_cdfs), one thread.
_CUMSUM_SOURCE = """
    DVG_THREAD_INDEX;
    (void) idx;
    int n = count[0];
    float c = 0.0f;
    for (int s = 0; s < n; s++) {
        c = s == 0 ? x[s] : x[s] + c;
        out[s] = c;
    }
"""

# matrix.h inverse (3x3) with clang's contraction (verified bit-exact).
_INVERSE_SOURCE = """
    DVG_THREAD_INDEX;
    int g = (int) idx;
    int o = 9 * g;
    float m00 = m[o], m01 = m[o + 1], m02 = m[o + 2];
    float m10 = m[o + 3], m11 = m[o + 4], m12 = m[o + 5];
    float m20 = m[o + 6], m21 = m[o + 7], m22 = m[o + 8];
    float a = sg_fma(m11, m22, -sg_mul(m21, m12));
    float b = sg_fma(m10, m22, -sg_mul(m12, m20));
    float c = sg_fma(m10, m21, -sg_mul(m11, m20));
    float det = sg_fma(m02, c, sg_fma(m00, a, -sg_mul(m01, b)));
    float invdet = 1.0f / det;
    out[o] = a * invdet;
    out[o + 1] = sg_fma(m02, m21, -sg_mul(m01, m22)) * invdet;
    out[o + 2] = sg_fma(m01, m12, -sg_mul(m02, m11)) * invdet;
    out[o + 3] = sg_fma(m12, m20, -sg_mul(m10, m22)) * invdet;
    out[o + 4] = sg_fma(m00, m22, -sg_mul(m02, m20)) * invdet;
    out[o + 5] = sg_fma(m10, m02, -sg_mul(m00, m12)) * invdet;
    out[o + 6] = sg_fma(m10, m21, -sg_mul(m20, m11)) * invdet;
    out[o + 7] = sg_fma(m20, m01, -sg_mul(m00, m21)) * invdet;
    out[o + 8] = sg_fma(m00, m11, -sg_mul(m10, m01)) * invdet;
"""

# aabb.h transform(shape_to_canvas, box): merge of the 4 transformed corners.
_XFORM_BOX_SOURCE = """
    DVG_THREAD_INDEX;
    int g = (int) idx;
    int o = 9 * g;
    int bo = 4 * g;
    float inf = sg_inf();
    float x0 = inf, y0 = inf, x1 = -inf, y1 = -inf;
    float cx[4] = {box[bo], box[bo], box[bo + 2], box[bo + 2]};
    float cy[4] = {box[bo + 1], box[bo + 3], box[bo + 1], box[bo + 3]};
    for (int k = 0; k < 4; k++) {
        float t0 = sg_xf(m[o], cx[k], m[o + 1], cy[k], m[o + 2]);
        float t1 = sg_xf(m[o + 3], cx[k], m[o + 4], cy[k], m[o + 5]);
        float t2 = sg_xf(m[o + 6], cx[k], m[o + 7], cy[k], m[o + 8]);
        float px = t0 / t2;
        float py = t1 / t2;
        x0 = px < x0 ? px : x0;
        y0 = py < y0 ? py : y0;
        x1 = px > x1 ? px : x1;
        y1 = py > y1 ? py : y1;
    }
    out[bo] = x0;
    out[bo + 1] = y0;
    out[bo + 2] = x1;
    out[bo + 3] = y1;
"""

# Ramanujan ellipse circumference as clang compiles it (compute_shape_length).
_ELLIPSE_SOURCE = """
    DVG_THREAD_INDEX;
    int i = (int) idx;
    float a = r[2 * i];
    float b = r[2 * i + 1];
    float pi = 3.14159265358979323846f;
    float s = sg_sqrt(sg_fma(3.0f, a, b) * sg_fma(3.0f, b, a));
    out[i] = 0.0f + pi * sg_fma(3.0f, a + b, -s);
"""

_KERNEL_DEFS = {
    'paths': (['pinfo', 'ctrl', 'pts', 'th', 'sw'], ['seg_f', 'path_f'], _PATHS_SOURCE),
    'cumsum': (['x', 'count'], ['out'], _CUMSUM_SOURCE),
    'inverse': (['m'], ['out'], _INVERSE_SOURCE),
    'xform_box': (['m', 'box'], ['out'], _XFORM_BOX_SOURCE),
    'ellipse': (['r'], ['out'], _ELLIPSE_SOURCE),
}
_kernels = {}          # (backend, name) -> kernel


class _SafeMetalKernel:
    """
        mx.fast.metal_kernel with the call signature of gpu_backend.Kernel.

        The scene kernels cannot go through gpu_backend.kernel() on Metal:
        that builds every Metal kernel with atomic_outputs (so an output buffer
        can only be written through atomic_store, and the 'paths' kernel reads
        its own output back) and with math_mode 'fast', which the rasterisation
        kernels need but the pools must not have -- fast math replaces the
        divisions with reciprocal approximations and breaks bit parity with the
        C++ core. On CUDA neither option exists, nvrtc's defaults are the ones
        wanted, and gpu_backend.kernel() is used unchanged; should it ever grow
        math_mode / atomic_outputs arguments, this class and the Metal branch
        of _kernel() disappear.
    """
    __slots__ = ('fn',)

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, inputs, grid, output_shapes, output_dtypes, init_value = None,
                 threadgroup = None):
        kwargs = {} if init_value is None else {'init_value': init_value}
        return self.fn(inputs = inputs, grid = grid, threadgroup = threadgroup,
                       output_shapes = output_shapes, output_dtypes = output_dtypes, **kwargs)


def supports_backend(backend):
    """
        Whether this module can build the pools on a GPU backend (render_mlx
        falls back to the C++ scene when it cannot). Both GPU backends are
        supported; anything else (no GPU at all) is not.
    """
    return backend in gb.BACKENDS


def _kernel(name):
    backend = gb.require_backend()
    key = (backend, name)
    k = _kernels.get(key)
    if k is None:
        inputs, outputs, source = _KERNEL_DEFS[name]
        if backend == 'cuda':
            k = gb.kernel('scene_gpu_' + name, inputs, outputs, source, _HEADER_FILES)
        else:
            k = _SafeMetalKernel(mx.fast.metal_kernel(
                name = 'diffvg_scene_gpu_' + name,
                input_names = list(inputs),
                output_names = list(outputs),
                source = gb.expand_source(source, backend),
                header = gb.header(_HEADER_FILES, backend = backend),
                ensure_row_contiguous = True,
                atomic_outputs = False,
                compile_options = {'math_mode': 'safe'}))
        _kernels[key] = k
    return k


def _run(name, inputs, grid, out_sizes, out_dtypes):
    # `grid` is a thread count on both backends; CUDA rounds it up to whole
    # blocks and the bounds check of DVG_THREAD_INDEX drops the extra threads.
    n = max(grid, 1)
    outs = _kernel(name)(
        inputs = [_pad(a) for a in inputs],
        grid = (n, 1, 1),
        threadgroup = (min(64, n), 1, 1),
        output_shapes = [(max(s, _MIN_BUFFER),) for s in out_sizes],
        output_dtypes = out_dtypes,
        init_value = 0)
    return [o[:s] for o, s in zip(outs, out_sizes)]


def reset_kernel_cache():
    """ Forget the built kernels and the loaded sources (after editing them). """
    _kernels.clear()
    gb.reset_kernel_cache()


# ---------------------------------------------------------------------------
# CPU-stream assembly
#
# A custom kernel only runs on the GPU ("[metal_kernel] Only supports the
# GPU"), so the five helper kernels above have MLX-op equivalents used when the
# pools are assembled on the CPU stream. They are backend-independent, and
# bit-identical to the kernels:
# - the C++ contractions that the kernels write as fma() are emulated in
#   float64 (CPU-only dtype: one rounding, as fma), the inner products that C++
#   rounds to float32 first are kept in float32;
# - CPU sqrt is correctly rounded (Metal's is not, hence sg_sqrt there);
# - the sequential float32 running sums (path length, path CDF) are cumsum over
#   a (paths, max segments) matrix zero-padded on the right, and the sample CDF
#   is a 1-D cumsum: MLX's CPU cumsum is sequential, so the rounding matches
#   (the GPU's is not, hence the kernels).

_scene_build_device = 'auto'

# 'auto' threshold on the work measure (points + segments + shapes + groups +
# members): the CPU stream is used when the work is at most this, 0 disables it.
#
# MEASURED (scratchpad/lazycpu, paired samples alternating CPU/GPU per
# repetition, median paired difference + sign test): the CPU stream never won
# at frame level. Assembling on its own is a wash below ~30 shapes (within
# 1 ms, CPU faster in 52-58% of pairs) and clearly worse from ~1000 shapes
# (synth3000 25 ms slower, hawaii 772 vs 26 ms), while a whole frame was
# slower with CPU assembly at almost every size (CPU faster in only 0-30% of
# pairs): the pools are consumed by GPU kernels, so building them on the CPU
# stream adds a cross-stream dependency instead of queueing ahead of the
# render kernels. Hence 0 (the GPU stream everywhere) until a measurement on
# an idle machine says otherwise - these runs shared the machine at load
# ~200, which penalises the CPU stream specifically. Re-derive with
# scratchpad/lazycpu/measure_paired.py and set this to the crossover work.
SCENE_BUILD_CPU_MAX_WORK = 0
# The per-path sequential sums need a (paths, max segments) matrix; skip the
# CPU stream when that matrix would be large in absolute terms (a scene mixing
# one huge path with many small ones), not merely larger than the segment count.
SCENE_BUILD_CPU_MAX_PADDED = 1 << 20


def set_scene_build_device(device):
    """
        Where the flat pools are assembled on the GPU backend: 'auto'
        (default: the CPU stream for small scenes, the GPU stream for large
        ones), 'cpu' or 'gpu'. Both give the same pools; the rasterisation
        kernels always run on the GPU.
    """
    global _scene_build_device
    if device not in ('auto', 'cpu', 'gpu'):
        raise ValueError("set_scene_build_device: expected 'auto', 'cpu' or 'gpu', got %r" % (device,))
    _scene_build_device = device


def get_scene_build_device():
    return _scene_build_device


def scene_build_work(topology):
    """ Cheap structural size measure of the assembly (host ints, no sync). """
    t = topology
    return t.num_points + t.num_segments + t.num_shapes + t.num_groups + t.num_total_shapes


def _use_cpu_build(t):
    mode = _scene_build_device
    if mode == 'gpu':
        return False
    if mode == 'cpu':
        return True
    if SCENE_BUILD_CPU_MAX_WORK <= 0 or scene_build_work(t) > SCENE_BUILD_CPU_MAX_WORK:
        return False
    return t.num_paths * t.max_segments_per_path <= SCENE_BUILD_CPU_MAX_PADDED


def _cpu_tables(t):
    """ Index tables of the CPU-stream path (built once per topology, host numpy). """
    c = getattr(t, '_cpu_tab', None)
    if c is not None:
        return c
    L = t.lay
    npts, nbp, thick, ctrl = L['npts'], L['nbp'], L['thick'], L['ctrl']
    P, Nseg = t.num_paths, t.num_segments
    point_base = np.cumsum(npts) - npts
    seg_start = np.cumsum(nbp) - nbp
    path_of_seg = np.repeat(np.arange(P), nbp)
    j_of_seg = np.arange(Nseg) - seg_start[path_of_seg]
    c1 = ctrl + 1
    cs = np.cumsum(c1) - c1
    pid = cs - cs[seg_start[path_of_seg]] if Nseg else cs
    base_seg = point_base[path_of_seg]
    npts_seg = np.maximum(npts[path_of_seg], 1)
    deg = ctrl
    qs = np.arange(4)[None, :]
    cc = deg[:, None]
    idx = np.where(qs >= cc + 1,
                   (base_seg + (pid + deg + 1) % npts_seg)[:, None],
                   base_seg[:, None] + pid[:, None] + np.minimum(qs, cc))
    th_start = np.cumsum(np.where(thick, npts, 0)) - np.where(thick, npts, 0)
    thick_seg = thick[path_of_seg] if Nseg else np.zeros(0, bool)
    tidx = np.where(thick_seg[:, None], th_start[path_of_seg][:, None] + (idx - base_seg[:, None]),
                    t.num_thickness)
    maxnbp = t.max_segments_per_path
    pad = np.full((P, maxnbp), Nseg, np.int64)
    if Nseg:
        pad[path_of_seg, j_of_seg] = np.arange(Nseg)
    c = dict(idx_m = mx.array(idx.reshape(-1).astype(np.int32)),
             tidx_m = mx.array(tidx.reshape(-1).astype(np.int32)),
             thick_seg_m = mx.array(thick_seg),
             deg_m = mx.array(deg.astype(np.int32)),
             path_of_seg_m = mx.array(path_of_seg.astype(np.int32)),
             pad_m = mx.array(pad.reshape(-1).astype(np.int32)),
             flat_m = mx.array((path_of_seg * maxnbp + j_of_seg).astype(np.int32)) if Nseg else None,
             jmat = mx.array((np.tile(np.arange(maxnbp), (P, 1)) + 1).astype(np.float32)) if P and maxnbp else None,
             nbp_col = mx.array(nbp.astype(np.float32).reshape(P, 1)) if P else None,
             path_of_point_m = mx.array(np.repeat(np.arange(P), npts).astype(np.int32)),
             maxnbp = maxnbp)
    t._cpu_tab = c
    return c


def _f64(a):
    return a.astype(mx.float64)


def _fma_cpu(a, b, c):
    """ float32 fma(a, b, c) (one rounding), a or b may be a Python float. """
    x = _f64(a) if isinstance(a, mx.array) else a
    y = _f64(b) if isinstance(b, mx.array) else b
    return (x * y + _f64(c)).astype(mx.float32)


def _fms_cpu(x, y, z, w):
    """ fma(x, y, -(z * w)) as the C++ / kernel code contracts it (z * w rounded to float32). """
    return _fma_cpu(x, y, -(z * w))


def _dist_cpu(a, b):
    d = a - b
    return mx.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])


def _paths_cpu(t, pts, th, sw_path):
    """ MLX-op equivalent of the 'paths' kernel (see its source above). """
    c = _cpu_tables(t)
    P, Nseg = t.num_paths, t.num_segments
    pts2 = pts.reshape(-1, 2)
    p4 = pts2[c['idx_m']].reshape(Nseg, 4, 2)
    x, y = p4[:, :, 0], p4[:, :, 1]
    seg_box = mx.stack([x.min(axis = 1), y.min(axis = 1), x.max(axis = 1), y.max(axis = 1)], axis = 1)
    sw_seg = sw_path[c['path_of_seg_m']]
    if t.num_thickness:
        th4 = mx.concatenate([th, mx.zeros((1,), mx.float32)])[c['tidx_m']].reshape(Nseg, 4)
        seg_r = mx.where(c['thick_seg_m'], th4.max(axis = 1), sw_seg)
    else:
        seg_r = sw_seg
    P0, P1, P2, P3 = p4[:, 0], p4[:, 1], p4[:, 2], p4[:, 3]
    t1 = np.float32(1.0) / np.float32(3.0)
    t2 = np.float32(2.0) / np.float32(3.0)
    def coef(tv):
        tt = np.float32(1.0) - tv
        return [float(tt * tt * tt), float(np.float32(3.0) * tt * tt * tv),
                float(np.float32(3.0) * tt * tv * tv), float(tv * tv * tv)]
    c1_, c2_ = coef(t1), coef(t2)
    def ev(cf):
        return ((cf[0] * P0 + cf[1] * P1) + cf[2] * P2) + cf[3] * P3
    q = (0.25 * P0 + 0.5 * P1) + 0.25 * P2
    d_line = _dist_cpu(P1, P0)
    d_quad = _dist_cpu(q, P0) + _dist_cpu(q, P2)
    v1, v2 = ev(c1_), ev(c2_)
    d_cub = (_dist_cpu(v1, P0) + _dist_cpu(v1, v2)) + _dist_cpu(v2, P3)
    deg = c['deg_m']
    d = mx.where(deg == 0, d_line, mx.where(deg == 1, d_quad, d_cub))
    # per-path sequential float32 sums over a zero-padded (P, max segments) matrix
    maxnbp = c['maxnbp']
    dpad = mx.concatenate([d, mx.zeros((1,), mx.float32)])[c['pad_m']].reshape(P, maxnbp)
    cum = mx.cumsum(dpad, axis = 1)
    path_len = mx.array(0.0, mx.float32) + cum[:, maxnbp - 1]
    inv = (mx.array(1.0, mx.float32) / path_len).reshape(P, 1)
    pmf_pad = dpad * inv
    cdf_pad = mx.cumsum(pmf_pad, axis = 1)
    positive = (path_len > mx.array(0.0, mx.float32)).reshape(P, 1)
    uni = mx.array(1.0, mx.float32) / c['nbp_col']
    pmf_pad = mx.where(positive, pmf_pad, uni)
    cdf_pad = mx.where(positive, cdf_pad, c['jmat'] / c['nbp_col'])
    seg_pmf = pmf_pad.reshape(-1)[c['flat_m']]
    seg_cdf = cdf_pad.reshape(-1)[c['flat_m']]
    inf = float(np.float32(np.inf))
    pmin = mx.full((P, 2), inf, mx.float32).at[c['path_of_point_m']].minimum(pts2)
    pmax = mx.full((P, 2), -inf, mx.float32).at[c['path_of_point_m']].maximum(pts2)
    path_box = mx.concatenate([pmin, pmax], axis = 1)
    return path_len, path_box, seg_pmf, seg_cdf, seg_box, seg_r


def _ellipse_cpu(rx, ry):
    """ MLX-op equivalent of the 'ellipse' kernel. """
    s = mx.sqrt(_fma_cpu(3.0, rx, ry) * _fma_cpu(3.0, ry, rx))
    pi = float(np.float32(np.pi))
    return mx.array(0.0, mx.float32) + pi * _fma_cpu(3.0, rx + ry, -s)


def _inverse_cpu(s2c, G):
    """ MLX-op equivalent of the 'inverse' kernel (matrix.h inverse with clang's contraction). """
    m = s2c.reshape(G, 9)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = [m[:, k] for k in range(9)]
    a = _fms_cpu(m11, m22, m21, m12)
    b = _fms_cpu(m10, m22, m12, m20)
    c = _fms_cpu(m10, m21, m11, m20)
    det = _fma_cpu(m02, c, _fma_cpu(m00, a, -(m01 * b)))
    invdet = mx.array(1.0, mx.float32) / det
    cols = [a, _fms_cpu(m02, m21, m01, m22), _fms_cpu(m01, m12, m02, m11),
            _fms_cpu(m12, m20, m10, m22), _fms_cpu(m00, m22, m02, m20), _fms_cpu(m10, m02, m00, m12),
            _fms_cpu(m10, m21, m20, m11), _fms_cpu(m20, m01, m00, m21), _fms_cpu(m00, m11, m10, m01)]
    return mx.stack([col * invdet for col in cols], axis = 1).reshape(-1)


def _xform_box_cpu(s2c, box, n):
    """ MLX-op equivalent of the 'xform_box' kernel (aabb.h transform). """
    m = s2c.reshape(n, 9)
    b = box.reshape(n, 4)
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    cx = [x0, x0, x1, x1]
    cy = [y0, y1, y0, y1]
    xs, ys = [], []
    for k in range(4):
        def xf(i, j, o):
            return _fma_cpu(m[:, i], cx[k], m[:, j] * cy[k]) + m[:, o]
        t0, t1, t2 = xf(0, 1, 2), xf(3, 4, 5), xf(6, 7, 8)
        xs.append(t0 / t2)
        ys.append(t1 / t2)
    sx, sy = mx.stack(xs, axis = 1), mx.stack(ys, axis = 1)
    return mx.stack([sx.min(axis = 1), sy.min(axis = 1), sx.max(axis = 1), sy.max(axis = 1)],
                    axis = 1).reshape(-1)


# ---------------------------------------------------------------------------
# Topology

class SceneTopology:
    """
        Structure-only description of a serialized scene (see module doc).
        Host arrays are numpy; `*_m` members are cached MLX uploads.
    """
    pass


def topology_signature(args):
    """ Cheap structural key (candidates are then verified fully). """
    S, G = int(args[2]), int(args[3])
    return (len(args), int(args[0]), int(args[1]), S, G, _enum_int(args[-2]))


def _shape_code(st):
    if st is _SP:
        return SHAPE_PATH
    if st is _SC:
        return SHAPE_CIRCLE
    if st is _SE:
        return SHAPE_ELLIPSE
    if st is _SR:
        return SHAPE_RECT
    return _enum_int(st)


def _color_code(ct):
    if ct is _CC:
        return 0
    if ct is _CL:
        return 1
    if ct is _CR:
        return 2
    return _enum_int(ct)


def _walk(args):
    """ One pass over the args list: per-shape / per-group structure. """
    S, G = int(args[2]), int(args[3])
    k = 7
    stype = [0] * S
    pos0 = [0] * S
    p_shape, p_npts, p_nbp, p_thick, p_closed, p_approx = [], [], [], [], [], []
    for s in range(S):
        c = _shape_code(args[k])
        stype[s] = c
        pos0[s] = k + 1
        if c == SHAPE_PATH:
            th = args[k + 3]
            p_shape.append(s)
            p_nbp.append(args[k + 1].shape[0])
            p_npts.append(args[k + 2].shape[0])
            p_thick.append(th is not None)
            p_closed.append(bool(args[k + 4]))
            p_approx.append(bool(args[k + 5]))
            k += 7
        elif c in (SHAPE_CIRCLE, SHAPE_ELLIPSE, SHAPE_RECT):
            k += 4
        else:
            raise ValueError('unknown shape type %r' % (args[k],))
    sids_pos = [0] * G
    nsh = [0] * G
    col = [[-1] * G, [-1] * G, [0] * G, [-1] * G, [-1] * G, [0] * G]  # ftype fpos fn stype spos sn
    eo = [0] * G
    mpos = [0] * G
    for g in range(G):
        sids_pos[g] = k
        nsh[g] = args[k].size
        k += 1
        for w in (0, 3):
            ct = args[k]
            k += 1
            if ct is None:
                continue
            c = _color_code(ct)
            col[w][g] = c
            col[w + 1][g] = k
            if c == 0:
                k += 1
            else:
                col[w + 2][g] = args[k + 2].shape[0]
                k += 4
        eo[g] = 1 if args[k] else 0
        k += 1
        mpos[g] = k
        k += 1
    assert k + 2 == len(args), 'malformed scene args'
    return dict(S = S, G = G, stype = stype, pos0 = pos0, p_shape = p_shape, p_npts = p_npts,
                p_nbp = p_nbp, p_thick = p_thick, p_closed = p_closed, p_approx = p_approx,
                sids_pos = sids_pos, nsh = nsh, col = col, eo = eo, mpos = mpos)


_SIZE = operator.attrgetter('size')


def _matches(t, args, trust_int_identity = False):
    """
        Full structural verification of a cached topology:
        - non-array structural values (canvas size, counts, shape / colour
          types, None thicknesses and colours, is_closed, use_distance_approx,
          use_even_odd_rule, filter type) by identity-then-equality;
        - Python-float vs mx.array for stroke widths and the filter radius
          (changes the gradient map);
        - sizes of the variable-size arrays (points, thickness, ctrl, shape
          ids, gradient offsets / stop colours), only for objects that are not
          the cached ones (an in-place MLX update cannot change the size);
        - contents of the int arrays (ctrl, shape ids): always re-read, since
          in-place updates are undetectable otherwise; with
          trust_int_identity, only for objects that are not the cached ones.
    """
    if len(args) != t.nargs:
        return False
    vals = _pick(args, t.nonarr_pos)
    if not all(map(operator.is_, vals, t.nonarr_vals)):
        # enums, None, bools and small ints are normally the identical objects;
        # compare the rest by value, never calling mx.array.__eq__
        A = mx.array
        for a, b in zip(vals, t.nonarr_vals):
            if a is b:
                continue
            if type(a) is A or type(b) is A or type(a) is not type(b) or not (a == b):
                return False
    if tuple(map(type, _pick(args, t.swr_pos))) != t.swr_types:
        return False
    objs = _pick(args, t.struct_pos)
    same = t.struct_objs
    if not all(map(operator.is_, objs, same)):
        try:
            idx = [i for i, (a, b) in enumerate(zip(objs, same)) if a is not b]
            sizes = t.struct_sizes
            for i in idx:
                if objs[i].size != sizes[i]:
                    return False
        except AttributeError:
            return False
        t.struct_objs = objs
    iobjs = _pick(args, t.int_pos)
    if trust_int_identity and all(map(operator.is_, iobjs, t.int_objs)):
        return True
    if _join(iobjs, t.int_count, np.int32).tobytes() != t.int_bytes:
        return False
    t.int_objs = iobjs
    return True


def _read_structure(args, w):
    """
        Structure arrays (the input of _build_core) and argument positions of
        serialized scene args, from a _walk result. Reads the int arrays
        (num_control_points, shape ids) once.
    """
    S, G = w['S'], w['G']
    stype = np.asarray(w['stype'], np.int32).reshape(S)
    pos0 = np.asarray(w['pos0'], np.int64).reshape(S)
    is_path = stype == SHAPE_PATH
    p_shape = np.asarray(w['p_shape'], np.int64)
    P = len(p_shape)
    npts = np.asarray(w['p_npts'], np.int64).reshape(P)
    nbp = np.asarray(w['p_nbp'], np.int64).reshape(P)
    thick = np.asarray(w['p_thick'], bool).reshape(P)
    nsh = np.asarray(w['nsh'], np.int64).reshape(G)
    ftype, fpos, fn, sctype, spos, sn = [np.asarray(c, np.int64).reshape(G) for c in w['col']]
    mpos = np.asarray(w['mpos'], np.int64).reshape(G)
    sids_pos = np.asarray(w['sids_pos'], np.int64).reshape(G)
    Nseg, T = int(nbp.sum()), int(nsh.sum())
    int_pos = tuple(pos0[p_shape].tolist()) + tuple(sids_pos.tolist())
    int_objs = _pick(args, int_pos)
    ints_host = _join(int_objs, Nseg + T, np.int32)
    st = dict(W = int(args[0]), H = int(args[1]), filter_type = _enum_int(args[-2]), stype = stype,
              p_shape = p_shape, npts = npts, nbp = nbp, thick = thick,
              closed = np.asarray(w['p_closed'], bool).reshape(P),
              approx = np.asarray(w['p_approx'], bool).reshape(P),
              ctrl = ints_host[:Nseg], nsh = nsh, sids = ints_host[Nseg:],
              ftype = ftype, fn = np.where(ftype > 0, fn, 0),
              sctype = sctype, sn = np.where(sctype > 0, sn, 0),
              eo = np.asarray(w['eo'], np.int32).reshape(G))
    ncolargs_f = np.where(ftype < 0, 0, np.where(ftype == 0, 1, 4))
    ncolargs_s = np.where(sctype < 0, 0, np.where(sctype == 0, 1, 4))
    cstart = np.stack([fpos, spos], axis = 1).reshape(-1)
    ccount = np.stack([ncolargs_f, ncolargs_s], axis = 1).reshape(-1)
    nonpath = np.nonzero(~is_path)[0]
    pos = dict(pos0 = pos0, fpos = fpos, spos = spos, mpos = mpos, sids_pos = sids_pos,
               ncolargs_f = ncolargs_f, int_pos = int_pos, int_objs = int_objs,
               int_bytes = ints_host.tobytes(),
               # source positions (sources_from_args)
               filter_radius = (len(args) - 1,),
               sw = tuple(np.where(is_path, pos0 + 5, pos0 + 2).tolist()),
               par = tuple(np.stack([pos0[nonpath], pos0[nonpath] + 1], axis = 1).reshape(-1).tolist()),
               pts = tuple((pos0[p_shape] + 1).tolist()),
               th = tuple((pos0[p_shape] + 2)[thick].tolist()),
               col = tuple(_ranges(np.where(cstart < 0, 0, cstart), ccount).tolist()),
               mat = tuple(mpos.tolist()))
    return st, pos


def _build_core(st):
    """
        SceneTopology from structure arrays only (no scene arguments):
          W, H, filter_type (int); stype (S,) shape codes; per path (shape
          order): p_shape (shape ids), npts, nbp (segments), thick, closed,
          approx; ctrl (sum nbp,) num_control_points of all paths; per group:
          nsh, ftype / sctype (-1 none, 0 constant, 1 linear, 2 radial), fn / sn
          (stop counts, 0 unless gradient), eo; sids (sum nsh,) shape ids.
        Everything the pools need except the float parameters: int pools,
        float-pool offsets (t.lay) and the gather permutation, kernel inputs,
        BVH leaf payloads, ip. The float parameters arrive as the "sources"
        of build_pools_from_sources.
    """
    t = SceneTopology()
    W, H = int(st['W']), int(st['H'])
    t.canvas_width, t.canvas_height = W, H
    t.filter_type = int(st['filter_type'])
    stype = np.asarray(st['stype'], np.int32).reshape(-1)
    S = int(stype.size)
    p_shape = np.asarray(st['p_shape'], np.int64).reshape(-1)
    P = int(p_shape.size)
    npts = np.asarray(st['npts'], np.int64).reshape(P)
    nbp = np.asarray(st['nbp'], np.int64).reshape(P)
    thick = np.asarray(st['thick'], bool).reshape(P)
    closed = np.asarray(st['closed'], bool).reshape(P)
    approx = np.asarray(st['approx'], bool).reshape(P)
    nsh = np.asarray(st['nsh'], np.int64).reshape(-1)
    G = int(nsh.size)
    ftype, fn, sctype, sn = [np.asarray(st[k], np.int64).reshape(G) for k in ('ftype', 'fn', 'sctype', 'sn')]
    fn = np.where(ftype > 0, fn, 0)
    sn = np.where(sctype > 0, sn, 0)
    eo = np.asarray(st['eo'], np.int32).reshape(G)
    ctrl = np.asarray(st['ctrl'], np.int64).reshape(-1)
    sids = np.asarray(st['sids'], np.int64).reshape(-1)
    is_path = stype == SHAPE_PATH
    if S and ((stype < 0) | (stype > 3)).any():
        raise ValueError('unknown shape type')
    if not np.array_equal(np.nonzero(is_path)[0], p_shape):
        raise ValueError('path shape ids do not match the shape types')
    if ctrl.size != int(nbp.sum()) or sids.size != int(nsh.sum()):
        raise ValueError('structure sizes are inconsistent')
    if ((ftype < -1) | (ftype > 2) | (sctype < -1) | (sctype > 2)).any():
        raise ValueError('unknown colour type')
    t.num_shapes, t.num_groups = S, G
    empty_group = nsh == 0
    E = int(empty_group.sum())
    t.num_empty_groups = E
    T = int(nsh.sum())
    t.num_total_shapes = T
    Nseg = int(nbp.sum())
    Npts = int(npts.sum())
    t.num_paths, t.num_segments, t.num_points = P, Nseg, Npts
    if T and (sids.min() < 0 or sids.max() >= S):
        raise ValueError('shape id out of range')
    if Nseg and (ctrl.min() < 0 or ctrl.max() > 2):
        raise ValueError('num_control_points must be 0, 1 or 2')

    circ = stype == SHAPE_CIRCLE
    er = (stype == SHAPE_ELLIPSE) | (stype == SHAPE_RECT)
    psize = np.select([circ, stype == SHAPE_ELLIPSE, stype == SHAPE_RECT], [3, 4, 4], 0)
    t.num_params = int(psize.sum())
    pbase = np.cumsum(psize) - psize
    # serialized (radius, center) / (p_min, p_max) values -> shape_params (S, 4)
    # rows [a b c d]; index num_params is an appended zero
    p2s = np.full((S, 4), t.num_params, np.int64)
    p2s[circ, :3] = pbase[circ, None] + np.arange(3)
    p2s[er] = pbase[er, None] + np.arange(4)
    t.par_to_sp_m = mx.array(p2s.reshape(-1).astype(np.int32))

    def csize(ct, n):
        return np.where(ct < 0, 0, np.where(ct == 0, 4, 4 + 5 * n))
    fsz, ssz = csize(ftype, fn), csize(sctype, sn)
    t.num_col = int(fsz.sum() + ssz.sum())

    # ---- node pool (scene BVH, group BVHs, path BVHs)
    # A group without shapes (the C++ core does not support it: Scene
    # construction hangs) keeps its record, gets a 3-node dummy block whose
    # root (bvh_base + 2*0 - 2) is internal with both children pointing at an
    # empty-box node, an empty scene-BVH leaf box and no sample-table entries,
    # so no kernel ever reaches a shape through it.
    n_scene = 2 * G - 1 if G > 0 else 0
    n_group = np.where(nsh > 0, 2 * nsh - 1, 3)
    n_path = np.where(nbp > 0, 2 * nbp - 1, 0)
    group_base = n_scene + np.cumsum(n_group) - n_group
    NG_nodes = int(n_group.sum())
    path_base = n_scene + NG_nodes + np.cumsum(n_path) - n_path
    N = n_scene + NG_nodes + int(n_path.sum())
    t.num_nodes = N

    # ---- float pool layout
    F0 = 1 + 5 * N
    fsz_shape = np.full(S, 5, np.int64)
    fsz_path = 2 * npts + thick * npts + 2 * nbp
    fsz_shape[p_shape] += fsz_path
    f_off = F0 + np.cumsum(fsz_shape) - fsz_shape
    GF0 = F0 + int(fsz_shape.sum())
    gsz = fsz + ssz + 18
    g_off = GF0 + np.cumsum(gsz) - gsz
    fill_off = np.where(ftype < 0, -1, g_off)
    stroke_off = np.where(sctype < 0, -1, g_off + fsz)
    gf_off = g_off + fsz + ssz
    SF0 = GF0 + int(gsz.sum())
    NF = SF0 + 2 * T
    if S == 0:
        NF = 1
    points_off = f_off[p_shape] + 5
    th_off = np.where(thick, points_off + 2 * npts, -1)
    cdf_off = points_off + 2 * npts + thick * npts
    t.lay = dict(f_off = f_off, points_off = points_off, th_off = th_off, fill_off = fill_off,
                 stroke_off = stroke_off, gf_off = gf_off, fsz = fsz, ssz = ssz, stype = stype,
                 p_shape = p_shape, npts = npts, nbp = nbp, thick = thick, circ = circ, er = er,
                 ftype = ftype, fn = fn, sctype = sctype, sn = sn, ctrl = ctrl)
    t.max_segments_per_path = int(nbp.max()) if P else 0

    # ---- int pool layout
    I0 = 12 * S + 12 * G
    IP0 = I0 + 2 * N
    ctrl_off = IP0 + np.cumsum(2 * nbp) - 2 * nbp
    pid_off = ctrl_off + nbp
    IG0 = IP0 + 2 * Nseg
    sid_off = IG0 + np.cumsum(nsh) - nsh
    NI = IG0 + 3 * T
    if S == 0:
        NI = 0

    srec = np.zeros((S, 12), np.int32)
    srec[:, 0] = stype
    srec[:, 1] = f_off
    srec[:, [2, 4, 6, 7, 8, 9]] = -1
    srec[p_shape, 2] = points_off
    srec[p_shape, 3] = npts
    srec[p_shape, 4] = ctrl_off
    srec[p_shape, 5] = nbp
    srec[p_shape, 6] = th_off
    srec[p_shape, 7] = path_base
    srec[p_shape, 8] = cdf_off
    srec[p_shape, 9] = pid_off
    srec[p_shape, 10] = closed.astype(np.int32)
    srec[p_shape, 11] = approx.astype(np.int32)
    grec = np.zeros((G, 12), np.int32)
    grec[:, 0] = sid_off
    grec[:, 1] = nsh
    grec[:, 2] = ftype
    grec[:, 3] = fill_off
    grec[:, 4] = fn
    grec[:, 5] = sctype
    grec[:, 6] = stroke_off
    grec[:, 7] = sn
    grec[:, 8] = eo
    grec[:, 9] = np.where(empty_group, group_base + 2, group_base)
    grec[:, 10] = gf_off

    path_of_seg = np.repeat(np.arange(P), nbp)
    seg_start = np.cumsum(nbp) - nbp
    j_of_seg = np.arange(Nseg) - seg_start[path_of_seg]
    c1 = ctrl + 1
    cs = np.cumsum(c1) - c1
    pid = cs - cs[seg_start[path_of_seg]] if Nseg else cs
    tail_paths = np.zeros(2 * Nseg, np.int32)
    tail_paths[2 * seg_start[path_of_seg] + j_of_seg] = ctrl
    tail_paths[2 * seg_start[path_of_seg] + nbp[path_of_seg] + j_of_seg] = pid
    gid_of_member = np.repeat(np.arange(G), nsh)
    head = np.concatenate([srec.reshape(-1), grec.reshape(-1)])
    tail = np.concatenate([tail_paths, sids.astype(np.int32), sids.astype(np.int32),
                           gid_of_member.astype(np.int32)])
    t.ints_head_m = mx.array(head.astype(np.int32))
    t.ints_tail_m = mx.array(tail.astype(np.int32))
    t.ints_head_np = head.astype(np.int32)
    t.ints_tail_np = tail.astype(np.int32)

    # ---- float pool gather permutation for [F0, NF)
    # source vector: [0 | stroke_width (S) | shape_params (4S) | path lengths (P) |
    #   points | thickness | segment cdf | segment pmf | colours | shape_to_canvas (9G) |
    #   canvas_to_shape (9G) | sample cdf (T) | sample pmf (T)]
    SRC_ZERO = 0
    SRC_SW = 1
    SRC_SP = SRC_SW + S
    SRC_D = SRC_SP + 4 * S
    SRC_PTS = SRC_D + P
    Nth = int(npts[thick].sum())
    SRC_TH = SRC_PTS + 2 * Npts
    SRC_CDF = SRC_TH + Nth
    SRC_PMF = SRC_CDF + Nseg
    SRC_COL = SRC_PMF + Nseg
    SRC_S2C = SRC_COL + t.num_col
    SRC_C2S = SRC_S2C + 9 * G
    SRC_SCDF = SRC_C2S + 9 * G
    SRC_SPMF = SRC_SCDF + T
    t.num_thickness = Nth
    if S > 0:
        rest = NF - F0
        perm = np.full(rest, -1, np.int64)
        perm[f_off - F0] = SRC_SW + np.arange(S)
        sidx = np.arange(S, dtype = np.int64)
        pmat = np.full((S, 4), SRC_ZERO, np.int64)
        pmat[circ, :3] = SRC_SP + 4 * sidx[circ, None] + np.arange(3)
        pmat[er] = SRC_SP + 4 * sidx[er, None] + np.arange(4)
        pmat[p_shape, 3] = SRC_D + np.arange(P)
        perm[(f_off - F0)[:, None] + 1 + np.arange(4)] = pmat
        perm[_ranges(points_off - F0, 2 * npts)] = SRC_PTS + np.arange(2 * Npts)
        perm[_ranges(th_off[thick] - F0, npts[thick])] = SRC_TH + np.arange(Nth)
        perm[_ranges(cdf_off - F0, nbp)] = SRC_CDF + np.arange(Nseg)
        perm[_ranges(cdf_off + nbp - F0, nbp)] = SRC_PMF + np.arange(Nseg)
        coff = np.stack([fill_off, stroke_off], axis = 1).reshape(-1)
        csz = np.stack([fsz, ssz], axis = 1).reshape(-1)
        perm[_ranges(np.where(coff < 0, 0, coff) - F0, csz)] = SRC_COL + np.arange(t.num_col)
        perm[_ranges(gf_off - F0, np.full(G, 9))] = SRC_S2C + np.arange(9 * G)
        perm[_ranges(gf_off + 9 - F0, np.full(G, 9))] = SRC_C2S + np.arange(9 * G)
        perm[SF0 - F0 + np.arange(T)] = SRC_SCDF + np.arange(T)
        perm[SF0 + T - F0 + np.arange(T)] = SRC_SPMF + np.arange(T)
        assert (perm >= 0).all(), 'float pool layout has holes'
        t.float_perm_m = mx.array(perm.astype(np.int32))
    # sizes of the float sources (build_pools_from_sources)
    t.src_sizes = dict(filter_radius = 1, stroke_width = S, shape_params = 4 * S, points = 2 * Npts,
                       thickness = Nth, colors = t.num_col, shape_to_canvas = 9 * G)

    # ---- per-call kernel inputs (static parts)
    th_start = np.cumsum(np.where(thick, npts, 0)) - np.where(thick, npts, 0)
    pinfo = np.stack([np.cumsum(npts) - npts, seg_start, nbp, npts, np.where(thick, th_start, -1)],
                     axis = 1).astype(np.int32)
    t.pinfo_m = mx.array(pinfo.reshape(-1))
    t.ctrl_m = mx.array(ctrl.astype(np.int32))
    t.path_shape_m = mx.array(p_shape.astype(np.int32))
    # shape lengths (sample tables) and shape bboxes: gather from [circle, ellipse, rect, path]
    is_ell, is_rect = stype == SHAPE_ELLIPSE, stype == SHAPE_RECT
    nc, ne, nr = int(circ.sum()), int(is_ell.sum()), int(is_rect.sum())
    order_idx = np.zeros(S, np.int64)
    order_idx[circ] = np.arange(nc)
    order_idx[is_ell] = nc + np.arange(ne)
    order_idx[is_rect] = nc + ne + np.arange(nr)
    order_idx[p_shape] = nc + ne + nr + np.arange(P)
    t.shape_gather_m = mx.array(order_idx.astype(np.int32))
    t.circle_sid_m = mx.array(np.nonzero(circ)[0].astype(np.int32))
    t.ellipse_sid_m = mx.array(np.nonzero(is_ell)[0].astype(np.int32))
    t.rect_sid_m = mx.array(np.nonzero(is_rect)[0].astype(np.int32))
    t.counts = (nc, ne, nr)
    t.sample_sid_m = mx.array(sids.astype(np.int32))

    # ---- BVH leaves
    # path BVHs (paths with >= 1 segment, shape order)
    t.path_bvh_counts = nbp[nbp > 0].astype(np.int32)
    t.path_payload_m = mx.array(np.stack([j_of_seg, -(pid + 1)], axis = 1).astype(np.int32).reshape(-1, 2))
    # shape max stroke radius: thickness paths use their path BVH root
    path_build_index = np.cumsum(nbp > 0) - 1
    thick_rooted = thick & (nbp > 0)
    t.thick_root_build_idx = path_build_index[thick_rooted].astype(np.int64)
    smr_idx = np.arange(S, dtype = np.int64)
    smr_idx[p_shape[thick_rooted]] = S + np.arange(int(thick_rooted.sum()))
    t.smr_gather_m = mx.array(smr_idx.astype(np.int32))
    t.num_thick_rooted = int(thick_rooted.sum())
    # group BVHs
    has_stroke = sctype >= 0
    t.group_bvh_counts = nsh[~empty_group].astype(np.int32)
    t.group_payload_m = mx.array(np.stack([sids, np.full(T, -1)], axis = 1).astype(np.int32).reshape(-1, 2))
    t.member_has_stroke_m = mx.array(has_stroke[gid_of_member])
    if E > 0:
        ne_idx = np.nonzero(~empty_group)[0]
        t.nonempty_groups_m = mx.array(ne_idx.astype(np.int32))
        big = np.float32(1e30)
        t.empty_box_m = mx.array(np.tile(np.array([big, big, -big, -big], np.float32), (E, 1)))
        # scene leaves: [non-empty groups (build order), empty groups] -> group order
        leaf_perm = np.zeros(G, np.int64)
        leaf_perm[ne_idx] = np.arange(len(ne_idx))
        leaf_perm[empty_group] = len(ne_idx) + np.arange(E)
        t.scene_leaf_perm_m = mx.array(leaf_perm.astype(np.int32))
        # group node region: [group build rows, dummy rows] -> pool layout
        NGb = int((2 * nsh[~empty_group] - 1).sum())
        build_base = np.cumsum(2 * nsh[~empty_group] - 1) - (2 * nsh[~empty_group] - 1)
        region = np.zeros(NG_nodes, np.int64)
        gb = group_base - n_scene
        region[_ranges(gb[~empty_group], 2 * nsh[~empty_group] - 1)] = _ranges(build_base, 2 * nsh[~empty_group] - 1)
        region[_ranges(gb[empty_group], np.full(E, 3))] = NGb + np.arange(3 * E)
        t.group_region_perm_m = mx.array(region.astype(np.int32))
        dummy_f = np.tile(np.array([big, big, -big, -big, 0], np.float32), (3 * E, 1))
        dummy_i = np.tile(np.array([[0, 0], [-1, -1], [-1, -1]], np.int32), (E, 1))
        t.dummy_nodes_f_m = mx.array(dummy_f)
        t.dummy_nodes_i_m = mx.array(dummy_i)
    t.scene_bvh_counts = np.array([G], np.int32)
    t.scene_payload_m = mx.array(np.stack([np.arange(G), np.full(G, -1)], axis = 1).astype(np.int32).reshape(-1, 2))

    # ---- ip (structure part)
    ip = np.zeros(IP_SIZE, np.int32)
    ip[IP_W], ip[IP_H], ip[IP_NSX], ip[IP_NSY] = W, H, 1, 1
    ip[IP_CANVAS_W], ip[IP_CANVAS_H] = W, H
    ip[IP_FILTER_TYPE] = t.filter_type
    ip[IP_FILTER_RADIUS_OFF] = 0
    ip[IP_NUM_FLOATS] = NF
    if S > 0:
        ip[IP_NUM_SHAPES], ip[IP_NUM_GROUPS], ip[IP_NUM_TOTAL_SHAPES] = S, G, T
        ip[IP_SCENE_BVH_BASE] = 0
        ip[IP_SHAPES_I_OFF] = 0
        ip[IP_GROUPS_I_OFF] = 12 * S
        ip[IP_BVH_I_OFF] = I0
        ip[IP_SAMPLE_SHAPE_ID_OFF] = IG0 + T
        ip[IP_SAMPLE_GROUP_ID_OFF] = IG0 + 2 * T
        ip[IP_BVH_F_OFF] = 1
        ip[IP_SAMPLE_CDF_OFF] = SF0
        ip[IP_SAMPLE_PMF_OFF] = SF0 + T
        ip[IP_NUM_INTS] = NI
        ip[IP_NUM_BVH_NODES] = N
    t.ip = ip
    t.num_floats, t.num_ints = NF, NI
    return t


def _attach_args(t, args, st, pos):
    """ Serialized-args specifics of a topology: argument positions, cache verification, gradient map. """
    A = mx.array
    S, P = t.num_shapes, t.num_paths
    L = t.lay
    t.nargs = len(args)
    t.filter_radius_pos = len(args) - 1
    t.src_pos = {k: pos[k] for k in ('filter_radius', 'sw', 'par', 'pts', 'th', 'col', 'mat')}
    t.sw_pos = pos['sw']
    # stroke widths and the filter radius may be Python floats or mx.arrays (gradient map)
    t.swr_pos = t.sw_pos + (len(args) - 1,)
    t.swr_types = tuple(map(type, _pick(args, t.swr_pos)))
    pos0, p_shape, thick = pos['pos0'], L['p_shape'], L['thick']
    fpos, spos, mpos, sids_pos = pos['fpos'], pos['spos'], pos['mpos'], pos['sids_pos']
    ftype, sctype = L['ftype'], L['sctype']
    # verification positions of the non-array structural args (enums, None, bools, counts):
    # canvas/counts, shape types, path is_closed / use_distance_approx / absent thickness,
    # fill / stroke colour types, use_even_odd_rule, filter type
    nonarr = np.concatenate([np.arange(4), pos0 - 1, pos0[p_shape] + 3, pos0[p_shape] + 4,
                             (pos0[p_shape] + 2)[~thick], sids_pos + 1, sids_pos + 2 + pos['ncolargs_f'],
                             mpos - 1, [len(args) - 2]])
    t.nonarr_pos = tuple(np.sort(nonarr).tolist())
    t.nonarr_vals = _pick(args, t.nonarr_pos)
    t.mat_pos = pos['mat']
    # int arrays (structure): ctrl in path order, shape ids in group order
    t.int_pos = pos['int_pos']
    t.int_count = t.num_segments + t.num_total_shapes
    t.int_objs = pos['int_objs']
    t.int_bytes = pos['int_bytes']
    pts_pos = pos0[p_shape] + 1
    th_pos = pos0[p_shape] + 2
    offs_pos = np.concatenate([np.where(ftype > 0, fpos + 2, -1), np.where(sctype > 0, spos + 2, -1)])
    offs_pos = offs_pos[offs_pos >= 0]
    struct_pos = np.concatenate([pts_pos, th_pos[thick], offs_pos, offs_pos + 1])
    t.struct_pos = tuple(struct_pos.tolist())
    t.struct_objs = _pick(args, t.struct_pos)
    t.struct_sizes = tuple(map(_SIZE, t.struct_objs))

    # ---- gradient index map as contiguous ranges: (arg position, pool start, count)
    f_off, points_off, th_off = L['f_off'], L['points_off'], L['th_off']
    circ, er, npts = L['circ'], L['er'], L['npts']
    gp, gs, gl = [], [], []
    def add(p, start, count):
        gp.append(np.asarray(p, np.int64).reshape(-1))
        gs.append(np.asarray(start, np.int64).reshape(-1))
        gl.append(np.broadcast_to(np.asarray(count, np.int64), np.shape(np.asarray(p).reshape(-1))))
    if S > 0:
        add(pos0[circ], f_off[circ] + 1, 1)
        add(pos0[circ] + 1, f_off[circ] + 2, 2)
        add(pos0[er], f_off[er] + 1, 2)
        add(pos0[er] + 1, f_off[er] + 3, 2)
        add(pts_pos, points_off, 2 * npts)
        add(th_pos[thick], th_off[thick], npts[thick])
        sw_arr = np.fromiter(map(operator.is_, t.swr_types[:S], (A,) * S), bool, count = S)
        sw_mapped = sw_arr.copy()
        sw_mapped[p_shape[thick]] = False
        add(np.asarray(t.sw_pos, np.int64)[sw_mapped], f_off[sw_mapped], 1)
        for ct, cp, cn, co in ((ftype, fpos, L['fn'], L['fill_off']), (sctype, spos, L['sn'], L['stroke_off'])):
            cst = ct == 0
            add(cp[cst], co[cst], 4)
            gr = ct > 0
            add(cp[gr], co[gr], 2)
            add(cp[gr] + 1, co[gr] + 2, 2)
            add(cp[gr] + 2, co[gr] + 4, cn[gr])
            add(cp[gr] + 3, co[gr] + 4 + cn[gr], 4 * cn[gr])
        add(mpos, L['gf_off'], 9)
    if t.swr_types[-1] is A:
        add([t.filter_radius_pos], [0], 1)
    t.grad_pos = np.concatenate(gp) if gp else np.zeros(0, np.int64)
    t.grad_start = np.concatenate(gs) if gs else np.zeros(0, np.int64)
    t.grad_len = np.concatenate(gl) if gl else np.zeros(0, np.int64)
    order = np.argsort(t.grad_pos, kind = 'stable')
    t.grad_pos, t.grad_start, t.grad_len = t.grad_pos[order], t.grad_start[order], t.grad_len[order]
    t._grad_list = None


def _build_topology(args):
    w = _walk(args)
    st, pos = _read_structure(args, w)
    t = _build_core(st)
    _attach_args(t, args, st, pos)
    return t


_TOPO_CACHE = OrderedDict()
_TOPO_CACHE_MAX = 8


def clear_topology_cache():
    _TOPO_CACHE.clear()


def topology_from_args(args, use_cache = True, trust_int_identity = False):
    """
        SceneTopology of serialized scene args. With use_cache, a cached
        topology is returned when the structure matches one of the last
        _TOPO_CACHE_MAX topologies (see _matches: structural values, array
        sizes and int array contents are verified).
        trust_int_identity: skip re-reading the int arrays (num_control_points,
        shape_ids) that are the same objects as in the cached topology. Only
        safe when those arrays are never modified in place.
    """
    if not use_cache:
        return _build_topology(args)
    sig = topology_signature(args)
    for key in reversed(list(_TOPO_CACHE.keys())):   # most recently used first
        if key[0] == sig and _matches(_TOPO_CACHE[key], args, trust_int_identity):
            _TOPO_CACHE.move_to_end(key)
            return _TOPO_CACHE[key]
    t = _build_topology(args)
    _TOPO_CACHE[(sig, id(t))] = t
    while len(_TOPO_CACHE) > _TOPO_CACHE_MAX:
        _TOPO_CACHE.popitem(last = False)
    return t


def grad_index_map(topology):
    """
        Same format as metal_scene.primal_offsets: a list aligned with args,
        an int64 index array into the floats pool (C order) or None.
    """
    t = topology
    if t._grad_list is None:
        out = [None] * t.nargs
        for p, s, n in zip(t.grad_pos.tolist(), t.grad_start.tolist(), t.grad_len.tolist()):
            out[p] = np.arange(s, s + n, dtype = np.int64)
        t._grad_list = out
    return t._grad_list


# ---------------------------------------------------------------------------
# Pools

def _category(args, pos, count):
    """
        mx.float32 array of the concatenated args at `pos`, read fresh on
        every call (in-place updates of the same mx.array are not detectable
        cheaply; see _join).
    """
    if count == 0:
        return mx.zeros((0,), mx.float32)
    return mx.array(_join(_pick(args, pos), count, np.float32))


def _cat(arrays):
    arrays = [a.reshape(-1) for a in arrays if a.size > 0]
    if not arrays:
        return mx.zeros((0,), mx.float32)
    return arrays[0] if len(arrays) == 1 else mx.concatenate(arrays)


def _fill_call_slots(ip, width, height, nsx, nsy, seed, use_prefiltering, eval_count, has_background):
    ip = ip.copy()
    ip[IP_W], ip[IP_H], ip[IP_NSX], ip[IP_NSY] = width, height, nsx, nsy
    ip[IP_USE_PREFILTERING] = 1 if use_prefiltering else 0
    ip[IP_HAS_BACKGROUND] = 1 if has_background else 0
    ip[IP_NUM_EVAL] = eval_count
    ip[IP_SEED_LO], ip[IP_SEED_HI] = _seed_bits(seed)
    ip[IP_USE_EVAL_POSITIONS] = 1 if eval_count > 0 else 0
    return ip


SOURCE_KEYS = ('filter_radius', 'stroke_width', 'shape_params', 'points', 'thickness', 'colors',
               'shape_to_canvas')


def _sources(args, pos, t):
    radius = _category(args, pos['filter_radius'], 1)
    if t.num_shapes == 0:
        return dict(filter_radius = radius)
    par = _category(args, pos['par'], t.num_params)
    sz = t.src_sizes
    return dict(filter_radius = radius,
                stroke_width = _category(args, pos['sw'], sz['stroke_width']),
                shape_params = mx.concatenate([par, mx.zeros((1,), mx.float32)])[t.par_to_sp_m],
                points = _category(args, pos['pts'], sz['points']),
                thickness = _category(args, pos['th'], sz['thickness']),
                colors = _category(args, pos['col'], sz['colors']),
                shape_to_canvas = _category(args, pos['mat'], sz['shape_to_canvas']))


def sources_from_args(topology, args):
    """
        The float sources of build_pools_from_sources gathered from serialized
        scene args (one bytes join per category, read fresh on every call).
    """
    return _sources(args, topology.src_pos, topology)


def build_pools(topology, args, width, height, nsx, nsy, seed, use_prefiltering = False,
                eval_count = 0, has_background = False, refit_from = None):
    """ build_pools_from_sources on the float parameters of serialized args (structure = topology). """
    return build_pools_from_sources(topology, sources_from_args(topology, args), width, height, nsx,
                                    nsy, seed, use_prefiltering, eval_count, has_background, refit_from)


def build_pools_from_sources(topology, sources, width, height, nsx, nsy, seed, use_prefiltering = False,
                             eval_count = 0, has_background = False, refit_from = None):
    """
        Flat pools for a topology and its float parameters. sources: dict of
        flat mx.float32 arrays (sizes in topology.src_sizes):
          filter_radius (1), stroke_width (S), shape_params (4S; rows [a b c d]
          as the pool shape record: circle radius cx cy -, ellipse rx ry cx cy,
          rect pmin pmax, path unused), points (2 * total points, x y per point,
          paths in shape order), thickness (per-point widths of thickness
          paths), colors (fill then stroke colour block of each group, pool
          layout), shape_to_canvas (9G, row major).
        Only filter_radius is needed when the scene has no shapes.
        Returns a dict:
          ip      mx.int32[64]   (per-call slots 0-3, 10-14, 28 filled)
          ip_np   numpy copy of ip
          ints    mx.int32[max(64, NI)], floats mx.float32[max(64, NF)] (lazy)
          num_ints, num_floats
          total_length  mx scalar: unnormalised total boundary length
                        (edge sampling needs > 0), lazy
          bvh     (path_build, group_build, scene_build) BVHBuild or None
          leaves  per-kind (boxes, radii) used by the builds (for refit)
        refit_from: a previous result for the same topology: the BVH leaf
        order and topology are reused (refit_bvhs) instead of rebuilding. The
        traversal stays valid but may differ from the CPU.
        Everything stays lazy. The assembly runs on the CPU or the GPU stream
        (set_scene_build_device); both give bit-identical pools.
    """
    use_cpu = _use_cpu_build(topology)
    if use_cpu:
        with mx.stream(mx.cpu):
            return _pools_impl(topology, sources, width, height, nsx, nsy, seed, use_prefiltering,
                               eval_count, has_background, refit_from, True)
    return _pools_impl(topology, sources, width, height, nsx, nsy, seed, use_prefiltering,
                       eval_count, has_background, refit_from, False)


def _pools_impl(topology, sources, width, height, nsx, nsy, seed, use_prefiltering, eval_count,
                has_background, refit_from, use_cpu):
    t = topology
    S, G, T, P = t.num_shapes, t.num_groups, t.num_total_shapes, t.num_paths
    ip = _fill_call_slots(t.ip, width, height, nsx, nsy, seed, use_prefiltering, eval_count,
                          has_background)
    radius = sources['filter_radius'].reshape(1)
    out = dict(ip_np = ip, ip = mx.array(ip), num_ints = t.num_ints, num_floats = t.num_floats,
               topology = t, bvh = None, leaves = None)
    if S == 0:
        out['floats'] = _pad(radius, MIN_POOL)
        out['ints'] = mx.zeros((MIN_POOL,), mx.int32)
        out['total_length'] = mx.array(0.0)
        return out
    sw = sources['stroke_width'].reshape(-1)
    sp = sources['shape_params'].reshape(-1)
    pts = sources['points'].reshape(-1)
    th = sources['thickness'].reshape(-1)
    col = sources['colors'].reshape(-1)
    s2c = sources['shape_to_canvas'].reshape(-1)

    # paths: lengths, CDFs, bboxes, segment leaves
    nc, ne, nr = t.counts
    if P > 0:
        sw_path = sw[t.path_shape_m]
        if use_cpu:
            path_len, path_box, seg_pmf, seg_cdf, seg_box, seg_r = _paths_cpu(t, pts, th, sw_path)
        else:
            seg_f, path_f = _run('paths', [t.pinfo_m, t.ctrl_m, pts, th, sw_path], P,
                                 [7 * t.num_segments, 5 * P], [mx.float32, mx.float32])
            seg_f = seg_f.reshape(-1, 7)
            path_f = path_f.reshape(P, 5)
            path_len = path_f[:, 0]
            path_box = path_f[:, 1:5]
            seg_pmf, seg_cdf = seg_f[:, 0], seg_f[:, 1]
            seg_box, seg_r = seg_f[:, 2:6], seg_f[:, 6]
    else:
        path_len = mx.zeros((0,), mx.float32)
        path_box = mx.zeros((0, 4), mx.float32)
        seg_pmf = seg_cdf = mx.zeros((0,), mx.float32)

    # shape lengths and bboxes (non-path shapes)
    lens, boxes = [], []
    two_pi = mx.array(np.float32(2.0 * np.pi))
    zero = mx.array(0.0, dtype = mx.float32)
    spm = sp.reshape(S, 4)
    if nc:
        c = spm[t.circle_sid_m]
        r, cx, cy = c[:, 0], c[:, 1], c[:, 2]
        lens.append(zero + two_pi * r)
        boxes.append(mx.stack([cx - r, cy - r, cx + r, cy + r], axis = 1))
    if ne:
        c = spm[t.ellipse_sid_m]
        rx, ry, cx, cy = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        lens.append(_ellipse_cpu(rx, ry) if use_cpu else
                    _run('ellipse', [mx.stack([rx, ry], axis = 1)], ne, [ne], [mx.float32])[0])
        boxes.append(mx.stack([cx - rx, cy - ry, cx + rx, cy + ry], axis = 1))
    if nr:
        c = spm[t.rect_sid_m]
        x0, y0, x1, y1 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        two = mx.array(2.0, dtype = mx.float32)
        lens.append(zero + two * (((x1 - x0) + y1) - y0))
        boxes.append(mx.stack([x0, y0, x1, y1], axis = 1))
    lens.append(path_len)
    boxes.append(path_box)
    shape_len = _cat(lens)[t.shape_gather_m]
    shape_box = mx.concatenate([b.reshape(-1, 4) for b in boxes], axis = 0)[t.shape_gather_m]

    # sample tables
    if T > 0:
        L = shape_len[t.sample_sid_m]
        raw = mx.cumsum(L) if use_cpu else \
            _run('cumsum', [L, mx.array(np.array([T] + [0] * 7, np.int32))], 1, [T], [mx.float32])[0]
        norm = raw[T - 1]
        nz = norm != 0
        scdf = mx.where(nz, raw / norm, zero)
        spmf = mx.where(nz, L / norm, zero)
        out['total_length'] = norm
    else:
        scdf = spmf = mx.zeros((0,), mx.float32)
        out['total_length'] = mx.array(0.0)

    if G == 0:
        c2s = mx.zeros((0,), mx.float32)
    elif use_cpu:
        c2s = _inverse_cpu(s2c, G)
    else:
        c2s = _run('inverse', [s2c], G, [9 * G], [mx.float32])[0]

    # BVHs
    prev = refit_from['bvh'] if refit_from is not None else None
    def build(kind, idx, boxes_, radii_, payload, key, counts):
        if prev is not None and prev[idx] is not None:
            nf = sgb.refit_bvhs(prev[idx], boxes_, radii_)
            b = prev[idx]
            return sgb.BVHBuild(nf, b.nodes_i, b.base, b.root, b.perm, b.plan)
        return sgb.build_bvhs(boxes_, radii_, payload, None, key, counts)
    half = mx.array(0.5, dtype = mx.float32)
    path_build = group_build = scene_build = None
    leaves = {}
    if len(t.path_bvh_counts):
        key = half * (seg_box[:, 1] + seg_box[:, 3])
        path_build = build('path', 0, seg_box, seg_r, t.path_payload_m, key, t.path_bvh_counts)
        leaves['path'] = (seg_box, seg_r)
    if t.num_thick_rooted:
        roots = mx.array(path_build.root[t.thick_root_build_idx].astype(np.int32))
        smr = mx.concatenate([sw, path_build.nodes_f[roots, 4]])[t.smr_gather_m]
    else:
        smr = sw
    group_nodes = None
    if G > 0:
        E = t.num_empty_groups
        NGE = G - E
        if NGE > 0:
            gbox = shape_box[t.sample_sid_m]
            gr = mx.where(t.member_has_stroke_m, smr[t.sample_sid_m], zero)
            key = sgb.morton2d_keys(half * (gbox[:, :2] + gbox[:, 2:]), t.canvas_width, t.canvas_height)
            group_build = build('group', 1, gbox, gr, t.group_payload_m, key, t.group_bvh_counts)
            leaves['group'] = (gbox, gr)
            groots = mx.array(group_build.root.astype(np.int32))
            groot_f = group_build.nodes_f[groots]
            s2c_ne = s2c if E == 0 else s2c.reshape(G, 9)[t.nonempty_groups_m]
            sbox = (_xform_box_cpu(s2c_ne, groot_f[:, :4], NGE) if use_cpu else
                    _run('xform_box', [s2c_ne, groot_f[:, :4]], NGE, [4 * NGE], [mx.float32])[0]).reshape(NGE, 4)
            sr = groot_f[:, 4]
            gnf, gni = group_build.nodes_f, group_build.nodes_i
        else:
            sbox = mx.zeros((0, 4), mx.float32)
            sr = mx.zeros((0,), mx.float32)
            gnf = mx.zeros((0, 5), mx.float32)
            gni = mx.zeros((0, 2), mx.int32)
        if E > 0:
            sbox = mx.concatenate([sbox, t.empty_box_m])[t.scene_leaf_perm_m]
            sr = mx.concatenate([sr, mx.zeros((E,), mx.float32)])[t.scene_leaf_perm_m]
            gnf = mx.concatenate([gnf, t.dummy_nodes_f_m])[t.group_region_perm_m]
            gni = mx.concatenate([gni, t.dummy_nodes_i_m])[t.group_region_perm_m]
        group_nodes = (gnf, gni)
        key = sgb.morton2d_keys(half * (sbox[:, :2] + sbox[:, 2:]), t.canvas_width, t.canvas_height)
        scene_build = build('scene', 2, sbox, sr, t.scene_payload_m, key, t.scene_bvh_counts)
        leaves['scene'] = (sbox, sr)
    parts = []
    if scene_build is not None:
        parts.append((scene_build.nodes_f, scene_build.nodes_i))
    if group_nodes is not None:
        parts.append(group_nodes)
    if path_build is not None:
        parts.append((path_build.nodes_f, path_build.nodes_i))
    nodes_f = _cat([p[0] for p in parts])
    nodes_i = mx.concatenate([p[1].reshape(-1) for p in parts]) if parts else mx.zeros((0,), mx.int32)

    src = _cat([zero.reshape(1), sw, sp, path_len, pts, th, seg_cdf, seg_pmf, col, s2c, c2s, scdf, spmf])
    floats = mx.concatenate([radius, nodes_f, src[t.float_perm_m]])
    ints = mx.concatenate([t.ints_head_m, nodes_i, t.ints_tail_m])
    out['floats'] = _pad(floats, MIN_POOL)
    out['ints'] = _pad(ints, MIN_POOL)
    out['bvh'] = (path_build, group_build, scene_build)
    out['leaves'] = leaves
    return out
