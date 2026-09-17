"""
    GPU backend for diffvg: colour forward and backward passes.

    The flat scene pools (layout in pydiffvg/metal/common.metal, mirrored by
    pydiffvg/cuda/common.cu) come either from pydiffvg/scene_gpu.py (MLX
    arrays built on the GPU; the default; for packed scenes,
    pydiffvg/packed.py, straight from the params arrays) or from the C++
    core's diffvg.Scene.export_flat() (numpy); every function taking pools
    accepts both (see render_mlx._scene_pools).

    They are rasterised by kernels built through pydiffvg/gpu_backend.py,
    which selects Apple Metal (pydiffvg/metal/*.metal, mx.fast.metal_kernel)
    or NVIDIA CUDA (pydiffvg/cuda/*.cu, mx.fast.cuda_kernel). The kernel
    bodies below are written ONCE, in the backend-neutral dialect documented
    in gpu_backend.py: plain C plus DVG_THREAD_INDEX (thread index and, on
    CUDA, the mandatory bounds check), DVG_ADD / DVG_STORE (atomics),
    DVG_XYZ (float4 -> float3 swizzle) and DVG_CEIL. Nothing else in a body
    differs between the two backends.

    Kernels: weight, render_color (forward); render_color_backward,
    sample_boundary, render_edge (backward: interior colour gradients, filter
    radius, background, boundary/edge-sampling gradients, translation image).
    Signed distance output and prefiltering: render_metal_stage3.py.
"""
import os
import numpy as np
import mlx.core as mx

from . import render_mlx
from . import metal_scene as ms
from . import gpu_backend as gb

# Kept for callers that used it; the live source directory is per backend.
_METAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metal')
# Extension-free base names; gpu_backend picks metal/*.metal or cuda/*.cu.
_HEADER_FILES = ('common', 'geometry', 'color')
_BACKWARD_HEADER_FILES = _HEADER_FILES + ('backward',)
_MIN_BUFFER = 8  # on Metal, smaller inputs arrive in the `constant` address space

# ip slot used by the backward kernels only (slots 32-63 are free): 1 when the
# screen-space translation gradient image is requested.
IP_WANT_D_TRANSLATION = 32
# Boundary sample record strides (sample_boundary kernel outputs)
BS_F_STRIDE = 13
BS_I_STRIDE = 5


def _load_header(files = _HEADER_FILES):
    return gb.header(files)


def reset_kernel_cache():
    """ Forget the loaded kernel sources and kernels (after editing them). """
    gb.reset_kernel_cache()


_WEIGHT_SOURCE = """
    DVG_THREAD_INDEX;
    SceneView s = make_scene_view(floats, ints, ip);
    int width = ip[IP_W];
    int height = ip[IP_H];
    int x = 0;
    int y = 0;
    float2 pt = sample_position(s, int(idx), x, y);
    int ftype = ip[IP_FILTER_TYPE];
    float radius = filter_radius(s);
    int ri = int(DVG_CEIL(radius));
    for (int dy = -ri; dy <= ri; dy++) {
        for (int dx = -ri; dx <= ri; dx++) {
            int xx = x + dx;
            int yy = y + dy;
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                float xc = xx + 0.5f;
                float yc = yy + 0.5f;
                float w = compute_filter_weight(ftype, radius, xc - pt.x, yc - pt.y);
                DVG_ADD(weight_image, yy * width + xx, w);
            }
        }
    }
"""

_WEIGHT_D_RADIUS_SOURCE = """
    DVG_THREAD_INDEX;
    SceneView s = make_scene_view(floats, ints, ip);
    int width = ip[IP_W];
    int height = ip[IP_H];
    int x = 0;
    int y = 0;
    float2 pt = sample_position(s, int(idx), x, y);
    int ftype = ip[IP_FILTER_TYPE];
    float radius = filter_radius(s);
    int ri = int(DVG_CEIL(radius));
    for (int dy = -ri; dy <= ri; dy++) {
        for (int dx = -ri; dx <= ri; dx++) {
            int xx = x + dx;
            int yy = y + dy;
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                float xc = xx + 0.5f;
                float yc = yy + 0.5f;
                float dw = filter_weight_d_radius(ftype, radius, xc - pt.x, yc - pt.y);
                DVG_ADD(d_weight_image, yy * width + xx, dw);
            }
        }
    }
"""

_RENDER_COLOR_SOURCE = """
    DVG_THREAD_INDEX;
    SceneView s = make_scene_view(floats, ints, ip);
    int width = ip[IP_W];
    int height = ip[IP_H];
    int x = 0;
    int y = 0;
    float2 pt = sample_position(s, int(idx), x, y);
    // normalize pt to [0, 1]
    float2 npt = pt;
    npt.x /= width;
    npt.y /= height;
    bool has_bg = ip[IP_HAS_BACKGROUND] != 0;
    float4 bg = float4(0);
    if (has_bg) {
        int b = 4 * (y * width + x);
        bg = float4(background[b], background[b + 1], background[b + 2], background[b + 3]);
    }
    float4 color = sample_color_scene(s, has_bg, bg, npt);
    int ftype = ip[IP_FILTER_TYPE];
    float radius = filter_radius(s);
    int ri = int(DVG_CEIL(radius));
    for (int dy = -ri; dy <= ri; dy++) {
        for (int dx = -ri; dx <= ri; dx++) {
            int xx = x + dx;
            int yy = y + dy;
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                float weight_sum = weight_image[yy * width + xx];
                if (weight_sum > 0) {
                    float xc = xx + 0.5f;
                    float yc = yy + 0.5f;
                    float fw = compute_filter_weight(ftype, radius, xc - pt.x, yc - pt.y);
                    float4 wc = fw * color / weight_sum;
                    int o = 4 * (yy * width + xx);
                    DVG_ADD(render_image, o, wc[0]);
                    DVG_ADD(render_image, o + 1, wc[1]);
                    DVG_ADD(render_image, o + 2, wc[2]);
                    DVG_ADD(render_image, o + 3, wc[3]);
                }
            }
        }
    }
"""

_RENDER_COLOR_BACKWARD_SOURCE = """
    DVG_THREAD_INDEX;
    SceneView s = make_scene_view(floats, ints, ip);
    int width = ip[IP_W];
    int height = ip[IP_H];
    int x = 0;
    int y = 0;
    float2 pt = sample_position(s, int(idx), x, y);
    float2 npt = pt;
    npt.x /= width;
    npt.y /= height;
    bool has_bg = ip[IP_HAS_BACKGROUND] != 0;
    bool want_translation = ip[IP_WANT_D_TRANSLATION] != 0;
    int pix = y * width + x;
    float4 bg = float4(0);
    if (has_bg) {
        bg = float4(background[4 * pix], background[4 * pix + 1],
                    background[4 * pix + 2], background[4 * pix + 3]);
    }
    float4 d_color = gather_d_color(s, d_render_image, weight_image, pt);

    Fragment fragments[MAX_HIT_SHAPES];
    int num_fragments = 0;
    float2 cpt = float2(npt.x * float(ip[IP_CANVAS_W]), npt.y * float(ip[IP_CANVAS_H]));
    if (ip[IP_NUM_GROUPS] > 0) {
        EdgeQuery eq;
        eq.shape_group_id = -1;
        eq.shape_id = -1;
        eq.hit = false;
        num_fragments = collect_fragments(s, cpt, false, eq, fragments);
    }
    float4 color = float4(0);
    float2 d_translation_px = float2(0);
    if (num_fragments <= 0) {
        if (has_bg) {
            color = bg;
            // Every sample of the pixel contributes: accumulate (as the CPU does).
            for (int k = 0; k < 4; k++) {
                DVG_ADD(d_background, 4 * pix + k, d_color[k]);
            }
        }
    } else {
        BlendState st;
        color = blend_forward(fragments, num_fragments, has_bg, bg, d_color, st);
        GradWrites g;
        for (int i = num_fragments - 1; i >= 0; i--) {
            float4 d_ci = blend_backward_step(fragments, i, st);
            int gid = fragments[i].group_id;
            gw_clear(g);
            if (fragments[i].is_stroke) {
                d_sample_color_param(s, group_stroke_type(s, gid), group_stroke_off(s, gid),
                                     group_stroke_num_stops(s, gid), cpt, d_ci, g, d_translation_px);
            } else {
                d_sample_color_param(s, group_fill_type(s, gid), group_fill_off(s, gid),
                                     group_fill_num_stops(s, gid), cpt, d_ci, g, d_translation_px);
            }
            for (int k = 0; k < g.n; k++) {
                DVG_ADD(d_floats, g.idx[k], g.val[k]);
            }
        }
        if (has_bg) {
            DVG_ADD(d_background, 4 * pix, st.d_curr_color.x);
            DVG_ADD(d_background, 4 * pix + 1, st.d_curr_color.y);
            DVG_ADD(d_background, 4 * pix + 2, st.d_curr_color.z);
            DVG_ADD(d_background, 4 * pix + 3, st.d_curr_alpha);
        }
    }
    if (want_translation) {
        DVG_ADD(d_translation, 2 * pix, d_translation_px.x);
        DVG_ADD(d_translation, 2 * pix + 1, d_translation_px.y);
    }
    // Filter radius (render_kernel splat loop)
    int ftype = ip[IP_FILTER_TYPE];
    float radius = filter_radius(s);
    int ri = int(DVG_CEIL(radius));
    float d_radius = 0;
    for (int dy = -ri; dy <= ri; dy++) {
        for (int dx = -ri; dx <= ri; dx++) {
            int xx = x + dx;
            int yy = y + dy;
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                int p = yy * width + xx;
                float weight_sum = weight_image[p];
                if (weight_sum > 0) {
                    float xc = xx + 0.5f;
                    float yc = yy + 0.5f;
                    float fw = compute_filter_weight(ftype, radius, xc - pt.x, yc - pt.y);
                    float4 d_pixel = float4(d_render_image[4 * p], d_render_image[4 * p + 1],
                                            d_render_image[4 * p + 2], d_render_image[4 * p + 3]);
                    // pixel = sum_i w_i c_i / W: d pixel / d r = sum_i (c_i - pixel) w_i' / W.
                    // The -pixel term is distributed over the samples as
                    // c_i w_i W' / W^2, with W' = d_weight_image (weight_d_radius kernel).
                    float dpc = dot(d_pixel, color);
                    d_radius += dpc / weight_sum *
                        (filter_weight_d_radius(ftype, radius, xc - pt.x, yc - pt.y) -
                         fw * d_weight_image[p] / weight_sum);
                }
            }
        }
    }
    DVG_ADD(d_floats, ip[IP_FILTER_RADIUS_OFF], d_radius);
"""

_SAMPLE_BOUNDARY_SOURCE = """
    DVG_THREAD_INDEX;
    SceneView s = make_scene_view(floats, ints, ip);
    int fo = BS_F_STRIDE * int(idx);
    int io = BS_I_STRIDE * int(idx);
    // Edge sampling needs a non-empty scene with non-zero total boundary
    // length (the normalised sample CDF ends in 1, else it is all zeros).
    // Checked here so that the host never has to read the pools.
    int nts = ip[IP_NUM_TOTAL_SHAPES];
    if (ip[IP_NUM_GROUPS] <= 0 || nts <= 0 || !(floats[ip[IP_SAMPLE_CDF_OFF] + nts - 1] > 0.0f)) {
        DVG_STORE(bs_i, io + 1, -1);
        return;
    }
    BoundarySample b;
    bool ok = generate_boundary_sample(s, int(idx), b);
    if (!ok) {
        DVG_STORE(bs_i, io + 1, -1);
        return;
    }
    DVG_STORE(bs_f, fo, b.pt.x);
    DVG_STORE(bs_f, fo + 1, b.pt.y);
    DVG_STORE(bs_f, fo + 2, b.local_pt.x);
    DVG_STORE(bs_f, fo + 3, b.local_pt.y);
    DVG_STORE(bs_f, fo + 4, b.normal.x);
    DVG_STORE(bs_f, fo + 5, b.normal.y);
    DVG_STORE(bs_f, fo + 6, b.local_normal.x);
    DVG_STORE(bs_f, fo + 7, b.local_normal.y);
    DVG_STORE(bs_f, fo + 8, b.local_velocity_scale);
    DVG_STORE(bs_f, fo + 9, b.t);
    DVG_STORE(bs_f, fo + 10, b.data.path.t);
    DVG_STORE(bs_f, fo + 11, b.pdf);
    DVG_STORE(bs_f, fo + 12, b.data.path.offset_dir);
    DVG_STORE(bs_i, io, b.shape_group_id);
    DVG_STORE(bs_i, io + 1, b.shape_id);
    DVG_STORE(bs_i, io + 2, b.data.path.base_point_id);
    DVG_STORE(bs_i, io + 3, b.data.path.point_id);
    DVG_STORE(bs_i, io + 4, b.data.is_stroke ? 1 : 0);
"""

_RENDER_EDGE_SOURCE = """
    DVG_THREAD_INDEX;
    SceneView s = make_scene_view(floats, ints, ip);
    int fo = BS_F_STRIDE * int(idx);
    int io = BS_I_STRIDE * int(idx);
    int shape_id = bs_i[io + 1];
    if (shape_id < 0) {
        return;
    }
    int width = ip[IP_W];
    int height = ip[IP_H];
    float2 boundary_pt = float2(bs_f[fo], bs_f[fo + 1]);
    float2 local_boundary_pt = float2(bs_f[fo + 2], bs_f[fo + 3]);
    float2 normal = float2(bs_f[fo + 4], bs_f[fo + 5]);
    float2 local_normal = float2(bs_f[fo + 6], bs_f[fo + 7]);
    float local_velocity_scale = bs_f[fo + 8];
    float t = bs_f[fo + 9];
    float pdf = bs_f[fo + 11];
    int shape_group_id = bs_i[io];
    BoundaryData data;
    data.path.base_point_id = bs_i[io + 2];
    data.path.point_id = bs_i[io + 3];
    data.path.t = bs_f[fo + 10];
    data.path.offset_dir = bs_f[fo + 12];
    data.is_stroke = bs_i[io + 4] != 0;
    if (!(pdf > 0)) {
        return;
    }
    int bx = int(boundary_pt.x * width);
    int by = int(boundary_pt.y * height);
    if (bx < 0 || bx >= width || by < 0 || by >= height) {
        return;
    }
    bool has_bg = ip[IP_HAS_BACKGROUND] != 0;
    int pix = by * width + bx;
    float4 bg = float4(0);
    if (has_bg) {
        bg = float4(background[4 * pix], background[4 * pix + 1],
                    background[4 * pix + 2], background[4 * pix + 3]);
    }
    EdgeQuery inside_query;
    inside_query.shape_group_id = shape_group_id;
    inside_query.shape_id = shape_id;
    inside_query.hit = false;
    EdgeQuery outside_query = inside_query;
    // Both sides through one call site (measured ~20% faster than two calls)
    float4 color_inside = float4(0);
    float4 color_outside = float4(0);
    for (int side = 0; side < 2; side++) {
        EdgeQuery q = inside_query;
        float2 qp = side == 0 ? boundary_pt - 1e-4f * normal : boundary_pt + 1e-4f * normal;
        float4 c = sample_color_scene_eq(s, has_bg, bg, qp, true, q);
        if (side == 0) {
            color_inside = c;
            inside_query = q;
        } else {
            color_outside = c;
            outside_query = q;
        }
    }
    if (!inside_query.hit && !outside_query.hit) {
        return;
    }
    if (!inside_query.hit) {
        normal = -normal;
        local_normal = -local_normal;
        float4 tmp = color_inside;
        color_inside = color_outside;
        color_outside = tmp;
    }
    float2 sboundary_pt = float2(boundary_pt.x * width, boundary_pt.y * height);
    float4 d_color = gather_d_color(s, d_render_image, weight_image, sboundary_pt);
    d_color /= float(ip[IP_CANVAS_W] * ip[IP_CANVAS_H]);
    float contrib = dot(color_inside - color_outside, d_color) / pdf;
    GradWrites g;
    gw_clear(g);
    accumulate_boundary_gradient(s, shape_id, contrib, t, local_normal, data, shape_group_id,
                                 local_boundary_pt, normal, local_velocity_scale, g);
    for (int k = 0; k < g.n; k++) {
        DVG_ADD(d_floats, g.idx[k], g.val[k]);
    }
    if (ip[IP_WANT_D_TRANSLATION] != 0) {
        DVG_ADD(d_translation, 2 * pix, normal.x * contrib);
        DVG_ADD(d_translation, 2 * pix + 1, normal.y * contrib);
    }
"""

# Compile-time constants the backward kernel bodies reference; gpu_backend
# spells them for each backend (constant / static constexpr).
_BACKWARD_CONSTANTS = (('IP_WANT_D_TRANSLATION', IP_WANT_D_TRANSLATION),
                       ('BS_F_STRIDE', BS_F_STRIDE),
                       ('BS_I_STRIDE', BS_I_STRIDE))

# name -> (header files, header constants, input names, output names, body)
_KERNEL_DEFS = {
    'weight': (_HEADER_FILES, (),
               ['ip', 'ints', 'floats'], ['weight_image'], _WEIGHT_SOURCE),
    'weight_d_radius': (_HEADER_FILES, (),
                        ['ip', 'ints', 'floats'], ['d_weight_image'], _WEIGHT_D_RADIUS_SOURCE),
    'render_color': (_HEADER_FILES, (),
                     ['ip', 'ints', 'floats', 'weight_image', 'background'],
                     ['render_image'], _RENDER_COLOR_SOURCE),
    'render_color_backward': (_BACKWARD_HEADER_FILES, _BACKWARD_CONSTANTS,
                              ['ip', 'ints', 'floats', 'weight_image', 'd_weight_image',
                               'd_render_image', 'background'],
                              ['d_floats', 'd_background', 'd_translation'],
                              _RENDER_COLOR_BACKWARD_SOURCE),
    'sample_boundary': (_BACKWARD_HEADER_FILES, _BACKWARD_CONSTANTS,
                        ['ip', 'ints', 'floats'], ['bs_f', 'bs_i'], _SAMPLE_BOUNDARY_SOURCE),
    'render_edge': (_BACKWARD_HEADER_FILES, _BACKWARD_CONSTANTS,
                    ['ip', 'ints', 'floats', 'bs_f', 'bs_i', 'weight_image',
                     'd_render_image', 'background'],
                    ['d_floats', 'd_translation'], _RENDER_EDGE_SOURCE),
}


def _kernel(name):
    """ The kernel for the active GPU backend (built once, cached there). """
    try:
        files, constants, inputs, outputs, source = _KERNEL_DEFS[name]
    except KeyError:
        raise KeyError(name)
    return gb.kernel(name, inputs, outputs, source, files, constants)


def _pad(a, n = _MIN_BUFFER):
    """ Flat array with at least n elements (zero padded). """
    a = a.reshape(-1)
    if a.size >= n:
        return a
    return mx.concatenate([a, mx.zeros((n - a.size,), dtype = a.dtype)])


def _seed_bits(seed):
    s = int(seed) & 0xFFFFFFFFFFFFFFFF
    lo = np.array([s & 0xFFFFFFFF], dtype = np.uint32).view(np.int32)[0]
    hi = np.array([s >> 32], dtype = np.uint32).view(np.int32)[0]
    return int(lo), int(hi)


def _threadgroup(n, kernel = None):
    """ Launch block for n threads (Metal 256; CUDA smaller, see gpu_backend). """
    return gb.threadgroup(n, kernel)


def is_supported(output_type, use_prefiltering, eval_positions):
    """ Whether the Metal forward pass implements this configuration. """
    if output_type != render_mlx.OutputType.color:
        return False
    if use_prefiltering:
        return False
    if eval_positions is not None and eval_positions.shape[0] > 0:
        return False
    return True


def _pool_m(a, dtype):
    """ Kernel-ready pool: mx arrays are used as they are (no host round trip). """
    if isinstance(a, mx.array):
        return _pad(a if a.dtype == dtype else a.astype(dtype))
    return _pad(mx.array(np.asarray(a, dtype = np.float32 if dtype == mx.float32 else np.int32)))


def prepare_flat(ip, ints, floats, width, height, num_samples_x, num_samples_y,
                 seed, background_image = None):
    """
        Prepares flattened pools for the kernels and fills the per-render ip
        slots; computes the (lazy) weight image. ip: host int32 array. ints /
        floats: numpy arrays (export_flat; uploaded here) or mx arrays
        (scene_gpu.build_pools; used as they are). Returns a dict reused by
        the forward colour pass and by the backward kernels:
        ip (numpy), ip_m, ints_m, floats_m, bg_m, weight_image, width, height,
        num_samples, has_background, has_edges.
    """
    ip = np.array(ip, dtype = np.int32, copy = True)
    ip[ms.IP_W] = width
    ip[ms.IP_H] = height
    ip[ms.IP_NSX] = num_samples_x
    ip[ms.IP_NSY] = num_samples_y
    ip[ms.IP_USE_PREFILTERING] = 0
    ip[ms.IP_HAS_BACKGROUND] = 1 if background_image is not None else 0
    ip[ms.IP_NUM_EVAL] = 0
    ip[ms.IP_SEED_LO], ip[ms.IP_SEED_HI] = _seed_bits(seed)
    ip[ms.IP_USE_EVAL_POSITIONS] = 0
    ip[IP_WANT_D_TRANSLATION] = 0

    ints_m = _pool_m(ints, mx.int32)
    floats_m = _pool_m(floats, mx.float32)
    if background_image is not None:
        bg = background_image
        if not isinstance(bg, mx.array):
            bg = mx.array(np.asarray(bg, dtype = np.float32))
        assert bg.shape[0] == height and bg.shape[1] == width and bg.shape[2] == 4
        bg_m = _pad(bg.astype(mx.float32))
    else:
        bg_m = mx.zeros((_MIN_BUFFER,), dtype = mx.float32)

    num_pixels = width * height
    num_samples = num_pixels * num_samples_x * num_samples_y
    # Edge sampling needs a non-empty scene with non-zero total boundary length.
    # The structural part is decided here; zero total length is detected by
    # the sample_boundary kernel itself (no pool read-back).
    num_total_shapes = int(ip[ms.IP_NUM_TOTAL_SHAPES])
    has_edges = int(ip[ms.IP_NUM_GROUPS]) > 0 and num_total_shapes > 0
    if has_edges and not isinstance(floats, mx.array):
        floats_np = np.asarray(floats, dtype = np.float32)
        has_edges = float(floats_np[int(ip[ms.IP_SAMPLE_CDF_OFF]) + num_total_shapes - 1]) > 0
    flat = dict(ip = ip, ints = ints, has_edges = has_edges, ip_m = _pad(mx.array(ip)), ints_m = ints_m,
                floats_m = floats_m, bg_m = bg_m, width = width, height = height,
                num_samples = num_samples, has_background = background_image is not None,
                weight_image = None)
    if num_samples > 0:
        flat['weight_image'] = _kernel('weight')(
            inputs = [flat['ip_m'], ints_m, floats_m],
            grid = (num_samples, 1, 1),
            output_shapes = [(max(num_pixels, _MIN_BUFFER),)],
            output_dtypes = [mx.float32],
            init_value = 0)[0]
    return flat


def render_color_prepared(flat):
    """ Colour forward pass on prepare_flat() pools; lazy (height, width, 4). """
    width, height, num_samples = flat['width'], flat['height'], flat['num_samples']
    if num_samples == 0:
        return mx.zeros((height, width, 4), dtype = mx.float32)
    render_image = _kernel('render_color')(
        inputs = [flat['ip_m'], flat['ints_m'], flat['floats_m'], flat['weight_image'], flat['bg_m']],
        grid = (num_samples, 1, 1),
        output_shapes = [(width * height * 4,)],
        output_dtypes = [mx.float32],
        init_value = 0)[0]
    return render_image.reshape(height, width, 4)


def render_color_flat(ip, ints, floats, width, height, num_samples_x, num_samples_y,
                      seed, background_image = None):
    """
        Runs the weight and colour kernels on flattened pools (numpy arrays
        from export_flat). ip is copied and its per-render slots filled.
        background_image: None or an (height, width, 4) array.
        Returns a lazy mx.array (height, width, 4).
    """
    return render_color_prepared(prepare_flat(ip, ints, floats, width, height, num_samples_x,
                                              num_samples_y, seed, background_image))


def weight_d_radius_image(flat):
    """
        d(weight_image)/d(filter radius) on prepare_flat() pools (lazy,
        float32[max(width * height, 8)]); computed on first use and cached in
        flat. Needed by the filter radius gradient (the normalisation by the
        weight image depends on the radius).
    """
    d_w = flat.get('d_weight_image')
    if d_w is None:
        num_samples = flat['num_samples']
        d_w = flat['d_weight_image'] = _kernel('weight_d_radius')(
            inputs = [flat['ip_m'], flat['ints_m'], flat['floats_m']],
            grid = (num_samples, 1, 1),
            output_shapes = [(max(flat['width'] * flat['height'], _MIN_BUFFER),)],
            output_dtypes = [mx.float32],
            init_value = 0)[0]
    return d_w


def _sort_boundary_samples(bs_f, bs_i, num_samples):
    """
        Reorders boundary-sample records by the Morton code of their screen
        position. render_edge reads only its own record, so this changes just
        the order of gradient accumulation, but coherent access to the scene
        pools makes the kernel much faster (tiger at 4x4: 3.56 s -> 0.39 s).
        Invalid samples (shape_id -1, pt 0) sort to the front and are skipped.
    """
    nf = BS_F_STRIDE * num_samples
    ni = BS_I_STRIDE * num_samples
    records_f = bs_f[:nf].reshape(num_samples, BS_F_STRIDE)
    records_i = bs_i[:ni].reshape(num_samples, BS_I_STRIDE)
    q = mx.clip((records_f[:, :2] * 1024).astype(mx.int32), 0, 1023)
    code = mx.zeros((num_samples,), dtype = mx.int32)
    for b in range(10):
        code = code | (((q[:, 0] >> b) & 1) << (2 * b + 1)) | (((q[:, 1] >> b) & 1) << (2 * b))
    order = mx.argsort(code)
    sorted_f = mx.take(records_f, order, axis = 0).reshape(-1)
    sorted_i = mx.take(records_i, order, axis = 0).reshape(-1)
    if bs_f.size > nf:
        sorted_f = mx.concatenate([sorted_f, bs_f[nf:]])
    if bs_i.size > ni:
        sorted_i = mx.concatenate([sorted_i, bs_i[ni:]])
    return sorted_f, sorted_i


def render_backward_prepared(flat, d_render_image, want_translation = False, edges = True):
    """
        Backward colour pass on prepare_flat() pools (the forward pass's pools
        and weight image). d_render_image: (height, width, 4) cotangent.
        Returns (d_floats (lazy, float32[len(floats pool)]), d_background
        ((height, width, 4) or None), d_translation ((height, width, 2) or None)).
        d_floats mirrors the floats pool layout (metal_scene.primal_offsets).
    """
    width, height, num_samples = flat['width'], flat['height'], flat['num_samples']
    nf = int(flat['floats_m'].size)
    has_bg = flat['has_background']
    if num_samples == 0:
        return (mx.zeros((nf,), dtype = mx.float32),
                mx.zeros((height, width, 4), dtype = mx.float32) if has_bg else None,
                mx.zeros((height, width, 2), dtype = mx.float32) if want_translation else None)
    ip_m = flat['ip_m']
    if want_translation:
        ip = np.array(flat['ip'], copy = True)
        ip[IP_WANT_D_TRANSLATION] = 1
        ip_m = _pad(mx.array(ip))
    ip = flat['ip']
    num_pixels = width * height
    d_img = _pad(mx.array(d_render_image).astype(mx.float32) if not isinstance(d_render_image, mx.array)
                 else d_render_image.astype(mx.float32))
    grid = (num_samples, 1, 1)
    tr_size = max(2 * num_pixels, _MIN_BUFFER) if want_translation else _MIN_BUFFER
    bg_size = max(4 * num_pixels, _MIN_BUFFER) if has_bg else _MIN_BUFFER
    d_weight_image = weight_d_radius_image(flat)
    d_floats, d_background, d_translation = _kernel('render_color_backward')(
        inputs = [ip_m, flat['ints_m'], flat['floats_m'], flat['weight_image'], d_weight_image,
                  d_img, flat['bg_m']],
        grid = grid,
        output_shapes = [(nf,), (bg_size,), (tr_size,)],
        output_dtypes = [mx.float32, mx.float32, mx.float32],
        init_value = 0)
    # Edge sampling (skipped for empty scenes and zero total boundary length,
    # like the CPU)
    if edges and flat['has_edges']:
        bs_f, bs_i = _kernel('sample_boundary')(
            inputs = [ip_m, flat['ints_m'], flat['floats_m']],
            grid = grid,
            output_shapes = [(max(BS_F_STRIDE * num_samples, _MIN_BUFFER),),
                             (max(BS_I_STRIDE * num_samples, _MIN_BUFFER),)],
            output_dtypes = [mx.float32, mx.int32],
            init_value = 0)
        bs_f, bs_i = _sort_boundary_samples(bs_f, bs_i, num_samples)
        d_floats_e, d_translation_e = _kernel('render_edge')(
            inputs = [ip_m, flat['ints_m'], flat['floats_m'], bs_f, bs_i, flat['weight_image'],
                      d_img, flat['bg_m']],
            grid = grid,
            output_shapes = [(nf,), (tr_size,)],
            output_dtypes = [mx.float32, mx.float32],
            init_value = 0)
        d_floats = d_floats + d_floats_e
        if want_translation:
            d_translation = d_translation + d_translation_e
    d_bg = d_background[:4 * num_pixels].reshape(height, width, 4) if has_bg else None
    d_tr = d_translation[:2 * num_pixels].reshape(height, width, 2) if want_translation else None
    return d_floats, d_bg, d_tr


def prepare_scene_gpu(scene_args, width, height, num_samples_x, num_samples_y, seed,
                      background_image, built = None):
    """
        Builds the C++ scene (unless `built` is given), flattens it and
        prepares the GPU pools and weight image. Returns (built, flat), with
        flat None when the configuration is not supported on Metal.
    """
    if built is None:
        built = render_mlx._build_scene(scene_args)
    scene, output_type, use_prefiltering, eval_positions, keep_alive = built
    if not is_supported(output_type, use_prefiltering, eval_positions):
        return built, None
    if background_image is not None:
        if background_image.shape[2] == 3:
            raise NotImplementedError('Background image must have 4 channels, not 3. Add a fourth channel with all ones via mx.ones().')
        assert background_image.shape[0] == height and background_image.shape[1] == width
    ip, ints, floats = ms.flatten_scene(scene)
    flat = prepare_flat(ip, ints, floats, width, height, num_samples_x, num_samples_y,
                        seed, background_image)
    return built, flat


def render_forward_gpu(scene_args, width, height, num_samples_x, num_samples_y, seed,
                       background_image, scene_cache = None):
    """
        Forward pass on the GPU. Same contract as render_mlx._forward (args
        is the output of serialize_scene; returns (height, width, 4) or the
        CPU result for configurations that are not supported on Metal yet).
        If scene_cache is given, the built scene is stored under 'scene' and
        the GPU pools + weight image under 'flat' for the backward pass.
    """
    built, flat = prepare_scene_gpu(scene_args, width, height, num_samples_x, num_samples_y,
                                    seed, background_image)
    if scene_cache is not None:
        scene_cache['scene'] = built
        if flat is not None:
            scene_cache['flat'] = flat
    if flat is None:
        return render_mlx._forward_built(width, height, num_samples_x, num_samples_y,
                                         seed, background_image, built)
    return render_color_prepared(flat)


class GradGather:
    """
        Maps a d_floats pool to per-argument gradients with one gather and one
        split. Built once per scene structure from contiguous pool ranges
        (arg position, pool start, count), e.g. scene_gpu's topology.grad_pos /
        grad_start / grad_len or ranges_from_offsets(metal_scene.primal_offsets).
    """
    __slots__ = ('nargs', 'pos', 'index_m', 'split')

    def __init__(self, nargs, pos, start, count):
        pos = np.asarray(pos, np.int64).reshape(-1)
        start = np.asarray(start, np.int64).reshape(-1)
        count = np.asarray(count, np.int64).reshape(-1)
        order = np.argsort(pos, kind = 'stable')
        pos, start, count = pos[order], start[order], count[order]
        self.nargs = nargs
        self.pos = pos.tolist()
        total = int(count.sum())
        ends = np.cumsum(count)
        index = np.repeat(start - ends + count, count) + np.arange(total, dtype = np.int64) \
            if total > 0 else np.zeros(0, np.int64)
        self.index_m = mx.array(index.astype(np.int32)) if total > 0 else None
        self.split = ends[:-1].tolist()

    def __call__(self, d_floats):
        """ Flat (1-D) gradients aligned with the args, None where unmapped. """
        out = [None] * self.nargs
        if not self.pos:
            return out
        if self.index_m is None:
            parts = [mx.zeros((0,), dtype = mx.float32)] * len(self.pos)
        else:
            g = d_floats[self.index_m]
            parts = mx.split(g, self.split) if len(self.pos) > 1 else [g]
        for p, d in zip(self.pos, parts):
            out[p] = d
        return out


def ranges_from_offsets(offsets):
    """ metal_scene.primal_offsets list -> (pos, start, count) arrays. """
    pos, start, count = [], [], []
    for k, idx in enumerate(offsets):
        if idx is None:
            continue
        pos.append(k)
        start.append(int(idx[0]) if idx.size else 0)
        count.append(int(idx.size))
    return pos, start, count


def grads_from_d_floats(flat, scene_args, d_floats):
    """
        Splits a d_floats pool into per-argument gradients aligned with
        scene_args (None where the argument has no pool storage).
        flat: a dict with export_flat's 'ip' and 'ints' (numpy).
    """
    offs = ms.primal_offsets(flat['ip'], flat['ints'], scene_args)
    grads = GradGather(len(scene_args), *ranges_from_offsets(offs))(d_floats)
    return [None if d is None else d.reshape(scene_args[k].shape) for k, d in enumerate(grads)]


def render_grad_gpu(grad_img, scene_args, width, height, num_samples_x, num_samples_y, seed,
                    background_image):
    """
        pydiffvg.RenderFunction.render_grad on the GPU: the screen-space
        translation gradient image (height, width, 2), or None when the
        configuration is not supported on Metal.
    """
    built, flat = prepare_scene_gpu(scene_args, width, height, num_samples_x, num_samples_y,
                                    seed, background_image)
    if flat is None:
        return None
    _, _, d_tr = render_backward_prepared(flat, grad_img, want_translation = True)
    mx.eval(d_tr)
    return d_tr
