"""
    GPU kernels for diffvg stage 3: signed distance output (forward +
    backward, also at eval_positions), prefiltered colour (forward +
    backward), and the screen-space translation gradient image.

    Built through pydiffvg/gpu_backend.py for either Apple Metal or NVIDIA
    CUDA; the kernel bodies below are written once in the backend-neutral
    dialect documented there (DVG_THREAD_INDEX / DVG_ADD / DVG_STORE /
    DVG_XYZ / DVG_CEIL).

    Shared device code: pydiffvg/metal/{common, geometry, color, backward,
    distance_grad, prefilter}.metal, mirrored by pydiffvg/cuda/*.cu.
    The *_flat functions take pools from
    scene_gpu.build_pools / build_pools_from_sources (mx arrays; the latter
    is also used by the packed-parameter API, pydiffvg/packed.py) or
    export_flat (numpy), as in
    render_metal.py; the scene-argument level helpers (flat_scene, *_gpu)
    build the scene with the C++ core.

    Gradients come back as `d_floats`, a pool with the layout of `floats`
    (map to the serialized arguments with metal_scene.primal_offsets).

    eval_positions: only for OutputType.sdf. The CPU core does not support
    colour output at eval positions (render() has no weight image then and
    render_kernel dereferences it in the splat loop), so neither does this.
"""
import os
import numpy as np
import mlx.core as mx

from . import render_mlx
from . import render_metal as rm
from . import metal_scene as ms
from . import gpu_backend as gb

# Extension-free base names; gpu_backend picks metal/*.metal or cuda/*.cu.
_HEADER_FILES = ('common', 'geometry', 'color', 'backward', 'distance_grad', 'prefilter')
_MIN = rm._MIN_BUFFER

# opts[] slots (prefilter source, PF_OPT_*)
OPT_WANT_TRANSLATION = 0
OPT_EXACT_SCAN = 1


def _load_header():
    return gb.header(_HEADER_FILES)


def reset_kernel_cache():
    gb.reset_kernel_cache()


# ---------------------------------------------------------------------------
# Kernel sources

# Sample position (grid mode) or eval position; canvas-space point `cpt`.
_POSITION = """
    DVG_THREAD_INDEX;
    SceneView s = make_scene_view(floats, ints, ip);
    int width = ip[IP_W];
    int height = ip[IP_H];
    bool use_eval = ip[IP_USE_EVAL_POSITIONS] != 0;
    int x = 0;
    int y = 0;
    float2 pt = float2(0);
    if (use_eval) {
        pt = float2(eval_positions[2 * idx], eval_positions[2 * idx + 1]);
        x = int(pt.x);
        y = int(pt.y);
    } else {
        pt = sample_position(s, int(idx), x, y);
    }
    float2 cpt = float2(pt.x / width * float(ip[IP_CANVAS_W]),
                        pt.y / height * float(ip[IP_CANVAS_H]));
    float weight = use_eval ? 1.f : 1.f / float(ip[IP_NSX] * ip[IP_NSY]);
"""

_SDF_FORWARD_SOURCE = _POSITION + """
    if (ip[IP_NUM_GROUPS] > 0) {
        SdfResult r;
        if (sdf_query(s, cpt, r)) {
            float dist = r.distance * weight;
            if (r.inside) {
                dist = -dist;
            }
            int o = use_eval ? int(idx) : y * width + x;
            DVG_ADD(sdf_image, o, dist);
        }
    }
"""

_SDF_BACKWARD_SOURCE = _POSITION + """
    float d_dist = use_eval ? d_sdf[idx] : d_sdf[y * width + x];
    if (ip[IP_NUM_GROUPS] > 0 && d_dist != 0) {
        SdfResult r;
        if (sdf_query(s, cpt, r)) {
            float d_abs_dist = weight * (r.inside ? -d_dist : d_dist);
            GradWrites g;
            gw_clear(g);
            float2 d_t = float2(0);
            d_compute_distance(s, r.group_id, r.shape_id, cpt, r.closest_pt, r.info, d_abs_dist, g, d_t);
            for (int k = 0; k < g.n; k++) {
                DVG_ADD(d_floats, g.idx[k], g.val[k]);
            }
            if (opts[PF_OPT_WANT_TRANSLATION] != 0 && x >= 0 && x < width && y >= 0 && y < height) {
                int o = 2 * (y * width + x);
                DVG_ADD(d_translation, o, d_t.x);
                DVG_ADD(d_translation, o + 1, d_t.y);
            }
        }
    }
"""

_PREFILTER_COMMON = """
    DVG_THREAD_INDEX;
    SceneView s = make_scene_view(floats, ints, ip);
    int width = ip[IP_W];
    int height = ip[IP_H];
    int x = 0;
    int y = 0;
    float2 pt = sample_position(s, int(idx), x, y);
    float2 cpt = float2(pt.x / width * float(ip[IP_CANVAS_W]),
                        pt.y / height * float(ip[IP_CANVAS_H]));
    bool has_bg = ip[IP_HAS_BACKGROUND] != 0;
    float4 bg = float4(0);
    if (has_bg) {
        int b = 4 * (y * width + x);
        bg = float4(background[b], background[b + 1], background[b + 2], background[b + 3]);
    }
    PrefilterFragment fragments[PF_MAX_FRAGMENTS];
    float3 accum_color[PF_MAX_FRAGMENTS];
    float accum_alpha[PF_MAX_FRAGMENTS];
    int num_fragments = pf_collect_fragments(s, cpt, opts[PF_OPT_EXACT_SCAN] != 0, fragments);
    float4 color = pf_blend(fragments, num_fragments, has_bg, bg, accum_color, accum_alpha);
    int ftype = ip[IP_FILTER_TYPE];
    float radius = filter_radius(s);
    int ri = int(DVG_CEIL(radius));
"""

_PREFILTER_FORWARD_SOURCE = _PREFILTER_COMMON + """
    for (int dy = -ri; dy <= ri; dy++) {
        for (int dx = -ri; dx <= ri; dx++) {
            int xx = x + dx;
            int yy = y + dy;
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                float weight_sum = weight_image[yy * width + xx];
                if (weight_sum > 0) {
                    float fw = compute_filter_weight(ftype, radius, xx + 0.5f - pt.x, yy + 0.5f - pt.y);
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

_PREFILTER_BACKWARD_SOURCE = _PREFILTER_COMMON + """
    float4 d_color = gather_d_color(s, d_render_image, weight_image, pt);
    float2 d_t = float2(0);
    float3 d_curr_color = DVG_XYZ(d_color);
    float d_curr_alpha = d_color.w;
    if (num_fragments > 0) {
        float final_alpha = color.w;
        if (final_alpha > 1e-6f) {
            d_curr_color = DVG_XYZ(d_color) / final_alpha;
            d_curr_alpha -= dot(DVG_XYZ(d_color), DVG_XYZ(color)) / final_alpha;
        }
        float first_alpha = has_bg ? bg.w : 0.f;
        float3 first_color = has_bg ? DVG_XYZ(bg) : float3(0);
        GradWrites g;
        for (int i = num_fragments - 1; i >= 0; i--) {
            float prev_alpha = i > 0 ? accum_alpha[i - 1] : first_alpha;
            float3 prev_color = i > 0 ? accum_color[i - 1] : first_color;
            float a = fragments[i].alpha;
            float d_prev_alpha = d_curr_alpha * (1.f - a);
            float d_alpha_i = d_curr_alpha * (1.f - prev_alpha) +
                              dot(d_curr_color, fragments[i].color - prev_color);
            float3 d_prev_color = d_curr_color * (1 - a);
            float3 d_color_i = d_curr_color * a;
            gw_clear(g);
            pf_d_fragment(s, fragments[i], cpt, d_color_i, d_alpha_i, g, d_t);
            for (int k = 0; k < g.n; k++) {
                DVG_ADD(d_floats, g.idx[k], g.val[k]);
            }
            d_curr_color = d_prev_color;
            d_curr_alpha = d_prev_alpha;
        }
    }
    if (has_bg) {
        int b = 4 * (y * width + x);
        DVG_ADD(d_background, b, d_curr_color.x);
        DVG_ADD(d_background, b + 1, d_curr_color.y);
        DVG_ADD(d_background, b + 2, d_curr_color.z);
        DVG_ADD(d_background, b + 3, d_curr_alpha);
    }
    if (opts[PF_OPT_WANT_TRANSLATION] != 0) {
        int o = 2 * (y * width + x);
        DVG_ADD(d_translation, o, d_t.x);
        DVG_ADD(d_translation, o + 1, d_t.y);
    }
    // Backprop to the filter weights; the weight fw of this sample is unused here
    float d_radius = 0;
    for (int dy = -ri; dy <= ri; dy++) {
        for (int dx = -ri; dx <= ri; dx++) {
            int xx = x + dx;
            int yy = y + dy;
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                float weight_sum = weight_image[yy * width + xx];
                if (weight_sum > 0) {
                    float ddx = xx + 0.5f - pt.x;
                    float ddy = yy + 0.5f - pt.y;
                    float fw = compute_filter_weight(ftype, radius, ddx, ddy);
                    int o = 4 * (yy * width + xx);
                    float4 d_pixel = float4(d_render_image[o], d_render_image[o + 1],
                                            d_render_image[o + 2], d_render_image[o + 3]);
                    // pixel = sum_i w_i c_i / sum_i w_i  =>  d w_i = d_pixel . (c_i - pixel) / sum_i w_i
                    // (render_kernel's formula is wrong; this is the corrected one)
                    float4 pixel = float4(render_image[o], render_image[o + 1],
                                          render_image[o + 2], render_image[o + 3]);
                    float d_weight = dot(d_pixel, color - pixel) / weight_sum;
                    d_radius += d_weight * pf_d_filter_weight_dr(ftype, radius, ddx, ddy);
                }
            }
        }
    }
    DVG_ADD(d_floats, ip[IP_FILTER_RADIUS_OFF], d_radius);
"""

_KERNEL_DEFS = {
    'sdf_forward': (['ip', 'ints', 'floats', 'eval_positions', 'opts'], ['sdf_image'], _SDF_FORWARD_SOURCE),
    'sdf_backward': (['ip', 'ints', 'floats', 'eval_positions', 'opts', 'd_sdf'],
                     ['d_floats', 'd_translation'], _SDF_BACKWARD_SOURCE),
    'prefilter_forward': (['ip', 'ints', 'floats', 'weight_image', 'background', 'opts'],
                          ['render_image'], _PREFILTER_FORWARD_SOURCE),
    'prefilter_backward': (['ip', 'ints', 'floats', 'weight_image', 'background', 'opts', 'd_render_image',
                            'render_image'],
                           ['d_floats', 'd_background', 'd_translation'], _PREFILTER_BACKWARD_SOURCE),
}


def _kernel(name):
    """ The kernel for the active GPU backend (built once, cached there). """
    inputs, outputs, source = _KERNEL_DEFS[name]
    return gb.kernel(name, inputs, outputs, source, _HEADER_FILES)


# ---------------------------------------------------------------------------
# Flat-pool level (numpy pools from export_flat)

def _fill_ip(ip, width, height, nsx, nsy, seed, use_prefiltering, has_background, num_eval):
    ip = np.array(ip, dtype = np.int32, copy = True)
    ip[ms.IP_W] = width
    ip[ms.IP_H] = height
    ip[ms.IP_NSX] = nsx
    ip[ms.IP_NSY] = nsy
    ip[ms.IP_USE_PREFILTERING] = 1 if use_prefiltering else 0
    ip[ms.IP_HAS_BACKGROUND] = 1 if has_background else 0
    ip[ms.IP_NUM_EVAL] = num_eval
    ip[ms.IP_SEED_LO], ip[ms.IP_SEED_HI] = rm._seed_bits(seed)
    ip[ms.IP_USE_EVAL_POSITIONS] = 1 if num_eval > 0 else 0
    return rm._pad(mx.array(ip))


def _pools(ints, floats):
    """ numpy pools are uploaded; mx pools (scene_gpu.build_pools) are used as they are """
    return rm._pool_m(ints, mx.int32), rm._pool_m(floats, mx.float32)


def _opts(want_translation = False, exact_scan = False):
    o = np.zeros(_MIN, dtype = np.int32)
    o[OPT_WANT_TRANSLATION] = 1 if want_translation else 0
    o[OPT_EXACT_SCAN] = 1 if exact_scan else 0
    return mx.array(o)


def _as_mx(a, dtype = mx.float32):
    if not isinstance(a, mx.array):
        a = mx.array(np.asarray(a, dtype = np.float32))
    return a.astype(dtype)


def _eval_array(eval_positions):
    """ (mx eval buffer padded, N) """
    if eval_positions is None or eval_positions.shape[0] == 0:
        return mx.zeros((_MIN,), dtype = mx.float32), 0
    ev = _as_mx(eval_positions)
    return rm._pad(ev), int(ev.shape[0])


def sdf_forward_flat(ip, ints, floats, width, height, nsx, nsy, seed, eval_positions = None,
                     use_prefiltering = False):
    """ use_prefiltering only moves the samples to sub-pixel centres (as render_kernel does). """
    ev, n_eval = _eval_array(eval_positions)
    ip_m = _fill_ip(ip, width, height, nsx, nsy, seed, use_prefiltering, False, n_eval)
    ints_m, floats_m = _pools(ints, floats)
    n = n_eval if n_eval > 0 else width * height * nsx * nsy
    out_n = n_eval if n_eval > 0 else width * height
    if n == 0:
        return mx.zeros((out_n, 1) if n_eval > 0 else (height, width, 1), dtype = mx.float32)
    out = _kernel('sdf_forward')(
        inputs = [ip_m, ints_m, floats_m, ev, _opts()],
        grid = (n, 1, 1),
        output_shapes = [(max(out_n, _MIN),)], output_dtypes = [mx.float32],
        init_value = 0)[0]
    out = out[:out_n]
    return out.reshape(n_eval, 1) if n_eval > 0 else out.reshape(height, width, 1)


def sdf_backward_flat(ip, ints, floats, width, height, nsx, nsy, seed, d_sdf,
                      eval_positions = None, want_translation = False, use_prefiltering = False):
    """ Returns (d_floats (len(floats),), d_translation (H, W, 2) or None). """
    ev, n_eval = _eval_array(eval_positions)
    ip_m = _fill_ip(ip, width, height, nsx, nsy, seed, use_prefiltering, False, n_eval)
    ints_m, floats_m = _pools(ints, floats)
    n = n_eval if n_eval > 0 else width * height * nsx * nsy
    nf = int(floats_m.size)
    if n == 0:
        return (mx.zeros((nf,)), mx.zeros((height, width, 2)) if want_translation else None)
    d_sdf_m = rm._pad(_as_mx(d_sdf))
    nt = max(2 * width * height, _MIN) if want_translation else _MIN
    d_floats, d_trans = _kernel('sdf_backward')(
        inputs = [ip_m, ints_m, floats_m, ev, _opts(want_translation), d_sdf_m],
        grid = (n, 1, 1),
        output_shapes = [(nf,), (nt,)], output_dtypes = [mx.float32, mx.float32],
        init_value = 0)
    return d_floats, (d_trans[:2 * width * height].reshape(height, width, 2) if want_translation else None)


def _background_m(background_image, width, height):
    if background_image is None:
        return mx.zeros((_MIN,), dtype = mx.float32)
    bg = _as_mx(background_image)
    if bg.shape[2] == 3:
        raise NotImplementedError('Background image must have 4 channels, not 3. Add a fourth channel with all ones via mx.ones().')
    assert bg.shape[0] == height and bg.shape[1] == width and bg.shape[2] == 4
    return rm._pad(bg)


def _weight_image(ip_m, ints_m, floats_m, n, width, height):
    # render_metal's weight kernel; sample_position honours IP_USE_PREFILTERING
    return rm._kernel('weight')(
        inputs = [ip_m, ints_m, floats_m],
        grid = (n, 1, 1),
        output_shapes = [(max(width * height, _MIN),)], output_dtypes = [mx.float32],
        init_value = 0)[0]


def prefiltered_weight_flat(ip, ints, floats, width, height, nsx, nsy, seed):
    """
        The (lazy) weight image of a prefiltered render, float32[max(W*H, 8)];
        pass it as weight_image= to prefiltered_forward_flat and
        prefiltered_backward_flat to share it between the passes.
    """
    n = width * height * nsx * nsy
    if n == 0:
        return None
    ip_m = _fill_ip(ip, width, height, nsx, nsy, seed, True, False, 0)
    ints_m, floats_m = _pools(ints, floats)
    return _weight_image(ip_m, ints_m, floats_m, n, width, height)


def prefiltered_forward_flat(ip, ints, floats, width, height, nsx, nsy, seed,
                             background_image = None, exact_scan = False, weight_image = None):
    ip_m = _fill_ip(ip, width, height, nsx, nsy, seed, True, background_image is not None, 0)
    ints_m, floats_m = _pools(ints, floats)
    n = width * height * nsx * nsy
    if n == 0:
        return mx.zeros((height, width, 4), dtype = mx.float32)
    weight = weight_image if weight_image is not None else \
        _weight_image(ip_m, ints_m, floats_m, n, width, height)
    out = _kernel('prefilter_forward')(
        inputs = [ip_m, ints_m, floats_m, weight, _background_m(background_image, width, height),
                  _opts(exact_scan = exact_scan)],
        grid = (n, 1, 1),
        output_shapes = [(max(width * height * 4, _MIN),)], output_dtypes = [mx.float32],
        init_value = 0)[0]
    return out[:width * height * 4].reshape(height, width, 4)


def prefiltered_backward_flat(ip, ints, floats, width, height, nsx, nsy, seed,
                              background_image, d_render_image, want_translation = False,
                              exact_scan = False, render_image = None, weight_image = None):
    """
        Returns (d_floats, d_background (H, W, 4) or None, d_translation (H, W, 2) or None).
        render_image: the forward output (H, W, 4) for the filter-radius gradient;
        recomputed when None. weight_image: see prefiltered_weight_flat.
    """
    has_bg = background_image is not None
    ip_m = _fill_ip(ip, width, height, nsx, nsy, seed, True, has_bg, 0)
    ints_m, floats_m = _pools(ints, floats)
    n = width * height * nsx * nsy
    nf = int(floats_m.size)
    npx = width * height
    if n == 0:
        return (mx.zeros((nf,)), mx.zeros((height, width, 4)) if has_bg else None,
                mx.zeros((height, width, 2)) if want_translation else None)
    weight = weight_image if weight_image is not None else \
        _weight_image(ip_m, ints_m, floats_m, n, width, height)
    d_img = rm._pad(_as_mx(d_render_image))
    if render_image is None:
        render_image = _kernel('prefilter_forward')(
            inputs = [ip_m, ints_m, floats_m, weight, _background_m(background_image, width, height),
                      _opts(exact_scan = exact_scan)],
            grid = (n, 1, 1),
            output_shapes = [(max(npx * 4, _MIN),)], output_dtypes = [mx.float32],
            init_value = 0)[0]
    fwd_img = rm._pad(_as_mx(render_image))
    nb = max(4 * npx, _MIN) if has_bg else _MIN
    nt = max(2 * npx, _MIN) if want_translation else _MIN
    d_floats, d_bg, d_trans = _kernel('prefilter_backward')(
        inputs = [ip_m, ints_m, floats_m, weight, _background_m(background_image, width, height),
                  _opts(want_translation, exact_scan), d_img, fwd_img],
        grid = (n, 1, 1),
        output_shapes = [(nf,), (nb,), (nt,)], output_dtypes = [mx.float32] * 3,
        init_value = 0)
    return (d_floats,
            d_bg[:4 * npx].reshape(height, width, 4) if has_bg else None,
            d_trans[:2 * npx].reshape(height, width, 2) if want_translation else None)


# ---------------------------------------------------------------------------
# Scene-argument level (output of serialize_scene, without the background)

def flat_scene(scene_args, built = None):
    """
        Builds the C++ scene (unless `built` from render_mlx._build_scene is
        given) and returns (ip, ints, floats, built). Empty scenes get the
        filter from the arguments, as in render_metal.render_forward_gpu.
    """
    if built is None:
        built = render_mlx._build_scene(scene_args)
    ip, ints, floats = ms.flatten_scene(built[0])
    if int(ip[ms.IP_NUM_GROUPS]) == 0:
        ip = np.array(ip, dtype = np.int32, copy = True)
        floats = np.array(floats, dtype = np.float32, copy = True)
        ftype = scene_args[-2]
        ip[ms.IP_FILTER_TYPE] = int(getattr(ftype, 'value', ftype))
        ip[ms.IP_FILTER_RADIUS_OFF] = 0
        floats[0] = render_mlx._floats(scene_args[-1])[0]
    return ip, ints, floats, built


def _eval_from_args(scene_args, eval_positions):
    if eval_positions is not None:
        return eval_positions
    ev = scene_args[6]
    return ev if ev is not None and ev.shape[0] > 0 else None


def sdf_forward_gpu(scene_args, width, height, nsx, nsy, seed, eval_positions = None, built = None):
    """ (height, width, 1), or (N, 1) with eval_positions (canvas pixel coordinates). """
    ip, ints, floats, _ = flat_scene(scene_args, built)
    return sdf_forward_flat(ip, ints, floats, width, height, nsx, nsy, seed,
                            _eval_from_args(scene_args, eval_positions), bool(scene_args[5]))


def sdf_backward_gpu(scene_args, width, height, nsx, nsy, seed, d_sdf, eval_positions = None,
                     want_translation = False, built = None):
    """ Returns (d_floats, d_translation (H, W, 2) or None). """
    ip, ints, floats, _ = flat_scene(scene_args, built)
    return sdf_backward_flat(ip, ints, floats, width, height, nsx, nsy, seed, d_sdf,
                             _eval_from_args(scene_args, eval_positions), want_translation,
                             bool(scene_args[5]))


def prefiltered_forward_gpu(scene_args, width, height, seed, background_image,
                            nsx = 1, nsy = 1, built = None):
    """ (height, width, 4) """
    ip, ints, floats, _ = flat_scene(scene_args, built)
    return prefiltered_forward_flat(ip, ints, floats, width, height, nsx, nsy, seed, background_image)


def prefiltered_backward_gpu(scene_args, width, height, seed, background_image, d_render_image,
                             want_translation = False, nsx = 1, nsy = 1, built = None, render_image = None):
    """ Returns (d_floats, d_background or None, d_translation or None). """
    ip, ints, floats, _ = flat_scene(scene_args, built)
    return prefiltered_backward_flat(ip, ints, floats, width, height, nsx, nsy, seed,
                                     background_image, d_render_image, want_translation,
                                     render_image = render_image)


def is_supported(output_type, use_prefiltering, eval_positions):
    has_eval = eval_positions is not None and eval_positions.shape[0] > 0
    if output_type == render_mlx.OutputType.sdf:
        return True
    return use_prefiltering and not has_eval
