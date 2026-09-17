// Signed distance output and prefiltered colour for the diffvg CUDA kernels.
//
// CUDA translation of pydiffvg/metal/prefilter.metal. Header order: common ->
// geometry -> color -> backward -> distance_grad -> prefilter.
//
// Ports of diffvg.cpp sample_distance and sample_color_prefiltered (forward
// and backward). Helpers here never touch output buffers (a Metal
// compiler-crash workaround, kept for structural parity with the Metal port):
// gradients go through GradWrites, fragments through local arrays; the kernel
// bodies (the Python dispatch layer) write outputs.
//
// See the TRANSLATION RULES header comment in common.cu. Mechanical changes:
// `inline` -> `__device__ inline`, `thread T&`/`thread T*` -> `T&`/`T*`,
// `constant` -> `static constexpr`, `v.xyz` -> `v.xyz()`, and the math renames.
//
// Deliberate differences from the CPU:
//  1. sample_distance scans the groups from last to first with the best
//     distance so far as compute_distance's max_radius (CPU: infinity for
//     every group). compute_distance only reports distances < max_radius, so
//     the selected group, shape and closest point are the same as the CPU's
//     strict `d < min_distance` rule.
//  2. Prefiltered strokes: the half-width is path_thickness_at (per-point
//     thickness interpolated at the closest point) and d_stroke_width goes
//     through d_path_thickness_at (the CPU used the dummy stroke_width 0 of
//     thickness paths before its fix).
//  3. Prefiltered strokes search with max_radius = (largest stroke radius in
//     the group) + 1 instead of infinity. A closest shape at distance
//     d >= that bound has weight smoothstep(d + w) - smoothstep(d - w) = 0,
//     so the set of fragments is unchanged (pass PF_OPT_EXACT_SCAN to disable).
//  4. More than PF_MAX_FRAGMENTS fragments: the extra ones are dropped (CPU
//     asserts).

// Overridable; see the note on DIFFVG_MAX_HIT_SHAPES in color.cu.
#ifndef DIFFVG_PF_MAX_FRAGMENTS
#define DIFFVG_PF_MAX_FRAGMENTS 64
#endif
static constexpr int PF_MAX_FRAGMENTS = DIFFVG_PF_MAX_FRAGMENTS;
static constexpr int PF_SCENE_BVH_STACK = 64;

// opts[] slots (owned by the Python dispatch layer)
static constexpr int PF_OPT_WANT_TRANSLATION = 0;
static constexpr int PF_OPT_EXACT_SCAN = 1;

// ---------------------------------------------------------------------------
// d compute_filter_weight / d radius, zero outside the filter support, for
// the prefiltered filter-radius gradient, which multiplies it by
// dot(d_pixel, c_i - pixel) / weight_sum.
__device__ inline float pf_d_filter_weight_dr(int type, float r, float dx, float dy) {
    float adx = fabsf(dx);
    float ady = fabsf(dy);
    if (adx > r || ady > r || !(r > 0)) {
        return 0.f;
    }
    if (type == FILTER_BOX) {
        // w = 1 / (4 r^2), so dw/dr = -0.5 / r^3 is the same for every sample
        // and its product with sum_i (c_i - pixel) = 0 vanishes. Return 0 so
        // the gradient is exactly zero (as on the CPU), not float noise.
        return 0.f;
    } else if (type == FILTER_TENT) {
        // w = (r - |dx|) (r - |dy|) / r^4
        float fx = r - adx;
        float fy = r - ady;
        float r4 = r * r * r * r;
        return (fx + fy) / r4 - 4 * fx * fy / (r4 * r);
    } else if (type == FILTER_RADIAL_PARABOLIC) {
        // w = (4/3)(1 - (dx/r)^2) (4/3)(1 - (dy/r)^2)
        float gx = 1 - square(dx / r);
        float gy = 1 - square(dy / r);
        float r3 = r * r * r;
        return (16.f / 9.f) * ((2 * dx * dx / r3) * gy + gx * (2 * dy * dy / r3));
    } else {
        // w = fx fy / r^2, fx = 0.5 (1 - cos(2 pi ndx)), ndx = dx / (2r) + 0.5
        float ndx = dx / (2 * r) + 0.5f;
        float ndy = dy / (2 * r) + 0.5f;
        float fx = 0.5f * (1 - cosf(2 * DIFFVG_PI * ndx));
        float fy = 0.5f * (1 - cosf(2 * DIFFVG_PI * ndy));
        float dfx = 0.5f * sinf(2 * DIFFVG_PI * ndx) * (2 * DIFFVG_PI) * (-dx / (2 * r * r));
        float dfy = 0.5f * sinf(2 * DIFFVG_PI * ndy) * (2 * DIFFVG_PI) * (-dy / (2 * r * r));
        return (dfx * fy + fx * dfy) / (r * r) - 2 * fx * fy / (r * r * r);
    }
}

// ---------------------------------------------------------------------------
// Signed distance (diffvg.cpp sample_distance)
struct SdfResult {
    int group_id;              // -1: no group found (distance 0)
    int shape_id;
    float distance;            // unsigned, unweighted, canvas space
    float2 closest_pt;         // canvas space
    ClosestPointPathInfo info;
    bool inside;
};

// canvas_pt in canvas space. Returns false when no group reports a distance.
__device__ inline bool sdf_query(SceneView s, float2 canvas_pt, SdfResult &r) {
    r.group_id = -1;
    r.shape_id = -1;
    r.distance = 0;
    r.closest_pt = float2(0);
    r.info.base_point_id = -1;
    r.info.point_id = -1;
    r.info.t = 0;
    r.inside = false;
    int num_groups = s.ip[IP_NUM_GROUPS];
    float best = DIFFVG_BIG;
    for (int group_id = num_groups - 1; group_id >= 0; group_id--) {
        if (group_num_shapes(s, group_id) <= 0) {
            continue;
        }
        int sid = -1;
        float2 p = float2(0);
        ClosestPointPathInfo li;
        li.base_point_id = -1;
        li.point_id = -1;
        li.t = 0;
        float d = DIFFVG_BIG;
        if (compute_distance(s, group_id, canvas_pt, best, sid, p, li, d)) {
            if (r.group_id == -1 || d < best) {
                best = d;
                r.group_id = group_id;
                r.shape_id = sid;
                r.closest_pt = p;
                r.info = li;
            }
        }
    }
    if (r.group_id < 0) {
        return false;
    }
    r.distance = best;
    if (group_has_fill(s, r.group_id)) {
        EdgeQuery eq;
        eq.shape_group_id = -1;
        eq.shape_id = -1;
        eq.hit = false;
        r.inside = is_inside(s, r.group_id, canvas_pt, false, eq);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Prefiltered colour (diffvg.cpp sample_color_prefiltered)
struct PrefilterFragment {
    float3 color;
    float alpha;               // colour alpha * coverage weight
    int group_id;
    int shape_id;
    float distance;            // stroke: unsigned; fill: signed (negative outside)
    float2 closest_pt;
    ClosestPointPathInfo info;
    bool is_stroke;
    bool within_distance;
};

// Largest stroke radius of any shape in a group (scene.cpp stroke_max_radius):
// for thickness paths the maximum thickness stored at the path BVH root.
__device__ inline float pf_group_max_stroke_radius(SceneView s, int group_id) {
    int n = group_num_shapes(s, group_id);
    float r = 0;
    for (int k = 0; k < n; k++) {
        int sid = group_shape_id(s, group_id, k);
        float w = shape_stroke_width(s, sid);
        if (shape_type(s, sid) == SHAPE_PATH && path_has_thickness(s, sid) &&
            path_num_base_points(s, sid) > 0) {
            w = node_max_radius(s, path_bvh_root(s, sid));
        }
        r = fmaxf(r, w);
    }
    return r;
}

// Collects the fragments at canvas_pt (canvas space), sorted back to front
// (increasing group id, stable). Returns their number.
__device__ inline int pf_collect_fragments(SceneView s, float2 pt, bool exact_scan,
                                           PrefilterFragment *fragments) {
    if (s.ip[IP_NUM_GROUPS] <= 0) {
        return 0;
    }
    int bvh_stack[PF_SCENE_BVH_STACK];
    int stack_size = 0;
    int num_fragments = 0;
    int base = s.ip[IP_SCENE_BVH_BASE];
    bvh_stack[stack_size++] = scene_bvh_root(s);
    while (stack_size > 0) {
        int n = bvh_stack[--stack_size];
        int child1 = node_child1(s, n);
        if (child1 < 0) {
            int group_id = node_child0(s, n);
            if (group_num_shapes(s, group_id) <= 0) {
                continue;
            }
            if (group_has_stroke(s, group_id)) {
                int sid = -1;
                float2 cp = float2(0);
                ClosestPointPathInfo info;
                info.base_point_id = -1;
                info.point_id = -1;
                info.t = 0;
                float d = DIFFVG_BIG;
                float max_radius = DIFFVG_BIG;
                if (!exact_scan) {
                    max_radius = fminf(pf_group_max_stroke_radius(s, group_id) + 1.f, DIFFVG_BIG);
                }
                if (compute_distance(s, group_id, pt, max_radius, sid, cp, info, d) && sid >= 0) {
                    float radius = path_thickness_at(s, sid, info);
                    float w = smoothstep(fabsf(d) + radius) - smoothstep(fabsf(d) - radius);
                    if (w > 0 && num_fragments < PF_MAX_FRAGMENTS) {
                        float4 ca = sample_color_param(s, group_stroke_type(s, group_id),
                                                       group_stroke_off(s, group_id),
                                                       group_stroke_num_stops(s, group_id), pt);
                        PrefilterFragment f;
                        f.color = ca.xyz();
                        f.alpha = ca.w * w;
                        f.group_id = group_id;
                        f.shape_id = sid;
                        f.distance = d;
                        f.closest_pt = cp;
                        f.info = info;
                        f.is_stroke = true;
                        f.within_distance = true;
                        fragments[num_fragments++] = f;
                    }
                }
            }
            if (group_has_fill(s, group_id)) {
                int sid = -1;
                float2 cp = float2(0);
                ClosestPointPathInfo info;
                info.base_point_id = -1;
                info.point_id = -1;
                info.t = 0;
                float d = DIFFVG_BIG;
                bool found = compute_distance(s, group_id, pt, 1.f, sid, cp, info, d);
                EdgeQuery eq;
                eq.shape_group_id = -1;
                eq.shape_id = -1;
                eq.hit = false;
                bool inside_ = is_inside(s, group_id, pt, false, eq);
                if (found || inside_) {
                    if (!inside_) {
                        d = -d;
                    }
                    float w = smoothstep(d);
                    if (w > 0 && num_fragments < PF_MAX_FRAGMENTS) {
                        float4 ca = sample_color_param(s, group_fill_type(s, group_id),
                                                       group_fill_off(s, group_id),
                                                       group_fill_num_stops(s, group_id), pt);
                        PrefilterFragment f;
                        f.color = ca.xyz();
                        f.alpha = ca.w * w;
                        f.group_id = group_id;
                        f.shape_id = sid;
                        f.distance = d;
                        f.closest_pt = cp;
                        f.info = info;
                        f.is_stroke = false;
                        f.within_distance = found;
                        fragments[num_fragments++] = f;
                    }
                }
            }
        } else {
            int c0 = base + node_child0(s, n);
            int c1 = base + child1;
            if (inside(node_box(s, c0), pt, node_max_radius(s, c0)) && stack_size < PF_SCENE_BVH_STACK) {
                bvh_stack[stack_size++] = c0;
            }
            if (inside(node_box(s, c1), pt, node_max_radius(s, c1)) && stack_size < PF_SCENE_BVH_STACK) {
                bvh_stack[stack_size++] = c1;
            }
        }
    }
    // Insertion sort, back to front (increasing group id), stable like the CPU
    for (int i = 1; i < num_fragments; i++) {
        int j = i;
        PrefilterFragment temp = fragments[j];
        while (j > 0 && fragments[j - 1].group_id > temp.group_id) {
            fragments[j] = fragments[j - 1];
            j--;
        }
        fragments[j] = temp;
    }
    return num_fragments;
}

// Back-to-front blend. accum_color/accum_alpha (PF_MAX_FRAGMENTS entries)
// receive the running premultiplied values for the backward pass.
__device__ inline float4 pf_blend(PrefilterFragment *fragments, int num_fragments, bool has_bg, float4 bg,
                                  float3 *accum_color, float *accum_alpha) {
    if (num_fragments <= 0) {
        return has_bg ? bg : float4(0);
    }
    float prev_alpha = has_bg ? bg.w : 0.f;
    float3 prev_color = has_bg ? bg.xyz() : float3(0);
    for (int i = 0; i < num_fragments; i++) {
        float a = fragments[i].alpha;
        prev_color = prev_color * (1 - a) + a * fragments[i].color;
        prev_alpha = prev_alpha * (1 - a) + a;
        accum_color[i] = prev_color;
        accum_alpha[i] = prev_alpha;
    }
    float3 final_color = prev_color;
    if (prev_alpha > 1e-6f) {
        final_color /= prev_alpha;
    }
    return float4(final_color, prev_alpha);
}

// Backward of one fragment's colour and coverage weight. d_color_i / d_alpha_i
// are the gradients of the fragment colour and (weighted) alpha from the
// reverse blend. Pushes at most 14 (colour) + 17 (distance) + 4 (thickness)
// entries into g; accumulates d_translation like the CPU.
__device__ inline void pf_d_fragment(SceneView s, PrefilterFragment f, float2 pt, float3 d_color_i, float d_alpha_i,
                                     GradWrites &g, float2 &d_translation) {
    int group_id = f.group_id;
    float d = f.distance;
    if (f.is_stroke) {
        float radius = path_thickness_at(s, f.shape_id, f.info);
        float a = fabsf(d) + radius;
        float b = fabsf(d) - radius;
        float w = smoothstep(a) - smoothstep(b);
        if (w != 0) {
            float d_w = w > 0 ? (f.alpha / w) * d_alpha_i : 0.f;
            d_alpha_i *= w;
            d_sample_color_param(s, group_stroke_type(s, group_id), group_stroke_off(s, group_id),
                                 group_stroke_num_stops(s, group_id), pt,
                                 float4(d_color_i, d_alpha_i), g, d_translation);
            float d_a = d_smoothstep(a, d_w);
            float d_b = -d_smoothstep(b, d_w);
            float d_d = d_a + d_b;
            if (d < 0) {
                d_d = -d_d;
            }
            float d_radius = d_a - d_b;
            // the interpolated per-point thickness also depends on the closest-point
            // parameter t (exactly 0 for uniform-width strokes)
            float d_t_root = d_radius * path_thickness_dt_at(s, f.shape_id, f.info);
            if (fabsf(d_d) > 1e-10f || d_t_root != 0) {
                d_compute_distance(s, group_id, f.shape_id, pt, f.closest_pt, f.info, d_d, d_t_root, g,
                                   d_translation);
            }
            d_path_thickness_at(s, f.shape_id, f.info, d_radius, g);
        }
    } else {
        float w = smoothstep(d);
        if (w != 0) {
            float d_w = w > 0 ? (f.alpha / w) * d_alpha_i : 0.f;
            d_alpha_i *= w;
            d_sample_color_param(s, group_fill_type(s, group_id), group_fill_off(s, group_id),
                                 group_fill_num_stops(s, group_id), pt,
                                 float4(d_color_i, d_alpha_i), g, d_translation);
            float d_d = d_smoothstep(d, d_w);
            if (d < 0) {
                d_d = -d_d;
            }
            if (fabsf(d_d) > 1e-10f && f.within_distance && f.shape_id >= 0) {
                d_compute_distance(s, group_id, f.shape_id, pt, f.closest_pt, f.info, d_d, g, d_translation);
            }
        }
    }
}
