// Colour sampling for the diffvg CUDA kernels.
//
// CUDA translation of pydiffvg/metal/color.metal: ports of
// sample_color(ColorType, ...) and the scene-level sample_color (fragment
// collection over the scene BVH + back-to-front blend) from diffvg.cpp. Built
// on common.cu and geometry.cu (is_inside, within_distance, EdgeQuery).
//
// See the TRANSLATION RULES header comment in common.cu. Mechanical changes
// here: `inline` -> `__device__ inline`, `thread T&` / `thread T*` -> `T&` /
// `T*`, `constant` -> `static constexpr`, the Metal swizzle `v.xyz` -> `v.xyz()`,
// and float min/max -> fminf/fmaxf, fabs -> fabsf.
//
// Rules (DESIGN.md "Mechanism"): float32 only, guarded divisions, helpers never
// touch output buffers. (The last rule is a Metal compiler-crash workaround
// that CUDA does not need, but it is kept so both backends share one
// structure.)
//
// PERFORMANCE NOTE for the CUDA port: `Fragment fragments[MAX_HIT_SHAPES]` and
// `BlendState` are per-thread arrays of 256 entries. On Metal these live in
// thread-private memory; in CUDA they become local memory (spilled to global,
// L1-cached). That is correct but costly, and it caps occupancy. See
// README-porting.md for the option of lowering MAX_HIT_SHAPES on CUDA.

// Overridable so the dispatch layer can trade fragment capacity for occupancy:
// these arrays are per-thread LOCAL memory on CUDA, and CUDA reserves the local
// pool for the device's full resident-thread capacity, so their size -- not the
// grid or the block size -- is what determines whether a kernel can launch.
// gpu_backend.set_fragment_capacity() emits these defines ahead of the sources.
#ifndef DIFFVG_MAX_HIT_SHAPES
#define DIFFVG_MAX_HIT_SHAPES 256
#endif
#ifndef DIFFVG_MAX_SCENE_BVH_STACK
#define DIFFVG_MAX_SCENE_BVH_STACK 64
#endif
static constexpr int MAX_HIT_SHAPES = DIFFVG_MAX_HIT_SHAPES;
static constexpr int MAX_SCENE_BVH_STACK = DIFFVG_MAX_SCENE_BVH_STACK;

// ---------------------------------------------------------------------------
// Per-colour sampling: diffvg.cpp sample_color(ColorType, void*, pt)
// type/off/nstops come from the group record (fill_* or stroke_*).
__device__ inline float4 color_stop(SceneView s, int off, int nstops, int i) {
    int o = off + 4 + nstops + 4 * i;
    return float4(s.f[o], s.f[o + 1], s.f[o + 2], s.f[o + 3]);
}

__device__ inline float4 sample_gradient_stops(SceneView s, int off, int nstops, float t) {
    if (nstops <= 0) {
        return float4(0);
    }
    if (t < s.f[off + 4]) {
        return color_stop(s, off, nstops, 0);
    }
    for (int i = 0; i < nstops - 1; i++) {
        float offset_curr = s.f[off + 4 + i];
        float offset_next = s.f[off + 4 + i + 1];
        if (t >= offset_curr && t < offset_next) {
            float4 color_curr = color_stop(s, off, nstops, i);
            float4 color_next = color_stop(s, off, nstops, i + 1);
            float denom = offset_next - offset_curr;
            float tt = (t - offset_curr) / (denom > 0 ? denom : 1.0f);
            return color_curr * (1 - tt) + color_next * tt;
        }
    }
    return color_stop(s, off, nstops, nstops - 1);
}

__device__ inline float4 sample_color_param(SceneView s, int type, int off, int nstops, float2 pt) {
    if (type == COLOR_CONSTANT) {
        return float4(s.f[off], s.f[off + 1], s.f[off + 2], s.f[off + 3]);
    } else if (type == COLOR_LINEAR) {
        float2 beg = float2(s.f[off], s.f[off + 1]);
        float2 end = float2(s.f[off + 2], s.f[off + 3]);
        float t = dot(pt - beg, end - beg) / fmaxf(dot(end - beg, end - beg), 1e-3f);
        return sample_gradient_stops(s, off, nstops, t);
    } else if (type == COLOR_RADIAL) {
        float2 center = float2(s.f[off], s.f[off + 1]);
        float2 radius = float2(s.f[off + 2], s.f[off + 3]);
        // CPU divides by the radius unguarded (inf/NaN for a zero radius);
        // keep the result finite here.
        float rx = fabsf(radius.x) > 1e-30f ? radius.x : 1e-30f;
        float ry = fabsf(radius.y) > 1e-30f ? radius.y : 1e-30f;
        float2 offset = pt - center;
        float2 normalized_offset = float2(offset.x / rx, offset.y / ry);
        float t = length(normalized_offset);
        return sample_gradient_stops(s, off, nstops, t);
    }
    return float4(0);
}

// ---------------------------------------------------------------------------
// Scene-level sampling: diffvg.cpp sample_color(const SceneData&, ...)
struct Fragment {
    float3 color;
    float alpha;
    int group_id;
    bool is_stroke;
};

// Collects the fragments hit at canvas point pt (scene BVH order, like the CPU)
// and sorts them back to front (increasing group id, stable insertion sort).
// fragments must point to a local array of MAX_HIT_SHAPES entries.
// With use_edge_query, within_distance / is_inside update eq.hit as on the CPU.
__device__ inline int collect_fragments(SceneView s, float2 pt, bool use_edge_query, EdgeQuery &eq,
                                        Fragment *fragments) {
    int bvh_stack[MAX_SCENE_BVH_STACK];
    int stack_size = 0;
    int num_fragments = 0;
    int base = s.ip[IP_SCENE_BVH_BASE];
    bvh_stack[stack_size++] = scene_bvh_root(s);
    while (stack_size > 0) {
        int n = bvh_stack[--stack_size];
        int child1 = node_child1(s, n);
        if (child1 < 0) {
            int group_id = node_child0(s, n);
            if (group_has_stroke(s, group_id)) {
                if (within_distance(s, group_id, pt, use_edge_query, eq)) {
                    float4 ca = sample_color_param(s, group_stroke_type(s, group_id),
                                                   group_stroke_off(s, group_id),
                                                   group_stroke_num_stops(s, group_id), pt);
                    if (num_fragments < MAX_HIT_SHAPES) {
                        Fragment f;
                        f.color = ca.xyz();
                        f.alpha = ca.w;
                        f.group_id = group_id;
                        f.is_stroke = true;
                        fragments[num_fragments++] = f;
                    }
                }
            }
            if (group_has_fill(s, group_id)) {
                if (is_inside(s, group_id, pt, use_edge_query, eq)) {
                    float4 ca = sample_color_param(s, group_fill_type(s, group_id),
                                                   group_fill_off(s, group_id),
                                                   group_fill_num_stops(s, group_id), pt);
                    if (num_fragments < MAX_HIT_SHAPES) {
                        Fragment f;
                        f.color = ca.xyz();
                        f.alpha = ca.w;
                        f.group_id = group_id;
                        f.is_stroke = false;
                        fragments[num_fragments++] = f;
                    }
                }
            }
        } else {
            int c0 = base + node_child0(s, n);
            int c1 = base + child1;
            if (inside(node_box(s, c0), pt, node_max_radius(s, c0)) && stack_size < MAX_SCENE_BVH_STACK) {
                bvh_stack[stack_size++] = c0;
            }
            if (inside(node_box(s, c1), pt, node_max_radius(s, c1)) && stack_size < MAX_SCENE_BVH_STACK) {
                bvh_stack[stack_size++] = c1;
            }
        }
    }
    for (int i = 1; i < num_fragments; i++) {
        int j = i;
        Fragment temp = fragments[j];
        while (j > 0 && fragments[j - 1].group_id > temp.group_id) {
            fragments[j] = fragments[j - 1];
            j--;
        }
        fragments[j] = temp;
    }
    return num_fragments;
}

// screen_pt in [0, 1)^2; scaled by the canvas size like the CPU.
// With use_edge_query, eq.hit is reset and updated exactly as the CPU does
// (render_edge_kernel); pass a dummy EdgeQuery and false otherwise.
__device__ inline float4 sample_color_scene_eq(SceneView s, bool has_bg, float4 bg, float2 screen_pt,
                                               bool use_edge_query, EdgeQuery &eq) {
    if (use_edge_query) {
        eq.hit = false;
    }
    if (s.ip[IP_NUM_GROUPS] <= 0) {
        return has_bg ? bg : float4(0);
    }
    float2 pt = screen_pt;
    pt.x *= (float)(s.ip[IP_CANVAS_W]);
    pt.y *= (float)(s.ip[IP_CANVAS_H]);

    Fragment fragments[MAX_HIT_SHAPES];
    int num_fragments = collect_fragments(s, pt, use_edge_query, eq, fragments);
    if (num_fragments <= 0) {
        return has_bg ? bg : float4(0);
    }
    // Blend (premultiplied accumulation)
    float accum_alpha = has_bg ? bg.w : 0.0f;
    float3 accum_color = has_bg ? bg.xyz() : float3(0);
    for (int i = 0; i < num_fragments; i++) {
        float3 new_color = fragments[i].color;
        float new_alpha = fragments[i].alpha;
        if (use_edge_query) {
            if (new_alpha >= 1.f && eq.hit) {
                eq.hit = false;
            }
            if (eq.shape_group_id == fragments[i].group_id) {
                eq.hit = true;
            }
        }
        accum_color = accum_color * (1 - new_alpha) + new_alpha * new_color;
        accum_alpha = accum_alpha * (1 - new_alpha) + new_alpha;
    }
    float3 final_color = accum_color;
    float final_alpha = accum_alpha;
    if (final_alpha > 1e-6f) {
        final_color /= final_alpha;
    }
    return float4(final_color, final_alpha);
}

// Backward-blend state for one sample (render_kernel's sample_color with
// d_color). The kernel body walks the fragments in reverse with
// blend_backward_step and flushes the colour gradients of each fragment.
struct BlendState {
    float3 accum_color[MAX_HIT_SHAPES];
    float accum_alpha[MAX_HIT_SHAPES];
    float3 first_color;
    float first_alpha;
    float3 d_curr_color;
    float d_curr_alpha;
};

// Forward blend over sorted fragments; returns the sample colour and seeds the
// reverse pass with d_color.
__device__ inline float4 blend_forward(Fragment *fragments, int num_fragments, bool has_bg, float4 bg,
                                       float4 d_color, BlendState &st) {
    st.first_alpha = has_bg ? bg.w : 0.0f;
    st.first_color = has_bg ? bg.xyz() : float3(0);
    for (int i = 0; i < num_fragments; i++) {
        float new_alpha = fragments[i].alpha;
        float3 prev_color = i > 0 ? st.accum_color[i - 1] : st.first_color;
        float prev_alpha = i > 0 ? st.accum_alpha[i - 1] : st.first_alpha;
        st.accum_color[i] = prev_color * (1 - new_alpha) + new_alpha * fragments[i].color;
        st.accum_alpha[i] = prev_alpha * (1 - new_alpha) + new_alpha;
    }
    float3 final_color = st.accum_color[num_fragments - 1];
    float final_alpha = st.accum_alpha[num_fragments - 1];
    if (final_alpha > 1e-6f) {
        final_color /= final_alpha;
    }
    st.d_curr_color = d_color.xyz();
    st.d_curr_alpha = d_color.w;
    if (final_alpha > 1e-6f) {
        st.d_curr_color = d_color.xyz() / final_alpha;
        st.d_curr_alpha -= dot(d_color.xyz(), final_color) / final_alpha;
    }
    return float4(final_color, final_alpha);
}

// One reverse step for fragment i; returns the d_color (rgb, alpha) of that
// fragment's colour and moves the state to the previous layer.
__device__ inline float4 blend_backward_step(Fragment *fragments, int i, BlendState &st) {
    float3 prev_color = i > 0 ? st.accum_color[i - 1] : st.first_color;
    float prev_alpha = i > 0 ? st.accum_alpha[i - 1] : st.first_alpha;
    float a = fragments[i].alpha;
    float d_prev_alpha = st.d_curr_alpha * (1.f - a);
    float d_alpha_i = st.d_curr_alpha * (1.f - prev_alpha);
    d_alpha_i += dot(st.d_curr_color, fragments[i].color - prev_color);
    float3 d_prev_color = st.d_curr_color * (1 - a);
    float3 d_color_i = st.d_curr_color * a;
    st.d_curr_color = d_prev_color;
    st.d_curr_alpha = d_prev_alpha;
    return float4(d_color_i, d_alpha_i);
}

__device__ inline float4 sample_color_scene(SceneView s, bool has_bg, float4 bg, float2 screen_pt) {
    EdgeQuery eq;
    eq.shape_group_id = -1;
    eq.shape_id = -1;
    eq.hit = false;
    return sample_color_scene_eq(s, has_bg, bg, screen_pt, false, eq);
}
