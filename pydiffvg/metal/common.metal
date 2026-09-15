// Shared definitions for the diffvg Metal kernels.
//
// This file is concatenated (with geometry.metal and color.metal) into the
// `header` of every mx.fast.metal_kernel. It mirrors the flattened scene
// produced by diffvg.Scene.export_flat(); see the layout tables there.
// Metal has no double: everything here is float32.

// ---------------------------------------------------------------------------
// Parameter block (int32[32])
constant int IP_W = 0;
constant int IP_H = 1;
constant int IP_NSX = 2;
constant int IP_NSY = 3;
constant int IP_CANVAS_W = 4;
constant int IP_CANVAS_H = 5;
constant int IP_NUM_SHAPES = 6;
constant int IP_NUM_GROUPS = 7;
constant int IP_NUM_TOTAL_SHAPES = 8;
constant int IP_FILTER_TYPE = 9;
constant int IP_USE_PREFILTERING = 10;
constant int IP_HAS_BACKGROUND = 11;
constant int IP_NUM_EVAL = 12;
constant int IP_SEED_LO = 13;
constant int IP_SEED_HI = 14;
constant int IP_SCENE_BVH_BASE = 15;
constant int IP_SHAPES_I_OFF = 16;
constant int IP_GROUPS_I_OFF = 17;
constant int IP_BVH_I_OFF = 18;
constant int IP_SAMPLE_SHAPE_ID_OFF = 19;
constant int IP_SAMPLE_GROUP_ID_OFF = 20;
constant int IP_BVH_F_OFF = 21;
constant int IP_SAMPLE_CDF_OFF = 22;
constant int IP_SAMPLE_PMF_OFF = 23;
constant int IP_FILTER_RADIUS_OFF = 24;
constant int IP_NUM_FLOATS = 25;
constant int IP_NUM_INTS = 26;
constant int IP_NUM_BVH_NODES = 27;
constant int IP_USE_EVAL_POSITIONS = 28;

constant int SHAPE_I_STRIDE = 12;
constant int GROUP_I_STRIDE = 12;
constant int BVH_F_STRIDE = 5;
constant int BVH_I_STRIDE = 2;

constant int SHAPE_CIRCLE = 0;
constant int SHAPE_ELLIPSE = 1;
constant int SHAPE_PATH = 2;
constant int SHAPE_RECT = 3;

constant int COLOR_NONE = -1;
constant int COLOR_CONSTANT = 0;
constant int COLOR_LINEAR = 1;
constant int COLOR_RADIAL = 2;

constant int FILTER_BOX = 0;
constant int FILTER_TENT = 1;
constant int FILTER_RADIAL_PARABOLIC = 2;
constant int FILTER_HANN = 3;

constant float DIFFVG_PI = 3.14159265358979323846f;

// Kernels are compiled with math_mode "fast": the AGX code generator hits a
// fatal LLVM error on these kernels in "safe" and "relaxed" modes. Fast math
// may assume there are no infinities or NaNs, so never rely on them: use this
// finite sentinel where the C++ code uses infinity<float>(), and guard divisions.
constant float DIFFVG_BIG = 1e30f;
inline float diffvg_infinity() { return DIFFVG_BIG; }
inline float square(float x) { return x * x; }
inline float cubic(float x) { return x * x * x; }

// ---------------------------------------------------------------------------
// View over the flattened scene
//
// mx.fast.metal_kernel passes inputs with fewer than 8 elements in the
// `constant` address space and larger ones as `device`. SceneView holds
// `device` pointers, so every pool (and any other buffer handed to a helper)
// must have at least 8 elements; export_flat pads the pools to >= 64.
struct SceneView {
    const device float *f;  // floats pool
    const device int *i;    // ints pool
    const device int *ip;   // parameter block
};

inline SceneView make_scene_view(const device float *f, const device int *i, const device int *ip) {
    SceneView s;
    s.f = f;
    s.i = i;
    s.ip = ip;
    return s;
}

// ---------------------------------------------------------------------------
// Axis-aligned bounding boxes (aabb.h)
struct AABB {
    float2 p_min;
    float2 p_max;
};

inline bool inside(AABB box, float2 p) {
    return p.x >= box.p_min.x && p.x <= box.p_max.x &&
           p.y >= box.p_min.y && p.y <= box.p_max.y;
}

inline bool inside(AABB box, float2 p, float radius) {
    return p.x >= box.p_min.x - radius && p.x <= box.p_max.x + radius &&
           p.y >= box.p_min.y - radius && p.y <= box.p_max.y + radius;
}

inline bool within_distance(AABB box, float2 pt, float r) {
    return pt.x >= box.p_min.x - r && pt.x <= box.p_max.x + r &&
           pt.y >= box.p_min.y - r && pt.y <= box.p_max.y + r;
}

// ---------------------------------------------------------------------------
// BVH nodes (absolute node index n)
inline AABB node_box(SceneView s, int n) {
    int o = s.ip[IP_BVH_F_OFF] + BVH_F_STRIDE * n;
    AABB b;
    b.p_min = float2(s.f[o], s.f[o + 1]);
    b.p_max = float2(s.f[o + 2], s.f[o + 3]);
    return b;
}
inline float node_max_radius(SceneView s, int n) { return s.f[s.ip[IP_BVH_F_OFF] + BVH_F_STRIDE * n + 4]; }
inline int node_child0(SceneView s, int n) { return s.i[s.ip[IP_BVH_I_OFF] + BVH_I_STRIDE * n]; }
inline int node_child1(SceneView s, int n) { return s.i[s.ip[IP_BVH_I_OFF] + BVH_I_STRIDE * n + 1]; }
inline int scene_bvh_root(SceneView s) { return s.ip[IP_SCENE_BVH_BASE] + 2 * s.ip[IP_NUM_GROUPS] - 2; }

// ---------------------------------------------------------------------------
// Shapes
inline int shape_rec(SceneView s, int sid) { return s.ip[IP_SHAPES_I_OFF] + SHAPE_I_STRIDE * sid; }
inline int shape_type(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 0]; }
inline int shape_f_off(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 1]; }
inline int path_points_off(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 2]; }
inline int path_num_points(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 3]; }
inline int path_ctrl_off(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 4]; }
inline int path_num_base_points(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 5]; }
inline int path_thickness_off(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 6]; }
inline bool path_has_thickness(SceneView s, int sid) { return path_thickness_off(s, sid) >= 0; }
inline int path_bvh_base(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 7]; }
inline int path_bvh_root(SceneView s, int sid) { return path_bvh_base(s, sid) + 2 * path_num_base_points(s, sid) - 2; }
inline int path_cdf_off(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 8]; }
inline int path_pmf_off(SceneView s, int sid) { return path_cdf_off(s, sid) + path_num_base_points(s, sid); }
inline int path_pid_off(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 9]; }
inline bool path_is_closed(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 10] != 0; }
inline bool path_use_distance_approx(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 11] != 0; }

inline float shape_stroke_width(SceneView s, int sid) { return s.f[shape_f_off(s, sid)]; }
inline float shape_param(SceneView s, int sid, int k) { return s.f[shape_f_off(s, sid) + 1 + k]; }  // a, b, c, d

inline float circle_radius(SceneView s, int sid) { return shape_param(s, sid, 0); }
inline float2 circle_center(SceneView s, int sid) { return float2(shape_param(s, sid, 1), shape_param(s, sid, 2)); }
inline float2 ellipse_radius(SceneView s, int sid) { return float2(shape_param(s, sid, 0), shape_param(s, sid, 1)); }
inline float2 ellipse_center(SceneView s, int sid) { return float2(shape_param(s, sid, 2), shape_param(s, sid, 3)); }
inline float2 rect_p_min(SceneView s, int sid) { return float2(shape_param(s, sid, 0), shape_param(s, sid, 1)); }
inline float2 rect_p_max(SceneView s, int sid) { return float2(shape_param(s, sid, 2), shape_param(s, sid, 3)); }

inline float2 path_point(SceneView s, int sid, int k) {
    int o = path_points_off(s, sid) + 2 * k;
    return float2(s.f[o], s.f[o + 1]);
}
inline float path_thickness(SceneView s, int sid, int k) { return s.f[path_thickness_off(s, sid) + k]; }
inline int path_num_control_points(SceneView s, int sid, int j) { return s.i[path_ctrl_off(s, sid) + j]; }
inline float path_length_cdf(SceneView s, int sid, int j) { return s.f[path_cdf_off(s, sid) + j]; }
inline float path_length_pmf(SceneView s, int sid, int j) { return s.f[path_pmf_off(s, sid) + j]; }
inline int path_point_id_map(SceneView s, int sid, int j) { return s.i[path_pid_off(s, sid) + j]; }

// ---------------------------------------------------------------------------
// Shape groups
inline int group_rec(SceneView s, int g) { return s.ip[IP_GROUPS_I_OFF] + GROUP_I_STRIDE * g; }
inline int group_num_shapes(SceneView s, int g) { return s.i[group_rec(s, g) + 1]; }
inline int group_shape_id(SceneView s, int g, int k) { return s.i[s.i[group_rec(s, g) + 0] + k]; }
inline int group_fill_type(SceneView s, int g) { return s.i[group_rec(s, g) + 2]; }
inline int group_fill_off(SceneView s, int g) { return s.i[group_rec(s, g) + 3]; }
inline int group_fill_num_stops(SceneView s, int g) { return s.i[group_rec(s, g) + 4]; }
inline int group_stroke_type(SceneView s, int g) { return s.i[group_rec(s, g) + 5]; }
inline int group_stroke_off(SceneView s, int g) { return s.i[group_rec(s, g) + 6]; }
inline int group_stroke_num_stops(SceneView s, int g) { return s.i[group_rec(s, g) + 7]; }
inline bool group_has_fill(SceneView s, int g) { return group_fill_type(s, g) != COLOR_NONE; }
inline bool group_has_stroke(SceneView s, int g) { return group_stroke_type(s, g) != COLOR_NONE; }
inline bool group_use_even_odd_rule(SceneView s, int g) { return s.i[group_rec(s, g) + 8] != 0; }
inline int group_bvh_base(SceneView s, int g) { return s.i[group_rec(s, g) + 9]; }
inline int group_bvh_root(SceneView s, int g) { return group_bvh_base(s, g) + 2 * group_num_shapes(s, g) - 2; }
inline int group_gf_off(SceneView s, int g) { return s.i[group_rec(s, g) + 10]; }

// ---------------------------------------------------------------------------
// 3x3 matrices, row-major like matrix.h: m(i, j) == r[i][j]
struct Mat3 {
    float3 r0;
    float3 r1;
    float3 r2;
};

inline float mat_at(Mat3 m, int i, int j) {
    return i == 0 ? m.r0[j] : (i == 1 ? m.r1[j] : m.r2[j]);
}

inline Mat3 mat3_from_pool(const device float *f, int off) {
    Mat3 m;
    m.r0 = float3(f[off], f[off + 1], f[off + 2]);
    m.r1 = float3(f[off + 3], f[off + 4], f[off + 5]);
    m.r2 = float3(f[off + 6], f[off + 7], f[off + 8]);
    return m;
}

inline Mat3 mat3_zero() {
    Mat3 m;
    m.r0 = float3(0);
    m.r1 = float3(0);
    m.r2 = float3(0);
    return m;
}

inline Mat3 group_shape_to_canvas(SceneView s, int g) { return mat3_from_pool(s.f, group_gf_off(s, g)); }
inline Mat3 group_canvas_to_shape(SceneView s, int g) { return mat3_from_pool(s.f, group_gf_off(s, g) + 9); }

// matrix.h xform_pt (same operation order)
inline float2 xform_pt(Mat3 m, float2 pt) {
    float t0 = m.r0[0] * pt[0] + m.r0[1] * pt[1] + m.r0[2];
    float t1 = m.r1[0] * pt[0] + m.r1[1] * pt[1] + m.r1[2];
    float t2 = m.r2[0] * pt[0] + m.r2[1] * pt[1] + m.r2[2];
    return float2(t0 / t2, t1 / t2);
}

// matrix.h d_xform_pt: accumulates into d_m and d_pt
inline void d_xform_pt(Mat3 m, float2 pt, float2 d_out, thread Mat3 &d_m, thread float2 &d_pt) {
    float t0 = m.r0[0] * pt[0] + m.r0[1] * pt[1] + m.r0[2];
    float t1 = m.r1[0] * pt[0] + m.r1[1] * pt[1] + m.r1[2];
    float t2 = m.r2[0] * pt[0] + m.r2[1] * pt[1] + m.r2[2];
    float2 out = float2(t0 / t2, t1 / t2);
    float3 d_t = float3(d_out[0] / t2,
                        d_out[1] / t2,
                        -(d_out[0] * out[0] + d_out[1] * out[1]) / t2);
    d_m.r0 += float3(d_t[0] * pt[0], d_t[0] * pt[1], d_t[0]);
    d_m.r1 += float3(d_t[1] * pt[0], d_t[1] * pt[1], d_t[1]);
    d_m.r2 += float3(d_t[2] * pt[0], d_t[2] * pt[1], d_t[2]);
    d_pt[0] += d_t[0] * m.r0[0] + d_t[1] * m.r1[0] + d_t[2] * m.r2[0];
    d_pt[1] += d_t[0] * m.r0[1] + d_t[1] * m.r1[1] + d_t[2] * m.r2[1];
}

// matrix.h xform_normal
inline float2 xform_normal(Mat3 m_inv, float2 n) {
    return normalize(float2(m_inv.r0[0] * n[0] + m_inv.r1[0] * n[1],
                            m_inv.r0[1] * n[0] + m_inv.r1[1] * n[1]));
}

// ---------------------------------------------------------------------------
// Gradient writes
//
// Passing a kernel's atomic output buffer into a helper function crashes the
// Metal compiler service (plain pointer, template or element reference alike).
// So helpers never touch output buffers: they push (index, value) pairs into a
// GradWrites, and the kernel body flushes them:
//     for (int k = 0; k < g.n; k++)
//         atomic_fetch_add_explicit(&d_floats[g.idx[k]], g.val[k], memory_order_relaxed);
// Flush after each fragment / boundary sample to keep n small.
constant int MAX_GRAD_WRITES = 64;

struct GradWrites {
    int idx[MAX_GRAD_WRITES];
    float val[MAX_GRAD_WRITES];
    int n;
};

inline void gw_clear(thread GradWrites &g) { g.n = 0; }

inline void gw_push(thread GradWrites &g, int k, float v) {
    if (g.n < MAX_GRAD_WRITES) {
        g.idx[g.n] = k;
        g.val[g.n] = v;
        g.n++;
    }
}

// Pushes a 3x3 gradient at `off` (row-major)
inline void gw_mat3(thread GradWrites &g, int off, Mat3 m) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            gw_push(g, off + 3 * i + j, mat_at(m, i, j));
        }
    }
}

// ---------------------------------------------------------------------------
// PCG32 (pcg.h). Bit-exact with the CPU implementation.
struct pcg32_state {
    uint64_t state;
    uint64_t inc;
};

inline uint32_t next_pcg32(thread pcg32_state &rng) {
    uint64_t oldstate = rng.state;
    rng.state = oldstate * 6364136223846793005ULL + (rng.inc | 1ULL);
    uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = uint32_t(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

inline float next_pcg32_float(thread pcg32_state &rng) {
    // CPU: the bits (r >> 9) | 0x3f800000 read as a float, minus 1.
    // That float is 1 + (r >> 9) * 2^-23, so this is exact.
    return float(next_pcg32(rng) >> 9) * (1.0f / 8388608.0f);
}

inline pcg32_state init_pcg32(int idx, uint64_t seed) {
    pcg32_state st;
    st.state = 0ULL;
    st.inc = ((uint64_t(idx) + 1ULL) << 1u) | 1ULL;
    next_pcg32(st);
    st.state += (0x853c49e6748fea9bULL + seed);
    next_pcg32(st);
    return st;
}

inline uint64_t scene_seed(SceneView s) {
    return (uint64_t(uint32_t(s.ip[IP_SEED_HI])) << 32) | uint64_t(uint32_t(s.ip[IP_SEED_LO]));
}

// ---------------------------------------------------------------------------
// Pixel filters (filter.h)
inline float filter_radius(SceneView s) { return s.f[s.ip[IP_FILTER_RADIUS_OFF]]; }

inline float compute_filter_weight(int type, float radius, float dx, float dy) {
    if (fabs(dx) > radius || fabs(dy) > radius) {
        return 0;
    }
    if (type == FILTER_BOX) {
        return 1.f / square(2 * radius);
    } else if (type == FILTER_TENT) {
        return (radius - fabs(dx)) * (radius - fabs(dy)) / square(square(radius));
    } else if (type == FILTER_RADIAL_PARABOLIC) {
        return (4.f / 3.f) * (1 - square(dx / radius)) *
               (4.f / 3.f) * (1 - square(dy / radius));
    } else {
        float ndx = (dx / (2 * radius)) + 0.5f;
        float ndy = (dy / (2 * radius)) + 0.5f;
        return 0.5f * (1.f - cos(2 * DIFFVG_PI * ndx)) *
               0.5f * (1.f - cos(2 * DIFFVG_PI * ndy)) /
               square(radius);
    }
}

// d compute_filter_weight / d radius (filter.h filter_weight_d_radius): zero
// outside the closed support; the box filter's jump at the border is not
// differentiated.
inline float filter_weight_d_radius(int type, float r, float dx, float dy) {
    if (fabs(dx) > r || fabs(dy) > r) {
        return 0;
    }
    if (type == FILTER_BOX) {
        return -2 / cubic(2 * r) * 2;
    } else if (type == FILTER_TENT) {
        float fx = r - fabs(dx);
        float fy = r - fabs(dy);
        float r4 = square(square(r));
        return (fx + fy) / r4 - 4 * fx * fy / (r4 * r);
    } else if (type == FILTER_RADIAL_PARABOLIC) {
        float gx = 1 - square(dx / r);
        float gy = 1 - square(dy / r);
        float r3 = r * r * r;
        float d_gx = 2 * square(dx) / r3;
        float d_gy = 2 * square(dy) / r3;
        return (16.f / 9.f) * (d_gx * gy + gx * d_gy);
    } else {
        float ndx = (dx / (2 * r)) + 0.5f;
        float ndy = (dy / (2 * r)) + 0.5f;
        float hx = 0.5f * (1.f - cos(2 * DIFFVG_PI * ndx));
        float hy = 0.5f * (1.f - cos(2 * DIFFVG_PI * ndy));
        float d_hx = 0.5f * sin(2 * DIFFVG_PI * ndx) * (2 * DIFFVG_PI) * (-dx / (2 * r * r));
        float d_hy = 0.5f * sin(2 * DIFFVG_PI * ndy) * (2 * DIFFVG_PI) * (-dy / (2 * r * r));
        return (d_hx * hy + hx * d_hy) / square(r) - 2 * hx * hy / cubic(r);
    }
}

// Returns d(radius) for filter.h d_compute_filter_weight
inline float d_compute_filter_weight(int type, float radius, float dx, float dy, float d_return) {
    return d_return * filter_weight_d_radius(type, radius, dx, dy);
}

// ---------------------------------------------------------------------------
// Sample positions (weight_kernel / render_kernel in diffvg.cpp)
// idx enumerates height * width * num_samples_y * num_samples_x.
inline float2 sample_position(SceneView s, int idx, thread int &x, thread int &y) {
    int nsx = s.ip[IP_NSX];
    int nsy = s.ip[IP_NSY];
    int width = s.ip[IP_W];
    pcg32_state rng = init_pcg32(idx, scene_seed(s));
    int sx = idx % nsx;
    int sy = (idx / nsx) % nsy;
    x = (idx / (nsx * nsy)) % width;
    y = (idx / (nsx * nsy * width));
    float rx = next_pcg32_float(rng);
    float ry = next_pcg32_float(rng);
    if (s.ip[IP_USE_PREFILTERING] != 0) {
        rx = ry = 0.5f;
    }
    return float2(x + ((float)sx + rx) / nsx,
                  y + ((float)sy + ry) / nsy);
}
