// Distance derivatives for the diffvg CUDA kernels.
//
// CUDA translation of pydiffvg/metal/distance_grad.metal: ports of diffvg.cpp
// smoothstep / d_smoothstep / prefilter_stroke_radius (+ its backward) and
// compute_distance.h d_closest_point / d_compute_distance.
// Header order: common -> geometry -> color -> backward -> distance_grad.
//
// See the TRANSLATION RULES header comment in common.cu. Mechanical changes:
// `inline` -> `__device__ inline`, `thread T&` -> `T&`, and the math renames
// (fabs -> fabsf, sqrt -> sqrtf, cos/sin -> cosf/sinf, atan2 -> atan2f,
// clamp -> clampf, max -> fmaxf).
//
// NOTE: Metal's built-in 3-argument smoothstep(edge0, edge1, x) coexists with
// the 1-argument overload defined below. CUDA has no smoothstep at all, so the
// 1-argument definition here is the only one; nothing in this port calls a
// 3-argument smoothstep.
//
// Rules (DESIGN.md "Mechanism"): float32 only, guarded divisions, no output
// buffers: every gradient is pushed into a GradWrites at the pool index of its
// value (mirror pool).
//
// d_t_root (optional, default 0): extra gradient w.r.t. the closest-point
// parameter t of a path segment, for callers whose output also depends on t
// (prefiltered strokes with per-point thickness: d_radius * dr/dt). It is
// applied through the same t derivative as the closest point (clamped line
// projection / implicit function theorem), exactly as compute_distance.h.
//
// Capacity: one d_compute_distance call pushes at most 8 (cubic path points)
// + 9 (shape_to_canvas) = 17 entries; d_path_thickness_at at most 4.
//
// DIFFERENCES FROM THE CPU DERIVATIVES (deliberate; checked against central
// finite differences of geometry compute_distance):
//  1. Path segments of every degree use one implicit-function-theorem
//     formulation on G(t) = (q(t) - pt) . q'(t) = 0 with the Bernstein basis:
//         dt = -(dG/dtheta) / G'(t),  G'(t) = q'.q' + (q - pt).q''
//     The CPU instead differentiates power-basis coefficients. That code has
//     bugs: the line branch drops the dt terms (d t/d p0, p1, pt), which only
//     matters for non-conformal shape_to_canvas; the quadratic branch
//     re-declares d_p0/d_p1/d_p2 inside the interior-t block, so all point
//     gradients of an interior closest point are discarded (and its d_D term
//     for p0 carries a spurious factor 2); the cubic branch divides by the
//     leading coefficient A, which is 0 for a degree-elevated line.
//  2. The ellipse, circle and rect derivatives follow the CPU formulas.

// ---------------------------------------------------------------------------
// diffvg.cpp smoothstep / d_smoothstep (1-argument forms)
__device__ inline float smoothstep(float d) {
    float t = clampf((d + 1.f) / 2.f, 0.f, 1.f);
    return t * t * (3 - 2 * t);
}

__device__ inline float d_smoothstep(float d, float d_ret) {
    if (d < -1.f || d > 1.f) {
        return 0.f;
    }
    float t = (d + 1.f) / 2.f;
    // ret = 3 t^2 - 2 t^3
    float d_t = d_ret * (6 * t - 6 * t * t);
    return d_t / 2.f;
}

// ---------------------------------------------------------------------------
// Stroke half-width at a closest point (diffvg.cpp prefilter_stroke_radius).
// Paths with per-point thickness serialize a dummy stroke_width of 0; for them
// the thickness is interpolated at (base_point_id, point_id, t) with the
// Bezier basis weights used by accumulate_boundary_gradient. Every other shape
// (or an invalid info) returns shape_stroke_width.
__device__ inline float path_thickness_at(SceneView s, int shape_id, ClosestPointPathInfo info) {
    if (shape_type(s, shape_id) != SHAPE_PATH || !path_has_thickness(s, shape_id) ||
        info.base_point_id < 0) {
        return shape_stroke_width(s, shape_id);
    }
    int num_points = path_num_points(s, shape_id);
    int point_id = info.point_id;
    float t = info.t;
    float tt = 1 - t;
    int ncp = path_num_control_points(s, shape_id, info.base_point_id);
    if (ncp == 0) {
        return tt * path_thickness(s, shape_id, point_id) +
               t * path_thickness(s, shape_id, (point_id + 1) % num_points);
    } else if (ncp == 1) {
        return (tt * tt) * path_thickness(s, shape_id, point_id) +
               (2 * tt * t) * path_thickness(s, shape_id, point_id + 1) +
               (t * t) * path_thickness(s, shape_id, (point_id + 2) % num_points);
    } else if (ncp == 2) {
        return (tt * tt * tt) * path_thickness(s, shape_id, point_id) +
               (3 * tt * tt * t) * path_thickness(s, shape_id, point_id + 1) +
               (3 * tt * t * t) * path_thickness(s, shape_id, point_id + 2) +
               (t * t * t) * path_thickness(s, shape_id, (point_id + 3) % num_points);
    }
    return shape_stroke_width(s, shape_id);
}

// d path_thickness_at / d t (diffvg.cpp prefilter_stroke_radius_dt): 0 unless
// the shape is a path with per-point thickness.
__device__ inline float path_thickness_dt_at(SceneView s, int shape_id, ClosestPointPathInfo info) {
    if (shape_type(s, shape_id) != SHAPE_PATH || !path_has_thickness(s, shape_id) ||
        info.base_point_id < 0) {
        return 0.f;
    }
    int num_points = path_num_points(s, shape_id);
    int point_id = info.point_id;
    float t = info.t;
    float tt = 1 - t;
    int ncp = path_num_control_points(s, shape_id, info.base_point_id);
    if (ncp == 0) {
        return path_thickness(s, shape_id, (point_id + 1) % num_points) - path_thickness(s, shape_id, point_id);
    } else if (ncp == 1) {
        float r0 = path_thickness(s, shape_id, point_id);
        float r1 = path_thickness(s, shape_id, point_id + 1);
        float r2 = path_thickness(s, shape_id, (point_id + 2) % num_points);
        return 2 * tt * (r1 - r0) + 2 * t * (r2 - r1);
    } else if (ncp == 2) {
        float r0 = path_thickness(s, shape_id, point_id);
        float r1 = path_thickness(s, shape_id, point_id + 1);
        float r2 = path_thickness(s, shape_id, point_id + 2);
        float r3 = path_thickness(s, shape_id, (point_id + 3) % num_points);
        return 3 * tt * tt * (r1 - r0) + 6 * tt * t * (r2 - r1) + 3 * t * t * (r3 - r2);
    }
    return 0.f;
}

// Backward of path_thickness_at with t held fixed (the t dependence goes
// through d_compute_distance's d_t_root, see path_thickness_dt_at). d_w goes
// to the per-point thickness entries, or to stroke_width when there are none.
__device__ inline void d_path_thickness_at(SceneView s, int shape_id, ClosestPointPathInfo info, float d_w,
                                           GradWrites &g) {
    if (shape_type(s, shape_id) != SHAPE_PATH || !path_has_thickness(s, shape_id) ||
        info.base_point_id < 0) {
        gw_push(g, shape_f_off(s, shape_id), d_w);
        return;
    }
    int num_points = path_num_points(s, shape_id);
    int off = path_thickness_off(s, shape_id);
    int point_id = info.point_id;
    float t = info.t;
    float tt = 1 - t;
    int ncp = path_num_control_points(s, shape_id, info.base_point_id);
    if (ncp == 0) {
        gw_push(g, off + point_id, tt * d_w);
        gw_push(g, off + (point_id + 1) % num_points, t * d_w);
    } else if (ncp == 1) {
        gw_push(g, off + point_id, (tt * tt) * d_w);
        gw_push(g, off + point_id + 1, (2 * tt * t) * d_w);
        gw_push(g, off + (point_id + 2) % num_points, (t * t) * d_w);
    } else if (ncp == 2) {
        gw_push(g, off + point_id, (tt * tt * tt) * d_w);
        gw_push(g, off + point_id + 1, (3 * tt * tt * t) * d_w);
        gw_push(g, off + point_id + 2, (3 * tt * t * t) * d_w);
        gw_push(g, off + (point_id + 3) % num_points, (t * t * t) * d_w);
    }
}

// ---------------------------------------------------------------------------
// d_closest_point (compute_distance.h). pt is the shape-local query point,
// d_q the gradient w.r.t. the shape-local closest point. Shape gradients are
// pushed into g; the gradient w.r.t. pt is accumulated into d_pt.

__device__ inline void d_closest_point_circle(SceneView s, int sid, float2 pt, float2 d_q,
                                              GradWrites &g, float2 &d_pt) {
    // q = c + r * normalize(pt - c)
    float2 c = circle_center(s, sid);
    float r = circle_radius(s, sid);
    float2 dir = pt - c;
    float l2 = length_squared(dir);
    if (l2 < 1e-20f) {
        return;
    }
    float l = sqrtf(l2);
    float2 n = dir / l;
    // d_normalize(dir, r * d_q) = (r / l) (d_q - n (n . d_q))
    float2 d_dir = (r / l) * (d_q - n * dot(n, d_q));
    float2 d_center = d_q - d_dir;
    int off = shape_f_off(s, sid);
    gw_push(g, off + 1, dot(d_q, n));
    gw_push(g, off + 2, d_center.x);
    gw_push(g, off + 3, d_center.y);
    d_pt += d_dir;
}

__device__ inline void d_closest_point_ellipse(SceneView s, int sid, float2 pt, float2 d_q,
                                               GradWrites &g, float2 &d_pt) {
    // q(theta) = c + (a cos(theta), b sin(theta)), theta the root of
    // F(theta) = (q - pt) . q'(theta) = 0. theta is differentiated implicitly.
    float2 cp = float2(0);
    if (!closest_point_ellipse(s, sid, pt, cp)) {
        return;
    }
    float2 radius = ellipse_radius(s, sid);
    float a = radius.x;
    float b = radius.y;
    float2 c = ellipse_center(s, sid);
    float2 rel = cp - c;
    float theta = atan2f(rel.y / b, rel.x / a);
    float ct = cosf(theta);
    float st = sinf(theta);
    float2 u = pt - c;
    float2 d_c = d_q;
    float d_a = d_q.x * ct;
    float d_b = d_q.y * st;
    float d_theta = -d_q.x * a * st + d_q.y * b * ct;
    // F = (b^2 - a^2) st ct + a ux st - b uy ct
    float F_theta = (b * b - a * a) * (ct * ct - st * st) + a * u.x * ct + b * u.y * st;
    float2 d_u = float2(0);
    if (fabsf(F_theta) > 1e-8f * fmaxf(a * a + b * b, 1e-8f)) {
        float k = -d_theta / F_theta;
        d_u.x = k * (a * st);
        d_u.y = k * (-b * ct);
        d_a += k * (-2 * a * st * ct + u.x * st);
        d_b += k * (2 * b * st * ct - u.y * ct);
    }
    // u = pt - c
    d_c -= d_u;
    d_pt += d_u;
    int off = shape_f_off(s, sid);
    gw_push(g, off + 1, d_a);
    gw_push(g, off + 2, d_b);
    gw_push(g, off + 3, d_c.x);
    gw_push(g, off + 4, d_c.y);
}

// Projection of pt onto segment p0p1 (closest_point_rect / the line branch of
// closest_point_path): q = p0 + t (p1 - p0), t = (pt - p0).(p1 - p0) / |p1 - p0|^2,
// clamped. Accumulates into d_p0, d_p1, d_pt. d_t_extra is an additional
// gradient w.r.t. t (used only in the unclamped regime).
__device__ inline void dg_d_project_segment(float2 p0, float2 p1, float2 pt, float t, float2 d_q, float d_t_extra,
                                            float2 &d_p0, float2 &d_p1, float2 &d_pt) {
    float2 e = p1 - p0;
    float ll = dot(e, e);
    if (t < 0 || !(ll > 0)) {
        d_p0 += d_q;
    } else if (t > 1) {
        d_p1 += d_q;
    } else {
        d_p0 += d_q * (1 - t);
        d_p1 += d_q * t;
        float d_t = dot(d_q, e) + d_t_extra;
        float d_num = d_t / ll;
        float d_den = d_t * (-t) / ll;
        // num = (pt - p0) . (p1 - p0)
        d_pt += e * d_num;
        d_p1 += (pt - p0) * d_num;
        d_p0 += ((p0 - p1) + (p0 - pt)) * d_num;
        // den = (p1 - p0) . (p1 - p0)
        d_p1 += 2 * e * d_den;
        d_p0 -= 2 * e * d_den;
    }
}

__device__ inline void d_closest_point_rect(SceneView s, int sid, float2 pt, float2 d_q,
                                            GradWrites &g, float2 &d_pt) {
    float2 p_min = rect_p_min(s, sid);
    float2 p_max = rect_p_max(s, sid);
    // corners: 0 left_top = p_min, 1 right_top, 2 left_bottom, 3 right_bottom = p_max
    float2 ps[4] = {p_min, float2(p_max.x, p_min.y), float2(p_min.x, p_max.y), p_max};
    // left, top, right, bottom (same order and tie-breaking as closest_point_rect)
    int e0[4] = {0, 0, 1, 2};
    int e1[4] = {2, 1, 3, 3};
    float min_dist = 0;
    int min_k = 0;
    float min_t = 0;
    for (int k = 0; k < 4; k++) {
        float2 p0 = ps[e0[k]];
        float2 p1 = ps[e1[k]];
        float ll = dot(p1 - p0, p1 - p0);
        float t = ll != 0 ? dot(pt - p0, p1 - p0) / ll : 0.f;
        float2 q = t < 0 ? p0 : (t > 1 ? p1 : p0 + t * (p1 - p0));
        float d = distance(q, pt);
        if (k == 0 || d < min_dist) {
            min_dist = d;
            min_k = k;
            min_t = t;
        }
    }
    float2 d_c[4] = {float2(0), float2(0), float2(0), float2(0)};
    float2 d_p0 = float2(0), d_p1 = float2(0);
    dg_d_project_segment(ps[e0[min_k]], ps[e1[min_k]], pt, min_t, d_q, 0.f, d_p0, d_p1, d_pt);
    d_c[e0[min_k]] += d_p0;
    d_c[e1[min_k]] += d_p1;
    float2 d_p_min = d_c[0] + float2(d_c[2].x, d_c[1].y);
    float2 d_p_max = d_c[3] + float2(d_c[1].x, d_c[2].y);
    int off = shape_f_off(s, sid);
    gw_push(g, off + 1, d_p_min.x);
    gw_push(g, off + 2, d_p_min.y);
    gw_push(g, off + 3, d_p_max.x);
    gw_push(g, off + 4, d_p_max.y);
}

__device__ inline void d_closest_point_path(SceneView s, int sid, float2 pt, float2 d_q, ClosestPointPathInfo info,
                                            float d_t_root, GradWrites &g, float2 &d_pt) {
    if (info.base_point_id < 0) {
        return;
    }
    int num_points = path_num_points(s, sid);
    int ncp = path_num_control_points(s, sid, info.base_point_id);
    if (ncp < 0 || ncp > 2) {
        return;
    }
    int n = ncp + 1;  // degree
    int idx[4];
    float2 P[4];      // control points relative to pt
    for (int i = 0; i < 4; i++) {
        idx[i] = 0;
        P[i] = float2(0);
    }
    for (int i = 0; i <= n; i++) {
        idx[i] = i == n ? (info.point_id + i) % num_points : info.point_id + i;
        P[i] = path_point(s, sid, idx[i]) - pt;
    }
    float t = info.t;
    float tt = 1 - t;
    // Bernstein basis B, B', B'' at t
    float B[4] = {0, 0, 0, 0};
    float dB[4] = {0, 0, 0, 0};
    float ddB[4] = {0, 0, 0, 0};
    if (n == 1) {
        B[0] = tt; B[1] = t;
        dB[0] = -1; dB[1] = 1;
    } else if (n == 2) {
        B[0] = tt * tt; B[1] = 2 * tt * t; B[2] = t * t;
        dB[0] = -2 * tt; dB[1] = 2 * (tt - t); dB[2] = 2 * t;
        ddB[0] = 2; ddB[1] = -4; ddB[2] = 2;
    } else {
        B[0] = tt * tt * tt; B[1] = 3 * tt * tt * t; B[2] = 3 * tt * t * t; B[3] = t * t * t;
        dB[0] = -3 * tt * tt; dB[1] = 3 * tt * (tt - 2 * t); dB[2] = 3 * t * (2 * tt - t); dB[3] = 3 * t * t;
        ddB[0] = 6 * tt; ddB[1] = 6 * (3 * t - 2); ddB[2] = 6 * (1 - 3 * t); ddB[3] = 6 * t;
    }
    float2 d_P[4] = {float2(0), float2(0), float2(0), float2(0)};
    if (n == 1) {
        // Straight line: the forward pass projects and clamps; t_root is 0 / 1
        // in the clamped regimes.
        float2 p0 = path_point(s, sid, idx[0]);
        float2 p1 = path_point(s, sid, idx[1]);
        float ll = dot(p1 - p0, p1 - p0);
        float tl = ll != 0 ? dot(pt - p0, p1 - p0) / ll : -1.f;
        dg_d_project_segment(p0, p1, pt, tl, d_q, d_t_root, d_P[0], d_P[1], d_pt);
    } else if (t == 0) {
        d_P[0] += d_q;
    } else if (t == 1) {
        d_P[n] += d_q;
    } else {
        float2 q = float2(0), dq = float2(0), ddq = float2(0);  // q is relative to pt
        for (int i = 0; i <= n; i++) {
            q += B[i] * P[i];
            dq += dB[i] * P[i];
            ddq += ddB[i] * P[i];
        }
        for (int i = 0; i <= n; i++) {
            d_P[i] += B[i] * d_q;
        }
        // implicit function theorem on G(t, P, pt) = (q - pt) . q'
        float G_t = dot(dq, dq) + dot(q, ddq);
        float scale = dot(dq, dq) + length(q) * length(ddq);
        if (fabsf(G_t) > 1e-6f * scale && scale > 0) {
            float d_t = dot(d_q, dq) + d_t_root;
            float k = -d_t / G_t;
            // dG/dp_i = B_i q' + B'_i (q - pt); dG/dpt = -q'
            for (int i = 0; i <= n; i++) {
                d_P[i] += k * (B[i] * dq + dB[i] * q);
            }
            d_pt -= k * dq;
        }
    }
    int off = path_points_off(s, sid);
    for (int i = 0; i <= n; i++) {
        gw_push(g, off + 2 * idx[i], d_P[i].x);
        gw_push(g, off + 2 * idx[i] + 1, d_P[i].y);
    }
}

__device__ inline void d_closest_point(SceneView s, int shape_id, float2 pt, float2 d_closest_pt,
                                       ClosestPointPathInfo info, float d_t_root, GradWrites &g,
                                       float2 &d_pt) {
    int type = shape_type(s, shape_id);
    if (type == SHAPE_CIRCLE) {
        d_closest_point_circle(s, shape_id, pt, d_closest_pt, g, d_pt);
    } else if (type == SHAPE_ELLIPSE) {
        d_closest_point_ellipse(s, shape_id, pt, d_closest_pt, g, d_pt);
    } else if (type == SHAPE_PATH) {
        d_closest_point_path(s, shape_id, pt, d_closest_pt, info, d_t_root, g, d_pt);
    } else if (type == SHAPE_RECT) {
        d_closest_point_rect(s, shape_id, pt, d_closest_pt, g, d_pt);
    }
}

__device__ inline void d_closest_point(SceneView s, int shape_id, float2 pt, float2 d_closest_pt,
                                       ClosestPointPathInfo info, GradWrites &g, float2 &d_pt) {
    d_closest_point(s, shape_id, pt, d_closest_pt, info, 0.f, g, d_pt);
}

// ---------------------------------------------------------------------------
// compute_distance.h d_compute_distance. canvas_pt and closest_pt are canvas
// space (closest_pt as returned by compute_distance). Pushes the shape
// parameter gradients and d shape_to_canvas (at gf_off); accumulates
// d_translation -= d canvas_pt, as the CPU does.
__device__ inline Mat3 dg_mat3_mul(Mat3 a, Mat3 b) {
    Mat3 r;
    float3 c0 = float3(b.r0[0], b.r1[0], b.r2[0]);
    float3 c1 = float3(b.r0[1], b.r1[1], b.r2[1]);
    float3 c2 = float3(b.r0[2], b.r1[2], b.r2[2]);
    r.r0 = float3(dot(a.r0, c0), dot(a.r0, c1), dot(a.r0, c2));
    r.r1 = float3(dot(a.r1, c0), dot(a.r1, c1), dot(a.r1, c2));
    r.r2 = float3(dot(a.r2, c0), dot(a.r2, c1), dot(a.r2, c2));
    return r;
}

__device__ inline Mat3 dg_mat3_transpose(Mat3 a) {
    Mat3 r;
    r.r0 = float3(a.r0[0], a.r1[0], a.r2[0]);
    r.r1 = float3(a.r0[1], a.r1[1], a.r2[1]);
    r.r2 = float3(a.r0[2], a.r1[2], a.r2[2]);
    return r;
}

__device__ inline void d_compute_distance(SceneView s, int group_id, int shape_id, float2 canvas_pt, float2 closest_pt,
                                          ClosestPointPathInfo info, float d_dist, float d_t_root, GradWrites &g,
                                          float2 &d_translation) {
    float2 diff = canvas_pt - closest_pt;
    float l2 = dot(diff, diff);
    if (l2 < 1e-10f) {
        // The derivative at distance = 0 is undefined
        return;
    }
    Mat3 canvas_to_shape = group_canvas_to_shape(s, group_id);
    Mat3 shape_to_canvas = group_shape_to_canvas(s, group_id);
    float2 local_pt = xform_pt(canvas_to_shape, canvas_pt);
    float2 local_closest_pt = xform_pt(canvas_to_shape, closest_pt);
    // dist = |canvas_pt - closest_pt|
    float2 d_v = (d_dist / sqrtf(l2)) * diff;
    float2 d_pt = d_v;
    float2 d_closest_pt = -d_v;
    // closest_pt = xform_pt(shape_to_canvas, local_closest_pt)
    Mat3 d_s2c = mat3_zero();
    float2 d_local_closest_pt = float2(0);
    d_xform_pt(shape_to_canvas, local_closest_pt, d_closest_pt, d_s2c, d_local_closest_pt);
    float2 d_local_pt = float2(0);
    d_closest_point(s, shape_id, local_pt, d_local_closest_pt, info, d_t_root, g, d_local_pt);
    // local_pt = xform_pt(canvas_to_shape, canvas_pt)
    Mat3 d_c2s = mat3_zero();
    d_xform_pt(canvas_to_shape, canvas_pt, d_local_pt, d_c2s, d_pt);
    // canvas_to_shape = inverse(shape_to_canvas): dS = -C^T dC C^T
    Mat3 tc2s = dg_mat3_transpose(canvas_to_shape);
    Mat3 m = dg_mat3_mul(dg_mat3_mul(tc2s, d_c2s), tc2s);
    d_s2c.r0 -= m.r0;
    d_s2c.r1 -= m.r1;
    d_s2c.r2 -= m.r2;
    gw_mat3(g, group_gf_off(s, group_id), d_s2c);
    d_translation -= d_pt;
}

// Original signature (d_t_root = 0)
__device__ inline void d_compute_distance(SceneView s, int group_id, int shape_id, float2 canvas_pt, float2 closest_pt,
                                          ClosestPointPathInfo info, float d_dist, GradWrites &g,
                                          float2 &d_translation) {
    d_compute_distance(s, group_id, shape_id, canvas_pt, closest_pt, info, d_dist, 0.f, g, d_translation);
}
