// Backward-pass leaf functions for the diffvg CUDA kernels.
//
// CUDA translation of pydiffvg/metal/backward.metal.
// Header order: common.cu, geometry.cu, color.cu, backward.cu.
// Only common.cu is required by this file.
//
// Mirrors (float32, GradWrites instead of atomicAdd):
//   diffvg.cpp        d_sample_color, gather_d_color, accumulate_boundary_gradient,
//                     sample_boundary_kernel (generate_boundary_sample)
//   sample_boundary.h BoundaryData, offset_curve_speed, sample_boundary (all shapes)
//   cdf.h             sample
//   scene.cpp         compute_shape_length (path length, not exported to the pool)
//
// Gradient indices follow the mirror pool: the gradient of a value is pushed at
// the index of that value in `floats`.
//
// See the TRANSLATION RULES header comment in common.cu. Mechanical changes
// here: `inline` -> `__device__ inline`, `thread T&` -> `T&`,
// `const device float *` -> `const float *`, and the math renames.
//
// NOTE on gather_d_color: on Metal, INPUT buffers may be passed to helpers but
// OUTPUT buffers may not (a compiler-crash workaround). CUDA has no such rule,
// so the restriction is irrelevant here; the signature is unchanged so the two
// backends stay interchangeable.
//
// Deviations from the C++ code (all in degenerate cases where the CPU produces
// inf/NaN):
//   * zero-length denominators (cdf intervals, tangents, rect with w + h == 0,
//     zero cap + path length, zero radial-gradient radius) reject the sample
//     (pdf = 0) or take the branch the NaN comparison would take on the CPU;
//   * the circle normal at radius 0 is (cos, sin) instead of NaN.
//   These guards were needed on Metal (fast math may not handle inf/NaN) and
//   are kept on CUDA so both backends produce identical results, even though
//   CUDA's IEEE semantics would reproduce the CPU's inf/NaN faithfully.

// ---------------------------------------------------------------------------
// sample_boundary.h data
struct PathBoundaryData {
    int base_point_id;
    int point_id;
    float t;
    // +-1 for a sample on a stroke flank (offset curve c + dir r n) of a path
    // with per-point thickness, whose normal is then the offset-curve normal;
    // 0 otherwise (fills, uniform-width strokes, caps / joins). sample_boundary.h.
    float offset_dir;
};

struct BoundaryData {
    PathBoundaryData path;
    bool is_stroke;
};

__device__ inline BoundaryData boundary_data_zero() {
    BoundaryData d;
    d.path.base_point_id = 0;
    d.path.point_id = 0;
    d.path.t = 0;
    d.path.offset_dir = 0;
    d.is_stroke = false;
    return d;
}

// ---------------------------------------------------------------------------
// d_sample_color (diffvg.cpp) for one colour record at `off`.
// Accumulates the derivative w.r.t. a translation of pt into d_translation
// (the kernel body adds it to the d_translation image).
__device__ inline void d_sample_color_param(SceneView s, int type, int off, int nstops, float2 pt,
                                            float4 d_color, GradWrites &g,
                                            float2 &d_translation) {
    if (type == COLOR_CONSTANT) {
        for (int k = 0; k < 4; k++) {
            gw_push(g, off + k, d_color[k]);
        }
        return;
    }
    if ((type != COLOR_LINEAR && type != COLOR_RADIAL) || nstops <= 0) {
        return;
    }
    const float *f = s.f;
    int so = off + 4;           // stop offsets
    int sc = off + 4 + nstops;  // stop colours (rgba per stop)
    int last = sc + 4 * (nstops - 1);

    float t = 0;
    float2 beg = float2(0), end = float2(0), l_vec = float2(0);
    float l = 1;
    float2 center = float2(0), radius = float2(1), offset = float2(0), normalized_offset = float2(0);
    if (type == COLOR_LINEAR) {
        beg = float2(f[off], f[off + 1]);
        end = float2(f[off + 2], f[off + 3]);
        l_vec = end - beg;
        l = fmaxf(dot(l_vec, l_vec), 1e-3f);
        t = dot(pt - beg, l_vec) / l;
    } else {
        center = float2(f[off], f[off + 1]);
        radius = float2(f[off + 2], f[off + 3]);
        if (radius.x == 0 || radius.y == 0) {
            // CPU: t is inf or NaN, which matches no stop interval.
            for (int k = 0; k < 4; k++) {
                gw_push(g, last + k, d_color[k]);
            }
            return;
        }
        offset = pt - center;
        normalized_offset = offset / radius;
        t = length(normalized_offset);
    }
    if (t < f[so]) {
        for (int k = 0; k < 4; k++) {
            gw_push(g, sc + k, d_color[k]);
        }
        return;
    }
    for (int i = 0; i < nstops - 1; i++) {
        float offset_curr = f[so + i];
        float offset_next = f[so + i + 1];
        if (t >= offset_curr && t < offset_next) {
            int ci = sc + 4 * i;
            int cn = sc + 4 * (i + 1);
            float4 color_curr = float4(f[ci], f[ci + 1], f[ci + 2], f[ci + 3]);
            float4 color_next = float4(f[cn], f[cn + 1], f[cn + 2], f[cn + 3]);
            float span = offset_next - offset_curr;  // > 0 inside this branch
            float tt = (t - offset_curr) / span;
            float4 d_color_curr = d_color * (1 - tt);
            float4 d_color_next = d_color * tt;
            float d_tt = dot(d_color, color_next - color_curr);
            float d_offset_next = -d_tt * tt / span;
            float d_offset_curr = d_tt * ((tt - 1.f) / span);
            float d_t = d_tt / span;
            for (int k = 0; k < 4; k++) {
                gw_push(g, ci + k, d_color_curr[k]);
            }
            for (int k = 0; k < 4; k++) {
                gw_push(g, cn + k, d_color_next[k]);
            }
            gw_push(g, so + i, d_offset_curr);
            gw_push(g, so + i + 1, d_offset_next);
            if (type == COLOR_LINEAR) {
                float2 d_beg = d_t * (-(pt - beg) - l_vec) / l;
                float2 d_end = d_t * (pt - beg) / l;
                float d_l = -d_t * t / l;
                if (dot(l_vec, l_vec) > 1e-3f) {
                    d_beg += 2 * d_l * (beg - end);
                    d_end += 2 * d_l * (end - beg);
                }
                gw_push(g, off + 0, d_beg.x);
                gw_push(g, off + 1, d_beg.y);
                gw_push(g, off + 2, d_end.x);
                gw_push(g, off + 3, d_end.y);
                d_translation += d_beg + d_end;
            } else {
                // d_length(normalized_offset, d_t)
                float2 d_normalized_offset = float2(0);
                if (t > 0) {
                    float d_l_sq = 0.5f * d_t / t;
                    d_normalized_offset = 2 * d_l_sq * normalized_offset;
                }
                float2 d_offset = d_normalized_offset / radius;
                float2 d_radius = -d_normalized_offset * offset / (radius * radius);
                float2 d_center = -d_offset;
                gw_push(g, off + 0, d_center.x);
                gw_push(g, off + 1, d_center.y);
                gw_push(g, off + 2, d_radius.x);
                gw_push(g, off + 3, d_radius.y);
                d_translation += d_center;
            }
            return;
        }
    }
    for (int k = 0; k < 4; k++) {
        gw_push(g, last + k, d_color[k]);
    }
}

// ---------------------------------------------------------------------------
// gather_d_color (diffvg.cpp). Width/height are the render size ip[IP_W], ip[IP_H].
// d_render_image (W*H*4) and weight_image (W*H) are kernel INPUT buffers.
__device__ inline float4 gather_d_color(SceneView s, const float *d_render_image,
                                        const float *weight_image, float2 pt) {
    int width = s.ip[IP_W];
    int height = s.ip[IP_H];
    int x = (int)(pt.x);
    int y = (int)(pt.y);
    float radius = filter_radius(s);
    int type = s.ip[IP_FILTER_TYPE];
    int ri = (int)(ceilf(radius));
    float4 d_color = float4(0);
    for (int dy = -ri; dy <= ri; dy++) {
        for (int dx = -ri; dx <= ri; dx++) {
            int xx = x + dx;
            int yy = y + dy;
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                float xc = xx + 0.5f;
                float yc = yy + 0.5f;
                float ddx = xc - pt.x;
                float ddy = yc - pt.y;
                float filter_weight = compute_filter_weight(type, radius, ddx, ddy);
                // Samples exactly on the border of a pixel footprint belong to
                // both neighbouring pixels: split the weight.
                if (fabsf(ddx) == radius) {
                    filter_weight *= 0.5f;
                }
                if (fabsf(ddy) == radius) {
                    filter_weight *= 0.5f;
                }
                int p = yy * width + xx;
                float weight_sum = weight_image[p];
                if (weight_sum > 0) {
                    d_color += (filter_weight / weight_sum) *
                        float4(d_render_image[4 * p], d_render_image[4 * p + 1],
                               d_render_image[4 * p + 2], d_render_image[4 * p + 3]);
                }
            }
        }
    }
    return d_color;
}

// ---------------------------------------------------------------------------
// cdf.h sample over floats[off .. off + n)
__device__ inline int cdf_sample(SceneView s, int off, int n, float u, float &updated_u) {
    const float *cdf = s.f;
    int lb = 0;
    int len = n - 1 - lb;
    while (len > 0) {
        int half_len = len / 2;
        int mid = lb + half_len;
        if (u < cdf[off + mid]) {
            len = half_len;
        } else {
            lb = mid + 1;
            len = len - half_len - 1;
        }
    }
    lb = iclamp(lb, 0, n - 1);
    float lo = lb > 0 ? cdf[off + lb - 1] : 0.f;
    float span = cdf[off + lb] - lo;
    // CPU: division by a zero-width interval gives inf/NaN; those intervals
    // have pmf 0, so any finite value leads to a rejected sample.
    updated_u = span > 0 ? (u - lo) / span : 0.f;
    if (lb == 0) {
        updated_u = cdf[off] > 0 ? u / cdf[off] : 0.f;
    }
    return lb;
}

__device__ inline int cdf_sample(SceneView s, int off, int n, float u) {
    float unused;
    return cdf_sample(s, off, n, u, unused);
}

// Picks a sample id (index into the edge sampling tables) from the scene CDF.
__device__ inline int sample_shape_cdf(SceneView s, float u) {
    return cdf_sample(s, s.ip[IP_SAMPLE_CDF_OFF], s.ip[IP_NUM_TOTAL_SHAPES], u);
}

// ---------------------------------------------------------------------------
// scene.cpp compute_shape_length for a path (scene.shapes_length is not in the
// exported pool; sample_boundary needs it for the stroke cap probability).
__device__ inline float path_shape_length(SceneView s, int sid) {
    // export_flat stores scene.shapes_length[sid] in the `d` slot of the shape
    // record (0 in older builds): use the CPU value when present.
    float exported = shape_param(s, sid, 3);
    if (exported > 0) {
        return exported;
    }
    int nbp = path_num_base_points(s, sid);
    int np = path_num_points(s, sid);
    float length = 0;
    for (int i = 0; i < nbp; i++) {
        int i0 = path_point_id_map(s, sid, i);
        int nc = path_num_control_points(s, sid, i);
        float2 p0 = path_point(s, sid, i0);
        if (nc == 0) {
            float2 p1 = path_point(s, sid, (i0 + 1) % np);
            length += distance(p1, p0);
        } else if (nc == 1) {
            float2 p1 = path_point(s, sid, i0 + 1);
            float2 p2 = path_point(s, sid, (i0 + 2) % np);
            float tt = 0.5f;
            float2 v1 = (tt * tt) * p0 + (2 * tt * 0.5f) * p1 + (0.5f * 0.5f) * p2;
            length += distance(v1, p0) + distance(v1, p2);
        } else {
            float2 p1 = path_point(s, sid, i0 + 1);
            float2 p2 = path_point(s, sid, i0 + 2);
            float2 p3 = path_point(s, sid, (i0 + 3) % np);
            float ta = 1.f / 3.f;
            float tb = 1 - ta;
            float2 v1 = (tb * tb * tb) * p0 + (3 * tb * tb * ta) * p1 + (3 * tb * ta * ta) * p2 + (ta * ta * ta) * p3;
            float tc = 2.f / 3.f;
            float td = 1 - tc;
            float2 v2 = (td * td * td) * p0 + (3 * td * td * tc) * p1 + (3 * td * tc * tc) * p2 + (tc * tc * tc) * p3;
            length += distance(v1, p0) + distance(v1, v2) + distance(v2, p3);
        }
    }
    return length;
}

// ---------------------------------------------------------------------------
// sample_boundary.h
__device__ inline float offset_curve_speed(float2 d1, float2 d2, float r, float dr, float dir) {
    float len = length(d1);
    if (!(len > 0)) {
        return 0;
    }
    float2 n = float2(-d1.y, d1.x) / len;
    float2 dn = float2(-d2.y, d2.x) / len - n * (dot(d1, d2) / (len * len));
    return length(d1 + dir * (dr * n + r * dn));
}

// Unit normal of the offset curve b = c + dir r n, oriented like n
// (sample_boundary.h offset_curve_normal). Requires |d1| > 0.
__device__ inline float2 offset_curve_normal(float2 d1, float2 d2, float r, float dr, float dir, float2 n) {
    float len = length(d1);
    if (!(len > 0)) {
        return n;
    }
    float2 dn = float2(-d2.y, d2.x) / len - n * (dot(d1, d2) / (len * len));
    float2 bt = d1 + dir * (dr * n + r * dn);
    float bl = length(bt);
    if (!(bl > 0)) {
        return n;
    }
    float2 nb = float2(-bt.y, bt.x) / bl;
    return dot(nb, n) < 0 ? -nb : nb;
}

__device__ inline float2 sample_boundary_circle(SceneView s, int sid, float t, float2 &normal,
                                                float &pdf, float dir, float stroke_radius) {
    float r = circle_radius(s, sid);
    float a = 2 * DIFFVG_PI * t;
    float ca = cosf(a);
    float sa = sinf(a);
    float2 offset = float2(r * ca, r * sa);
    float ol = length(offset);
    normal = ol > 0 ? offset / ol : float2(ca, sa);
    float boundary_radius = fabsf(r + dir * stroke_radius);
    if (!(boundary_radius > 0)) {
        pdf = 0;
        return float2(0);
    }
    pdf /= (2 * DIFFVG_PI * boundary_radius);
    float2 ret = circle_center(s, sid) + offset;
    if (dir != 0.f) {
        ret += dir * stroke_radius * normal;
        if (dir < 0) {
            normal = -normal;
        }
    }
    return ret;
}

__device__ inline float2 sample_boundary_ellipse(SceneView s, int sid, float t, float2 &normal,
                                                 float &pdf, BoundaryData &data,
                                                 float dir, float stroke_radius) {
    data.path.t = t;
    float2 r = ellipse_radius(s, sid);
    float a = 2 * DIFFVG_PI * t;
    float ca = cosf(a);
    float sa = sinf(a);
    float2 offset = float2(r.x * ca, r.y * sa);
    float dxdt = -r.x * sa * 2 * DIFFVG_PI;
    float dydt = r.y * ca * 2 * DIFFVG_PI;
    float tan_len = sqrtf(square(dxdt) + square(dydt));
    if (!(tan_len > 0)) {
        pdf = 0;
        return float2(0);
    }
    normal = float2(dydt, -dxdt) / tan_len;
    if (dir != 0.f) {
        float two_pi_sq = square(2 * DIFFVG_PI);
        float2 d2 = float2(-r.x * ca * two_pi_sq, -r.y * sa * two_pi_sq);
        float speed = offset_curve_speed(float2(dxdt, dydt), d2, stroke_radius, 0.f, -dir);
        if (!(speed > 0)) {
            pdf = 0;
            return float2(0);
        }
        pdf /= speed;
    } else {
        pdf /= tan_len;
    }
    float2 ret = ellipse_center(s, sid) + offset;
    if (dir != 0.f) {
        ret += dir * stroke_radius * normal;
        if (dir < 0) {
            normal = -normal;
        }
    }
    return ret;
}

__device__ inline int path_vertex_point_id(SceneView s, int sid, int k) {
    return k < path_num_base_points(s, sid) ? path_point_id_map(s, sid, k) : path_num_points(s, sid) - 1;
}

__device__ inline float2 sample_boundary_path(SceneView s, int sid, float t, float2 &normal,
                                              float &pdf, BoundaryData &data,
                                              float dir, float stroke_radius) {
    int nbp = path_num_base_points(s, sid);
    int np = path_num_points(s, sid);
    bool has_thickness = path_has_thickness(s, sid);
    if (nbp <= 0) {
        pdf = 0;
        return float2(0);
    }
    if (dir != 0.f) {
        // Round joins and caps: full circles around every vertex.
        int num_circles = path_is_closed(s, sid) ? nbp : nbp + 1;
        float cap_length = 0;
        if (has_thickness) {
            for (int k = 0; k < num_circles; k++) {
                cap_length += 2 * DIFFVG_PI * path_thickness(s, sid, path_vertex_point_id(s, sid, k));
            }
        } else {
            cap_length = 2 * DIFFVG_PI * stroke_radius * num_circles;
        }
        float total = cap_length + path_shape_length(s, sid);
        if (!(total > 0)) {
            pdf = 0;
            return float2(0);
        }
        float cap_prob = cap_length / total;
        if (t < cap_prob) {
            t = t / cap_prob;
            int k = imin((int)(t * num_circles), num_circles - 1);
            t = t * num_circles - k;
            int pid = path_vertex_point_id(s, sid, k);
            float r = has_thickness ? path_thickness(s, sid, pid) : stroke_radius;
            if (!(r > 0)) {
                pdf = 0;
                return float2(0);
            }
            pdf *= 2 * cap_prob / (float)(num_circles) / (2 * DIFFVG_PI * r);
            float2 p0 = path_point(s, sid, pid);
            float a = 2 * DIFFVG_PI * t;
            float2 dirv = float2(cosf(a), sinf(a));
            float2 offset = r * dirv;
            normal = dirv;  // normalize(offset), r > 0
            if (k < nbp) {
                data.path.base_point_id = k;
                data.path.point_id = pid;
                data.path.t = 0;
            } else {
                data.path.base_point_id = nbp - 1;
                data.path.point_id = path_point_id_map(s, sid, nbp - 1);
                data.path.t = 1;
            }
            return p0 + offset;
        } else {
            if (!(cap_prob < 1)) {
                pdf = 0;
                return float2(0);
            }
            t = (t - cap_prob) / (1 - cap_prob);
            pdf *= (1 - cap_prob);
        }
    }
    int sample_id = cdf_sample(s, path_cdf_off(s, sid), nbp, t, t);
    int point_id = path_point_id_map(s, sid, sample_id);
    int nc = path_num_control_points(s, sid, sample_id);
    data.path.base_point_id = sample_id;
    data.path.point_id = point_id;
    data.path.t = t;
    if (t < -1e-3f || t > 1 + 1e-3f) {
        pdf = 0;
        return float2(0);
    }
    float tt = 1 - t;
    float2 ret;
    float2 tangent;
    float2 d2 = float2(0);
    float r = stroke_radius;
    float dr = 0;
    if (nc == 0) {
        int i0 = point_id;
        int i1 = (i0 + 1) % np;
        float2 p0 = path_point(s, sid, i0);
        float2 p1 = path_point(s, sid, i1);
        tangent = p1 - p0;
        ret = p0 + t * (p1 - p0);
        if (has_thickness) {
            float r0 = path_thickness(s, sid, i0);
            float r1 = path_thickness(s, sid, i1);
            r = r0 + t * (r1 - r0);
            dr = r1 - r0;
        }
    } else if (nc == 1) {
        int i0 = point_id;
        int i1 = i0 + 1;
        int i2 = (i0 + 2) % np;
        float2 p0 = path_point(s, sid, i0);
        float2 p1 = path_point(s, sid, i1);
        float2 p2 = path_point(s, sid, i2);
        tangent = 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1);
        ret = (tt * tt) * p0 + (2 * tt * t) * p1 + (t * t) * p2;
        d2 = 2 * (p2 - 2 * p1 + p0);
        if (has_thickness) {
            float r0 = path_thickness(s, sid, i0);
            float r1 = path_thickness(s, sid, i1);
            float r2 = path_thickness(s, sid, i2);
            r = (tt * tt) * r0 + (2 * tt * t) * r1 + (t * t) * r2;
            dr = 2 * tt * (r1 - r0) + 2 * t * (r2 - r1);
        }
    } else {
        int i0 = point_id;
        int i1 = point_id + 1;
        int i2 = point_id + 2;
        int i3 = (point_id + 3) % np;
        float2 p0 = path_point(s, sid, i0);
        float2 p1 = path_point(s, sid, i1);
        float2 p2 = path_point(s, sid, i2);
        float2 p3 = path_point(s, sid, i3);
        tangent = 3 * square(1 - t) * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t * t * (p3 - p2);
        ret = (tt * tt * tt) * p0 + (3 * tt * tt * t) * p1 + (3 * tt * t * t) * p2 + (t * t * t) * p3;
        d2 = 6 * tt * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1);
        if (has_thickness) {
            float r0 = path_thickness(s, sid, i0);
            float r1 = path_thickness(s, sid, i1);
            float r2 = path_thickness(s, sid, i2);
            float r3 = path_thickness(s, sid, i3);
            r = (tt * tt * tt) * r0 + (3 * tt * tt * t) * r1 + (3 * tt * t * t) * r2 + (t * t * t) * r3;
            dr = 3 * tt * tt * (r1 - r0) + 6 * tt * t * (r2 - r1) + 3 * t * t * (r3 - r2);
        }
    }
    float tan_len = length(tangent);
    if (!(tan_len > 0)) {
        pdf = 0;
        return float2(0);
    }
    normal = float2(-tangent.y, tangent.x) / tan_len;
    pdf *= path_length_pmf(s, sid, sample_id) / tan_len;
    if (dir != 0.f) {
        float speed = offset_curve_speed(tangent, d2, r, dr, dir);
        if (!(speed > 0)) {
            pdf = 0;
            return float2(0);
        }
        pdf *= tan_len / speed;
        ret += dir * r * normal;
        if (has_thickness) {
            normal = offset_curve_normal(tangent, d2, r, dr, dir, normal);
            data.path.offset_dir = dir;
        }
        if (dir < 0) {
            normal = -normal;
        }
    }
    return ret;
}

__device__ inline float2 sample_boundary_rect(SceneView s, int sid, float t, float2 &normal,
                                              float &pdf, float dir, float stroke_radius) {
    float2 p_min = rect_p_min(s, sid);
    float2 p_max = rect_p_max(s, sid);
    float w = p_max.x - p_min.x;
    float h = p_max.y - p_min.y;
    if (dir > 0 && stroke_radius > 0) {
        float arc_length = 2 * DIFFVG_PI * stroke_radius;
        float edge_length = 2 * (w + h);
        float arc_prob = arc_length / (arc_length + edge_length);
        if (t < arc_prob) {
            t = t / arc_prob;
            pdf *= arc_prob / arc_length;
            int k = imin((int)(t * 4), 3);
            float theta = ((float)(k) + (t * 4 - (float)(k))) * DIFFVG_PI / 2;
            float2 corner = k == 0 ? p_max :
                            k == 1 ? float2(p_min.x, p_max.y) :
                            k == 2 ? p_min :
                                     float2(p_max.x, p_min.y);
            normal = float2(cosf(theta), sinf(theta));
            return corner + stroke_radius * normal;
        }
        if (!(arc_prob < 1)) {
            pdf = 0;
            return float2(0);
        }
        t = (t - arc_prob) / (1 - arc_prob);
        pdf *= (1 - arc_prob);
    }
    if (!(w + h > 0)) {
        pdf = 0;
        return float2(0);
    }
    pdf /= (2 * (w + h));
    float2 ret;
    if (t <= w / (w + h)) {
        t = w > 0 ? t * ((w + h) / w) : 0.f;
        if (t < 0.5f) {
            normal = float2(0, -1);
            ret = p_min + 2 * t * float2(w, 0.f);
        } else {
            normal = float2(0, 1);
            ret = float2(p_min.x, p_max.y) + 2 * (t - 0.5f) * float2(w, 0.f);
        }
    } else {
        t = h > 0 ? (t - w / (w + h)) * ((w + h) / h) : 0.f;
        if (t < 0.5f) {
            normal = float2(-1, 0);
            ret = p_min + 2 * t * float2(0.f, h);
        } else {
            normal = float2(1, 0);
            ret = float2(p_max.x, p_min.y) + 2 * (t - 0.5f) * float2(0.f, h);
        }
    }
    if (dir != 0.f) {
        ret += dir * stroke_radius * normal;
        if (dir < 0) {
            normal = -normal;
        }
    }
    return ret;
}

// Returns the LOCAL boundary point; normal is the local normal, pdf is w.r.t.
// local arc length (without the shape pmf).
__device__ inline float2 sample_boundary(SceneView s, int group_id, int shape_id, float t,
                                         float2 &normal, float &pdf,
                                         BoundaryData &data) {
    data = boundary_data_zero();
    normal = float2(0);
    pdf = 1;
    bool has_fill = group_has_fill(s, group_id);
    bool has_stroke = group_has_stroke(s, group_id);
    bool stroke_perturb = false;
    if (has_fill && has_stroke) {
        if (t < 0.5f) {
            stroke_perturb = false;
            t = 2 * t;
        } else {
            stroke_perturb = true;
            t = 2 * (t - 0.5f);
        }
        pdf = 0.5f;
    } else if (has_stroke) {
        stroke_perturb = true;
    }
    data.is_stroke = stroke_perturb;
    float dir = 0.f;
    if (stroke_perturb) {
        if (t < 0.5f) {
            dir = -1.f;
            t = 2 * t;
        } else {
            dir = 1.f;
            t = 2 * (t - 0.5f);
        }
        pdf *= 0.5f;
    }
    float stroke_radius = shape_stroke_width(s, shape_id);
    int type = shape_type(s, shape_id);
    if (type == SHAPE_CIRCLE) {
        return sample_boundary_circle(s, shape_id, t, normal, pdf, dir, stroke_radius);
    } else if (type == SHAPE_ELLIPSE) {
        return sample_boundary_ellipse(s, shape_id, t, normal, pdf, data, dir, stroke_radius);
    } else if (type == SHAPE_PATH) {
        return sample_boundary_path(s, shape_id, t, normal, pdf, data, dir, stroke_radius);
    } else if (type == SHAPE_RECT) {
        return sample_boundary_rect(s, shape_id, t, normal, pdf, dir, stroke_radius);
    }
    pdf = 0;
    return float2(0);
}

// ---------------------------------------------------------------------------
// sample_boundary_kernel (diffvg.cpp) for sample index idx. Returns false when
// the sample is invalid (C++: shape_id == -1). Uses sample_shapes_pmf[sample_id].
struct BoundarySample {
    float2 pt;            // canvas point normalised to [0, 1)
    float2 local_pt;
    float2 normal;        // canvas space
    float2 local_normal;  // shape space
    float local_velocity_scale;
    int shape_group_id;
    int shape_id;
    float t;
    BoundaryData data;
    float pdf;
};

__device__ inline bool generate_boundary_sample(SceneView s, int idx, BoundarySample &b) {
    b.pt = float2(0);
    b.local_pt = float2(0);
    b.normal = float2(0);
    b.local_normal = float2(0);
    b.local_velocity_scale = 0;
    b.shape_group_id = -1;
    b.shape_id = -1;
    b.t = 0;
    b.data = boundary_data_zero();
    b.pdf = 0;
    if (s.ip[IP_NUM_GROUPS] == 0 || s.ip[IP_NUM_TOTAL_SHAPES] == 0) {
        return false;
    }
    pcg32_state rng = init_pcg32(idx, scene_seed(s));
    float u = next_pcg32_float(rng);
    int sample_id = sample_shape_cdf(s, u);
    int shape_id = s.i[s.ip[IP_SAMPLE_SHAPE_ID_OFF] + sample_id];
    int group_id = s.i[s.ip[IP_SAMPLE_GROUP_ID_OFF] + sample_id];
    float shape_pmf = s.f[s.ip[IP_SAMPLE_PMF_OFF] + sample_id];
    if (!(shape_pmf > 0)) {
        return false;
    }
    float t = next_pcg32_float(rng);
    float2 normal;
    float boundary_pdf;
    BoundaryData data;
    float2 local_pt = sample_boundary(s, group_id, shape_id, t, normal, boundary_pdf, data);
    if (!(boundary_pdf > 0)) {
        return false;
    }
    Mat3 m = group_shape_to_canvas(s, group_id);
    Mat3 mi = group_canvas_to_shape(s, group_id);
    float2 boundary_pt = xform_pt(m, local_pt);
    float2 local_normal = normal;
    float2 local_tangent = float2(-local_normal.y, local_normal.x);
    float2 canvas_tangent = float2(m.r0[0] * local_tangent.x + m.r0[1] * local_tangent.y,
                                   m.r1[0] * local_tangent.x + m.r1[1] * local_tangent.y);
    float2 cn = float2(mi.r0[0] * local_normal.x + mi.r1[0] * local_normal.y,
                       mi.r0[1] * local_normal.x + mi.r1[1] * local_normal.y);
    float arc_length_jacobian = length(canvas_tangent);
    float canvas_normal_length = length(cn);
    if (!(arc_length_jacobian > 0) || !(canvas_normal_length > 0)) {
        return false;
    }
    boundary_pdf /= arc_length_jacobian;
    b.pt = float2(boundary_pt.x / (float)(s.ip[IP_CANVAS_W]), boundary_pt.y / (float)(s.ip[IP_CANVAS_H]));
    b.local_pt = local_pt;
    b.normal = cn / canvas_normal_length;
    b.local_normal = local_normal;
    b.local_velocity_scale = 1 / canvas_normal_length;
    b.shape_group_id = group_id;
    b.shape_id = shape_id;
    b.t = t;
    b.data = data;
    b.pdf = shape_pmf * boundary_pdf;
    return true;
}

// ---------------------------------------------------------------------------
// accumulate_boundary_gradient (diffvg.cpp), including the offset-curve
// (variable thickness) flank terms. At most 4 (thickness) + 8 (points)
// + 9 (shape_to_canvas) = 21 writes.
__device__ inline void accumulate_boundary_gradient(SceneView s, int shape_id, float canvas_contrib, float t,
                                                    float2 local_normal, BoundaryData data, int group_id,
                                                    float2 local_boundary_pt, float2 canvas_normal,
                                                    float local_velocity_scale, GradWrites &g) {
    (void)t;
    float contrib = canvas_contrib * local_velocity_scale;
    float2 normal = local_normal;
    int type = shape_type(s, shape_id);
    int f_off = shape_f_off(s, shape_id);
    float pt = data.path.t;
    float pt1 = 1 - pt;
    // Bezier weights of the sampled path segment (straight / quadratic / cubic)
    int pidx[4] = {0, 0, 0, 0};
    float pw[4] = {0, 0, 0, 0};
    int pn = 0;
    if (type == SHAPE_PATH) {
        int np = path_num_points(s, shape_id);
        int nc = path_num_control_points(s, shape_id, data.path.base_point_id);
        int i0 = data.path.point_id;
        if (nc == 0) {
            pidx[0] = i0; pw[0] = pt1;
            pidx[1] = (i0 + 1) % np; pw[1] = pt;
            pn = 2;
        } else if (nc == 1) {
            pidx[0] = i0; pw[0] = square(pt1);
            pidx[1] = i0 + 1; pw[1] = 2 * pt1 * pt;
            pidx[2] = (i0 + 2) % np; pw[2] = pt * pt;
            pn = 3;
        } else if (nc == 2) {
            pidx[0] = i0; pw[0] = cubic(pt1);
            pidx[1] = i0 + 1; pw[1] = 3 * square(pt1) * pt;
            pidx[2] = i0 + 2; pw[2] = 3 * pt1 * pt * pt;
            pidx[3] = (i0 + 3) % np; pw[3] = pt * pt * pt;
            pn = 4;
        }
    }
    // Stroke flank of a path with per-point thickness (diffvg.cpp): the
    // boundary point is b = c + dir r n and `normal` is the offset-curve
    // normal, so thickness velocities are dir B_i n and point velocities get
    // the rotation term dir r dn/dp_i.
    float flank_dir = 0;
    float2 flank_n = float2(0);
    float flank_len = 0;
    float flank_r = 0;
    float pdb[4] = {0, 0, 0, 0};
    if (data.is_stroke && type == SHAPE_PATH && data.path.offset_dir != 0 &&
            path_has_thickness(s, shape_id) && pn > 0) {
        float db[4] = {0, 0, 0, 0};
        if (pn == 2) {
            db[0] = -1; db[1] = 1;
        } else if (pn == 3) {
            db[0] = -2 * pt1; db[1] = 2 * (pt1 - pt); db[2] = 2 * pt;
        } else {
            db[0] = -3 * pt1 * pt1; db[1] = 3 * pt1 * (pt1 - 2 * pt);
            db[2] = 3 * pt * (2 * pt1 - pt); db[3] = 3 * pt * pt;
        }
        float2 d1 = float2(0);
        float r = 0;
        for (int k = 0; k < pn; k++) {
            d1 += db[k] * path_point(s, shape_id, pidx[k]);
            r += pw[k] * path_thickness(s, shape_id, pidx[k]);
            pdb[k] = db[k];
        }
        float len = length(d1);
        if (len > 0) {
            flank_dir = data.path.offset_dir;
            flank_n = float2(-d1.y, d1.x) / len;
            flank_len = len;
            flank_r = r;
        }
    }
    if (data.is_stroke) {
        if (type == SHAPE_PATH && path_has_thickness(s, shape_id)) {
            int th_off = path_thickness_off(s, shape_id);
            float thickness_contrib = flank_dir != 0 ? contrib * flank_dir * dot(flank_n, normal) : contrib;
            for (int k = 0; k < pn; k++) {
                gw_push(g, th_off + pidx[k], pw[k] * thickness_contrib);
            }
        } else {
            gw_push(g, f_off, contrib);
        }
    }
    if (type == SHAPE_CIRCLE) {
        gw_push(g, f_off + 2, normal.x * contrib);
        gw_push(g, f_off + 3, normal.y * contrib);
        float2 radial = local_boundary_pt - circle_center(s, shape_id);
        float radial_len = length(radial);
        if (radial_len > 0) {
            gw_push(g, f_off + 1, dot(radial / radial_len, normal) * contrib);
        }
    } else if (type == SHAPE_ELLIPSE) {
        gw_push(g, f_off + 3, normal.x * contrib);
        gw_push(g, f_off + 4, normal.y * contrib);
        float a = 2 * DIFFVG_PI * pt;
        gw_push(g, f_off + 1, cosf(a) * normal.x * contrib);
        gw_push(g, f_off + 2, sinf(a) * normal.y * contrib);
    } else if (type == SHAPE_PATH) {
        int p_off = path_points_off(s, shape_id);
        float gx = 0;
        float gy = 0;
        float fs = 0;
        if (flank_dir != 0) {
            float nx = flank_n.x;
            float ny = flank_n.y;
            gx = dot(float2(-nx * ny, 1 - ny * ny), normal);
            gy = dot(float2(nx * nx - 1, nx * ny), normal);
            fs = contrib * flank_dir * flank_r / flank_len;
        }
        for (int k = 0; k < pn; k++) {
            gw_push(g, p_off + 2 * pidx[k], pw[k] * normal.x * contrib + pdb[k] * fs * gx);
            gw_push(g, p_off + 2 * pidx[k] + 1, pw[k] * normal.y * contrib + pdb[k] * fs * gy);
        }
    } else if (type == SHAPE_RECT) {
        float2 center = 0.5f * (rect_p_min(s, shape_id) + rect_p_max(s, shape_id));
        gw_push(g, local_boundary_pt.x < center.x ? f_off + 1 : f_off + 3, normal.x * contrib);
        gw_push(g, local_boundary_pt.y < center.y ? f_off + 2 : f_off + 4, normal.y * contrib);
    }
    Mat3 d_m = mat3_zero();
    float2 d_local_pt = float2(0);
    d_xform_pt(group_shape_to_canvas(s, group_id), local_boundary_pt,
               canvas_normal * canvas_contrib, d_m, d_local_pt);
    gw_mat3(g, group_gf_off(s, group_id), d_m);
}
