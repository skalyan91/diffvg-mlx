// Geometry queries for the diffvg CUDA kernels.
//
// CUDA translation of pydiffvg/metal/geometry.metal (ports of winding_number.h,
// is_inside (diffvg.cpp), compute_distance.h (closest_point + compute_distance)
// and within_distance.h). Built on common.cu (SceneView, BVH/shape accessors,
// AABB, Mat3, DIFFVG_BIG, vector types, scalar helpers).
//
// See the TRANSLATION RULES header comment in common.cu. In this file the
// mechanical changes are: `inline` -> `__device__ inline`, `thread T&` -> `T&`,
// `constant` -> `static constexpr`, and the math renames (fabs -> fabsf, sqrt ->
// sqrtf, acos -> acosf, cos -> cosf, pow -> powf, float min/max -> fminf/fmaxf,
// int min/max -> imin/imax, float clamp -> clampf, int abs -> iabs). No
// algorithm, constant, branch or operation order is changed.
//
// Rules this file follows (see DESIGN.md "Mechanism"):
//  * float32 only (no double), no recursion, fixed-size stacks as in C++;
//  * no inf/NaN logic: every division / sqrt / acos / pow argument is guarded
//    explicitly. (The Metal build needs this because it is compiled with
//    math_mode "fast"; CUDA keeps IEEE semantics, but the guards are retained
//    verbatim so both backends take identical branches.)
//  * no output buffers are touched here: helpers return values or write into
//    reference parameters.
//
// FLOAT32 DEVIATIONS FROM THE CPU (marked "F32:" below):
//  1. Path winding number. Same half-open crossing rule as winding_number.h
//     (see wn_cubic_crossings): end points classified from the stored points,
//     curves split into y-monotone pieces at the roots of y'(t). The CPU runs
//     it in double; here quadratics/cubics are formed in Bernstein form in a
//     frame centred on the query point and scaled to O(1). Results differ only
//     for points within float precision of the curve.
//  2. Cubic Bezier closest point / within distance (quintic). The CPU builds
//     power-basis coefficients, normalises by the leading coefficient (NaN
//     when it is 0, e.g. a line written as a cubic with evenly spaced control
//     points: the CPU then only considers the end points) and uses isolator
//     polynomials solved in double with a 20-iteration Newton/bisection
//     (|f| < 1e-5 stop). Here the quintic (q(t)-pt).q'(t) is formed in
//     Bernstein form (translated to pt, scaled by the control-point extent)
//     and all its roots in [0,1] are isolated by subdivision (no division by
//     the leading coefficient). Root t agrees to ~1e-7; the GPU finds roots
//     the CPU misses (e.g. the line-as-cubic case, where the CPU is wrong by
//     up to the curve length). A segment whose capsule lower bound on the
//     distance exceeds the current best (closest point) or the maximum
//     radius (within distance) skips the root search; this cannot change the
//     result beyond float rounding (a relative tolerance guards the bound).
//  3. Ellipse closest point. The CPU bisects Eberly's root in double. Here the
//     bisection runs on u = s + 1 in float with r0 - 1 computed as
//     (e0 - e1)(e0 + e1)/e1^2, which avoids cancellation in s + 1 and s + r0;
//     the closest point agrees to ~1e-6 relative to the radius.
//  3b. Performance: root refinements stop as soon as the Newton step is below
//     1e-7; wn_cubic_crossings returns [below0] - [below3] without splitting
//     when the whole hull is right of the point and decides each piece's side
//     from its x-hull (blossoms) before root solving; within_distance rejects
//     cubics with a split-at-1/2 capsule bound before the quintic.
//  4. CUDA NOTE: where the Metal file warns that math_mode "fast" may
//     reassociate float expressions (so even the CPU-float code paths can
//     differ in the last bits), CUDA instead uses IEEE division and a
//     correctly rounded sqrtf, with fma contraction enabled. The CUDA results
//     therefore sit closer to the C++ core than the Metal ones do, but are not
//     bit-identical to either; this only matters for points exactly on a
//     boundary.

// ---------------------------------------------------------------------------
// edge_query.h / compute_distance.h structs
struct EdgeQuery {
    int shape_group_id;
    int shape_id;
    bool hit;  // Do we hit the specified shape_group_id & shape_id?
};

struct ClosestPointPathInfo {
    int base_point_id;
    int point_id;
    float t;  // t_root in C++
};

static constexpr int GEOM_PATH_BVH_STACK = 128;
static constexpr int GEOM_GROUP_BVH_STACK = 64;

// ---------------------------------------------------------------------------
// solve.h (float instantiation, same branch structure)
__device__ inline bool solve_quadratic(float a, float b, float c, float &t0, float &t1) {
    // From https://github.com/mmp/pbrt-v3/blob/master/src/core/pbrt.h#L419
    float discrim = b * b - 4 * a * c;
    if (discrim < 0) {
        return false;
    }
    float root_discrim = sqrtf(discrim);
    float q;
    if (b < 0) {
        q = -0.5f * (b - root_discrim);
    } else {
        q = -0.5f * (b + root_discrim);
    }
    // C++ divides by zero here (a == 0 gives t0 = +-inf, q == 0 gives
    // t1 = inf/NaN). Those roots are always rejected by the callers' [0, 1]
    // tests, so substitute an out-of-range sentinel instead.
    t0 = a != 0 ? q / a : DIFFVG_BIG;
    t1 = q != 0 ? c / q : DIFFVG_BIG;
    if (t0 > t1) {
        float tmp = t0;
        t0 = t1;
        t1 = tmp;
    }
    return true;
}

__device__ inline int solve_cubic(float a, float b, float c, float d,
                                  float &t0, float &t1, float &t2) {
    if (fabsf(a) < 1e-6f) {
        if (solve_quadratic(b, c, d, t0, t1)) {
            return 2;
        } else {
            return 0;
        }
    }
    // normalize cubic equation
    b /= a;
    c /= a;
    d /= a;
    float Q = (b * b - 3 * c) / 9.f;
    float R = (2 * b * b * b - 9 * b * c + 27 * d) / 54.f;
    float Q3 = Q * Q * Q;
    if (R * R < Q3) {
        // 3 real roots (Q3 > 0 here)
        float sQ = sqrtf(fmaxf(Q, 0.f));
        float theta = acosf(clampf(R / fmaxf(sqrtf(Q3), 1e-30f), -1.f, 1.f));
        t0 = -2.f * sQ * cosf(theta / 3.f) - b / 3.f;
        t1 = -2.f * sQ * cosf((theta + 2.f * DIFFVG_PI) / 3.f) - b / 3.f;
        t2 = -2.f * sQ * cosf((theta - 2.f * DIFFVG_PI) / 3.f) - b / 3.f;
        return 3;
    } else {
        float disc = sqrtf(fmaxf(R * R - Q3, 0.f));
        float A = R > 0 ? -powf(fmaxf(R + disc, 0.f), 1.f / 3.f) :
                           powf(fmaxf(-R + disc, 0.f), 1.f / 3.f);
        float B = fabsf(A) > 1e-6f ? Q / A : 0.f;
        t0 = (A + B) - b / 3.f;
        return 1;
    }
}

// ---------------------------------------------------------------------------
// F32: Bernstein-form polynomials on [0, 1] and robust root isolation.
// A polynomial of degree n <= 5 is stored by its Bernstein coefficients.
// Roots are isolated by de Casteljau subdivision with Descartes' rule of
// signs for Bernstein coefficients (the number of real roots in the interval
// is at most the number of coefficient sign changes, with the same parity):
// 0 changes -> no root, 1 change -> exactly one simple root (refined by
// safeguarded Newton), >= 2 -> split at the midpoint (depth limited).
static constexpr int BPOLY_MAX_DEPTH = 22;
static constexpr int BPOLY_STACK = 32;

struct BPoly {
    float c[6];
    int n;
};

struct Roots5 {
    float t[5];
    int n;
};

__device__ inline BPoly bpoly_zero(int n) {
    BPoly p;
    for (int i = 0; i < 6; i++) {
        p.c[i] = 0;
    }
    p.n = n;
    return p;
}

__device__ inline Roots5 roots5_empty() {
    Roots5 r;
    for (int i = 0; i < 5; i++) {
        r.t[i] = 0;
    }
    r.n = 0;
    return r;
}

// Value and derivative (w.r.t. t on the polynomial's own [0, 1]) in one pass.
__device__ inline float bpoly_eval_d(BPoly p, float t, float &d) {
    float b[6];
    for (int i = 0; i < 6; i++) {
        b[i] = p.c[i];
    }
    float s = 1 - t;
    for (int k = p.n; k > 1; k--) {
        for (int i = 0; i < k; i++) {
            b[i] = s * b[i] + t * b[i + 1];
        }
    }
    if (p.n >= 1) {
        d = (float)(p.n) * (b[1] - b[0]);
        return s * b[0] + t * b[1];
    }
    d = 0;
    return b[0];
}

__device__ inline float bpoly_eval(BPoly p, float t) {
    float d = 0;
    return bpoly_eval_d(p, t, d);
}

__device__ inline void bpoly_split(BPoly p, BPoly &left, BPoly &right) {
    float b[6];
    for (int i = 0; i < 6; i++) {
        b[i] = p.c[i];
    }
    left = bpoly_zero(p.n);
    right = bpoly_zero(p.n);
    left.c[0] = b[0];
    right.c[p.n] = b[p.n];
    for (int k = 1; k <= p.n; k++) {
        for (int i = 0; i <= p.n - k; i++) {
            b[i] = 0.5f * (b[i] + b[i + 1]);
        }
        left.c[k] = b[0];
        right.c[p.n - k] = b[p.n - k];
    }
}

// Number of sign changes of the nonzero coefficients; also the signs of the
// first and last nonzero coefficients (= signs of p just inside 0 and 1).
__device__ inline int bpoly_sign_changes(BPoly p, float &first, float &last) {
    int changes = 0;
    first = 0;
    last = 0;
    for (int i = 0; i <= p.n; i++) {
        float v = p.c[i];
        if (v != 0) {
            if (last != 0 && (v < 0) != (last < 0)) {
                changes++;
            }
            if (first == 0) {
                first = v;
            }
            last = v;
        }
    }
    return changes;
}

// The unique simple root in (0, 1) of p (one coefficient sign change).
// `first` is the sign of p just right of 0.
__device__ inline float bpoly_refine01(BPoly p, float first) {
    bool neg_lo = first < 0;
    float lo = 0, hi = 1;
    float f0 = p.c[0], f1 = p.c[p.n];
    float u = 0.5f;
    if (f0 != 0 && f1 != 0 && (f0 < 0) != (f1 < 0)) {
        u = clampf(f0 / (f0 - f1), 0.f, 1.f);  // regula falsi start
    }
    for (int it = 0; it < 40; it++) {
        float du = 0;
        float fu = bpoly_eval_d(p, u, du);
        if (fu == 0) {
            break;
        }
        if ((fu < 0) == neg_lo) {
            lo = u;
        } else {
            hi = u;
        }
        float mid = 0.5f * (lo + hi);
        float un = mid;
        if (du != 0) {
            float step = fu / du;
            float nt = u - step;
            if (fabsf(step) < 1e-7f) {
                // Converged: the Newton step is below the tolerance. (It can
                // round onto a bracket end, which would otherwise reject it
                // and bisect until the bracket is exhausted.)
                u = clampf(nt, lo, hi);
                break;
            }
            if (nt > lo && nt < hi) {
                un = nt;
            }
        }
        if (fabsf(un - u) < 1e-7f || !(mid > lo && mid < hi)) {
            u = un;
            break;
        }
        u = un;
    }
    return u;
}

struct BPolyEntry {
    BPoly p;
    float l;
    float r;
    int depth;
    bool own_left;  // report a root exactly at l (false for left children: shared with the parent)
};

// Roots of p in [0, 1] (unordered; at most 5). A root exactly at a
// subdivision point or at t = 0 / 1 is reported once. At the depth limit
// (clusters / multiple roots closer than 2^-22) the interval midpoint is
// reported. An identically zero polynomial has no roots.
__device__ inline Roots5 bpoly_roots01(BPoly p) {
    Roots5 res = roots5_empty();
    if (p.n <= 0) {
        return res;
    }
    if (p.c[p.n] == 0) {
        bool all_zero = true;
        for (int i = 0; i <= p.n; i++) {
            if (p.c[i] != 0) {
                all_zero = false;
            }
        }
        if (all_zero) {
            return res;
        }
        res.t[res.n++] = 1;
    }
    BPolyEntry stack[BPOLY_STACK];
    int sp = 0;
    stack[sp].p = p;
    stack[sp].l = 0;
    stack[sp].r = 1;
    stack[sp].depth = 0;
    stack[sp].own_left = true;
    sp++;
    while (sp > 0 && res.n < 5) {
        BPolyEntry e = stack[--sp];
        if (e.own_left && e.p.c[0] == 0) {
            res.t[res.n++] = e.l;
            if (res.n >= 5) {
                break;
            }
        }
        float first = 0, last = 0;
        int ch = bpoly_sign_changes(e.p, first, last);
        if (ch == 0) {
            continue;
        }
        if (ch == 1) {
            float u = bpoly_refine01(e.p, first);
            res.t[res.n++] = e.l + u * (e.r - e.l);
            continue;
        }
        float m = 0.5f * (e.l + e.r);
        if (e.depth >= BPOLY_MAX_DEPTH || !(m > e.l && m < e.r) || sp + 2 > BPOLY_STACK) {
            res.t[res.n++] = m;
            continue;
        }
        BPoly left, right;
        bpoly_split(e.p, left, right);
        // push right first so the left half is processed first
        stack[sp].p = right;
        stack[sp].l = m;
        stack[sp].r = e.r;
        stack[sp].depth = e.depth + 1;
        stack[sp].own_left = true;
        sp++;
        stack[sp].p = left;
        stack[sp].l = e.l;
        stack[sp].r = m;
        stack[sp].depth = e.depth + 1;
        stack[sp].own_left = false;
        sp++;
    }
    return res;
}

// Fixed-degree (cubic) Bernstein helpers on float4 coefficients.
__device__ inline float bez3_eval_d(float4 c, float t, float &d) {
    float s = 1 - t;
    float a0 = s * c[0] + t * c[1];
    float a1 = s * c[1] + t * c[2];
    float a2 = s * c[2] + t * c[3];
    float b0 = s * a0 + t * a1;
    float b1 = s * a1 + t * a2;
    d = 3 * (b1 - b0);
    return s * b0 + t * b1;
}

// ---------------------------------------------------------------------------
// Winding number crossing rule (shared with winding_number.h, which runs the
// same algorithm in double).
//
// A horizontal ray {y = pt.y, x > pt.x} is intersected with y-monotone curve
// pieces under a HALF-OPEN rule: a piece from value ya to yb crosses the ray
// iff (ya <= pt.y) != (yb <= pt.y); it contributes +1 if it starts below
// (ya <= pt.y, i.e. it runs upwards in y) and -1 otherwise, provided the
// crossing lies right of pt. The classification of a segment END POINT is
// computed from the stored point (p.y <= pt.y) with no arithmetic, so two
// segments sharing a vertex classify it identically: a ray through (or within
// float noise of) a vertex is counted once for a crossing and zero or two
// times (with opposite signs) for a touch, independent of root finding.
// Curves are split at the roots of y'(t) into y-monotone pieces; the value at
// a split point is evaluated once and its classification is shared by the two
// pieces it separates. Errors in the split values or in the root location only
// matter for points within float noise of the curve (a misplaced extremum can
// only add or remove a pair of opposite crossings next to it).

// Roots of a0 t^2 + a1 t + a2 = 0 strictly inside (0, 1), sorted; returns the
// count. Divisions are guarded so no result exceeds 1 in magnitude.
__device__ inline int wn_quadratic_roots01(float a0, float a1, float a2, float &r0, float &r1) {
    int n = 0;
    float t0 = 0, t1 = 0;
    if (a0 == 0) {
        if (fabsf(a2) < fabsf(a1)) {
            t0 = -a2 / a1;
            n = 1;
        }
    } else {
        float disc = a1 * a1 - 4 * a0 * a2;
        if (disc >= 0) {
            float sq = sqrtf(disc);
            float q = a1 < 0 ? -0.5f * (a1 - sq) : -0.5f * (a1 + sq);
            if (fabsf(q) < fabsf(a0)) {
                t0 = q / a0;
                n = 1;
            }
            if (fabsf(a2) < fabsf(q)) {
                if (n == 0) {
                    t0 = a2 / q;
                } else {
                    t1 = a2 / q;
                }
                n++;
            }
        }
    }
    int m = 0;
    r0 = 0;
    r1 = 0;
    if (n > 0 && t0 > 0 && t0 < 1) {
        r0 = t0;
        m = 1;
    }
    if (n > 1 && t1 > 0 && t1 < 1) {
        if (m == 0) {
            r0 = t1;
            m = 1;
        } else if (t1 != r0) {
            if (t1 < r0) {
                r1 = r0;
                r0 = t1;
            } else {
                r1 = t1;
            }
            m = 2;
        }
    }
    return m;
}

// Root of the cubic Bernstein polynomial c on [lo, hi], where c is monotone
// and changes classification (c <= 0) between lo and hi; below_lo is the
// classification at lo. Safeguarded Newton on a shrinking bracket; the result
// always lies in [lo, hi].
__device__ inline float wn_bez3_piece_root(float4 c, float lo, float hi, float ylo, float yhi, bool below_lo) {
    float u = 0.5f * (lo + hi);
    float den = ylo - yhi;
    if (fabsf(ylo) < fabsf(den)) {
        float r = ylo / den;
        if (r > 0 && r < 1) {
            u = lo + r * (hi - lo);
        }
    }
    for (int it = 0; it < 48; it++) {
        float du = 0;
        float fu = bez3_eval_d(c, u, du);
        if (fu == 0) {
            return u;
        }
        if ((fu <= 0) == below_lo) {
            lo = u;
        } else {
            hi = u;
        }
        float mid = 0.5f * (lo + hi);
        if (!(mid > lo && mid < hi)) {
            return u;  // bracket exhausted at float resolution
        }
        float un = mid;
        if (fabsf(fu) < fabsf(du) * (hi - lo)) {
            float step = fu / du;
            float nt = u - step;
            if (fabsf(step) < 1e-7f) {
                return clampf(nt, lo, hi);  // converged (see bpoly_refine01)
            }
            if (nt > lo && nt < hi) {
                un = nt;
            }
        }
        if (fabsf(un - u) < 1e-7f) {
            return un;
        }
        u = un;
    }
    return u;
}

// Blossom (polar form) of the cubic Bernstein polynomial c.
__device__ inline float bez3_blossom(float4 c, float u1, float u2, float u3) {
    float a0 = c[0] + u1 * (c[1] - c[0]);
    float a1 = c[1] + u1 * (c[2] - c[1]);
    float a2 = c[2] + u1 * (c[3] - c[2]);
    float b0 = a0 + u2 * (a1 - a0);
    float b1 = a1 + u2 * (a2 - a1);
    return b0 + u3 * (b1 - b0);
}

// Signed crossings of the cubic Bezier with Bernstein coefficients y, x (in a
// frame translated to the query point and scaled to O(1)) with the ray
// {y = 0, x > 0}, under the half-open rule above. below0 / below3 classify
// the end points and must come from the stored points (p.y <= pt.y).
__device__ inline int wn_cubic_crossings(float4 y, float4 x, bool below0, bool below3) {
    float xmax = fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3]));
    if (!(xmax > 0)) {
        return 0;  // the curve lies in the half plane x <= 0
    }
    if (fminf(fminf(x[0], x[1]), fminf(x[2], x[3])) > 0) {
        // Every crossing is right of the point. The signed crossings of the
        // consecutive y-monotone pieces telescope: a piece from class b to b'
        // contributes [b] - [b'] (+1 upwards, -1 downwards, 0 if equal), so
        // the sum is [below0] - [below3] whatever the interior extrema are.
        return (below0 ? 1 : 0) - (below3 ? 1 : 0);
    }
    // y'(t)/3 in Bernstein form (d0, d1, d2) -> power form
    float d0 = y[1] - y[0];
    float d1 = y[2] - y[1];
    float d2 = y[3] - y[2];
    float e0 = 0, e1 = 0;
    int ne = wn_quadratic_roots01(d0 - 2 * d1 + d2, 2 * (d1 - d0), d0, e0, e1);
    float ts[4];
    float ys[4];
    bool bs[4];
    ts[0] = 0;
    ys[0] = y[0];
    bs[0] = below0;
    int n = 1;
    for (int k = 0; k < ne; k++) {
        float tk = k == 0 ? e0 : e1;
        float dd = 0;
        float yk = bez3_eval_d(y, tk, dd);
        ts[n] = tk;
        ys[n] = yk;
        bs[n] = yk <= 0;
        n++;
    }
    ts[n] = 1;
    ys[n] = y[3];
    bs[n] = below3;
    n++;
    int winding = 0;
    for (int k = 0; k + 1 < n; k++) {
        if (bs[k] == bs[k + 1]) {
            continue;
        }
        // x-hull of the piece [ta, tb] (blossom control values): decides
        // the side without a root solve unless the piece straddles x = 0.
        float ta = ts[k], tb = ts[k + 1];
        float h0 = bez3_blossom(x, ta, ta, ta);
        float h1 = bez3_blossom(x, ta, ta, tb);
        float h2 = bez3_blossom(x, ta, tb, tb);
        float h3 = bez3_blossom(x, tb, tb, tb);
        bool right;
        if (fminf(fminf(h0, h1), fminf(h2, h3)) > 0) {
            right = true;
        } else if (!(fmaxf(fmaxf(h0, h1), fmaxf(h2, h3)) > 0)) {
            right = false;
        } else {
            float u = wn_bez3_piece_root(y, ta, tb, ys[k], ys[k + 1], bs[k]);
            float dx = 0;
            right = bez3_eval_d(x, u, dx) > 0;
        }
        if (right) {
            winding += bs[k] ? 1 : -1;
        }
    }
    return winding;
}

// ---------------------------------------------------------------------------
// Curve evaluation helpers (same expressions as the C++ lambdas)
__device__ inline float2 eval_quadratic(float2 p0, float2 p1, float2 p2, float t) {
    float tt = 1 - t;
    return (tt * tt) * p0 + (2 * tt * t) * p1 + (t * t) * p2;
}

__device__ inline float2 eval_cubic(float2 p0, float2 p1, float2 p2, float2 p3, float t) {
    float tt = 1 - t;
    return (tt * tt * tt) * p0 + (3 * tt * tt * t) * p1 + (3 * tt * t * t) * p2 + (t * t * t) * p3;
}

__device__ inline float max_abs4(float2 a, float2 b, float2 c, float2 d) {
    float m = fmaxf(fabsf(a.x), fabsf(a.y));
    m = fmaxf(m, fmaxf(fabsf(b.x), fabsf(b.y)));
    m = fmaxf(m, fmaxf(fabsf(c.x), fabsf(c.y)));
    m = fmaxf(m, fmaxf(fabsf(d.x), fabsf(d.y)));
    return m;
}

__device__ inline float dist_point_segment(float2 p, float2 a, float2 b) {
    float2 ab = b - a;
    float ll = dot(ab, ab);
    float t = ll > 0 ? clampf(dot(p - a, ab) / ll, 0.f, 1.f) : 0.f;
    return distance(p, a + t * ab);
}

// Lower bound on the distance from pt to a cubic Bezier: the curve lies in
// the convex hull of its control points, which lies in the capsule of radius
// h = max(dist(p1, p0p3), dist(p2, p0p3)) around the chord p0p3. Returns
// bound - tolerance (float rounding of the coordinates).
__device__ inline float cubic_distance_lower_bound(float2 p0, float2 p1, float2 p2, float2 p3, float2 pt) {
    float h = fmaxf(dist_point_segment(p1, p0, p3), dist_point_segment(p2, p0, p3));
    float tol = 1e-5f * (1 + max_abs4(p0 - pt, p1 - pt, p2 - pt, p3 - pt));
    return dist_point_segment(pt, p0, p3) - h - tol;
}

// Tighter lower bound: split the cubic at t = 1/2 (de Casteljau) and take
// the smaller of the two halves' capsule bounds. Each half lies in the hull
// of its own control points; the tolerance of cubic_distance_lower_bound
// covers the rounding of the split points.
__device__ inline float cubic_split_lower_bound(float2 p0, float2 p1, float2 p2, float2 p3, float2 pt) {
    float2 m01 = 0.5f * (p0 + p1);
    float2 m12 = 0.5f * (p1 + p2);
    float2 m23 = 0.5f * (p2 + p3);
    float2 m012 = 0.5f * (m01 + m12);
    float2 m123 = 0.5f * (m12 + m23);
    float2 mid = 0.5f * (m012 + m123);
    return fminf(cubic_distance_lower_bound(p0, m01, m012, mid, pt),
                 cubic_distance_lower_bound(mid, m123, m23, p3, pt));
}

// F32: roots in [0, 1] of (q(t) - pt) . q'(t) for the cubic q (deviation 2).
__device__ inline Roots5 cubic_closest_roots(float2 p0, float2 p1, float2 p2, float2 p3, float2 pt) {
    float2 P0 = p0 - pt;
    float2 P1 = p1 - pt;
    float2 P2 = p2 - pt;
    float2 P3 = p3 - pt;
    float S = max_abs4(P0, P1, P2, P3);
    BPoly g;
    g.n = 5;
    for (int i = 0; i < 6; i++) {
        g.c[i] = 0;
    }
    if (!(S > 0)) {
        Roots5 none;
        none.n = 0;
        for (int i = 0; i < 5; i++) {
            none.t[i] = 0;
        }
        return none;
    }
    float inv_s = 1 / S;
    P0 *= inv_s;
    P1 *= inv_s;
    P2 *= inv_s;
    P3 *= inv_s;
    // q' in Bernstein form (degree 2), without the factor 3 (roots unchanged)
    float2 D0 = P1 - P0;
    float2 D1 = P2 - P1;
    float2 D2 = P3 - P2;
    // product of Bernstein polynomials: c_k = sum_{i+j=k} C(3,i)C(2,j)/C(5,k) P_i.D_j
    g.c[0] = dot(P0, D0);
    g.c[1] = (2 * dot(P0, D1) + 3 * dot(P1, D0)) / 5.f;
    g.c[2] = (dot(P0, D2) + 6 * dot(P1, D1) + 3 * dot(P2, D0)) / 10.f;
    g.c[3] = (3 * dot(P1, D2) + 6 * dot(P2, D1) + dot(P3, D0)) / 10.f;
    g.c[4] = (3 * dot(P2, D2) + 2 * dot(P3, D1)) / 5.f;
    g.c[5] = dot(P3, D2);
    return bpoly_roots01(g);
}

// vector.h quadratic_closest_pt_approx
__device__ inline float det2(float2 a, float2 b) {
    return a.x * b.y - b.x * a.y;
}

__device__ inline float2 quadratic_closest_pt_approx(float2 p0, float2 p1, float2 p2, float2 pt, float &t_out) {
    // From http://w3.impa.br/~diego/publications/NehHop08.pdf
    float2 b0 = p0 - pt;
    float2 b1 = p1 - pt;
    float2 b2 = p2 - pt;
    float a = det2(b0, b2), b = 2 * det2(b1, b0), d = 2 * det2(b2, b1);
    float f = b * d - a * a;
    float2 d21 = b2 - b1, d10 = b1 - b0, d20 = b2 - b0;
    float2 gf = 2 * (b * d21 + d * d10 + a * d20);
    gf = float2(gf.y, -gf.x);
    float gg = dot(gf, gf);
    // C++ divides by zero for degenerate (collinear) control polygons -> NaN
    // point; guard with a zero offset instead.
    float2 pp = gg != 0 ? -f * gf / gg : float2(0);
    float2 d0p = b0 - pp;
    float ap = det2(d0p, d20), bp = 2 * det2(d10, d0p);
    float denom = 2 * a + b + d;
    float t = clampf(denom != 0 ? (ap + bp) / denom : 0.f, 0.f, 1.f);
    float tt = 1 - t;
    t_out = t;
    return (tt * tt) * b0 + (2 * tt * t) * b1 + (t * t) * b2 + pt;
}

// ---------------------------------------------------------------------------
// winding_number.h
__device__ inline int winding_number_circle(SceneView s, int sid, float2 pt) {
    float2 c = circle_center(s, sid);
    float r = circle_radius(s, sid);
    return distance_squared(c, pt) < r * r ? 1 : 0;
}

__device__ inline int winding_number_ellipse(SceneView s, int sid, float2 pt) {
    float2 c = ellipse_center(s, sid);
    float2 r = ellipse_radius(s, sid);
    float rx2 = r.x * r.x;
    float ry2 = r.y * r.y;
    if (!(rx2 > 0) || !(ry2 > 0)) {
        return 0;  // C++: division by zero -> inf/NaN -> "< 1" fails
    }
    return square(c.x - pt.x) / rx2 + square(c.y - pt.y) / ry2 < 1 ? 1 : 0;
}

__device__ inline int winding_number_rect(SceneView s, int sid, float2 pt) {
    float2 p_min = rect_p_min(s, sid);
    float2 p_max = rect_p_max(s, sid);
    return (pt.x > p_min.x && pt.x < p_max.x && pt.y > p_min.y && pt.y < p_max.y) ? 1 : 0;
}

__device__ inline bool winding_intersect(AABB box, float2 pt) {
    if (pt.y < box.p_min.y || pt.y > box.p_max.y) {
        return false;
    }
    if (pt.x > box.p_max.x) {
        return false;
    }
    return true;
}

__device__ inline int winding_number_path(SceneView s, int sid, float2 pt) {
    int num_points = path_num_points(s, sid);
    int base = path_bvh_base(s, sid);
    int bvh_stack[GEOM_PATH_BVH_STACK];
    int stack_size = 0;
    int winding_number = 0;
    bvh_stack[stack_size++] = path_bvh_root(s, sid);
    while (stack_size > 0) {
        int n = bvh_stack[--stack_size];
        int child1 = node_child1(s, n);
        if (child1 < 0) {
            int base_point_id = node_child0(s, n);
            int point_id = -child1 - 1;
            int ncp = path_num_control_points(s, sid, base_point_id);
            // Half-open crossing rule (see wn_cubic_crossings): end points are
            // classified from the stored points, never from derived values.
            if (ncp == 0) {
                // Straight line
                int i0 = point_id;
                int i1 = (point_id + 1) % num_points;
                float2 p0 = path_point(s, sid, i0);
                float2 p1 = path_point(s, sid, i1);
                bool b0 = p0.y <= pt.y;
                bool b1 = p1.y <= pt.y;
                if (b0 != b1) {
                    // crossing x >= pt.x  <=>  sign(cross) matches the direction
                    float2 q0 = p0 - pt;
                    float2 q1 = p1 - pt;
                    float cr = q0.x * q1.y - q0.y * q1.x;
                    if (b0 ? cr >= 0 : cr <= 0) {
                        winding_number += b0 ? 1 : -1;
                    }
                }
            } else if (ncp == 1) {
                // Quadratic Bezier curve, degree-elevated to a cubic (end points
                // unchanged) in a pt-centred, extent-scaled frame.
                int i0 = point_id;
                int i1 = point_id + 1;
                int i2 = (point_id + 2) % num_points;
                float2 p0 = path_point(s, sid, i0);
                float2 p1 = path_point(s, sid, i1);
                float2 p2 = path_point(s, sid, i2);
                bool b0 = p0.y <= pt.y;
                bool b1 = p1.y <= pt.y;
                bool b2 = p2.y <= pt.y;
                if (!(b0 == b1 && b1 == b2)) {
                    float2 q0 = p0 - pt;
                    float2 q1 = p1 - pt;
                    float2 q2 = p2 - pt;
                    float S = max_abs4(q0, q1, q2, q2);
                    if (S > 0) {
                        float inv_s = 1 / S;
                        float inv_3s = inv_s / 3;
                        float2 c1 = (q0 + 2 * q1) * inv_3s;
                        float2 c2 = (2 * q1 + q2) * inv_3s;
                        float4 cy = float4(q0.y * inv_s, c1.y, c2.y, q2.y * inv_s);
                        float4 cx = float4(q0.x * inv_s, c1.x, c2.x, q2.x * inv_s);
                        winding_number += wn_cubic_crossings(cy, cx, b0, b2);
                    }
                }
            } else if (ncp == 2) {
                // Cubic Bezier curve. F32 (deviation 1): Bernstein form in a
                // pt-centred, extent-scaled frame.
                int i0 = point_id;
                int i1 = point_id + 1;
                int i2 = point_id + 2;
                int i3 = (point_id + 3) % num_points;
                float2 p0 = path_point(s, sid, i0);
                float2 p1 = path_point(s, sid, i1);
                float2 p2 = path_point(s, sid, i2);
                float2 p3 = path_point(s, sid, i3);
                bool b0 = p0.y <= pt.y;
                bool b1 = p1.y <= pt.y;
                bool b2 = p2.y <= pt.y;
                bool b3 = p3.y <= pt.y;
                if (!(b0 == b1 && b1 == b2 && b2 == b3)) {
                    float2 q0 = p0 - pt;
                    float2 q1 = p1 - pt;
                    float2 q2 = p2 - pt;
                    float2 q3 = p3 - pt;
                    float S = max_abs4(q0, q1, q2, q3);
                    if (S > 0) {
                        float inv_s = 1 / S;
                        float4 cy = float4(q0.y, q1.y, q2.y, q3.y) * inv_s;
                        float4 cx = float4(q0.x, q1.x, q2.x, q3.x) * inv_s;
                        winding_number += wn_cubic_crossings(cy, cx, b0, b3);
                    }
                }
            }
        } else {
            int c0 = base + node_child0(s, n);
            int c1 = base + child1;
            if (winding_intersect(node_box(s, c0), pt) && stack_size < GEOM_PATH_BVH_STACK) {
                bvh_stack[stack_size++] = c0;
            }
            if (winding_intersect(node_box(s, c1), pt) && stack_size < GEOM_PATH_BVH_STACK) {
                bvh_stack[stack_size++] = c1;
            }
        }
    }
    return winding_number;
}

__device__ inline int compute_winding_number(SceneView s, int shape_id, float2 local_pt) {
    int type = shape_type(s, shape_id);
    if (type == SHAPE_CIRCLE) {
        return winding_number_circle(s, shape_id, local_pt);
    } else if (type == SHAPE_ELLIPSE) {
        return winding_number_ellipse(s, shape_id, local_pt);
    } else if (type == SHAPE_PATH) {
        return winding_number_path(s, shape_id, local_pt);
    } else if (type == SHAPE_RECT) {
        return winding_number_rect(s, shape_id, local_pt);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// diffvg.cpp is_inside
__device__ inline bool is_inside(SceneView s, int group_id, float2 canvas_pt, bool use_edge_query, EdgeQuery &eq) {
    // pt is in canvas space, transform it to shape's local space
    float2 local_pt = xform_pt(group_canvas_to_shape(s, group_id), canvas_pt);
    int base = group_bvh_base(s, group_id);
    int root = group_bvh_root(s, group_id);
    if (!inside(node_box(s, root), local_pt)) {
        return false;
    }
    bool even_odd = group_use_even_odd_rule(s, group_id);
    int winding_number = 0;
    int bvh_stack[GEOM_GROUP_BVH_STACK];
    int stack_size = 0;
    bvh_stack[stack_size++] = root;
    while (stack_size > 0) {
        int n = bvh_stack[--stack_size];
        int child1 = node_child1(s, n);
        if (child1 < 0) {
            int shape_id = node_child0(s, n);
            int w = compute_winding_number(s, shape_id, local_pt);
            winding_number += w;
            if (use_edge_query) {
                if (eq.shape_group_id == group_id && eq.shape_id == shape_id) {
                    if ((even_odd && iabs(w) % 2 == 1) || (!even_odd && w != 0)) {
                        eq.hit = true;
                    }
                }
            }
        } else {
            int c0 = base + node_child0(s, n);
            int c1 = base + child1;
            if (inside(node_box(s, c0), local_pt) && stack_size < GEOM_GROUP_BVH_STACK) {
                bvh_stack[stack_size++] = c0;
            }
            if (inside(node_box(s, c1), local_pt) && stack_size < GEOM_GROUP_BVH_STACK) {
                bvh_stack[stack_size++] = c1;
            }
        }
    }
    if (even_odd) {
        return iabs(winding_number) % 2 == 1;
    } else {
        return winding_number != 0;
    }
}

// ---------------------------------------------------------------------------
// compute_distance.h closest_point
__device__ inline bool closest_point_circle(SceneView s, int sid, float2 pt, float2 &result) {
    float2 c = circle_center(s, sid);
    float r = circle_radius(s, sid);
    float2 dir = pt - c;
    float l2 = length_squared(dir);
    if (l2 < 1e-20f) {
        // Every boundary point is equally close; pick one.
        dir = float2(1, 0);
        l2 = 1;
    }
    result = c + r * (dir / sqrtf(l2));
    return true;
}

// F32 (deviation 3): Eberly's root, bisected on u = s + 1.
__device__ inline float ellipse_get_root_u(float r0, float r0m1, float z0, float z1, float g) {
    float n0 = r0 * z0;
    float u0 = z1;                                          // s0 + 1
    float u1 = g < 0 ? 1.f : sqrtf(n0 * n0 + z1 * z1);      // s1 + 1
    float u = u0;
    for (int i = 0; i < 200; i++) {
        u = 0.5f * (u0 + u1);
        if (u == u0 || u == u1) {
            break;
        }
        // s + r0 = u + (r0 - 1) >= u > 0
        float ratio0 = n0 / (u + r0m1);
        float ratio1 = z1 / u;
        float gg = ratio0 * ratio0 + ratio1 * ratio1 - 1;
        if (gg > 0) {
            u0 = u;
        } else if (gg < 0) {
            u1 = u;
        } else {
            break;
        }
    }
    return u;
}

__device__ inline bool closest_point_ellipse(SceneView s, int sid, float2 pt, float2 &result) {
    float2 radius = ellipse_radius(s, sid);
    float2 center = ellipse_center(s, sid);
    float e[2] = {fabsf(radius.x), fabsf(radius.y)};
    float u[2] = {pt.x - center.x, pt.y - center.y};
    if (!(e[0] > 0) || !(e[1] > 0)) {
        return false;
    }
    // Reflect into the first quadrant and order the axes so that e0 >= e1.
    int i0 = e[0] >= e[1] ? 0 : 1;
    int i1 = 1 - i0;
    float e0 = e[i0], e1 = e[i1];
    float y0 = fabsf(u[i0]), y1 = fabsf(u[i1]);
    float x0 = 0, x1 = 0;
    if (y1 > 0) {
        if (y0 > 0) {
            float z0 = y0 / e0;
            float z1 = y1 / e1;
            float g = z0 * z0 + z1 * z1 - 1;
            if (g != 0) {
                float r0 = (e0 / e1) * (e0 / e1);
                float r0m1 = ((e0 - e1) / e1) * ((e0 + e1) / e1);
                float ub = ellipse_get_root_u(r0, r0m1, z0, z1, g);
                x0 = r0 * y0 / (ub + r0m1);
                x1 = y1 / ub;
            } else {
                x0 = y0;
                x1 = y1;
            }
        } else {
            x0 = 0;
            x1 = e1;
        }
    } else {
        float numer0 = e0 * y0;
        float denom0 = (e0 - e1) * (e0 + e1);
        if (numer0 < denom0) {
            float xde0 = numer0 / denom0;
            x0 = e0 * xde0;
            x1 = e1 * sqrtf(fmaxf(1 - xde0 * xde0, 0.f));
        } else {
            x0 = e0;
            x1 = 0;
        }
    }
    float x[2];
    x[i0] = u[i0] < 0 ? -x0 : x0;
    x[i1] = u[i1] < 0 ? -x1 : x1;
    result = float2(center.x + x[0], center.y + x[1]);
    return true;
}

__device__ inline bool closest_point_rect(SceneView s, int sid, float2 pt, float2 &result) {
    float2 p_min = rect_p_min(s, sid);
    float2 p_max = rect_p_max(s, sid);
    float2 ps[4] = {p_min, float2(p_max.x, p_min.y), float2(p_min.x, p_max.y), p_max};
    // left, top, right, bottom (same order as C++)
    int e0[4] = {0, 0, 1, 2};
    int e1[4] = {2, 1, 3, 3};
    float min_dist = 0;
    float2 closest_pt = float2(0);
    for (int k = 0; k < 4; k++) {
        float2 p0 = ps[e0[k]];
        float2 p1 = ps[e1[k]];
        float2 q = p0;
        float ll = dot(p1 - p0, p1 - p0);
        // C++: 0/0 = NaN for a zero-length edge -> falls to the p0 + t*(..) branch
        // with a NaN point; use p0 instead.
        float t = ll != 0 ? dot(pt - p0, p1 - p0) / ll : 0.f;
        if (t < 0) {
            q = p0;
        } else if (t > 1) {
            q = p1;
        } else {
            q = p0 + t * (p1 - p0);
        }
        float d = distance(q, pt);
        if (k == 0 || d < min_dist) {
            min_dist = d;
            closest_pt = q;
        }
    }
    result = closest_pt;
    return true;
}

__device__ inline bool closest_point_path(SceneView s, int sid, float2 pt, float max_radius,
                                          ClosestPointPathInfo &path_info, float2 &result) {
    float min_dist = max_radius;
    float2 ret_pt = float2(0);
    bool found = false;
    int num_points = path_num_points(s, sid);
    bool use_approx = path_use_distance_approx(s, sid);
    int base = path_bvh_base(s, sid);
    int bvh_stack[GEOM_PATH_BVH_STACK];
    int stack_size = 0;
    bvh_stack[stack_size++] = path_bvh_root(s, sid);
    while (stack_size > 0) {
        int n = bvh_stack[--stack_size];
        int child1 = node_child1(s, n);
        if (child1 < 0) {
            int base_point_id = node_child0(s, n);
            int point_id = -child1 - 1;
            int ncp = path_num_control_points(s, sid, base_point_id);
            float dist = 0;
            float2 closest_pt = float2(0);
            float t_root = 0;
            if (ncp == 0) {
                // Straight line
                int i0 = point_id;
                int i1 = (point_id + 1) % num_points;
                float2 p0 = path_point(s, sid, i0);
                float2 p1 = path_point(s, sid, i1);
                float ll = dot(p1 - p0, p1 - p0);
                // C++: zero-length segment gives t = NaN -> middle branch with a
                // NaN distance that never wins; treat it as the point p0.
                float t = ll != 0 ? dot(pt - p0, p1 - p0) / ll : -1.f;
                if (t < 0) {
                    dist = distance(p0, pt);
                    closest_pt = p0;
                    t_root = 0;
                } else if (t > 1) {
                    dist = distance(p1, pt);
                    closest_pt = p1;
                    t_root = 1;
                } else {
                    closest_pt = p0 + t * (p1 - p0);
                    dist = distance(closest_pt, pt);
                    t_root = t;
                }
            } else if (ncp == 1) {
                // Quadratic Bezier curve (float on the CPU too)
                int i0 = point_id;
                int i1 = point_id + 1;
                int i2 = (point_id + 2) % num_points;
                float2 p0 = path_point(s, sid, i0);
                float2 p1 = path_point(s, sid, i1);
                float2 p2 = path_point(s, sid, i2);
                if (use_approx) {
                    closest_pt = quadratic_closest_pt_approx(p0, p1, p2, pt, t_root);
                    dist = distance(closest_pt, pt);
                } else {
                    float2 pt0 = eval_quadratic(p0, p1, p2, 0);
                    float2 pt1 = eval_quadratic(p0, p1, p2, 1);
                    float dist0 = distance(pt0, pt);
                    float dist1 = distance(pt1, pt);
                    dist = dist0;
                    closest_pt = pt0;
                    t_root = 0;
                    if (dist1 < dist) {
                        dist = dist1;
                        closest_pt = pt1;
                        t_root = 1;
                    }
                    float2 a2 = p0 - 2 * p1 + p2;
                    float2 a1 = -p0 + p1;
                    float2 a0 = p0 - pt;
                    float A = dot(a2, a2);
                    float B = 3 * dot(a2, a1);
                    float C = 2 * dot(a1, a1) + dot(a2, a0);
                    float D = dot(a1, a0);
                    float t[3] = {0, 0, 0};
                    int num_sol = solve_cubic(A, B, C, D, t[0], t[1], t[2]);
                    for (int j = 0; j < num_sol; j++) {
                        if (t[j] >= 0 && t[j] <= 1) {
                            float2 p = eval_quadratic(p0, p1, p2, t[j]);
                            float distp = distance(p, pt);
                            if (distp < dist) {
                                dist = distp;
                                closest_pt = p;
                                t_root = t[j];
                            }
                        }
                    }
                }
            } else if (ncp == 2) {
                // Cubic Bezier curve. F32 (deviation 2).
                int i0 = point_id;
                int i1 = point_id + 1;
                int i2 = point_id + 2;
                int i3 = (point_id + 3) % num_points;
                float2 p0 = path_point(s, sid, i0);
                float2 p1 = path_point(s, sid, i1);
                float2 p2 = path_point(s, sid, i2);
                float2 p3 = path_point(s, sid, i3);
                float2 pt0 = eval_cubic(p0, p1, p2, p3, 0);
                float2 pt1 = eval_cubic(p0, p1, p2, p3, 1);
                float dist0 = distance(pt0, pt);
                float dist1 = distance(pt1, pt);
                dist = dist0;
                closest_pt = pt0;
                t_root = 0;
                if (dist1 < dist) {
                    dist = dist1;
                    closest_pt = pt1;
                    t_root = 1;
                }
                Roots5 roots = roots5_empty();
                if (!(cubic_distance_lower_bound(p0, p1, p2, p3, pt) > fminf(dist, min_dist))) {
                    roots = cubic_closest_roots(p0, p1, p2, p3, pt);
                }
                for (int j = 0; j < roots.n; j++) {
                    float2 p = eval_cubic(p0, p1, p2, p3, roots.t[j]);
                    float distp = distance(p, pt);
                    if (distp < dist) {
                        dist = distp;
                        closest_pt = p;
                        t_root = roots.t[j];
                    }
                }
            }
            if (dist < min_dist) {
                min_dist = dist;
                ret_pt = closest_pt;
                path_info.base_point_id = base_point_id;
                path_info.point_id = point_id;
                path_info.t = t_root;
                found = true;
            }
        } else {
            int c0 = base + node_child0(s, n);
            int c1 = base + child1;
            if (within_distance(node_box(s, c0), pt, min_dist) && stack_size < GEOM_PATH_BVH_STACK) {
                bvh_stack[stack_size++] = c0;
            }
            if (within_distance(node_box(s, c1), pt, min_dist) && stack_size < GEOM_PATH_BVH_STACK) {
                bvh_stack[stack_size++] = c1;
            }
        }
    }
    result = ret_pt;
    return found;
}

__device__ inline bool closest_point(SceneView s, int shape_id, float2 pt, float max_radius,
                                     ClosestPointPathInfo &path_info, float2 &result) {
    int type = shape_type(s, shape_id);
    if (type == SHAPE_CIRCLE) {
        return closest_point_circle(s, shape_id, pt, result);
    } else if (type == SHAPE_ELLIPSE) {
        return closest_point_ellipse(s, shape_id, pt, result);
    } else if (type == SHAPE_PATH) {
        return closest_point_path(s, shape_id, pt, max_radius, path_info, result);
    } else if (type == SHAPE_RECT) {
        return closest_point_rect(s, shape_id, pt, result);
    }
    return false;
}

// compute_distance.h compute_distance. Pass DIFFVG_BIG where C++ passes
// infinity<float>(); max_radius >= DIFFVG_BIG is treated as infinite.
__device__ inline bool compute_distance(SceneView s, int group_id, float2 canvas_pt, float max_radius,
                                        int &min_shape_id, float2 &closest_pt,
                                        ClosestPointPathInfo &path_info, float &distance_out) {
    // pt is in canvas space, transform it to shape's local space
    Mat3 c = group_canvas_to_shape(s, group_id);
    Mat3 shape_to_canvas = group_shape_to_canvas(s, group_id);
    float2 local_pt = xform_pt(c, canvas_pt);
    int base = group_bvh_base(s, group_id);
    int bvh_stack[GEOM_GROUP_BVH_STACK];
    int stack_size = 0;
    bvh_stack[stack_size++] = group_bvh_root(s, group_id);

    float min_dist = max_radius;
    bool found = false;
    // max_radius is a canvas-space distance, but the BVH and closest_point
    // queries below run in shape space. A canvas distance d corresponds to a
    // shape-space distance of at most ||canvas_to_shape||_2 * d, so search
    // with that (conservative) radius and filter by canvas distance afterwards.
    float local_max_radius = max_radius;
    if (max_radius < DIFFVG_BIG) {
        // largest singular value of the 2x2 linear part
        float a00 = c.r0[0] * c.r0[0] + c.r1[0] * c.r1[0];
        float a01 = c.r0[0] * c.r0[1] + c.r1[0] * c.r1[1];
        float a11 = c.r0[1] * c.r0[1] + c.r1[1] * c.r1[1];
        float half_tr = (a00 + a11) / 2;
        float det = a00 * a11 - a01 * a01;
        float lambda_max = half_tr + sqrtf(fmaxf(half_tr * half_tr - det, 0.f));
        local_max_radius = fminf(max_radius * sqrtf(fmaxf(lambda_max, 0.f)), DIFFVG_BIG);
    }

    while (stack_size > 0) {
        int n = bvh_stack[--stack_size];
        int child1 = node_child1(s, n);
        if (child1 < 0) {
            int shape_id = node_child0(s, n);
            ClosestPointPathInfo local_path_info;
            local_path_info.base_point_id = -1;
            local_path_info.point_id = -1;
            local_path_info.t = 0;
            float2 local_closest_pt = float2(0);
            if (closest_point(s, shape_id, local_pt, local_max_radius, local_path_info, local_closest_pt)) {
                float2 cp = xform_pt(shape_to_canvas, local_closest_pt);
                float dist = distance(cp, canvas_pt);
                // min_dist starts at max_radius; shapes whose closest_point
                // does not test max_radius itself (circle, ellipse, rect)
                // must not be reported beyond it.
                if (dist < min_dist) {
                    found = true;
                    min_dist = dist;
                    min_shape_id = shape_id;
                    closest_pt = cp;
                    path_info = local_path_info;
                }
            }
        } else {
            int c0 = base + node_child0(s, n);
            int c1 = base + child1;
            if (inside(node_box(s, c0), local_pt, local_max_radius) && stack_size < GEOM_GROUP_BVH_STACK) {
                bvh_stack[stack_size++] = c0;
            }
            if (inside(node_box(s, c1), local_pt, local_max_radius) && stack_size < GEOM_GROUP_BVH_STACK) {
                bvh_stack[stack_size++] = c1;
            }
        }
    }
    distance_out = min_dist;
    return found;
}

// ---------------------------------------------------------------------------
// within_distance.h
__device__ inline bool within_distance_circle(SceneView s, int sid, float2 pt, float r) {
    float dist_to_center = distance(circle_center(s, sid), pt);
    return fabsf(dist_to_center - circle_radius(s, sid)) < r;
}

__device__ inline bool within_distance_ellipse(SceneView s, int sid, float2 pt, float r) {
    // Cheap rejection using the annulus bounds min(|rx|,|ry|) .. max(|rx|,|ry|).
    float2 radius = ellipse_radius(s, sid);
    float d_center = distance(ellipse_center(s, sid), pt);
    float r_min = fminf(fabsf(radius.x), fabsf(radius.y));
    float r_max = fmaxf(fabsf(radius.x), fabsf(radius.y));
    if (d_center >= r_max + r || d_center <= r_min - r) {
        return false;
    }
    float2 cp = float2(0);
    if (!closest_point_ellipse(s, sid, pt, cp)) {
        return false;
    }
    return distance_squared(cp, pt) < r * r;
}

__device__ inline bool within_distance_segment(float2 p0, float2 p1, float2 pt, float r0, float r1) {
    float ll = dot(p1 - p0, p1 - p0);
    if (ll == 0) {
        // C++: t = NaN -> middle branch with a NaN distance -> false.
        return false;
    }
    float t = dot(pt - p0, p1 - p0) / ll;
    if (t < 0) {
        return distance_squared(p0, pt) < r0 * r0;
    } else if (t > 1) {
        return distance_squared(p1, pt) < r1 * r1;
    } else {
        float rr = r0 + t * (r1 - r0);
        return distance_squared(p0 + t * (p1 - p0), pt) < rr * rr;
    }
}

__device__ inline bool within_distance_path(SceneView s, int sid, float2 pt, float r) {
    int num_points = path_num_points(s, sid);
    bool has_thickness = path_has_thickness(s, sid);
    bool use_approx = path_use_distance_approx(s, sid);
    int base = path_bvh_base(s, sid);
    int bvh_stack[GEOM_PATH_BVH_STACK];
    int stack_size = 0;
    bvh_stack[stack_size++] = path_bvh_root(s, sid);
    while (stack_size > 0) {
        int n = bvh_stack[--stack_size];
        int child1 = node_child1(s, n);
        if (child1 < 0) {
            int base_point_id = node_child0(s, n);
            int point_id = -child1 - 1;
            int ncp = path_num_control_points(s, sid, base_point_id);
            if (ncp == 0) {
                // Straight line
                int i0 = point_id;
                int i1 = (point_id + 1) % num_points;
                float2 p0 = path_point(s, sid, i0);
                float2 p1 = path_point(s, sid, i1);
                float r0 = r;
                float r1 = r;
                if (has_thickness) {
                    r0 = path_thickness(s, sid, i0);
                    r1 = path_thickness(s, sid, i1);
                }
                if (within_distance_segment(p0, p1, pt, r0, r1)) {
                    return true;
                }
            } else if (ncp == 1) {
                // Quadratic Bezier curve (float on the CPU too)
                int i0 = point_id;
                int i1 = point_id + 1;
                int i2 = (point_id + 2) % num_points;
                float2 p0 = path_point(s, sid, i0);
                float2 p1 = path_point(s, sid, i1);
                float2 p2 = path_point(s, sid, i2);
                if (use_approx) {
                    float t_unused = 0;
                    float2 cp = quadratic_closest_pt_approx(p0, p1, p2, pt, t_unused);
                    if (distance_squared(cp, pt) < r * r) {
                        return true;
                    }
                    // C++ returns the approximate test's result for the whole
                    // path here (it stops at the first approximated quadratic).
                    return false;
                }
                float r0 = r;
                float r1 = r;
                float r2 = r;
                if (has_thickness) {
                    r0 = path_thickness(s, sid, i0);
                    r1 = path_thickness(s, sid, i1);
                    r2 = path_thickness(s, sid, i2);
                }
                if (distance_squared(eval_quadratic(p0, p1, p2, 0), pt) < r0 * r0) {
                    return true;
                }
                if (distance_squared(eval_quadratic(p0, p1, p2, 1), pt) < r2 * r2) {
                    return true;
                }
                float2 a2 = p0 - 2 * p1 + p2;
                float2 a1 = -p0 + p1;
                float2 a0 = p0 - pt;
                float A = dot(a2, a2);
                float B = 3 * dot(a2, a1);
                float C = 2 * dot(a1, a1) + dot(a2, a0);
                float D = dot(a1, a0);
                float t[3] = {0, 0, 0};
                int num_sol = solve_cubic(A, B, C, D, t[0], t[1], t[2]);
                for (int j = 0; j < num_sol; j++) {
                    if (t[j] >= 0 && t[j] <= 1) {
                        float tt = 1 - t[j];
                        float rr = (tt * tt) * r0 + (2 * tt * t[j]) * r1 + (t[j] * t[j]) * r2;
                        float2 p = eval_quadratic(p0, p1, p2, t[j]);
                        if (distance_squared(p, pt) < rr * rr) {
                            return true;
                        }
                    }
                }
            } else if (ncp == 2) {
                // Cubic Bezier curve. F32 (deviation 2).
                int i0 = point_id;
                int i1 = point_id + 1;
                int i2 = point_id + 2;
                int i3 = (point_id + 3) % num_points;
                float2 p0 = path_point(s, sid, i0);
                float2 p1 = path_point(s, sid, i1);
                float2 p2 = path_point(s, sid, i2);
                float2 p3 = path_point(s, sid, i3);
                float r0 = r;
                float r1 = r;
                float r2 = r;
                float r3 = r;
                if (has_thickness) {
                    r0 = path_thickness(s, sid, i0);
                    r1 = path_thickness(s, sid, i1);
                    r2 = path_thickness(s, sid, i2);
                    r3 = path_thickness(s, sid, i3);
                }
                if (distance_squared(eval_cubic(p0, p1, p2, p3, 0), pt) < r0 * r0) {
                    return true;
                }
                if (distance_squared(eval_cubic(p0, p1, p2, p3, 1), pt) < r3 * r3) {
                    return true;
                }
                float rmax = fmaxf(fmaxf(fabsf(r0), fabsf(r1)), fmaxf(fabsf(r2), fabsf(r3)));
                Roots5 roots = roots5_empty();
                if (!(cubic_distance_lower_bound(p0, p1, p2, p3, pt) >= rmax) &&
                    !(cubic_split_lower_bound(p0, p1, p2, p3, pt) >= rmax)) {
                    roots = cubic_closest_roots(p0, p1, p2, p3, pt);
                }
                for (int j = 0; j < roots.n; j++) {
                    float tj = roots.t[j];
                    float tt = 1 - tj;
                    float rr = (tt * tt * tt) * r0 + (3 * tt * tt * tj) * r1 +
                               (3 * tt * tj * tj) * r2 + (tj * tj * tj) * r3;
                    if (distance_squared(eval_cubic(p0, p1, p2, p3, tj), pt) < rr * rr) {
                        return true;
                    }
                }
            }
        } else {
            int c0 = base + node_child0(s, n);
            int c1 = base + child1;
            if (within_distance(node_box(s, c0), pt, node_max_radius(s, c0)) && stack_size < GEOM_PATH_BVH_STACK) {
                bvh_stack[stack_size++] = c0;
            }
            if (within_distance(node_box(s, c1), pt, node_max_radius(s, c1)) && stack_size < GEOM_PATH_BVH_STACK) {
                bvh_stack[stack_size++] = c1;
            }
        }
    }
    return false;
}

__device__ inline bool within_distance_rect(SceneView s, int sid, float2 pt, float r) {
    float2 left_top = rect_p_min(s, sid);
    float2 right_bottom = rect_p_max(s, sid);
    float2 right_top = float2(right_bottom.x, left_top.y);
    float2 left_bottom = float2(left_top.x, right_bottom.y);
    return within_distance_segment(left_top, left_bottom, pt, r, r) ||   // left
           within_distance_segment(left_top, right_top, pt, r, r) ||     // top
           within_distance_segment(right_top, right_bottom, pt, r, r) || // right
           within_distance_segment(left_bottom, right_bottom, pt, r, r); // bottom
}

__device__ inline bool within_distance_shape(SceneView s, int shape_id, float2 pt, float r) {
    int type = shape_type(s, shape_id);
    if (type == SHAPE_CIRCLE) {
        return within_distance_circle(s, shape_id, pt, r);
    } else if (type == SHAPE_ELLIPSE) {
        return within_distance_ellipse(s, shape_id, pt, r);
    } else if (type == SHAPE_PATH) {
        return within_distance_path(s, shape_id, pt, r);
    } else if (type == SHAPE_RECT) {
        return within_distance_rect(s, shape_id, pt, r);
    }
    return false;
}

// within_distance(scene, group, pt[, edge_query]). The edge-query version
// visits every shape (it must set eq.hit) and does not reset eq.hit.
__device__ inline bool within_distance(SceneView s, int group_id, float2 canvas_pt, bool use_edge_query, EdgeQuery &eq) {
    bool specialized = !use_edge_query || group_id != eq.shape_group_id;
    float2 local_pt = xform_pt(group_canvas_to_shape(s, group_id), canvas_pt);
    int base = group_bvh_base(s, group_id);
    int bvh_stack[GEOM_GROUP_BVH_STACK];
    int stack_size = 0;
    bvh_stack[stack_size++] = group_bvh_root(s, group_id);
    bool ret = false;
    while (stack_size > 0) {
        int n = bvh_stack[--stack_size];
        int child1 = node_child1(s, n);
        if (child1 < 0) {
            int shape_id = node_child0(s, n);
            if (within_distance_shape(s, shape_id, local_pt, shape_stroke_width(s, shape_id))) {
                if (specialized) {
                    return true;
                }
                ret = true;
                if (shape_id == eq.shape_id) {
                    eq.hit = true;
                }
            }
        } else {
            int c0 = base + node_child0(s, n);
            int c1 = base + child1;
            if (inside(node_box(s, c0), local_pt, node_max_radius(s, c0)) && stack_size < GEOM_GROUP_BVH_STACK) {
                bvh_stack[stack_size++] = c0;
            }
            if (inside(node_box(s, c1), local_pt, node_max_radius(s, c1)) && stack_size < GEOM_GROUP_BVH_STACK) {
                bvh_stack[stack_size++] = c1;
            }
        }
    }
    return ret;
}
