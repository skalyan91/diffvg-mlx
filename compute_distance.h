#pragma once

#include "diffvg.h"
#include "edge_query.h"
#include "scene.h"
#include "shape.h"
#include "solve.h"
#include "vector.h"

#include <cassert>

struct ClosestPointPathInfo {
    int base_point_id;
    int point_id;
    float t_root;
};

DEVICE
inline
bool closest_point(const Circle &circle, const Vector2f &pt,
                   Vector2f *result) {
    auto dir = pt - circle.center;
    if (length_squared(dir) < 1e-20f) {
        // Every boundary point is equally close; pick one.
        dir = Vector2f{1, 0};
    }
    *result = circle.center + circle.radius * normalize(dir);
    return true;
}

// Closest point on an axis-aligned ellipse.
// Eberly, "Distance from a Point to an Ellipse, an Ellipsoid, or a Hyperellipsoid"
// https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf
// Works in the first quadrant with e0 >= e1 and uses bisection on the
// (unique) root of the characteristic function.
DEVICE
inline
double ellipse_get_root(double r0, double z0, double z1, double g) {
    auto n0 = r0 * z0;
    auto s0 = z1 - 1;
    auto s1 = g < 0 ? 0 : sqrt(n0 * n0 + z1 * z1) - 1;
    auto s = 0.0;
    for (int i = 0; i < 200; i++) {
        s = (s0 + s1) / 2;
        if (s == s0 || s == s1) {
            break;
        }
        auto ratio0 = n0 / (s + r0);
        auto ratio1 = z1 / (s + 1);
        g = ratio0 * ratio0 + ratio1 * ratio1 - 1;
        if (g > 0) {
            s0 = s;
        } else if (g < 0) {
            s1 = s;
        } else {
            break;
        }
    }
    return s;
}

DEVICE
inline
bool closest_point(const Ellipse &ellipse, const Vector2f &pt,
                   Vector2f *result) {
    double e[2] = {fabs(double(ellipse.radius.x)), fabs(double(ellipse.radius.y))};
    double u[2] = {double(pt.x) - double(ellipse.center.x),
                   double(pt.y) - double(ellipse.center.y)};
    if (!(e[0] > 0) || !(e[1] > 0)) {
        return false;
    }
    // Reflect into the first quadrant and order the axes so that e0 >= e1.
    int i0 = e[0] >= e[1] ? 0 : 1;
    int i1 = 1 - i0;
    auto e0 = e[i0], e1 = e[i1];
    auto y0 = fabs(u[i0]), y1 = fabs(u[i1]);
    auto x0 = 0.0, x1 = 0.0;
    if (y1 > 0) {
        if (y0 > 0) {
            auto z0 = y0 / e0;
            auto z1 = y1 / e1;
            auto g = z0 * z0 + z1 * z1 - 1;
            if (g != 0) {
                auto r0 = (e0 / e1) * (e0 / e1);
                auto sbar = ellipse_get_root(r0, z0, z1, g);
                x0 = r0 * y0 / (sbar + r0);
                x1 = y1 / (sbar + 1);
            } else {
                x0 = y0;
                x1 = y1;
            }
        } else {
            x0 = 0;
            x1 = e1;
        }
    } else {
        auto numer0 = e0 * y0;
        auto denom0 = e0 * e0 - e1 * e1;
        if (numer0 < denom0) {
            auto xde0 = numer0 / denom0;
            x0 = e0 * xde0;
            x1 = e1 * sqrt(std::max(1 - xde0 * xde0, 0.0));
        } else {
            x0 = e0;
            x1 = 0;
        }
    }
    double x[2];
    x[i0] = u[i0] < 0 ? -x0 : x0;
    x[i1] = u[i1] < 0 ? -x1 : x1;
    *result = Vector2f{float(double(ellipse.center.x) + x[0]),
                       float(double(ellipse.center.y) + x[1])};
    return true;
}

// ---------------------------------------------------------------------------
// Robust real roots on [0, 1] of polynomials of degree <= 5 in Bernstein form
// (double precision port of the Metal backend's bpoly_* helpers).
// Roots are isolated by de Casteljau subdivision with Descartes' rule of signs
// for Bernstein coefficients (the number of roots in the interval is at most the
// number of coefficient sign changes, with the same parity): 0 changes -> no
// root, 1 change -> exactly one simple root (refined by safeguarded Newton),
// >= 2 -> split at the midpoint (depth limited). Unlike the former power-basis
// solver this never divides by the leading coefficient.
struct BernsteinPoly {
    double c[6];
    int n;
};

// Value and derivative (w.r.t. t on the polynomial's own [0, 1]).
DEVICE
inline
double bernstein_eval(const BernsteinPoly &p, double t, double *d) {
    double b[6];
    for (int i = 0; i <= p.n; i++) {
        b[i] = p.c[i];
    }
    double s = 1 - t;
    for (int k = p.n; k > 1; k--) {
        for (int i = 0; i < k; i++) {
            b[i] = s * b[i] + t * b[i + 1];
        }
    }
    if (p.n >= 1) {
        *d = double(p.n) * (b[1] - b[0]);
        return s * b[0] + t * b[1];
    }
    *d = 0;
    return b[0];
}

DEVICE
inline
void bernstein_split(const BernsteinPoly &p, BernsteinPoly *left, BernsteinPoly *right) {
    double b[6];
    for (int i = 0; i <= p.n; i++) {
        b[i] = p.c[i];
    }
    left->n = p.n;
    right->n = p.n;
    left->c[0] = b[0];
    right->c[p.n] = b[p.n];
    for (int k = 1; k <= p.n; k++) {
        for (int i = 0; i <= p.n - k; i++) {
            b[i] = 0.5 * (b[i] + b[i + 1]);
        }
        left->c[k] = b[0];
        right->c[p.n - k] = b[p.n - k];
    }
}

// Number of sign changes of the nonzero coefficients; *first = first nonzero
// coefficient (its sign is the sign of p just right of 0).
DEVICE
inline
int bernstein_sign_changes(const BernsteinPoly &p, double *first) {
    int changes = 0;
    double f = 0, last = 0;
    for (int i = 0; i <= p.n; i++) {
        double v = p.c[i];
        if (v != 0) {
            if (last != 0 && (v < 0) != (last < 0)) {
                changes++;
            }
            if (f == 0) {
                f = v;
            }
            last = v;
        }
    }
    *first = f;
    return changes;
}

// The unique simple root in (0, 1) of p (exactly one coefficient sign change).
DEVICE
inline
double bernstein_refine01(const BernsteinPoly &p, double first) {
    bool neg_lo = first < 0;
    double lo = 0, hi = 1;
    double f0 = p.c[0], f1 = p.c[p.n];
    double u = 0.5;
    if (f0 != 0 && f1 != 0 && (f0 < 0) != (f1 < 0)) {
        // regula falsi start
        u = f0 / (f0 - f1);
        u = u < 0 ? 0 : (u > 1 ? 1 : u);
    }
    for (int it = 0; it < 60; it++) {
        double du = 0;
        double fu = bernstein_eval(p, u, &du);
        if (fu == 0) {
            break;
        }
        if ((fu < 0) == neg_lo) {
            lo = u;
        } else {
            hi = u;
        }
        double mid = 0.5 * (lo + hi);
        double un = mid;
        if (du != 0) {
            double nt = u - fu / du;
            if (nt > lo && nt < hi) {
                un = nt;
            }
        }
        if (fabs(un - u) < 1e-15 || !(mid > lo && mid < hi)) {
            u = un;
            break;
        }
        u = un;
    }
    return u;
}

// Roots of p in [0, 1] (unordered, at most 5) written to roots; returns the count.
// A root exactly at a subdivision point or at t = 0 / 1 is reported once. At the
// depth limit (roots closer than 2^-40) the interval midpoint is reported. An
// identically zero polynomial has no roots.
DEVICE
inline
int bernstein_roots01(const BernsteinPoly &p, double *roots) {
    int num_roots = 0;
    if (p.n <= 0) {
        return 0;
    }
    if (p.c[p.n] == 0) {
        bool all_zero = true;
        for (int i = 0; i <= p.n; i++) {
            if (p.c[i] != 0) {
                all_zero = false;
            }
        }
        if (all_zero) {
            return 0;
        }
        roots[num_roots++] = 1;
    }
    struct Entry {
        BernsteinPoly p;
        double l, r;
        int depth;
        bool own_left; // report a root exactly at l (false for left children: shared with the parent)
    };
    constexpr int max_stack = 64;
    constexpr int max_depth = 40;
    Entry stack[max_stack];
    int sp = 0;
    stack[sp++] = Entry{p, 0, 1, 0, true};
    while (sp > 0 && num_roots < 5) {
        Entry e = stack[--sp];
        if (e.own_left && e.p.c[0] == 0) {
            roots[num_roots++] = e.l;
            if (num_roots >= 5) {
                break;
            }
        }
        double first = 0;
        int changes = bernstein_sign_changes(e.p, &first);
        if (changes == 0) {
            continue;
        }
        if (changes == 1) {
            double u = bernstein_refine01(e.p, first);
            roots[num_roots++] = e.l + u * (e.r - e.l);
            continue;
        }
        double m = 0.5 * (e.l + e.r);
        if (e.depth >= max_depth || !(m > e.l && m < e.r) || sp + 2 > max_stack) {
            roots[num_roots++] = m;
            continue;
        }
        BernsteinPoly left, right;
        bernstein_split(e.p, &left, &right);
        // push right first so that the left half is processed first
        stack[sp++] = Entry{right, m, e.r, e.depth + 1, true};
        stack[sp++] = Entry{left, e.l, m, e.depth + 1, false};
    }
    return num_roots;
}

// Roots in [0, 1] of (q(t) - pt) . q'(t) for the cubic Bezier q with control
// points p0..p3 (the stationary points of the distance to pt). The quintic is
// formed in Bernstein form in a frame translated to pt and scaled by the
// control-point extent, so it is well conditioned and never normalised by its
// leading coefficient (which vanishes e.g. for a degree-elevated line).
DEVICE
inline
int cubic_closest_roots(const Vector2f &p0, const Vector2f &p1,
                        const Vector2f &p2, const Vector2f &p3,
                        const Vector2f &pt, double *roots) {
    double X[4] = {double(p0.x) - double(pt.x), double(p1.x) - double(pt.x),
                   double(p2.x) - double(pt.x), double(p3.x) - double(pt.x)};
    double Y[4] = {double(p0.y) - double(pt.y), double(p1.y) - double(pt.y),
                   double(p2.y) - double(pt.y), double(p3.y) - double(pt.y)};
    double S = 0;
    for (int i = 0; i < 4; i++) {
        S = fabs(X[i]) > S ? fabs(X[i]) : S;
        S = fabs(Y[i]) > S ? fabs(Y[i]) : S;
    }
    if (!(S > 0)) {
        return 0;
    }
    for (int i = 0; i < 4; i++) {
        X[i] /= S;
        Y[i] /= S;
    }
    // q' in Bernstein form (degree 2), without the factor 3 (roots unchanged)
    double DX[3] = {X[1] - X[0], X[2] - X[1], X[3] - X[2]};
    double DY[3] = {Y[1] - Y[0], Y[2] - Y[1], Y[3] - Y[2]};
    auto pd = [&](int i, int j) { return X[i] * DX[j] + Y[i] * DY[j]; };
    // product of Bernstein polynomials:
    // c_k = sum_{i+j=k} C(3,i) C(2,j) / C(5,k) P_i . D_j
    BernsteinPoly g;
    g.n = 5;
    g.c[0] = pd(0, 0);
    g.c[1] = (2 * pd(0, 1) + 3 * pd(1, 0)) / 5;
    g.c[2] = (pd(0, 2) + 6 * pd(1, 1) + 3 * pd(2, 0)) / 10;
    g.c[3] = (3 * pd(1, 2) + 6 * pd(2, 1) + pd(3, 0)) / 10;
    g.c[4] = (3 * pd(2, 2) + 2 * pd(3, 1)) / 5;
    g.c[5] = pd(3, 2);
    return bernstein_roots01(g, roots);
}

DEVICE
inline
float point_to_segment_distance(const Vector2f &p, const Vector2f &a, const Vector2f &b) {
    auto ab = b - a;
    auto ll = dot(ab, ab);
    auto t = ll > 0 ? dot(p - a, ab) / ll : 0.f;
    t = t < 0 ? 0.f : (t > 1 ? 1.f : t);
    return distance(p, a + t * ab);
}

// Lower bound on the distance from pt to a cubic Bezier: the curve lies in the
// convex hull of its control points, which lies in the capsule of radius
// h = max(dist(p1, p0p3), dist(p2, p0p3)) around the chord p0p3. A relative
// tolerance is subtracted to absorb float rounding, so skipping the root search
// when the bound exceeds the current best distance cannot change the result.
DEVICE
inline
float cubic_distance_lower_bound(const Vector2f &p0, const Vector2f &p1,
                                 const Vector2f &p2, const Vector2f &p3,
                                 const Vector2f &pt) {
    auto h = max(point_to_segment_distance(p1, p0, p3), point_to_segment_distance(p2, p0, p3));
    auto m = 0.f;
    for (const Vector2f &q : {p0 - pt, p1 - pt, p2 - pt, p3 - pt}) {
        // explicit float: with <math.h> in scope fabs() returns double, and
        // max(float, double) does not compile (clang accepts it, g++ does not)
        m = max(m, max(float(fabs(q.x)), float(fabs(q.y))));
    }
    return point_to_segment_distance(pt, p0, p3) - h - 1e-5f * (1 + m);
}

DEVICE
inline
bool closest_point(const Path &path, const BVHNode *bvh_nodes, const Vector2f &pt, float max_radius,
                   ClosestPointPathInfo *path_info,
                   Vector2f *result) {
    auto min_dist = max_radius;
    auto ret_pt = Vector2f{0, 0};
    auto found = false;
    auto num_segments = path.num_base_points;
    constexpr auto max_bvh_size = 128;
    int bvh_stack[max_bvh_size];
    auto stack_size = 0;
    bvh_stack[stack_size++] = 2 * num_segments - 2;
    while (stack_size > 0) {
        const BVHNode &node = bvh_nodes[bvh_stack[--stack_size]];
        if (node.child1 < 0) {
            // leaf
            auto base_point_id = node.child0;
            auto point_id = - node.child1 - 1;
            assert(base_point_id < num_segments);
            assert(point_id < path.num_points);
            auto dist = 0.f;
            auto closest_pt = Vector2f{0, 0};
            auto t_root = 0.f;
            if (path.num_control_points[base_point_id] == 0) {
                // Straight line
                auto i0 = point_id;
                auto i1 = (point_id + 1) % path.num_points;
                auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
                auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
                // project pt to line
                // zero-length segment: 0/0 = NaN would skip every branch below with a NaN
                // distance; treat it as the point p0 (as geometry.metal does)
                auto t = dot(p1 - p0, p1 - p0) > 0 ? dot(pt - p0, p1 - p0) / dot(p1 - p0, p1 - p0) : -1.f;
                if (t < 0) {
                    dist = distance(p0, pt);
                    closest_pt = p0;
                    t_root = 0;
                } else if (t > 1) {
                    dist = distance(p1, pt);
                    closest_pt = p1;
                    t_root = 1;
                } else {
                    dist = distance(p0 + t * (p1 - p0), pt);
                    closest_pt = p0 + t * (p1 - p0);
                    t_root = t;
                }
            } else if (path.num_control_points[base_point_id] == 1) {
                // Quadratic Bezier curve
                auto i0 = point_id;
                auto i1 = point_id + 1;
                auto i2 = (point_id + 2) % path.num_points;
                auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
                auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
                auto p2 = Vector2f{path.points[2 * i2], path.points[2 * i2 + 1]};
                if (path.use_distance_approx) {
                    closest_pt = quadratic_closest_pt_approx(p0, p1, p2, pt, &t_root);
                    dist = distance(closest_pt, pt);
                } else {
                    auto eval = [&](float t) -> Vector2f {
                        auto tt = 1 - t;
                        return (tt*tt)*p0 + (2*tt*t)*p1 + (t*t)*p2;
                    };
                    auto pt0 = eval(0);
                    auto pt1 = eval(1);
                    auto dist0 = distance(pt0, pt);
                    auto dist1 = distance(pt1, pt);
                    {
                        dist = dist0;
                        closest_pt = pt0;
                        t_root = 0;
                    }
                    if (dist1 < dist) {
                        dist = dist1;
                        closest_pt = pt1;
                        t_root = 1;
                    }
                    // The curve is (1-t)^2p0 + 2(1-t)tp1 + t^2p2
                    // = (p0-2p1+p2)t^2+(-2p0+2p1)t+p0 = q
                    // Want to solve (q - pt) dot q' = 0
                    // q' = (p0-2p1+p2)t + (-p0+p1)
                    // Expanding (p0-2p1+p2)^2 t^3 +
                    //           3(p0-2p1+p2)(-p0+p1) t^2 +
                    //           (2(-p0+p1)^2+(p0-2p1+p2)(p0-pt))t +
                    //           (-p0+p1)(p0-pt) = 0
                    auto A = sum((p0-2*p1+p2)*(p0-2*p1+p2));
                    auto B = sum(3*(p0-2*p1+p2)*(-p0+p1));
                    auto C = sum(2*(-p0+p1)*(-p0+p1)+(p0-2*p1+p2)*(p0-pt));
                    auto D = sum((-p0+p1)*(p0-pt));
                    float t[3];
                    int num_sol = solve_cubic(A, B, C, D, t);
                    for (int j = 0; j < num_sol; j++) {
                        if (t[j] >= 0 && t[j] <= 1) {
                            auto p = eval(t[j]);
                            auto distp = distance(p, pt);
                            if (distp < dist) {
                                dist = distp;
                                closest_pt = p;
                                t_root = t[j];
                            }
                        }
                    }
                }
            } else if (path.num_control_points[base_point_id] == 2) {
                // Cubic Bezier curve
                auto i0 = point_id;
                auto i1 = point_id + 1;
                auto i2 = point_id + 2;
                auto i3 = (point_id + 3) % path.num_points;
                auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
                auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
                auto p2 = Vector2f{path.points[2 * i2], path.points[2 * i2 + 1]};
                auto p3 = Vector2f{path.points[2 * i3], path.points[2 * i3 + 1]};
                auto eval = [&](float t) -> Vector2f {
                    auto tt = 1 - t;
                    return (tt*tt*tt)*p0 + (3*tt*tt*t)*p1 + (3*tt*t*t)*p2 + (t*t*t)*p3;
                };
                auto pt0 = eval(0);
                auto pt1 = eval(1);
                auto dist0 = distance(pt0, pt);
                auto dist1 = distance(pt1, pt);
                {
                    dist = dist0;
                    closest_pt = pt0;
                    t_root = 0;
                }
                if (dist1 < dist) {
                    dist = dist1;
                    closest_pt = pt1;
                    t_root = 1;
                }
                // Stationary points of the distance: roots in [0, 1] of the quintic
                // (q(t) - pt) . q'(t), found robustly in Bernstein form (see
                // cubic_closest_roots). Skip the search when the convex-hull
                // capsule is farther than the best distance found so far.
                if (cubic_distance_lower_bound(p0, p1, p2, p3, pt) < min(dist, min_dist)) {
                    double roots[5];
                    int num_roots = cubic_closest_roots(p0, p1, p2, p3, pt, roots);
                    for (int j = 0; j < num_roots; j++) {
                        auto t = float(roots[j]);
                        auto p = eval(t);
                        auto distp = distance(p, pt);
                        if (distp < dist) {
                            dist = distp;
                            closest_pt = p;
                            t_root = t;
                        }
                    }
                }
            } else {
                assert(false);
            }
            if (dist < min_dist) {
                min_dist = dist;
                ret_pt = closest_pt;
                path_info->base_point_id = base_point_id;
                path_info->point_id = point_id;
                path_info->t_root = t_root;
                found = true;
            }
        } else {
            assert(node.child0 >= 0 && node.child1 >= 0);
            const AABB &b0 = bvh_nodes[node.child0].box;
            if (within_distance(b0, pt, min_dist)) {
                bvh_stack[stack_size++] = node.child0;
            }
            const AABB &b1 = bvh_nodes[node.child1].box;
            if (within_distance(b1, pt, min_dist)) {
                bvh_stack[stack_size++] = node.child1;
            }
            assert(stack_size <= max_bvh_size);
        }
    }
    if (found) {
        assert(path_info->base_point_id < num_segments);
    }
    *result = ret_pt;
    return found;
}

DEVICE
inline
bool closest_point(const Rect &rect, const Vector2f &pt,
                   Vector2f *result) {
    auto min_dist = 0.f;
    auto closest_pt = Vector2f{0, 0};
    auto update = [&](const Vector2f &p0, const Vector2f &p1, bool first) {
        // project pt to line
        // zero-length segment: 0/0 = NaN would skip every branch below with a NaN
        // distance; treat it as the point p0 (as geometry.metal does)
        auto t = dot(p1 - p0, p1 - p0) > 0 ? dot(pt - p0, p1 - p0) / dot(p1 - p0, p1 - p0) : -1.f;
        if (t < 0) {
            auto d = distance(p0, pt);
            if (first || d < min_dist) {
                min_dist = d;
                closest_pt = p0;
            }
        } else if (t > 1) {
            auto d = distance(p1, pt);
            if (first || d < min_dist) {
                min_dist = d;
                closest_pt = p1;
            }
        } else {
            auto p = p0 + t * (p1 - p0);
            auto d = distance(p, pt);
            if (first || d < min_dist) {
                min_dist = d;
                closest_pt = p;
            }
        }
    };
    auto left_top = rect.p_min;
    auto right_top = Vector2f{rect.p_max.x, rect.p_min.y};
    auto left_bottom = Vector2f{rect.p_min.x, rect.p_max.y};
    auto right_bottom = rect.p_max;
    update(left_top, left_bottom, true);
    update(left_top, right_top, false);
    update(right_top, right_bottom, false);
    update(left_bottom, right_bottom, false);
    *result = closest_pt;
    return true;
}

DEVICE
inline
bool closest_point(const Shape &shape, const BVHNode *bvh_nodes, const Vector2f &pt, float max_radius,
                   ClosestPointPathInfo *path_info,
                   Vector2f *result) {
    switch (shape.type) {
        case ShapeType::Circle:
            return closest_point(*(const Circle *)shape.ptr, pt, result);
        case ShapeType::Ellipse:
            return closest_point(*(const Ellipse *)shape.ptr, pt, result);
        case ShapeType::Path:
            return closest_point(*(const Path *)shape.ptr, bvh_nodes, pt, max_radius, path_info, result);
        case ShapeType::Rect:
            return closest_point(*(const Rect *)shape.ptr, pt, result);
    }
    assert(false);
    return false;
}

DEVICE
inline
bool compute_distance(const SceneData &scene,
                      int shape_group_id,
                      const Vector2f &pt,
                      float max_radius,
                      int *min_shape_id,
                      Vector2f *closest_pt_,
                      ClosestPointPathInfo *path_info,
                      float *result) {
    const ShapeGroup &shape_group = scene.shape_groups[shape_group_id];
    // pt is in canvas space, transform it to shape's local space
    auto local_pt = xform_pt(shape_group.canvas_to_shape, pt);

    constexpr auto max_bvh_stack_size = 64;
    int bvh_stack[max_bvh_stack_size];
    auto stack_size = 0;
    bvh_stack[stack_size++] = 2 * shape_group.num_shapes - 2;
    const auto &bvh_nodes = scene.shape_groups_bvh_nodes[shape_group_id];

    auto min_dist = max_radius;
    auto found = false;
    // max_radius is a canvas-space distance, but the BVH and closest_point
    // queries below run in shape space. A canvas distance d corresponds to a
    // shape-space distance of at most ||canvas_to_shape||_2 * d, so search
    // with that (conservative) radius and filter by canvas distance afterwards.
    auto local_max_radius = max_radius;
    if (isfinite(max_radius)) {
        const auto &c = shape_group.canvas_to_shape;
        // largest singular value of the 2x2 linear part
        auto a00 = c(0, 0) * c(0, 0) + c(1, 0) * c(1, 0);
        auto a01 = c(0, 0) * c(0, 1) + c(1, 0) * c(1, 1);
        auto a11 = c(0, 1) * c(0, 1) + c(1, 1) * c(1, 1);
        auto half_tr = (a00 + a11) / 2;
        auto det = a00 * a11 - a01 * a01;
        // explicit float: with <math.h> in scope sqrt() returns double, and
        // max(double, float) does not compile (clang accepts it, g++ does not)
        auto lambda_max = half_tr + float(sqrt(max(half_tr * half_tr - det, 0.f)));
        local_max_radius = max_radius * float(sqrt(max(lambda_max, 0.f)));
    }

    while (stack_size > 0) {
        const BVHNode &node = bvh_nodes[bvh_stack[--stack_size]];
        if (node.child1 < 0) {
            // leaf
            auto shape_id = node.child0;
            const auto &shape = scene.shapes[shape_id];
            ClosestPointPathInfo local_path_info{-1, -1};
            auto local_closest_pt = Vector2f{0, 0};
            if (closest_point(shape, scene.path_bvhs[shape_id], local_pt, local_max_radius, &local_path_info, &local_closest_pt)) {
                auto closest_pt = xform_pt(shape_group.shape_to_canvas, local_closest_pt);
                auto dist = distance(closest_pt, pt);
                // min_dist starts at max_radius; shapes whose closest_point
                // does not test max_radius itself (circle, ellipse, rect)
                // must not be reported beyond it.
                if (dist < min_dist) {
                    found = true;
                    min_dist = dist;
                    if (min_shape_id != nullptr) {
                        *min_shape_id = shape_id;
                    }
                    if (closest_pt_ != nullptr) {
                        *closest_pt_ = closest_pt;
                    }
                    if (path_info != nullptr) {
                        *path_info = local_path_info;
                    }
                }
            }
        } else {
            assert(node.child0 >= 0 && node.child1 >= 0);
            const AABB &b0 = bvh_nodes[node.child0].box;
            if (inside(b0, local_pt, local_max_radius)) {
                bvh_stack[stack_size++] = node.child0;
            }
            const AABB &b1 = bvh_nodes[node.child1].box;
            if (inside(b1, local_pt, local_max_radius)) {
                bvh_stack[stack_size++] = node.child1;
            }
            assert(stack_size <= max_bvh_stack_size);
        }
    }

    *result = min_dist;
    return found;
}


DEVICE
inline
void d_closest_point(const Circle &circle,
                     const Vector2f &pt,
                     const Vector2f &d_closest_pt,
                     Circle &d_circle,
                     Vector2f &d_pt) {
    // dir = pt - circle.center
    // return circle.center + circle.radius * normalize(dir);
    auto dir = pt - circle.center;
    if (length_squared(dir) < 1e-20f) {
        return;
    }
    auto n = normalize(dir);
    auto d_dir = d_normalize(dir, circle.radius * d_closest_pt);
    auto d_center = d_closest_pt - d_dir;
    atomic_add(&d_circle.center.x, d_center);
    atomic_add(&d_circle.radius, dot(d_closest_pt, n));
    d_pt += d_dir;
}

DEVICE
inline
void d_closest_point(const Ellipse &ellipse,
                     const Vector2f &pt,
                     const Vector2f &d_closest_pt,
                     Ellipse &d_ellipse,
                     Vector2f &d_pt) {
    // The closest point is q(theta) = c + (a cos(theta), b sin(theta)) where
    // theta is the root of F(theta) = (q - pt) . q'(theta) = 0.
    // Differentiate q directly and theta implicitly: d_theta = -dF / F_theta.
    auto closest_pt = Vector2f{0, 0};
    if (!closest_point(ellipse, pt, &closest_pt)) {
        return;
    }
    auto a = ellipse.radius.x;
    auto b = ellipse.radius.y;
    auto c = ellipse.center;
    auto rel = closest_pt - c;
    auto theta = atan2(rel.y / b, rel.x / a);
    auto ct = cos(theta);
    auto st = sin(theta);
    auto u = pt - c;
    // q = c + (a ct, b st)
    auto d_c = d_closest_pt;
    auto d_a = d_closest_pt.x * ct;
    auto d_b = d_closest_pt.y * st;
    auto d_theta = -d_closest_pt.x * a * st + d_closest_pt.y * b * ct;
    // F = (b^2 - a^2) st ct + a ux st - b uy ct
    auto F_theta = (b * b - a * a) * (ct * ct - st * st) + a * u.x * ct + b * u.y * st;
    auto d_u = Vector2f{0, 0};
    if (fabs(F_theta) > 1e-8f * max(a * a + b * b, 1e-8f)) {
        auto k = -d_theta / F_theta;
        d_u.x = k * (a * st);
        d_u.y = k * (-b * ct);
        d_a += k * (-2 * a * st * ct + u.x * st);
        d_b += k * (2 * b * st * ct - u.y * ct);
    }
    // u = pt - c
    d_c -= d_u;
    d_pt += d_u;
    atomic_add(&d_ellipse.center.x, d_c);
    atomic_add(&d_ellipse.radius.x, Vector2f{d_a, d_b});
}

DEVICE
inline
void d_closest_point(const Path &path,
                     const Vector2f &pt,
                     const Vector2f &d_closest_pt,
                     const ClosestPointPathInfo &path_info,
                     Path &d_path,
                     Vector2f &d_pt,
                     float d_t_root = 0) {
    auto base_point_id = path_info.base_point_id;
    auto point_id = path_info.point_id;
    auto min_t_root = path_info.t_root;
    
    if (base_point_id < 0) {
        return;
    }
    auto ncp = path.num_control_points[base_point_id];
    assert(ncp >= 0 && ncp <= 2);
    // Segment of degree n with control points p_i = points[idx[i]] and closest
    // point q(t) = sum_i B_i(t) p_i (Bernstein basis).
    //  * Lines: q is the clamped projection p0 + t (p1 - p0),
    //    t = (pt - p0).(p1 - p0) / |p1 - p0|^2; the dt/dp0, dt/dp1, dt/dpt terms are
    //    included (they cancel only under conformal shape_to_canvas).
    //  * Curves with interior t: t solves G(t) = (q(t) - pt) . q'(t) = 0, so by the
    //    implicit function theorem dt/dtheta = -(dG/dtheta) / G'(t) with
    //    G'(t) = q'.q' + (q - pt).q'', dG/dp_i = B_i q' + B'_i (q - pt), dG/dpt = -q'.
    //    The t term is skipped when |G'| <= 1e-6 (|q'|^2 + |q - pt| |q''|).
    //  * t == 0 or t == 1: end point, no t dependence.
    // (Replaces per-degree power-basis code that discarded the quadratic point
    // gradients, dropped the line dt terms and divided by the cubic leading
    // coefficient.)
    int n = ncp + 1;
    int idx[4] = {0, 0, 0, 0};
    Vector2f P[4] = {Vector2f{0, 0}, Vector2f{0, 0}, Vector2f{0, 0}, Vector2f{0, 0}};
    for (int i = 0; i <= n; i++) {
        idx[i] = i == n ? (point_id + i) % path.num_points : point_id + i;
        // control points relative to pt
        P[i] = Vector2f{path.points[2 * idx[i]], path.points[2 * idx[i] + 1]} - pt;
    }
    Vector2f d_P[4] = {Vector2f{0, 0}, Vector2f{0, 0}, Vector2f{0, 0}, Vector2f{0, 0}};
    if (n == 1) {
        // Straight line
        auto p0 = Vector2f{path.points[2 * idx[0]], path.points[2 * idx[0] + 1]};
        auto p1 = Vector2f{path.points[2 * idx[1]], path.points[2 * idx[1] + 1]};
        auto e = p1 - p0;
        auto ll = dot(e, e);
        auto t = ll > 0 ? dot(pt - p0, e) / ll : -1.f;
        if (t < 0) {
            d_P[0] += d_closest_pt;
        } else if (t > 1) {
            d_P[1] += d_closest_pt;
        } else {
            // q = p0 + t (p1 - p0)
            d_P[0] += d_closest_pt * (1 - t);
            d_P[1] += d_closest_pt * t;
            // t = num / den, num = (pt - p0).(p1 - p0), den = (p1 - p0).(p1 - p0)
            auto d_t = dot(d_closest_pt, e) + d_t_root;
            auto d_num = d_t / ll;
            auto d_den = -d_t * t / ll;
            d_pt += e * d_num;
            d_P[1] += (pt - p0) * d_num;
            d_P[0] += ((p0 - p1) + (p0 - pt)) * d_num;
            d_P[1] += e * (2 * d_den);
            d_P[0] += e * (-2 * d_den);
        }
    } else {
        auto t = min_t_root;
        if (t == 0) {
            d_P[0] += d_closest_pt;
        } else if (t == 1) {
            d_P[n] += d_closest_pt;
        } else {
            auto tt = 1 - t;
            // Bernstein basis B, B', B'' at t
            float B[4] = {0, 0, 0, 0};
            float dB[4] = {0, 0, 0, 0};
            float ddB[4] = {0, 0, 0, 0};
            if (n == 2) {
                B[0] = tt * tt; B[1] = 2 * tt * t; B[2] = t * t;
                dB[0] = -2 * tt; dB[1] = 2 * (tt - t); dB[2] = 2 * t;
                ddB[0] = 2; ddB[1] = -4; ddB[2] = 2;
            } else {
                B[0] = tt * tt * tt; B[1] = 3 * tt * tt * t; B[2] = 3 * tt * t * t; B[3] = t * t * t;
                dB[0] = -3 * tt * tt; dB[1] = 3 * tt * (tt - 2 * t); dB[2] = 3 * t * (2 * tt - t); dB[3] = 3 * t * t;
                ddB[0] = 6 * tt; ddB[1] = 6 * (3 * t - 2); ddB[2] = 6 * (1 - 3 * t); ddB[3] = 6 * t;
            }
            // q is relative to pt
            auto q = Vector2f{0, 0};
            auto dq = Vector2f{0, 0};
            auto ddq = Vector2f{0, 0};
            for (int i = 0; i <= n; i++) {
                q += B[i] * P[i];
                dq += dB[i] * P[i];
                ddq += ddB[i] * P[i];
            }
            for (int i = 0; i <= n; i++) {
                d_P[i] += B[i] * d_closest_pt;
            }
            auto G_t = dot(dq, dq) + dot(q, ddq);
            auto scale = dot(dq, dq) + length(q) * length(ddq);
            if (scale > 0 && fabs(G_t) > 1e-6f * scale) {
                // d_t_root: extra gradient w.r.t. t from callers whose output also
                // depends on the closest-point parameter (prefiltered thickness)
                auto k = -(dot(d_closest_pt, dq) + d_t_root) / G_t;
                for (int i = 0; i <= n; i++) {
                    d_P[i] += k * (B[i] * dq + dB[i] * q);
                }
                d_pt += (-k) * dq;
            }
        }
    }
    for (int i = 0; i <= n; i++) {
        atomic_add(d_path.points + 2 * idx[i], d_P[i]);
    }
}

DEVICE
inline
void d_closest_point(const Rect &rect,
                     const Vector2f &pt,
                     const Vector2f &d_closest_pt,
                     Rect &d_rect,
                     Vector2f &d_pt) {
    auto dist = [&](const Vector2f &p0, const Vector2f &p1) -> float {
        // project pt to line
        // zero-length segment: 0/0 = NaN would skip every branch below with a NaN
        // distance; treat it as the point p0 (as geometry.metal does)
        auto t = dot(p1 - p0, p1 - p0) > 0 ? dot(pt - p0, p1 - p0) / dot(p1 - p0, p1 - p0) : -1.f;
        if (t < 0) {
            return distance(p0, pt);
        } else if (t > 1) {
            return distance(p1, pt);
        } else {
            return distance(p0 + t * (p1 - p0), pt);
        }
        // return 0;
    };
    auto left_top = rect.p_min;
    auto right_top = Vector2f{rect.p_max.x, rect.p_min.y};
    auto left_bottom = Vector2f{rect.p_min.x, rect.p_max.y};
    auto right_bottom = rect.p_max;
    auto left_dist = dist(left_top, left_bottom);
    auto top_dist = dist(left_top, right_top);
    auto right_dist = dist(right_top, right_bottom);
    auto bottom_dist = dist(left_bottom, right_bottom);
    int min_id = 0;
    auto min_dist = left_dist;
    if (top_dist < min_dist) { min_dist = top_dist; min_id = 1; }
    if (right_dist < min_dist) { min_dist = right_dist; min_id = 2; }
    if (bottom_dist < min_dist) { min_dist = bottom_dist; min_id = 3; }

    auto d_update = [&](const Vector2f &p0, const Vector2f &p1,
                        const Vector2f &d_closest_pt,
                        Vector2f &d_p0, Vector2f &d_p1) {
        // project pt to line
        // zero-length segment: 0/0 = NaN would skip every branch below with a NaN
        // distance; treat it as the point p0 (as geometry.metal does)
        auto t = dot(p1 - p0, p1 - p0) > 0 ? dot(pt - p0, p1 - p0) / dot(p1 - p0, p1 - p0) : -1.f;
        if (t < 0) {
            d_p0 += d_closest_pt;
        } else if (t > 1) {
            d_p1 += d_closest_pt;
        } else {
            // p = p0 + t * (p1 - p0)
            auto d_p = d_closest_pt;
            d_p0 += d_p * (1 - t);
            d_p1 += d_p * t;
            auto d_t = sum(d_p * (p1 - p0));
            // t = dot(pt - p0, p1 - p0) / dot(p1 - p0, p1 - p0)
            auto d_numerator = d_t / dot(p1 - p0, p1 - p0);
            auto d_denominator = d_t * (-t) / dot(p1 - p0, p1 - p0);
            // numerator = dot(pt - p0, p1 - p0)
            d_pt += (p1 - p0) * d_numerator;
            d_p1 += (pt - p0) * d_numerator;
            d_p0 += ((p0 - p1) + (p0 - pt)) * d_numerator;
            // denominator = dot(p1 - p0, p1 - p0)
            d_p1 += 2 * (p1 - p0) * d_denominator;
            d_p0 += 2 * (p0 - p1) * d_denominator;
        }
    };
    auto d_left_top = Vector2f{0, 0};
    auto d_right_top = Vector2f{0, 0};
    auto d_left_bottom = Vector2f{0, 0};
    auto d_right_bottom = Vector2f{0, 0};
    if (min_id == 0) {
        d_update(left_top, left_bottom, d_closest_pt, d_left_top, d_left_bottom);
    } else if (min_id == 1) {
        d_update(left_top, right_top, d_closest_pt, d_left_top, d_right_top);
    } else if (min_id == 2) {
        d_update(right_top, right_bottom, d_closest_pt, d_right_top, d_right_bottom);
    } else {
        assert(min_id == 3);
        d_update(left_bottom, right_bottom, d_closest_pt, d_left_bottom, d_right_bottom);
    }
    auto d_p_min = Vector2f{0, 0};
    auto d_p_max = Vector2f{0, 0};
    // left_top = rect.p_min
    // right_top = Vector2f{rect.p_max.x, rect.p_min.y}
    // left_bottom = Vector2f{rect.p_min.x, rect.p_max.y}
    // right_bottom = rect.p_max
    d_p_min += d_left_top;
    d_p_max.x += d_right_top.x;
    d_p_min.y += d_right_top.y;
    d_p_min.x += d_left_bottom.x;
    d_p_max.y += d_left_bottom.y;
    d_p_max += d_right_bottom;
    atomic_add(d_rect.p_min, d_p_min);
    atomic_add(d_rect.p_max, d_p_max);
}

DEVICE
inline
void d_closest_point(const Shape &shape,
                     const Vector2f &pt,
                     const Vector2f &d_closest_pt,
                     const ClosestPointPathInfo &path_info,
                     Shape &d_shape,
                     Vector2f &d_pt,
                     float d_t_root = 0) {
    switch (shape.type) {
        case ShapeType::Circle:
            d_closest_point(*(const Circle *)shape.ptr,
                            pt,
                            d_closest_pt,
                            *(Circle *)d_shape.ptr,
                            d_pt);
            break;
        case ShapeType::Ellipse:
            d_closest_point(*(const Ellipse *)shape.ptr,
                            pt,
                            d_closest_pt,
                            *(Ellipse *)d_shape.ptr,
                            d_pt);
            break;
        case ShapeType::Path:
            d_closest_point(*(const Path *)shape.ptr,
                            pt,
                            d_closest_pt,
                            path_info,
                            *(Path *)d_shape.ptr,
                            d_pt,
                            d_t_root);
            break;
        case ShapeType::Rect:
            d_closest_point(*(const Rect *)shape.ptr,
                            pt,
                            d_closest_pt,
                            *(Rect *)d_shape.ptr,
                            d_pt);
            break;
    }
}

DEVICE
inline
void d_compute_distance(const Matrix3x3f &canvas_to_shape,
                        const Matrix3x3f &shape_to_canvas,
                        const Shape &shape,
                        const Vector2f &pt,
                        const Vector2f &closest_pt,
                        const ClosestPointPathInfo &path_info,
                        float d_dist,
                        Matrix3x3f &d_shape_to_canvas,
                        Shape &d_shape,
                        float *d_translation,
                        float d_t_root = 0) {
    if (distance_squared(pt, closest_pt) < 1e-10f) {
        // The derivative at distance=0 is undefined
        return;
    }
    assert(isfinite(d_dist));
    // pt is in canvas space, transform it to shape's local space
    auto local_pt = xform_pt(canvas_to_shape, pt);
    auto local_closest_pt = xform_pt(canvas_to_shape, closest_pt);
    // auto local_closest_pt = closest_point(shape, local_pt);
    // auto closest_pt = xform_pt(shape_group.shape_to_canvas, local_closest_pt);
    // auto dist = distance(closest_pt, pt);
    auto d_pt = Vector2f{0, 0};
    auto d_closest_pt = Vector2f{0, 0};
    d_distance(closest_pt, pt, d_dist, d_closest_pt, d_pt);
    assert(isfinite(d_pt));
    assert(isfinite(d_closest_pt));
    // auto closest_pt = xform_pt(shape_group.shape_to_canvas, local_closest_pt);
    auto d_local_closest_pt = Vector2f{0, 0};
    auto d_shape_to_canvas_ = Matrix3x3f();
    d_xform_pt(shape_to_canvas, local_closest_pt, d_closest_pt,
               d_shape_to_canvas_, d_local_closest_pt);
    assert(isfinite(d_local_closest_pt));
    auto d_local_pt = Vector2f{0, 0};
    d_closest_point(shape, local_pt, d_local_closest_pt, path_info, d_shape, d_local_pt, d_t_root);
    assert(isfinite(d_local_pt));
    auto d_canvas_to_shape = Matrix3x3f();
    d_xform_pt(canvas_to_shape,
               pt,
               d_local_pt,
               d_canvas_to_shape,
               d_pt);
    // http://jack.valmadre.net/notes/2016/09/04/back-prop-differentials/#back-propagation-using-differentials
    auto tc2s = transpose(canvas_to_shape);
    d_shape_to_canvas_ += -tc2s * d_canvas_to_shape * tc2s;
    atomic_add(&d_shape_to_canvas(0, 0), d_shape_to_canvas_);
    if (d_translation != nullptr) {
        atomic_add(d_translation, -d_pt);
    }
}
