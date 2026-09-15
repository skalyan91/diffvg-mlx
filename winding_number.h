#pragma once

#include "diffvg.h"
#include "scene.h"
#include "shape.h"
#include "solve.h"
#include "vector.h"

DEVICE
int compute_winding_number(const Circle &circle, const Vector2f &pt) {
    const auto &c = circle.center;
    auto r = circle.radius;
    // inside the circle: return 1, outside the circle: return 0
    if (distance_squared(c, pt) < r * r) {
        return 1;
    } else {
        return 0;
    }
}

DEVICE
int compute_winding_number(const Ellipse &ellipse, const Vector2f &pt) {
    const auto &c = ellipse.center;
    const auto &r = ellipse.radius;
    // inside the ellipse: return 1, outside the ellipse: return 0
    if (square(c.x - pt.x) / square(r.x) + square(c.y - pt.y) / square(r.y) < 1) {
        return 1;
    } else {
        return 0;
    }
}

DEVICE
bool intersect(const AABB &box, const Vector2f &pt) {
    if (pt.y < box.p_min.y || pt.y > box.p_max.y) {
        return false;
    }
    if (pt.x > box.p_max.x) {
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Crossing rule for the path winding number (mirrored by winding_number_path
// in pydiffvg/metal/geometry.metal, which runs the same algorithm in float32).
//
// A horizontal ray {y = pt.y, x > pt.x} is intersected with y-monotone curve
// pieces under a HALF-OPEN rule: a piece from value ya to yb crosses the ray
// iff (ya <= pt.y) != (yb <= pt.y); it contributes +1 if it starts below
// (it runs upwards in y) and -1 otherwise, provided the crossing lies right
// of pt. A segment END POINT is classified from the stored point
// (p.y <= pt.y) with no arithmetic, so two segments sharing a vertex classify
// it identically: a ray through (or within float noise of) a vertex is counted
// once for a crossing and zero or two times (with opposite signs) for a touch,
// independent of root finding. Curves are split at the roots of y'(t) into
// y-monotone pieces; the value at a split point is evaluated once and its
// classification is shared by the two pieces it separates. Errors in the split
// values or in the root location only matter for points within float noise of
// the curve.

// Cubic Bernstein polynomial and its derivative.
DEVICE
inline double wn_bez3_eval_d(const double c[4], double t, double &d) {
    double s = 1 - t;
    double a0 = s * c[0] + t * c[1];
    double a1 = s * c[1] + t * c[2];
    double a2 = s * c[2] + t * c[3];
    double b0 = s * a0 + t * a1;
    double b1 = s * a1 + t * a2;
    d = 3 * (b1 - b0);
    return s * b0 + t * b1;
}

// Roots of a0 t^2 + a1 t + a2 = 0 strictly inside (0, 1), sorted.
DEVICE
inline int wn_quadratic_roots01(double a0, double a1, double a2, double &r0, double &r1) {
    int n = 0;
    double t0 = 0, t1 = 0;
    if (a0 == 0) {
        if (fabs(a2) < fabs(a1)) {
            t0 = -a2 / a1;
            n = 1;
        }
    } else {
        double disc = a1 * a1 - 4 * a0 * a2;
        if (disc >= 0) {
            double sq = sqrt(disc);
            double q = a1 < 0 ? -0.5 * (a1 - sq) : -0.5 * (a1 + sq);
            if (fabs(q) < fabs(a0)) {
                t0 = q / a0;
                n = 1;
            }
            if (fabs(a2) < fabs(q)) {
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

// Root of the cubic Bernstein polynomial c on [lo, hi], where c is monotone and
// changes classification (c <= 0) between lo and hi; below_lo is the
// classification at lo. Safeguarded Newton on a shrinking bracket.
DEVICE
inline double wn_bez3_piece_root(const double c[4], double lo, double hi,
                                 double ylo, double yhi, bool below_lo) {
    double u = 0.5 * (lo + hi);
    double den = ylo - yhi;
    if (fabs(ylo) < fabs(den)) {
        double r = ylo / den;
        if (r > 0 && r < 1) {
            u = lo + r * (hi - lo);
        }
    }
    for (int it = 0; it < 100; it++) {
        double du = 0;
        double fu = wn_bez3_eval_d(c, u, du);
        if (fu == 0) {
            return u;
        }
        if ((fu <= 0) == below_lo) {
            lo = u;
        } else {
            hi = u;
        }
        double mid = 0.5 * (lo + hi);
        if (!(mid > lo && mid < hi)) {
            return u;
        }
        double un = mid;
        if (fabs(fu) < fabs(du) * (hi - lo)) {
            double nt = u - fu / du;
            if (nt > lo && nt < hi) {
                un = nt;
            }
        }
        if (fabs(un - u) < 1e-15) {
            return un;
        }
        u = un;
    }
    return u;
}

// Signed crossings of the cubic Bezier with Bernstein coefficients y, x
// (relative to the query point) with the ray {y = 0, x > 0}. below0 / below3
// classify the end points and must come from the stored points.
DEVICE
inline int wn_cubic_crossings(const double y[4], const double x[4], bool below0, bool below3) {
    double xmax = std::max(std::max(x[0], x[1]), std::max(x[2], x[3]));
    if (!(xmax > 0)) {
        return 0;
    }
    bool x_all_right = std::min(std::min(x[0], x[1]), std::min(x[2], x[3])) > 0;
    double d0 = y[1] - y[0];
    double d1 = y[2] - y[1];
    double d2 = y[3] - y[2];
    double e[2];
    int ne = wn_quadratic_roots01(d0 - 2 * d1 + d2, 2 * (d1 - d0), d0, e[0], e[1]);
    double ts[4], ys[4];
    bool bs[4];
    ts[0] = 0;
    ys[0] = y[0];
    bs[0] = below0;
    int n = 1;
    for (int k = 0; k < ne; k++) {
        double dd = 0;
        double yk = wn_bez3_eval_d(y, e[k], dd);
        ts[n] = e[k];
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
        bool right = x_all_right;
        if (!right) {
            double u = wn_bez3_piece_root(y, ts[k], ts[k + 1], ys[k], ys[k + 1], bs[k]);
            double dx = 0;
            right = wn_bez3_eval_d(x, u, dx) > 0;
        }
        if (right) {
            winding += bs[k] ? 1 : -1;
        }
    }
    return winding;
}

DEVICE
int compute_winding_number(const Path &path, const BVHNode *bvh_nodes, const Vector2f &pt) {
    // Shoot a horizontal ray from pt to right, intersect with all curves of the path,
    // count intersection (half-open rule, see wn_cubic_crossings)
    auto num_segments = path.num_base_points;
    constexpr auto max_bvh_size = 128;
    int bvh_stack[max_bvh_size];
    auto stack_size = 0;
    auto winding_number = 0;
    bvh_stack[stack_size++] = 2 * num_segments - 2;
    const double px = pt.x, py = pt.y;
    while (stack_size > 0) {
        const BVHNode &node = bvh_nodes[bvh_stack[--stack_size]];
        if (node.child1 < 0) {
            // leaf
            auto base_point_id = node.child0;
            auto point_id = - node.child1 - 1;
            assert(base_point_id < num_segments);
            assert(point_id < path.num_points);
            if (path.num_control_points[base_point_id] == 0) {
                // Straight line
                auto i0 = point_id;
                auto i1 = (point_id + 1) % path.num_points;
                auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
                auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
                bool b0 = p0.y <= pt.y;
                bool b1 = p1.y <= pt.y;
                if (b0 != b1) {
                    // crossing x >= pt.x  <=>  sign(cross) matches the direction
                    double q0x = p0.x - px, q0y = p0.y - py;
                    double q1x = p1.x - px, q1y = p1.y - py;
                    double cr = q0x * q1y - q0y * q1x;
                    if (b0 ? cr >= 0 : cr <= 0) {
                        winding_number += b0 ? 1 : -1;
                    }
                }
            } else if (path.num_control_points[base_point_id] == 1) {
                // Quadratic Bezier curve, degree-elevated to a cubic (end points unchanged)
                auto i0 = point_id;
                auto i1 = point_id + 1;
                auto i2 = (point_id + 2) % path.num_points;
                auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
                auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
                auto p2 = Vector2f{path.points[2 * i2], path.points[2 * i2 + 1]};
                bool b0 = p0.y <= pt.y;
                bool b1 = p1.y <= pt.y;
                bool b2 = p2.y <= pt.y;
                if (!(b0 == b1 && b1 == b2)) {
                    double q0x = p0.x - px, q1x = p1.x - px, q2x = p2.x - px;
                    double q0y = p0.y - py, q1y = p1.y - py, q2y = p2.y - py;
                    double cy[4] = {q0y, (q0y + 2 * q1y) / 3, (2 * q1y + q2y) / 3, q2y};
                    double cx[4] = {q0x, (q0x + 2 * q1x) / 3, (2 * q1x + q2x) / 3, q2x};
                    winding_number += wn_cubic_crossings(cy, cx, b0, b2);
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
                bool b0 = p0.y <= pt.y;
                bool b1 = p1.y <= pt.y;
                bool b2 = p2.y <= pt.y;
                bool b3 = p3.y <= pt.y;
                if (!(b0 == b1 && b1 == b2 && b2 == b3)) {
                    double cy[4] = {p0.y - py, p1.y - py, p2.y - py, p3.y - py};
                    double cx[4] = {p0.x - px, p1.x - px, p2.x - px, p3.x - px};
                    winding_number += wn_cubic_crossings(cy, cx, b0, b3);
                }
            } else {
                assert(false);
            }
        } else {
            assert(node.child0 >= 0 && node.child1 >= 0);
            const AABB &b0 = bvh_nodes[node.child0].box;
            if (intersect(b0, pt)) {
                bvh_stack[stack_size++] = node.child0;
            }
            const AABB &b1 = bvh_nodes[node.child1].box;
            if (intersect(b1, pt)) {
                bvh_stack[stack_size++] = node.child1;
            }
            assert(stack_size <= max_bvh_size);
        }
    }
    return winding_number;
}

DEVICE
int compute_winding_number(const Rect &rect, const Vector2f &pt) {
    const auto &p_min = rect.p_min;
    const auto &p_max = rect.p_max;
    // inside the rectangle: return 1, outside the rectangle: return 0
    if (pt.x > p_min.x && pt.x < p_max.x && pt.y > p_min.y && pt.y < p_max.y) {
        return 1;
    } else {
        return 0;
    }
}

DEVICE
int compute_winding_number(const Shape &shape, const BVHNode *bvh_nodes, const Vector2f &pt) {
    switch (shape.type) {
        case ShapeType::Circle:
            return compute_winding_number(*(const Circle *)shape.ptr, pt);
        case ShapeType::Ellipse:
            return compute_winding_number(*(const Ellipse *)shape.ptr, pt);
        case ShapeType::Path:
            return compute_winding_number(*(const Path *)shape.ptr, bvh_nodes, pt);
        case ShapeType::Rect:
            return compute_winding_number(*(const Rect *)shape.ptr, pt);
    }
    assert(false);
    return 0;
}
