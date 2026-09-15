#pragma once

#include "diffvg.h"
#include "shape.h"
#include "scene.h"
#include "vector.h"
#include "cdf.h"

struct PathBoundaryData {
    int base_point_id;
    int point_id;
    float t;
    // +-1 for a sample on a stroke flank (offset curve c + dir r n) of a path
    // with per-point thickness, whose normal is then the offset-curve normal;
    // 0 otherwise (fills, uniform-width strokes, end caps / joins).
    float offset_dir;
};

struct BoundaryData {
    PathBoundaryData path;
    bool is_stroke;
};

// Speed |d/dt (c(t) + dir * r(t) * n(t))| of a stroke offset curve, where
// d1 = c'(t), d2 = c''(t), n = perp(c') / |c'|, r = stroke radius, dr = r'(t).
// Stroke boundaries are offset curves whose arc length differs from the
// center curve by (1 + dir * r * curvature), so the sampling pdf
// (per unit length of the sampled boundary) must use this speed instead of |c'|.
DEVICE
inline
float offset_curve_speed(const Vector2f &d1, const Vector2f &d2,
                         float r, float dr, float dir) {
    auto len = length(d1);
    auto n = Vector2f{-d1.y, d1.x} / len;
    auto dn = Vector2f{-d2.y, d2.x} / len - n * (dot(d1, d2) / (len * len));
    return length(d1 + dir * (dr * n + r * dn));
}

// Unit normal of the offset curve b(t) = c(t) + dir * r(t) * n(t) (tangent
// d1 + dir * (dr * n + r * dn), see offset_curve_speed), oriented to agree
// with n = perp(c') / |c'|. For a varying radius (dr != 0) it is tilted from n
// by atan(dr / ((1 + dir r curvature) |c'|)).
DEVICE
inline
Vector2f offset_curve_normal(const Vector2f &d1, const Vector2f &d2,
                             float r, float dr, float dir, const Vector2f &n) {
    auto len = length(d1);
    auto dn = Vector2f{-d2.y, d2.x} / len - n * (dot(d1, d2) / (len * len));
    auto bt = d1 + dir * (dr * n + r * dn);
    auto bl = length(bt);
    if (!(bl > 0)) {
        return n;
    }
    auto nb = Vector2f{-bt.y, bt.x} / bl;
    return dot(nb, n) < 0 ? -nb : nb;
}

DEVICE
Vector2f sample_boundary(const Circle &circle,
                         float t,
                         Vector2f &normal,
                         float &pdf,
                         BoundaryData &,
                         float stroke_perturb_direction,
                         float stroke_radius) {
    // Parametric form of a circle (t in [0, 1)):
    // x = center.x + r * cos(2pi * t)
    // y = center.y + r * sin(2pi * t)
    auto offset = Vector2f{
        circle.radius * cos(2 * float(M_PI) * t),
        circle.radius * sin(2 * float(M_PI) * t)
    };
    normal = normalize(offset);
    // The stroke boundary is a circle of radius (radius +- stroke_radius)
    auto boundary_radius = fabs(circle.radius + stroke_perturb_direction * stroke_radius);
    if (boundary_radius <= 0) {
        pdf = 0;
        return Vector2f{0, 0};
    }
    pdf /= (2 * float(M_PI) * boundary_radius);
    auto ret = circle.center + offset;
    if (stroke_perturb_direction != 0.f) {
        ret += stroke_perturb_direction * stroke_radius * normal;
        if (stroke_perturb_direction < 0) {
            // normal should point towards the perturb direction
            normal = -normal;
        }
    }
    return ret;
}

DEVICE
Vector2f sample_boundary(const Ellipse &ellipse,
                         float t,
                         Vector2f &normal,
                         float &pdf,
                         BoundaryData &data,
                         float stroke_perturb_direction,
                         float stroke_radius) {
    // t may have been remapped by the fill/stroke selection; record the
    // parameter actually used so that the radius derivative uses it.
    data.path.t = t;
    // Parametric form of a ellipse (t in [0, 1)):
    // x = center.x + r.x * cos(2pi * t)
    // y = center.y + r.y * sin(2pi * t)
    const auto &r = ellipse.radius;
    auto offset = Vector2f{
        r.x * cos(2 * float(M_PI) * t),
        r.y * sin(2 * float(M_PI) * t)
    };
    auto dxdt = -r.x * sin(2 * float(M_PI) * t) * 2 * float(M_PI);
    auto dydt = r.y * cos(2 * float(M_PI) * t) * 2 * float(M_PI);
    // tangent is normalize(dxdt, dydt)
    normal = normalize(Vector2f{dydt, -dxdt});
    if (stroke_perturb_direction != 0.f) {
        auto two_pi_sq = square(2 * float(M_PI));
        auto d2 = Vector2f{-r.x * cos(2 * float(M_PI) * t) * two_pi_sq,
                           -r.y * sin(2 * float(M_PI) * t) * two_pi_sq};
        // offset_curve_speed uses n = perp(c') = (-c'.y, c'.x)/|c'|, while the
        // ellipse normal is (c'.y, -c'.x)/|c'|: flip the direction accordingly.
        auto speed = offset_curve_speed(Vector2f{dxdt, dydt}, d2,
                                        stroke_radius, 0.f, -stroke_perturb_direction);
        if (!(speed > 0)) {
            pdf = 0;
            return Vector2f{0, 0};
        }
        pdf /= speed;
    } else {
        pdf /= sqrt(square(dxdt) + square(dydt));
    }
    auto ret = ellipse.center + offset;
    if (stroke_perturb_direction != 0.f) {
        ret += stroke_perturb_direction * stroke_radius * normal;
        if (stroke_perturb_direction < 0) {
            // normal should point towards the perturb direction
            normal = -normal;
        }
    }
    return ret;
}

DEVICE
Vector2f sample_boundary(const Path &path,
                         const float *path_length_cdf,
                         const float *path_length_pmf,
                         const int *point_id_map,
                         float path_length,
                         float t,
                         Vector2f &normal,
                         float &pdf,
                         BoundaryData &data,
                         float stroke_perturb_direction,
                         float stroke_radius) {
    if (stroke_perturb_direction != 0.f) {
        // A stroke is the union of round-capped segments (see within_distance),
        // so besides the offset curves its boundary contains circular arcs
        // around the path vertices: the end caps of open paths *and* the round
        // joins between consecutive segments (for open and closed paths).
        // We sample full circles around every vertex; the parts of a circle
        // that lie inside the stroke contribute nothing, since the colours on
        // both sides are equal.
        auto num_circles = path.is_closed ? path.num_base_points : path.num_base_points + 1;
        auto vertex_point_id = [&](int k) -> int {
            return k < path.num_base_points ? point_id_map[k] : path.num_points - 1;
        };
        auto cap_length = 0.f;
        if (path.thickness != nullptr) {
            for (int k = 0; k < num_circles; k++) {
                cap_length += 2 * float(M_PI) * path.thickness[vertex_point_id(k)];
            }
        } else {
            cap_length = 2 * float(M_PI) * stroke_radius * num_circles;
        }
        auto cap_prob = cap_length / (cap_length + path_length);
        if (t < cap_prob) {
            t = t / cap_prob;
            // pick a vertex uniformly, then an angle uniformly
            auto k = min(int(t * num_circles), num_circles - 1);
            t = t * num_circles - k;
            auto pid = vertex_point_id(k);
            auto r = path.thickness != nullptr ? path.thickness[pid] : stroke_radius;
            if (!(r > 0)) {
                pdf = 0;
                return Vector2f{0, 0};
            }
            // Both perturbation directions sample the same set of circles, so
            // undo the factor 0.5 that was applied when choosing the direction.
            pdf *= 2 * cap_prob / float(num_circles) / (2 * float(M_PI) * r);
            auto p0 = Vector2f{path.points[2 * pid], path.points[2 * pid + 1]};
            auto offset = Vector2f{
                r * cos(2 * float(M_PI) * t),
                r * sin(2 * float(M_PI) * t)
            };
            normal = normalize(offset);
            if (k < path.num_base_points) {
                // start vertex of segment k: t = 0 puts all weight on it
                data.path.base_point_id = k;
                data.path.point_id = pid;
                data.path.t = 0;
            } else {
                // end vertex of the last segment of an open path
                data.path.base_point_id = path.num_base_points - 1;
                data.path.point_id = point_id_map[path.num_base_points - 1];
                data.path.t = 1;
            }
            return p0 + offset;
        } else {
            t = (t - cap_prob) / (1 - cap_prob);
            pdf *= (1 - cap_prob);
        }
    }
    // Binary search on path_length_cdf
    auto sample_id = sample(path_length_cdf,
                            path.num_base_points,
                            t,
                            &t);
    assert(sample_id >= 0 && sample_id < path.num_base_points);
    auto point_id = point_id_map[sample_id];
    if (path.num_control_points[sample_id] == 0) {
        // Straight line
        auto i0 = point_id;
        auto i1 = (i0 + 1) % path.num_points;
        assert(i0 < path.num_points);
        auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
        auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
        data.path.base_point_id = sample_id;
        data.path.point_id = point_id;
        data.path.t = t;
        if (t < -1e-3f || t > 1+1e-3f) {
            // return invalid sample
            pdf = 0;
            return Vector2f{0, 0};
        }
        auto tangent = (p1 - p0);
        auto tan_len = length(tangent);
        if (tan_len == 0) {
            // return invalid sample
            pdf = 0;
            return Vector2f{0, 0};
        }
        normal = Vector2f{-tangent.y, tangent.x} / tan_len;
        // length of tangent is the Jacobian of the sampling transformation
        pdf *= path_length_pmf[sample_id] / tan_len;
        auto ret = p0 + t * (p1 - p0);
        if (stroke_perturb_direction != 0.f) {
            auto r0 = stroke_radius;
            auto r1 = stroke_radius;
            if (path.thickness != nullptr) {
                r0 = path.thickness[i0];
                r1 = path.thickness[i1];
            }
            auto r = r0 + t * (r1 - r0);
            // Jacobian of the offset curve instead of the center curve
            auto dr = r1 - r0;
            auto d2 = Vector2f{0, 0};
            auto speed = offset_curve_speed(tangent, d2, r, dr, stroke_perturb_direction);
            if (!(speed > 0)) {
                pdf = 0;
                return Vector2f{0, 0};
            }
            pdf *= tan_len / speed;
            ret += stroke_perturb_direction * r * normal;
            if (path.thickness != nullptr) {
                // The offset curve is not parallel to the centre curve when the
                // thickness varies: use its own normal for the Reynolds term.
                // (The point b = c + dir r n and the pdf speed |b'| are unchanged.)
                normal = offset_curve_normal(tangent, d2, r, dr, stroke_perturb_direction, normal);
                data.path.offset_dir = stroke_perturb_direction;
            }
            if (stroke_perturb_direction < 0) {
                // normal should point towards the perturb direction
                normal = -normal;
            }
        }
        return ret;
    } else if (path.num_control_points[sample_id] == 1) {
        // Quadratic Bezier curve
        auto i0 = point_id;
        auto i1 = i0 + 1;
        auto i2 = (i0 + 2) % path.num_points;
        auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
        auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
        auto p2 = Vector2f{path.points[2 * i2], path.points[2 * i2 + 1]};
        auto eval = [&](float t) -> Vector2f {
            auto tt = 1 - t;
            return (tt*tt)*p0 + (2*tt*t)*p1 + (t*t)*p2;
        };
        data.path.base_point_id = sample_id;
        data.path.point_id = point_id;
        data.path.t = t;
        if (t < -1e-3f || t > 1+1e-3f) {
            // return invalid sample
            pdf = 0;
            return Vector2f{0, 0};
        }
        auto tangent = 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1);
        auto tan_len = length(tangent);
        if (tan_len == 0) {
            // return invalid sample
            pdf = 0;
            return Vector2f{0, 0};
        }
        normal = Vector2f{-tangent.y, tangent.x} / tan_len;
        // length of tangent is the Jacobian of the sampling transformation
        pdf *= path_length_pmf[sample_id] / tan_len;
        auto ret = eval(t);
        if (stroke_perturb_direction != 0.f) {
            auto r0 = stroke_radius;
            auto r1 = stroke_radius;
            auto r2 = stroke_radius;
            if (path.thickness != nullptr) {
                r0 = path.thickness[i0];
                r1 = path.thickness[i1];
                r2 = path.thickness[i2];
            }
            auto tt = 1 - t;
            auto r = (tt*tt)*r0 + (2*tt*t)*r1 + (t*t)*r2;
            auto dr = 2 * tt * (r1 - r0) + 2 * t * (r2 - r1);
            auto d2 = 2 * (p2 - 2 * p1 + p0);
            // Jacobian of the offset curve instead of the center curve
            auto speed = offset_curve_speed(tangent, d2, r, dr, stroke_perturb_direction);
            if (!(speed > 0)) {
                pdf = 0;
                return Vector2f{0, 0};
            }
            pdf *= tan_len / speed;
            ret += stroke_perturb_direction * r * normal;
            if (path.thickness != nullptr) {
                // The offset curve is not parallel to the centre curve when the
                // thickness varies: use its own normal for the Reynolds term.
                // (The point b = c + dir r n and the pdf speed |b'| are unchanged.)
                normal = offset_curve_normal(tangent, d2, r, dr, stroke_perturb_direction, normal);
                data.path.offset_dir = stroke_perturb_direction;
            }
            if (stroke_perturb_direction < 0) {
                // normal should point towards the perturb direction
                normal = -normal;
            }
        }
        return ret;
    } else if (path.num_control_points[sample_id] == 2) {
        // Cubic Bezier curve
        auto i0 = point_id;
        auto i1 = point_id + 1;
        auto i2 = point_id + 2;
        auto i3 = (point_id + 3) % path.num_points;
        assert(i0 >= 0 && i2 < path.num_points);
        auto p0 = Vector2f{path.points[2 * i0], path.points[2 * i0 + 1]};
        auto p1 = Vector2f{path.points[2 * i1], path.points[2 * i1 + 1]};
        auto p2 = Vector2f{path.points[2 * i2], path.points[2 * i2 + 1]};
        auto p3 = Vector2f{path.points[2 * i3], path.points[2 * i3 + 1]};
        auto eval = [&](float t) -> Vector2f {
            auto tt = 1 - t;
            return (tt*tt*tt)*p0 + (3*tt*tt*t)*p1 + (3*tt*t*t)*p2 + (t*t*t)*p3;
        };
        data.path.base_point_id = sample_id;
        data.path.point_id = point_id;
        data.path.t = t;
        if (t < -1e-3f || t > 1+1e-3f) {
            // return invalid sample
            pdf = 0;
            return Vector2f{0, 0};
        }
        auto tangent = 3 * square(1 - t) * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t * t * (p3 - p2);
        auto tan_len = length(tangent);
        if (tan_len == 0) {
            // return invalid sample
            pdf = 0;
            return Vector2f{0, 0};
        }
        normal = Vector2f{-tangent.y, tangent.x} / tan_len;
        // length of tangent is the Jacobian of the sampling transformation
        pdf *= path_length_pmf[sample_id] / tan_len;
        auto ret = eval(t);
        if (stroke_perturb_direction != 0.f) {
            auto r0 = stroke_radius;
            auto r1 = stroke_radius;
            auto r2 = stroke_radius;
            auto r3 = stroke_radius;
            if (path.thickness != nullptr) {
                r0 = path.thickness[i0];
                r1 = path.thickness[i1];
                r2 = path.thickness[i2];
                r3 = path.thickness[i3];
            }
            auto tt = 1 - t;
            auto r = (tt*tt*tt)*r0 + (3*tt*tt*t)*r1 + (3*tt*t*t)*r2 + (t*t*t)*r3;
            auto dr = 3 * tt * tt * (r1 - r0) + 6 * tt * t * (r2 - r1) + 3 * t * t * (r3 - r2);
            auto d2 = 6 * tt * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1);
            // Jacobian of the offset curve instead of the center curve
            auto speed = offset_curve_speed(tangent, d2, r, dr, stroke_perturb_direction);
            if (!(speed > 0)) {
                pdf = 0;
                return Vector2f{0, 0};
            }
            pdf *= tan_len / speed;
            ret += stroke_perturb_direction * r * normal;
            if (path.thickness != nullptr) {
                // The offset curve is not parallel to the centre curve when the
                // thickness varies: use its own normal for the Reynolds term.
                // (The point b = c + dir r n and the pdf speed |b'| are unchanged.)
                normal = offset_curve_normal(tangent, d2, r, dr, stroke_perturb_direction, normal);
                data.path.offset_dir = stroke_perturb_direction;
            }
            if (stroke_perturb_direction < 0) {
                // normal should point towards the perturb direction
                normal = -normal;
            }
        }
        return ret;
    } else {
        assert(false);
    }
    assert(false);
    return Vector2f{0, 0};
}

DEVICE
Vector2f sample_boundary(const Rect &rect,
                         float t, Vector2f &normal,
                         float &pdf,
                         BoundaryData &,
                         float stroke_perturb_direction,
                         float stroke_radius) {
    // Roll a dice to decide whether to sample width or height
    auto w = rect.p_max.x - rect.p_min.x;
    auto h = rect.p_max.y - rect.p_min.y;
    if (stroke_perturb_direction > 0 && stroke_radius > 0) {
        // The outer boundary of a stroked rectangle (within_distance of the
        // four edges) has rounded corners: quarter circles of radius
        // stroke_radius around each corner. Sample those arcs too.
        auto arc_length = 2 * float(M_PI) * stroke_radius;
        auto edge_length = 2 * (w + h);
        auto arc_prob = arc_length / (arc_length + edge_length);
        if (t < arc_prob) {
            t = t / arc_prob;
            pdf *= arc_prob / arc_length;
            auto k = min(int(t * 4), 3);
            auto theta = (float(k) + (t * 4 - float(k))) * float(M_PI) / 2;
            // angle in [0, pi/2): +x,+y quadrant (y points down) -> p_max corner, etc.
            auto corner = k == 0 ? rect.p_max :
                          k == 1 ? Vector2f{rect.p_min.x, rect.p_max.y} :
                          k == 2 ? rect.p_min :
                                   Vector2f{rect.p_max.x, rect.p_min.y};
            normal = Vector2f{cos(theta), sin(theta)};
            return corner + stroke_radius * normal;
        }
        t = (t - arc_prob) / (1 - arc_prob);
        pdf *= (1 - arc_prob);
    }
    pdf /= (2 * (w +h));
    if (t <= w / (w + h)) {
        // Sample width
        // reuse t for the next dice
        t *= (w + h) / w;
        // Roll a dice to decide whether to sample upper width or lower width
        if (t < 0.5f) {
            // Sample upper width
            normal = Vector2f{0, -1};
            auto ret = rect.p_min + 2 * t * Vector2f{rect.p_max.x - rect.p_min.x, 0.f};
            if (stroke_perturb_direction != 0.f) {
                ret += stroke_perturb_direction * stroke_radius * normal;
                if (stroke_perturb_direction < 0) {
                    // normal should point towards the perturb direction
                    normal = -normal;
                }
            }
            return ret;
        } else {
            // Sample lower width
            normal = Vector2f{0, 1};
            auto ret = Vector2f{rect.p_min.x, rect.p_max.y} +
                2 * (t - 0.5f) * Vector2f{rect.p_max.x - rect.p_min.x, 0.f};
            if (stroke_perturb_direction != 0.f) {
                ret += stroke_perturb_direction * stroke_radius * normal;
                if (stroke_perturb_direction < 0) {
                    // normal should point towards the perturb direction
                    normal = -normal;
                }
            }
            return ret;
        }
    } else {
        // Sample height
        // reuse t for the next dice
        assert(h > 0);
        t = (t - w / (w + h)) * (w + h) / h;
        // Roll a dice to decide whether to sample left height or right height
        if (t < 0.5f) {
            // Sample left height
            normal = Vector2f{-1, 0};
            auto ret = rect.p_min + 2 * t * Vector2f{0.f, rect.p_max.y - rect.p_min.y};
            if (stroke_perturb_direction != 0.f) {
                ret += stroke_perturb_direction * stroke_radius * normal;
                if (stroke_perturb_direction < 0) {
                    // normal should point towards the perturb direction
                    normal = -normal;
                }
            }
            return ret;
        } else {
            // Sample right height
            normal = Vector2f{1, 0};
            auto ret = Vector2f{rect.p_max.x, rect.p_min.y} +
                2 * (t - 0.5f) * Vector2f{0.f, rect.p_max.y - rect.p_min.y};
            if (stroke_perturb_direction != 0.f) {
                ret += stroke_perturb_direction * stroke_radius * normal;
                if (stroke_perturb_direction < 0) {
                    // normal should point towards the perturb direction
                    normal = -normal;
                }
            }
            return ret;
        }
    }
}

DEVICE
Vector2f sample_boundary(const SceneData &scene,
                         int shape_group_id,
                         int shape_id,
                         float t,
                         Vector2f &normal,
                         float &pdf,
                         BoundaryData &data) {
    const ShapeGroup &shape_group = scene.shape_groups[shape_group_id];
    const Shape &shape = scene.shapes[shape_id];
    pdf = 1;
    // Choose which one to sample: stroke discontinuities or fill discontinuities.
    // TODO: we don't need to sample fill discontinuities when stroke alpha is 1 and both
    // fill and stroke color exists
    auto stroke_perturb = false;
    if (shape_group.fill_color != nullptr && shape_group.stroke_color != nullptr) {
        if (t < 0.5f) {
            stroke_perturb = false;
            t = 2 * t;
            pdf = 0.5f;
        } else {
            stroke_perturb = true;
            t = 2 * (t - 0.5f);
            pdf = 0.5f;
        }
    } else if (shape_group.stroke_color != nullptr) {
        stroke_perturb = true;
    }
    data.is_stroke = stroke_perturb;
    data.path.offset_dir = 0;
    auto stroke_perturb_direction = 0.f;
    if (stroke_perturb) {
        if (t < 0.5f) {
            stroke_perturb_direction = -1.f;
            t = 2 * t;
            pdf *= 0.5f;
        } else {
            stroke_perturb_direction = 1.f;
            t = 2 * (t - 0.5f);
            pdf *= 0.5f;
        }
    }
    switch (shape.type) {
        case ShapeType::Circle:
            return sample_boundary(
                *(const Circle *)shape.ptr, t, normal, pdf, data, stroke_perturb_direction, shape.stroke_width);
        case ShapeType::Ellipse:
            return sample_boundary(
                *(const Ellipse *)shape.ptr, t, normal, pdf, data, stroke_perturb_direction, shape.stroke_width);
        case ShapeType::Path:
            return sample_boundary(
                *(const Path *)shape.ptr,
                scene.path_length_cdf[shape_id],
                scene.path_length_pmf[shape_id],
                scene.path_point_id_map[shape_id],
                scene.shapes_length[shape_id],
                t,
                normal,
                pdf,
                data,
                stroke_perturb_direction,
                shape.stroke_width);
        case ShapeType::Rect:
            return sample_boundary(
                *(const Rect *)shape.ptr, t, normal, pdf, data, stroke_perturb_direction, shape.stroke_width);
    }
    assert(false);
    return Vector2f{};
}

