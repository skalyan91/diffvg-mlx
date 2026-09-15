#pragma once

#include "diffvg.h"
#include "compute_distance.h"
#include "edge_query.h"
#include "shape.h"
#include "vector.h"

DEVICE
inline
bool within_distance(const Circle &circle, const Vector2f &pt, float r) {
    auto dist_to_center = distance(circle.center, pt);
    if (fabs(dist_to_center - circle.radius) < r) {
        return true;
    }
    return false;
}

DEVICE
inline
bool within_distance(const Ellipse &ellipse, const Vector2f &pt, float r) {
    // Cheap rejection/acceptance using the axis-aligned annulus bounds:
    // the boundary lies between the radii min(|rx|,|ry|) and max(|rx|,|ry|).
    auto d_center = distance(ellipse.center, pt);
    auto r_min = min(fabs(ellipse.radius.x), fabs(ellipse.radius.y));
    auto r_max = max(fabs(ellipse.radius.x), fabs(ellipse.radius.y));
    if (d_center >= r_max + r || d_center <= r_min - r) {
        return false;
    }
    auto closest_pt = Vector2f{0, 0};
    if (!closest_point(ellipse, pt, &closest_pt)) {
        return false;
    }
    return distance_squared(closest_pt, pt) < r * r;
}

DEVICE
inline
bool within_distance(const Path &path, const BVHNode *bvh_nodes, const Vector2f &pt, float r) {
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
                auto r0 = r;
                auto r1 = r;
                // override radius if path has thickness
                if (path.thickness != nullptr) {
                    r0 = path.thickness[i0];
                    r1 = path.thickness[i1];
                }
                if (t < 0) {
                    if (distance_squared(p0, pt) < r0 * r0) {
                        return true;
                    }
                } else if (t > 1) {
                    if (distance_squared(p1, pt) < r1 * r1) {
                        return true;
                    }
                } else {
                    auto r = r0 + t * (r1 - r0);
                    if (distance_squared(p0 + t * (p1 - p0), pt) < r * r) {
                        return true;
                    }
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
                    auto cp = quadratic_closest_pt_approx(p0, p1, p2, pt);
                    return distance_squared(cp, pt) < r * r;
                }
                auto eval = [&](float t) -> Vector2f {
                    auto tt = 1 - t;
                    return (tt*tt)*p0 + (2*tt*t)*p1 + (t*t)*p2;
                };
                auto r0 = r;
                auto r1 = r;
                auto r2 = r;
                // override radius if path has thickness
                if (path.thickness != nullptr) {
                    r0 = path.thickness[i0];
                    r1 = path.thickness[i1];
                    r2 = path.thickness[i2];
                }
                if (distance_squared(eval(0), pt) < r0 * r0) {
                    return true;
                }
                if (distance_squared(eval(1), pt) < r2 * r2) {
                    return true;
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
                        auto tt = 1 - t[j];
                        auto r = (tt*tt)*r0 + (2*tt*t[j])*r1 + (t[j]*t[j])*r2;
                        auto p = eval(t[j]);
                        if (distance_squared(p, pt) < r*r) {
                            return true;
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
                auto r0 = r;
                auto r1 = r;
                auto r2 = r;
                auto r3 = r;
                // override radius if path has thickness
                if (path.thickness != nullptr) {
                    r0 = path.thickness[i0];
                    r1 = path.thickness[i1];
                    r2 = path.thickness[i2];
                    r3 = path.thickness[i3];
                }
                if (distance_squared(eval(0), pt) < r0*r0) {
                    return true;
                }
                if (distance_squared(eval(1), pt) < r3*r3) {
                    return true;
                }
                // Stationary points of the distance to the centre line: roots in
                // [0, 1] of the quintic (q(t) - pt) . q'(t), found robustly in
                // Bernstein form (see cubic_closest_roots). The interpolated
                // radius is at most max(r0..r3), so skip the search when the
                // convex-hull capsule is farther than that.
                auto max_r = max(max(r0, r1), max(r2, r3));
                if (cubic_distance_lower_bound(p0, p1, p2, p3, pt) < max_r) {
                    double roots[5];
                    int num_roots = cubic_closest_roots(p0, p1, p2, p3, pt, roots);
                    for (int j = 0; j < num_roots; j++) {
                        auto t = float(roots[j]);
                        auto tt = 1 - t;
                        auto r = (tt*tt*tt)*r0 + (3*tt*tt*t)*r1 + (3*tt*t*t)*r2 + (t*t*t)*r3;
                        if (distance_squared(eval(t), pt) < r * r) {
                            return true;
                        }
                    }
                }
            } else {
                assert(false);
            }
        } else {
            assert(node.child0 >= 0 && node.child1 >= 0);
            const AABB &b0 = bvh_nodes[node.child0].box;
            if (within_distance(b0, pt, bvh_nodes[node.child0].max_radius)) {
                bvh_stack[stack_size++] = node.child0;
            }
            const AABB &b1 = bvh_nodes[node.child1].box;
            if (within_distance(b1, pt, bvh_nodes[node.child1].max_radius)) {
                bvh_stack[stack_size++] = node.child1;
            }
            assert(stack_size <= max_bvh_size);
        }
    }
    return false;
}

DEVICE
inline
int within_distance(const Rect &rect, const Vector2f &pt, float r) {
    auto test = [&](const Vector2f &p0, const Vector2f &p1) {
        // project pt to line
        // zero-length segment: 0/0 = NaN would skip every branch below with a NaN
        // distance; treat it as the point p0 (as geometry.metal does)
        auto t = dot(p1 - p0, p1 - p0) > 0 ? dot(pt - p0, p1 - p0) / dot(p1 - p0, p1 - p0) : -1.f;
        if (t < 0) {
            if (distance_squared(p0, pt) < r * r) {
                return true;
            }
        } else if (t > 1) {
            if (distance_squared(p1, pt) < r * r) {
                return true;
            }
        } else {
            if (distance_squared(p0 + t * (p1 - p0), pt) < r * r) {
                return true;
            }
        }
        return false;
    };
    auto left_top = rect.p_min;
    auto right_top = Vector2f{rect.p_max.x, rect.p_min.y};
    auto left_bottom = Vector2f{rect.p_min.x, rect.p_max.y};
    auto right_bottom = rect.p_max;
    // left
    if (test(left_top, left_bottom)) {
        return true;
    }
    // top
    if (test(left_top, right_top)) {
        return true;
    }
    // right
    if (test(right_top, right_bottom)) {
        return true;
    }
    // bottom
    if (test(left_bottom, right_bottom)) {
        return true;
    }
    return false;
}

DEVICE
inline
bool within_distance(const Shape &shape, const BVHNode *bvh_nodes, const Vector2f &pt, float r) {
    switch (shape.type) {
        case ShapeType::Circle:
            return within_distance(*(const Circle *)shape.ptr, pt, r);
        case ShapeType::Ellipse:
            return within_distance(*(const Ellipse *)shape.ptr, pt, r);
        case ShapeType::Path:
            return within_distance(*(const Path *)shape.ptr, bvh_nodes, pt, r);
        case ShapeType::Rect:
            return within_distance(*(const Rect *)shape.ptr, pt, r);
    }
    assert(false);
    return false;
}

DEVICE
inline
bool within_distance(const SceneData &scene,
                     int shape_group_id,
                     const Vector2f &pt) {
    const ShapeGroup &shape_group = scene.shape_groups[shape_group_id];
    // pt is in canvas space, transform it to shape's local space
    auto local_pt = xform_pt(shape_group.canvas_to_shape, pt);

    constexpr auto max_bvh_stack_size = 64;
    int bvh_stack[max_bvh_stack_size];
    auto stack_size = 0;
    bvh_stack[stack_size++] = 2 * shape_group.num_shapes - 2;
    const auto &bvh_nodes = scene.shape_groups_bvh_nodes[shape_group_id];

    while (stack_size > 0) {
        const BVHNode &node = bvh_nodes[bvh_stack[--stack_size]];
        if (node.child1 < 0) {
            // leaf
            auto shape_id = node.child0;
            const auto &shape = scene.shapes[shape_id];
            if (within_distance(shape, scene.path_bvhs[shape_id],
                                local_pt, shape.stroke_width)) {
                return true;
            }
        } else {
            assert(node.child0 >= 0 && node.child1 >= 0);
            const AABB &b0 = bvh_nodes[node.child0].box;
            if (inside(b0, local_pt, bvh_nodes[node.child0].max_radius)) {
                bvh_stack[stack_size++] = node.child0;
            }
            const AABB &b1 = bvh_nodes[node.child1].box;
            if (inside(b1, local_pt, bvh_nodes[node.child1].max_radius)) {
                bvh_stack[stack_size++] = node.child1;
            }
            assert(stack_size <= max_bvh_stack_size);
        }
    }

    return false;
}

DEVICE
inline
bool within_distance(const SceneData &scene,
                     int shape_group_id,
                     const Vector2f &pt,
                     EdgeQuery *edge_query) {
    if (edge_query == nullptr || shape_group_id != edge_query->shape_group_id) {
        // Specialized version
        return within_distance(scene, shape_group_id, pt);
    }
    const ShapeGroup &shape_group = scene.shape_groups[shape_group_id];
    // pt is in canvas space, transform it to shape's local space
    auto local_pt = xform_pt(shape_group.canvas_to_shape, pt);

    constexpr auto max_bvh_stack_size = 64;
    int bvh_stack[max_bvh_stack_size];
    auto stack_size = 0;
    bvh_stack[stack_size++] = 2 * shape_group.num_shapes - 2;
    const auto &bvh_nodes = scene.shape_groups_bvh_nodes[shape_group_id];

    auto ret = false;
    while (stack_size > 0) {
        const BVHNode &node = bvh_nodes[bvh_stack[--stack_size]];
        if (node.child1 < 0) {
            // leaf
            auto shape_id = node.child0;
            const auto &shape = scene.shapes[shape_id];
            if (within_distance(shape, scene.path_bvhs[shape_id],
                                local_pt, shape.stroke_width)) {
                ret = true;
                if (shape_id == edge_query->shape_id) {
                    edge_query->hit = true;
                }
            }
        } else {
            assert(node.child0 >= 0 && node.child1 >= 0);
            const AABB &b0 = bvh_nodes[node.child0].box;
            if (inside(b0, local_pt, bvh_nodes[node.child0].max_radius)) {
                bvh_stack[stack_size++] = node.child0;
            }
            const AABB &b1 = bvh_nodes[node.child1].box;
            if (inside(b1, local_pt, bvh_nodes[node.child1].max_radius)) {
                bvh_stack[stack_size++] = node.child1;
            }
            assert(stack_size <= max_bvh_stack_size);
        }
    }

    return ret;
}
