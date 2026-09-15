#include "diffvg.h"
#include "aabb.h"
#include "shape.h"
#include "sample_boundary.h"
#include "atomic.h"
#include "cdf.h"
#include "compute_distance.h"
#include "cuda_utils.h"
#include "edge_query.h"
#include "filter.h"
#include "matrix.h"
#include "parallel.h"
#include "pcg.h"
#include "ptr.h"
#include "scene.h"
#include "vector.h"
#include "winding_number.h"
#include "within_distance.h"
#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace py = pybind11;

struct Command {
    int shape_group_id;
    int shape_id;
    int point_id; // Only used by path
};

DEVICE
bool is_inside(const SceneData &scene_data,
               int shape_group_id,
               const Vector2f &pt,
               EdgeQuery *edge_query) {
    const ShapeGroup &shape_group = scene_data.shape_groups[shape_group_id];
    // pt is in canvas space, transform it to shape's local space
    auto local_pt = xform_pt(shape_group.canvas_to_shape, pt);
    const auto &bvh_nodes = scene_data.shape_groups_bvh_nodes[shape_group_id];
    const AABB &bbox = bvh_nodes[2 * shape_group.num_shapes - 2].box;
    if (!inside(bbox, local_pt)) {
        return false;
    }
    auto winding_number = 0;
    // Traverse the shape group BVH
    constexpr auto max_bvh_stack_size = 64;
    int bvh_stack[max_bvh_stack_size];
    auto stack_size = 0;
    bvh_stack[stack_size++] = 2 * shape_group.num_shapes - 2;
    while (stack_size > 0) {
        const BVHNode &node = bvh_nodes[bvh_stack[--stack_size]];
        if (node.child1 < 0) {
            // leaf
            auto shape_id = node.child0;
            auto w = compute_winding_number(
                scene_data.shapes[shape_id], scene_data.path_bvhs[shape_id], local_pt);
            winding_number += w;
            if (edge_query != nullptr) {
                if (edge_query->shape_group_id == shape_group_id &&
                        edge_query->shape_id == shape_id) {
                    if ((shape_group.use_even_odd_rule && abs(w) % 2 == 1) ||
                        (!shape_group.use_even_odd_rule && w != 0)) {
                        edge_query->hit = true;
                    }
                }
            }
        } else {
            assert(node.child0 >= 0 && node.child1 >= 0);
            const AABB &b0 = bvh_nodes[node.child0].box;
            if (inside(b0, local_pt)) {
                bvh_stack[stack_size++] = node.child0;
            }
            const AABB &b1 = bvh_nodes[node.child1].box;
            if (inside(b1, local_pt)) {
                bvh_stack[stack_size++] = node.child1;
            }
            assert(stack_size <= max_bvh_stack_size);
        }
    }
    if (shape_group.use_even_odd_rule) {
        return abs(winding_number) % 2 == 1;
    } else {
        return winding_number != 0;
    }
}

DEVICE void accumulate_boundary_gradient(const Shape &shape,
                                         float canvas_contrib,
                                         float t,
                                         const Vector2f &normal, // shape (local) space
                                         const BoundaryData &boundary_data,
                                         Shape &d_shape,
                                         const Matrix3x3f &shape_to_canvas,
                                         const Vector2f &local_boundary_pt,
                                         Matrix3x3f &d_shape_to_canvas,
                                         const Vector2f &canvas_normal,
                                         float local_velocity_scale) {
    assert(isfinite(canvas_contrib));
    assert(isfinite(normal));
    assert(isfinite(canvas_normal));
    // According to Reynold transport theorem,
    // the Jacobian of the boundary integral is dot(velocity, normal),
    // where the velocity depends on the variable being differentiated with.
    // canvas_contrib is the Monte Carlo weight of the canvas-space boundary
    // integral, so it must be multiplied by dot(canvas velocity, canvas normal).
    // Shape parameters move the boundary in local space; with
    // M = linear part of shape_to_canvas and
    // canvas_normal = normalize(M^{-T} normal):
    //   dot(M v_local, canvas_normal) = dot(v_local, normal) / |M^{-T} normal|
    // so we use the local normal with contrib scaled by local_velocity_scale.
    auto contrib = canvas_contrib * local_velocity_scale;
    // Stroke flank of a path with per-point thickness (sample_boundary sets
    // offset_dir): the boundary point is b = c(t) + dir r(t) n(t) with
    // n = perp(c') / |c'|, and `normal` is the normal of the offset curve b,
    // which differs from +-n when r varies. The Reynolds term is
    // dot(velocity of b, normal), so
    //  * thickness r_i: velocity dir B_i n      -> B_i dir dot(n, normal)
    //  * point p_i:     velocity B_i e_k + dir r dn/dp_i,
    //                   dn/dp_i[k] = B'_i (perp(e_k) - n dot(n, perp(e_k))) / |c'|
    // For uniform-width strokes (and caps / joins) normal = +-n, the rotation
    // term vanishes and the code below reduces to the plain velocities.
    auto flank_dir = 0.f;
    auto flank_n = Vector2f{0, 0};
    auto flank_len = 0.f;
    auto flank_r = 0.f;
    int flank_degree = 0;
    int flank_idx[4] = {0, 0, 0, 0};
    float flank_dB[4] = {0, 0, 0, 0};
    if (boundary_data.is_stroke && shape.type == ShapeType::Path &&
            boundary_data.path.offset_dir != 0) {
        const Path &path = *(const Path *)shape.ptr;
        if (path.thickness != nullptr) {
            auto point_id = boundary_data.path.point_id;
            auto t = boundary_data.path.t;
            auto tt = 1 - t;
            int n = path.num_control_points[boundary_data.path.base_point_id] + 1;
            float B[4] = {0, 0, 0, 0};
            float dB[4] = {0, 0, 0, 0};
            if (n == 1) {
                B[0] = tt; B[1] = t;
                dB[0] = -1; dB[1] = 1;
            } else if (n == 2) {
                B[0] = tt * tt; B[1] = 2 * tt * t; B[2] = t * t;
                dB[0] = -2 * tt; dB[1] = 2 * (tt - t); dB[2] = 2 * t;
            } else {
                B[0] = tt * tt * tt; B[1] = 3 * tt * tt * t; B[2] = 3 * tt * t * t; B[3] = t * t * t;
                dB[0] = -3 * tt * tt; dB[1] = 3 * tt * (tt - 2 * t); dB[2] = 3 * t * (2 * tt - t); dB[3] = 3 * t * t;
            }
            auto d1 = Vector2f{0, 0};
            auto r = 0.f;
            for (int i = 0; i <= n; i++) {
                flank_idx[i] = i == n ? (point_id + i) % path.num_points : point_id + i;
                d1 += dB[i] * Vector2f{path.points[2 * flank_idx[i]], path.points[2 * flank_idx[i] + 1]};
                r += B[i] * path.thickness[flank_idx[i]];
                flank_dB[i] = dB[i];
            }
            auto len = length(d1);
            if (len > 0) {
                flank_dir = boundary_data.path.offset_dir;
                flank_n = Vector2f{-d1.y, d1.x} / len;
                flank_len = len;
                flank_r = r;
                flank_degree = n;
            }
        }
    }
    if (boundary_data.is_stroke) {
        auto has_path_thickness = false;
        if (shape.type == ShapeType::Path) {
            const Path &path = *(const Path *)shape.ptr;
            has_path_thickness = path.thickness != nullptr;
        }
        // differentiate stroke width: velocity is the same as normal
        if (has_path_thickness) {
            auto thickness_contrib = flank_dir != 0 ?
                contrib * flank_dir * dot(flank_n, normal) : contrib;
            Path *d_p = (Path*)d_shape.ptr;
            auto base_point_id = boundary_data.path.base_point_id;
            auto point_id = boundary_data.path.point_id;
            auto t = boundary_data.path.t;
            const Path &path = *(const Path *)shape.ptr;
            if (path.num_control_points[base_point_id] == 0) {
                // Straight line
                auto i0 = point_id;
                auto i1 = (point_id + 1) % path.num_points;
                // r = r0 + t * (r1 - r0)
                atomic_add(&d_p->thickness[i0], (1 - t) * thickness_contrib);
                atomic_add(&d_p->thickness[i1], (    t) * thickness_contrib);
            } else if (path.num_control_points[base_point_id] == 1) {
                // Quadratic Bezier curve
                auto i0 = point_id;
                auto i1 = point_id + 1;
                auto i2 = (point_id + 2) % path.num_points;
                // r = (1-t)^2r0 + 2(1-t)t r1 + t^2 r2
                atomic_add(&d_p->thickness[i0], square(1 - t) * thickness_contrib);
                atomic_add(&d_p->thickness[i1], (2*(1-t)*t) * thickness_contrib);
                atomic_add(&d_p->thickness[i2], (t*t) * thickness_contrib);
            } else if (path.num_control_points[base_point_id] == 2) {
                auto i0 = point_id;
                auto i1 = point_id + 1;
                auto i2 = point_id + 2;
                auto i3 = (point_id + 3) % path.num_points;
                // r = (1-t)^3r0 + 3*(1-t)^2tr1 + 3*(1-t)t^2r2 + t^3r3
                atomic_add(&d_p->thickness[i0], cubic(1 - t) * thickness_contrib);
                atomic_add(&d_p->thickness[i1], 3 * square(1 - t) * t * thickness_contrib);
                atomic_add(&d_p->thickness[i2], 3 * (1 - t) * t * t * thickness_contrib);
                atomic_add(&d_p->thickness[i3], t * t * t * thickness_contrib);
            } else {
                assert(false);
            }
        } else {
            atomic_add(&d_shape.stroke_width, contrib);
        }
    }
    switch (shape.type) {
        case ShapeType::Circle: {
            Circle *d_p = (Circle*)d_shape.ptr;
            const Circle &circle = *(const Circle *)shape.ptr;
            // velocity for the center is (1, 0) for x and (0, 1) for y
            atomic_add(&d_p->center[0], normal * contrib);
            // velocity for the radius is the radial direction. This equals the
            // normal only on an outward-facing boundary: on the inner side of a
            // stroke (or after the normal is flipped) it is minus the normal.
            auto radial = local_boundary_pt - circle.center;
            auto radial_len = length(radial);
            if (radial_len > 0) {
                atomic_add(&d_p->radius, dot(radial / radial_len, normal) * contrib);
            }
            break;
        } case ShapeType::Ellipse: {
            Ellipse *d_p = (Ellipse*)d_shape.ptr;
            // velocity for the center is (1, 0) for x and (0, 1) for y
            atomic_add(&d_p->center[0], normal * contrib);
            // velocity for the radius:
            // x = center.x + r.x * cos(2pi * t)
            // y = center.y + r.y * sin(2pi * t)
            // for r.x: (cos(2pi * t), 0)
            // for r.y: (0, sin(2pi * t))
            // use the (possibly remapped) parameter stored by sample_boundary,
            // not the raw random number
            auto te = boundary_data.path.t;
            atomic_add(&d_p->radius.x, cos(2 * float(M_PI) * te) * normal.x * contrib);
            atomic_add(&d_p->radius.y, sin(2 * float(M_PI) * te) * normal.y * contrib);
            break;
        } case ShapeType::Path: {
            Path *d_p = (Path*)d_shape.ptr;
            auto base_point_id = boundary_data.path.base_point_id;
            auto point_id = boundary_data.path.point_id;
            auto t = boundary_data.path.t;
            const Path &path = *(const Path *)shape.ptr;
            if (path.num_control_points[base_point_id] == 0) {
                // Straight line
                auto i0 = point_id;
                auto i1 = (point_id + 1) % path.num_points;
                // pt = p0 + t * (p1 - p0)
                // velocity for p0.x: (1 - t,     0)
                //              p0.y: (    0, 1 - t)
                //              p1.x: (    t,     0)
                //              p1.y: (    0,     t)
                atomic_add(&d_p->points[2 * i0 + 0], (1 - t) * normal.x * contrib);
                atomic_add(&d_p->points[2 * i0 + 1], (1 - t) * normal.y * contrib);
                atomic_add(&d_p->points[2 * i1 + 0], (    t) * normal.x * contrib);
                atomic_add(&d_p->points[2 * i1 + 1], (    t) * normal.y * contrib);
            } else if (path.num_control_points[base_point_id] == 1) {
                // Quadratic Bezier curve
                auto i0 = point_id;
                auto i1 = point_id + 1;
                auto i2 = (point_id + 2) % path.num_points;
                // pt = (1-t)^2p0 + 2(1-t)t p1 + t^2 p2
                // velocity for p0.x: ((1-t)^2,       0)
                //              p0.y: (      0, (1-t)^2)
                //              p1.x: (2(1-t)t,       0)
                //              p1.y: (      0, 2(1-t)t)
                //              p1.x: (    t^2,       0)
                //              p1.y: (      0,     t^2)
                atomic_add(&d_p->points[2 * i0 + 0], square(1 - t) * normal.x * contrib);
                atomic_add(&d_p->points[2 * i0 + 1], square(1 - t) * normal.y * contrib);
                atomic_add(&d_p->points[2 * i1 + 0], (2*(1-t)*t) * normal.x * contrib);
                atomic_add(&d_p->points[2 * i1 + 1], (2*(1-t)*t) * normal.y * contrib);
                atomic_add(&d_p->points[2 * i2 + 0], (t*t) * normal.x * contrib);
                atomic_add(&d_p->points[2 * i2 + 1], (t*t) * normal.y * contrib);
            } else if (path.num_control_points[base_point_id] == 2) {
                auto i0 = point_id;
                auto i1 = point_id + 1;
                auto i2 = point_id + 2;
                auto i3 = (point_id + 3) % path.num_points;
                // pt = (1-t)^3p0 + 3*(1-t)^2tp1 + 3*(1-t)t^2p2 + t^3p3
                // velocity for p0.x: (   (1-t)^3,          0)
                //              p0.y: (         0,    (1-t)^3)
                //              p1.x: (3*(1-t)^2t,          0)
                //              p1.y: (         0, 3*(1-t)^2t)
                //              p2.x: (3*(1-t)t^2,          0)
                //              p2.y: (         0, 3*(1-t)t^2)
                //              p2.x: (       t^3,          0)
                //              p2.y: (         0,        t^3)
                atomic_add(&d_p->points[2 * i0 + 0], cubic(1 - t) * normal.x * contrib);
                atomic_add(&d_p->points[2 * i0 + 1], cubic(1 - t) * normal.y * contrib);
                atomic_add(&d_p->points[2 * i1 + 0], 3 * square(1 - t) * t * normal.x * contrib);
                atomic_add(&d_p->points[2 * i1 + 1], 3 * square(1 - t) * t * normal.y * contrib);
                atomic_add(&d_p->points[2 * i2 + 0], 3 * (1 - t) * t * t * normal.x * contrib);
                atomic_add(&d_p->points[2 * i2 + 1], 3 * (1 - t) * t * t * normal.y * contrib);
                atomic_add(&d_p->points[2 * i3 + 0], t * t * t * normal.x * contrib);
                atomic_add(&d_p->points[2 * i3 + 1], t * t * t * normal.y * contrib);
            } else {
                assert(false);
            }
            if (flank_dir != 0) {
                // rotation of the offset direction: dir r dn/dp_i . normal
                auto nx = flank_n.x;
                auto ny = flank_n.y;
                // perp(e_x) = (0, 1), perp(e_y) = (-1, 0)
                auto gx = dot(Vector2f{-nx * ny, 1 - ny * ny}, normal);
                auto gy = dot(Vector2f{nx * nx - 1, nx * ny}, normal);
                auto s = contrib * flank_dir * flank_r / flank_len;
                for (int i = 0; i <= flank_degree; i++) {
                    atomic_add(&d_p->points[2 * flank_idx[i] + 0], flank_dB[i] * s * gx);
                    atomic_add(&d_p->points[2 * flank_idx[i] + 1], flank_dB[i] * s * gy);
                }
            }
            break;
        } case ShapeType::Rect: {
            Rect *d_p = (Rect*)d_shape.ptr;
            const Rect &rect = *(const Rect *)shape.ptr;
            // The velocity depends on which edge the sample lies on. Identify
            // the edge from the sample position rather than from the normal:
            // the normal is flipped on the inner side of a stroke (and whenever
            // render_edge_kernel swaps inside/outside), which previously
            // attributed the gradient to the opposite edge.
            // The x and y components are handled separately so that samples on
            // the rounded stroke corners (normal not axis aligned) move with
            // both coordinates of their corner. For points on an edge the
            // normal has a single non-zero component.
            auto center = 0.5f * (rect.p_min + rect.p_max);
            if (local_boundary_pt.x < center.x) {
                // left edge / left corners: velocity for p_min.x is (1, 0)
                atomic_add(&d_p->p_min.x, normal.x * contrib);
            } else {
                // right edge / right corners: velocity for p_max.x is (1, 0)
                atomic_add(&d_p->p_max.x, normal.x * contrib);
            }
            if (local_boundary_pt.y < center.y) {
                // top edge / top corners: velocity for p_min.y is (0, 1)
                atomic_add(&d_p->p_min.y, normal.y * contrib);
            } else {
                // bottom edge / bottom corners: velocity for p_max.y is (0, 1)
                atomic_add(&d_p->p_max.y, normal.y * contrib);
            }
            break;
        } default: {
            assert(false);
            break;
        }
    }
    // for shape_to_canvas we have the following relationship:
    // boundary_pt = xform_pt(shape_to_canvas, local_pt)
    // the velocity is the derivative of boundary_pt with respect to shape_to_canvas
    // we can use reverse-mode AD to compute the dot product of the velocity and the Jacobian
    // by passing the normal in d_xform_pt
    auto d_shape_to_canvas_ = Matrix3x3f();
    auto d_local_boundary_pt = Vector2f{0, 0};
    d_xform_pt(shape_to_canvas,
               local_boundary_pt,
               canvas_normal * canvas_contrib,
               d_shape_to_canvas_,
               d_local_boundary_pt);
    atomic_add(&d_shape_to_canvas(0, 0), d_shape_to_canvas_);
}

DEVICE
Vector4f sample_color(const ColorType &color_type,
                      void *color,
                      const Vector2f &pt) {
    switch (color_type) {
        case ColorType::Constant: {
            auto c = (const Constant*)color;
            assert(isfinite(c->color));
            return c->color;
        } case ColorType::LinearGradient: {
            auto c = (const LinearGradient*)color;
            // Project pt to (c->begin, c->end)
            auto beg = c->begin;
            auto end = c->end;
            auto t = dot(pt - beg, end - beg) / max(dot(end - beg, end - beg), 1e-3f);
            // Find the correponding stop:
            if (t < c->stop_offsets[0]) {
                return Vector4f{c->stop_colors[0],
                                c->stop_colors[1],
                                c->stop_colors[2],
                                c->stop_colors[3]};
            }
            for (int i = 0; i < c->num_stops - 1; i++) {
                auto offset_curr = c->stop_offsets[i];
                auto offset_next = c->stop_offsets[i + 1];
                assert(offset_next > offset_curr);
                if (t >= offset_curr && t < offset_next) {
                    auto color_curr = Vector4f{
                        c->stop_colors[4 * i + 0],
                        c->stop_colors[4 * i + 1],
                        c->stop_colors[4 * i + 2],
                        c->stop_colors[4 * i + 3]};
                    auto color_next = Vector4f{
                        c->stop_colors[4 * (i + 1) + 0],
                        c->stop_colors[4 * (i + 1) + 1],
                        c->stop_colors[4 * (i + 1) + 2],
                        c->stop_colors[4 * (i + 1) + 3]};
                    auto tt = (t - offset_curr) / (offset_next - offset_curr);
                    assert(isfinite(tt));
                    assert(isfinite(color_curr));
                    assert(isfinite(color_next));
                    return color_curr * (1 - tt) + color_next * tt;
                }
            }
            return Vector4f{c->stop_colors[4 * (c->num_stops - 1) + 0],
                            c->stop_colors[4 * (c->num_stops - 1) + 1],
                            c->stop_colors[4 * (c->num_stops - 1) + 2],
                            c->stop_colors[4 * (c->num_stops - 1) + 3]};
        } case ColorType::RadialGradient: {
            auto c = (const RadialGradient*)color;
            // Distance from pt to center
            auto offset = pt - c->center;
            auto normalized_offset = offset / c->radius;
            auto t = length(normalized_offset);
            // Find the correponding stop:
            if (t < c->stop_offsets[0]) {
                return Vector4f{c->stop_colors[0],
                                c->stop_colors[1],
                                c->stop_colors[2],
                                c->stop_colors[3]};
            }
            for (int i = 0; i < c->num_stops - 1; i++) {
                auto offset_curr = c->stop_offsets[i];
                auto offset_next = c->stop_offsets[i + 1];
                assert(offset_next > offset_curr);
                if (t >= offset_curr && t < offset_next) {
                    auto color_curr = Vector4f{
                        c->stop_colors[4 * i + 0],
                        c->stop_colors[4 * i + 1],
                        c->stop_colors[4 * i + 2],
                        c->stop_colors[4 * i + 3]};
                    auto color_next = Vector4f{
                        c->stop_colors[4 * (i + 1) + 0],
                        c->stop_colors[4 * (i + 1) + 1],
                        c->stop_colors[4 * (i + 1) + 2],
                        c->stop_colors[4 * (i + 1) + 3]};
                    auto tt = (t - offset_curr) / (offset_next - offset_curr);
                    assert(isfinite(tt));
                    assert(isfinite(color_curr));
                    assert(isfinite(color_next));
                    return color_curr * (1 - tt) + color_next * tt;
                }
            }
            return Vector4f{c->stop_colors[4 * (c->num_stops - 1) + 0],
                            c->stop_colors[4 * (c->num_stops - 1) + 1],
                            c->stop_colors[4 * (c->num_stops - 1) + 2],
                            c->stop_colors[4 * (c->num_stops - 1) + 3]};
        } default: {
            assert(false);
        }
    }
    return Vector4f{};
}

DEVICE
void d_sample_color(const ColorType &color_type,
                    void *color_ptr,
                    const Vector2f &pt,
                    const Vector4f &d_color,
                    void *d_color_ptr,
                    float *d_translation) {
    switch (color_type) {
        case ColorType::Constant: {
            auto d_c = (Constant*)d_color_ptr;
            atomic_add(&d_c->color[0], d_color);
            return;
        } case ColorType::LinearGradient: {
            auto c = (const LinearGradient*)color_ptr;
            auto d_c = (LinearGradient*)d_color_ptr;
            // Project pt to (c->begin, c->end)
            auto beg = c->begin;
            auto end = c->end;
            auto t = dot(pt - beg, end - beg) / max(dot(end - beg, end - beg), 1e-3f);
            // Find the correponding stop:
            if (t < c->stop_offsets[0]) {
                atomic_add(&d_c->stop_colors[0], d_color);
                return;
            }
            for (int i = 0; i < c->num_stops - 1; i++) {
                auto offset_curr = c->stop_offsets[i];
                auto offset_next = c->stop_offsets[i + 1];
                assert(offset_next > offset_curr);
                if (t >= offset_curr && t < offset_next) {
                    auto color_curr = Vector4f{
                        c->stop_colors[4 * i + 0],
                        c->stop_colors[4 * i + 1],
                        c->stop_colors[4 * i + 2],
                        c->stop_colors[4 * i + 3]};
                    auto color_next = Vector4f{
                        c->stop_colors[4 * (i + 1) + 0],
                        c->stop_colors[4 * (i + 1) + 1],
                        c->stop_colors[4 * (i + 1) + 2],
                        c->stop_colors[4 * (i + 1) + 3]};
                    auto tt = (t - offset_curr) / (offset_next - offset_curr);
                    // return color_curr * (1 - tt) + color_next * tt;
                    auto d_color_curr = d_color * (1 - tt);
                    auto d_color_next = d_color * tt;
                    auto d_tt = sum(d_color * (color_next - color_curr));
                    auto d_offset_next = -d_tt * tt / (offset_next - offset_curr);
                    auto d_offset_curr = d_tt * ((tt - 1.f) / (offset_next - offset_curr));
                    auto d_t = d_tt / (offset_next - offset_curr);
                    assert(isfinite(d_tt));
                    atomic_add(&d_c->stop_colors[4 * i], d_color_curr);
                    atomic_add(&d_c->stop_colors[4 * (i + 1)], d_color_next);
                    atomic_add(&d_c->stop_offsets[i], d_offset_curr);
                    atomic_add(&d_c->stop_offsets[i + 1], d_offset_next);
                    // auto t = dot(pt - beg, end - beg) / max(dot(end - beg, end - beg), 1e-6f);
                    // l = max(dot(end - beg, end - beg), 1e-3f)
                    // t = dot(pt - beg, end - beg) / l;
                    auto l = max(dot(end - beg, end - beg), 1e-3f);
                    auto d_beg = d_t * (-(pt - beg)-(end - beg)) / l;
                    auto d_end = d_t * (pt - beg) / l;
                    auto d_l = -d_t * t / l;
                    if (dot(end - beg, end - beg) > 1e-3f) {
                        d_beg += 2 * d_l * (beg - end);
                        d_end += 2 * d_l * (end - beg);
                    }
                    atomic_add(&d_c->begin[0], d_beg);
                    atomic_add(&d_c->end[0], d_end);
                    if (d_translation != nullptr) {
                        atomic_add(d_translation, (d_beg + d_end));
                    }
                    return;
                }
            }
            atomic_add(&d_c->stop_colors[4 * (c->num_stops - 1)], d_color);
            return;
        } case ColorType::RadialGradient: {
            auto c = (const RadialGradient*)color_ptr;
            auto d_c = (RadialGradient*)d_color_ptr;
            // Distance from pt to center
            auto offset = pt - c->center;
            auto normalized_offset = offset / c->radius;
            auto t = length(normalized_offset);
            // Find the correponding stop:
            if (t < c->stop_offsets[0]) {
                atomic_add(&d_c->stop_colors[0], d_color);
                return;
            }
            for (int i = 0; i < c->num_stops - 1; i++) {
                auto offset_curr = c->stop_offsets[i];
                auto offset_next = c->stop_offsets[i + 1];
                assert(offset_next > offset_curr);
                if (t >= offset_curr && t < offset_next) {
                    auto color_curr = Vector4f{
                        c->stop_colors[4 * i + 0],
                        c->stop_colors[4 * i + 1],
                        c->stop_colors[4 * i + 2],
                        c->stop_colors[4 * i + 3]};
                    auto color_next = Vector4f{
                        c->stop_colors[4 * (i + 1) + 0],
                        c->stop_colors[4 * (i + 1) + 1],
                        c->stop_colors[4 * (i + 1) + 2],
                        c->stop_colors[4 * (i + 1) + 3]};
                    auto tt = (t - offset_curr) / (offset_next - offset_curr);
                    assert(isfinite(tt));
                    // return color_curr * (1 - tt) + color_next * tt;
                    auto d_color_curr = d_color * (1 - tt);
                    auto d_color_next = d_color * tt;
                    auto d_tt = sum(d_color * (color_next - color_curr));
                    auto d_offset_next = -d_tt * tt / (offset_next - offset_curr);
                    auto d_offset_curr = d_tt * ((tt - 1.f) / (offset_next - offset_curr));
                    auto d_t = d_tt / (offset_next - offset_curr);
                    assert(isfinite(d_t));
                    atomic_add(&d_c->stop_colors[4 * i], d_color_curr);
                    atomic_add(&d_c->stop_colors[4 * (i + 1)], d_color_next);
                    atomic_add(&d_c->stop_offsets[i], d_offset_curr);
                    atomic_add(&d_c->stop_offsets[i + 1], d_offset_next);
                    // offset = pt - c->center
                    // normalized_offset = offset / c->radius
                    // t = length(normalized_offset)
                    auto d_normalized_offset = d_length(normalized_offset, d_t);
                    auto d_offset = d_normalized_offset / c->radius;
                    auto d_radius = -d_normalized_offset * offset / (c->radius * c->radius);
                    auto d_center = -d_offset;
                    atomic_add(&d_c->center[0], d_center);
                    atomic_add(&d_c->radius[0], d_radius);
                    if (d_translation != nullptr) {
                        atomic_add(d_translation, d_center);
                    }
                    // Without this return the gradient also fell through to
                    // the "beyond the last stop" case below and was added a
                    // second time to the last stop colour.
                    return;
                }
            }
            atomic_add(&d_c->stop_colors[4 * (c->num_stops - 1)], d_color);
            return;
        } default: {
            assert(false);
        }
    }
}

struct Fragment {
    Vector3f color;
    float alpha;
    int group_id;
    bool is_stroke;
};

struct PrefilterFragment {
    Vector3f color;
    float alpha;
    int group_id;
    bool is_stroke;
    int shape_id;
    float distance;
    Vector2f closest_pt;
    ClosestPointPathInfo path_info;
    bool within_distance;
};

DEVICE
Vector4f sample_color(const SceneData &scene,
                      const Vector4f *background_color,
                      const Vector2f &screen_pt,
                      const Vector4f *d_color = nullptr,
                      EdgeQuery *edge_query = nullptr,
                      Vector4f *d_background_color = nullptr,
                      float *d_translation = nullptr) {
    if (edge_query != nullptr) {
        edge_query->hit = false;
    }

    // screen_pt is in screen space ([0, 1), [0, 1)),
    // need to transform to canvas space
    auto pt = screen_pt;
    pt.x *= scene.canvas_width;
    pt.y *= scene.canvas_height;
    constexpr auto max_hit_shapes = 256;
    constexpr auto max_bvh_stack_size = 64;
    Fragment fragments[max_hit_shapes];
    int bvh_stack[max_bvh_stack_size];
    auto stack_size = 0;
    auto num_fragments = 0;
    if (scene.num_shape_groups > 0) {
        bvh_stack[stack_size++] = 2 * scene.num_shape_groups - 2;
    }
    while (stack_size > 0) {
        const BVHNode &node = scene.bvh_nodes[bvh_stack[--stack_size]];
        if (node.child1 < 0) {
            // leaf
            auto group_id = node.child0;
            const ShapeGroup &shape_group = scene.shape_groups[group_id];
            if (shape_group.stroke_color != nullptr) {
                if (within_distance(scene, group_id, pt, edge_query)) {
                    auto color_alpha = sample_color(shape_group.stroke_color_type,
                                                    shape_group.stroke_color,
                                                    pt);
                    Fragment f;
                    f.color = Vector3f{color_alpha[0], color_alpha[1], color_alpha[2]};
                    f.alpha = color_alpha[3];
                    f.group_id = group_id;
                    f.is_stroke = true;
                    assert(num_fragments < max_hit_shapes);
                    fragments[num_fragments++] = f;
                }
            }
            if (shape_group.fill_color != nullptr) {
                if (is_inside(scene, group_id, pt, edge_query)) {
                    auto color_alpha = sample_color(shape_group.fill_color_type,
                                                    shape_group.fill_color,
                                                    pt);
                    Fragment f;
                    f.color = Vector3f{color_alpha[0], color_alpha[1], color_alpha[2]};
                    f.alpha = color_alpha[3];
                    f.group_id = group_id;
                    f.is_stroke = false;
                    assert(num_fragments < max_hit_shapes);
                    fragments[num_fragments++] = f;
                }
            }
        } else {
            assert(node.child0 >= 0 && node.child1 >= 0);
            const AABB &b0 = scene.bvh_nodes[node.child0].box;
            if (inside(b0, pt, scene.bvh_nodes[node.child0].max_radius)) {
                bvh_stack[stack_size++] = node.child0;
            }
            const AABB &b1 = scene.bvh_nodes[node.child1].box;
            if (inside(b1, pt, scene.bvh_nodes[node.child1].max_radius)) {
                bvh_stack[stack_size++] = node.child1;
            }
            assert(stack_size <= max_bvh_stack_size);
        }
    }
    if (num_fragments <= 0) {
        if (background_color != nullptr) {
            if (d_background_color != nullptr) {
                // Several samples (and threads) share one background pixel:
                // accumulate atomically instead of overwriting.
                atomic_add(*d_background_color, *d_color);
            }
            return *background_color;
        }
        return Vector4f{0, 0, 0, 0};
    }
    // Sort the fragments from back to front (i.e. increasing order of group id)
    // https://github.com/frigaut/yorick-imutil/blob/master/insort.c#L37
    for (int i = 1; i < num_fragments; i++) {
        auto j = i;
        auto temp = fragments[j];
        while (j > 0 && fragments[j - 1].group_id > temp.group_id) {
            fragments[j] = fragments[j - 1];
            j--;
        }
        fragments[j] = temp;
    }
    // Blend the color
    Vector3f accum_color[max_hit_shapes];
    float accum_alpha[max_hit_shapes];
    // auto hit_opaque = false;
    auto first_alpha = 0.f;
    auto first_color = Vector3f{0, 0, 0};
    if (background_color != nullptr) {
        first_alpha = background_color->w;
        first_color = Vector3f{background_color->x,
                               background_color->y,
                               background_color->z};
    }
    for (int i = 0; i < num_fragments; i++) {
        const Fragment &fragment = fragments[i];
        auto new_color = fragment.color;
        auto new_alpha = fragment.alpha;
        auto prev_alpha = i > 0 ? accum_alpha[i - 1] : first_alpha;
        auto prev_color = i > 0 ? accum_color[i - 1] : first_color;
        if (edge_query != nullptr) {
            // Do we hit the target shape?
            if (new_alpha >= 1.f && edge_query->hit) {
                // A fully opaque shape in front of the target occludes it
                edge_query->hit = false;
            }
            if (edge_query->shape_group_id == fragment.group_id) {
                edge_query->hit = true;
            }
        }
        // prev_color is alpha premultiplied, don't need to multiply with
        // prev_alpha
        accum_color[i] = prev_color * (1 - new_alpha) + new_alpha * new_color;
        accum_alpha[i] = prev_alpha * (1 - new_alpha) + new_alpha;
    }
    auto final_color = accum_color[num_fragments - 1];
    auto final_alpha = accum_alpha[num_fragments - 1];
    if (final_alpha > 1e-6f) {
        final_color /= final_alpha;
    }
    assert(isfinite(final_color));
    assert(isfinite(final_alpha));
    if (d_color != nullptr) {
        // Backward pass
        auto d_final_color = Vector3f{(*d_color)[0], (*d_color)[1], (*d_color)[2]};
        auto d_final_alpha = (*d_color)[3];
        auto d_curr_color = d_final_color;
        auto d_curr_alpha = d_final_alpha;
        if (final_alpha > 1e-6f) {
            // final_color = curr_color / final_alpha
            d_curr_color = d_final_color / final_alpha;
            d_curr_alpha -= sum(d_final_color * final_color) / final_alpha;
        }
        assert(isfinite(*d_color));
        assert(isfinite(d_curr_color));
        assert(isfinite(d_curr_alpha));
        for (int i = num_fragments - 1; i >= 0; i--) {
            // color[n] = prev_color * (1 - new_alpha) + new_alpha * new_color;
            // alpha[n] = prev_alpha * (1 - new_alpha) + new_alpha;
            auto prev_alpha = i > 0 ? accum_alpha[i - 1] : first_alpha;
            auto prev_color = i > 0 ? accum_color[i - 1] : first_color;
            auto d_prev_alpha = d_curr_alpha * (1.f - fragments[i].alpha);
            auto d_alpha_i = d_curr_alpha * (1.f - prev_alpha);
            d_alpha_i += sum(d_curr_color * (fragments[i].color - prev_color));
            auto d_prev_color = d_curr_color * (1 - fragments[i].alpha);
            auto d_color_i = d_curr_color * fragments[i].alpha;
            auto group_id = fragments[i].group_id;
            if (fragments[i].is_stroke) {
                d_sample_color(scene.shape_groups[group_id].stroke_color_type,
                               scene.shape_groups[group_id].stroke_color,
                               pt,
                               Vector4f{d_color_i[0], d_color_i[1], d_color_i[2], d_alpha_i},
                               scene.d_shape_groups[group_id].stroke_color,
                               d_translation);
            } else {
                d_sample_color(scene.shape_groups[group_id].fill_color_type,
                               scene.shape_groups[group_id].fill_color,
                               pt,
                               Vector4f{d_color_i[0], d_color_i[1], d_color_i[2], d_alpha_i},
                               scene.d_shape_groups[group_id].fill_color,
                               d_translation);
            }
            d_curr_color = d_prev_color;
            d_curr_alpha = d_prev_alpha;
        }
        if (d_background_color != nullptr) {
            atomic_add(d_background_color->x, d_curr_color.x);
            atomic_add(d_background_color->y, d_curr_color.y);
            atomic_add(d_background_color->z, d_curr_color.z);
            atomic_add(d_background_color->w, d_curr_alpha);
        }
    }
    return Vector4f{final_color[0], final_color[1], final_color[2], final_alpha};
}

DEVICE
float sample_distance(const SceneData &scene,
                      const Vector2f &screen_pt,
                      float weight,
                      const float *d_dist = nullptr,
                      float *d_translation = nullptr) {
    // screen_pt is in screen space ([0, 1), [0, 1)),
    // need to transform to canvas space
    auto pt = screen_pt;
    pt.x *= scene.canvas_width;
    pt.y *= scene.canvas_height;
    // for each shape
    auto min_group_id = -1;
    auto min_distance = 0.f;
    auto min_shape_id = -1;
    auto closest_pt = Vector2f{0, 0};
    auto min_path_info = ClosestPointPathInfo{-1, -1, 0};
    for (int group_id = scene.num_shape_groups - 1; group_id >= 0; group_id--) {
        auto s = -1;
        auto p = Vector2f{0, 0};
        ClosestPointPathInfo local_path_info;
        auto d = infinity<float>();
        if (compute_distance(scene, group_id, pt, infinity<float>(), &s, &p, &local_path_info, &d)) {
            if (min_group_id == -1 || d < min_distance) {
                min_distance = d;
                min_group_id = group_id;
                min_shape_id = s;
                closest_pt = p;
                min_path_info = local_path_info;
            }
        }
    }
    if (min_group_id == -1) {
        return min_distance;
    }
    min_distance *= weight;
    auto inside = false;
    const ShapeGroup &shape_group = scene.shape_groups[min_group_id];
    if (shape_group.fill_color != nullptr) {
        inside = is_inside(scene,
                           min_group_id,
                           pt,
                           nullptr);
        if (inside) {
            min_distance = -min_distance;
        }
    }
    assert((min_group_id >= 0 && min_shape_id >= 0) || scene.num_shape_groups == 0);
    if (d_dist != nullptr) {
        // forward: returned distance = weight * (+-distance)
        auto d_abs_dist = weight * (inside ? -(*d_dist) : (*d_dist));
        const ShapeGroup &shape_group = scene.shape_groups[min_group_id];
        const Shape &shape = scene.shapes[min_shape_id];
        ShapeGroup &d_shape_group = scene.d_shape_groups[min_group_id];
        Shape &d_shape = scene.d_shapes[min_shape_id];
        d_compute_distance(shape_group.canvas_to_shape,
                           shape_group.shape_to_canvas,
                           shape,
                           pt,
                           closest_pt,
                           min_path_info,
                           d_abs_dist,
                           d_shape_group.shape_to_canvas,
                           d_shape,
                           d_translation);
    }
    return min_distance;
}

// Gather d_color from d_image inside the filter kernel, normalize by
// weight_image.
DEVICE
Vector4f gather_d_color(const Filter &filter,
                        const float *d_color_image,
                        const float *weight_image,
                        int width,
                        int height,
                        const Vector2f &pt) {
    auto x = int(pt.x);
    auto y = int(pt.y);
    auto radius = filter.radius;
    assert(radius > 0);
    auto ri = (int)ceil(radius);
    auto d_color = Vector4f{0, 0, 0, 0};
    for (int dy = -ri; dy <= ri; dy++) {
        for (int dx = -ri; dx <= ri; dx++) {
            auto xx = x + dx;
            auto yy = y + dy;
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                auto xc = xx + 0.5f;
                auto yc = yy + 0.5f;
                auto filter_weight =
                    compute_filter_weight(filter, xc - pt.x, yc - pt.y);
                // A point exactly on the border of a pixel footprint belongs to
                // both neighbouring pixels (the filter support is closed). For
                // point samples this is a measure-zero event, but boundary
                // samples of pixel-aligned edges (e.g. integer rectangles) hit
                // it systematically and were counted twice. Split the weight.
                if (fabs(xc - pt.x) == radius) {
                    filter_weight *= 0.5f;
                }
                if (fabs(yc - pt.y) == radius) {
                    filter_weight *= 0.5f;
                }
                // pixel = \sum weight * color / \sum weight
                auto weight_sum = weight_image[yy * width + xx];
                if (weight_sum > 0) {
                    d_color += (filter_weight / weight_sum) * Vector4f{
                        d_color_image[4 * (yy * width + xx) + 0],
                        d_color_image[4 * (yy * width + xx) + 1],
                        d_color_image[4 * (yy * width + xx) + 2],
                        d_color_image[4 * (yy * width + xx) + 3],
                    };
                }
            }
        }
    }
    return d_color;
}

DEVICE
float smoothstep(float d) {
    auto t = clamp((d + 1.f) / 2.f, 0.f, 1.f);
    return t * t * (3 - 2 * t);
}

DEVICE
float d_smoothstep(float d, float d_ret) {
    if (d < -1.f || d > 1.f) {
        return 0.f;
    }
    auto t = (d + 1.f) / 2.f;
    // ret = t * t * (3 - 2 * t)
    //     = 3 * t * t - 2 * t * t * t
    auto d_t = d_ret * (6 * t - 6 * t * t);
    return d_t / 2.f;
}

// Stroke radius used by prefiltered rendering at the closest point of a shape.
// For a path with per-point thickness the serialized stroke_width is a dummy 0,
// so interpolate the thickness at the closest point (segment base_point_id,
// first point point_id, parameter t_root) with the Bezier basis weights, exactly
// as accumulate_boundary_gradient does.
DEVICE
inline
float prefilter_stroke_radius(const Shape &shape, const ClosestPointPathInfo &info) {
    if (shape.type != ShapeType::Path) {
        return shape.stroke_width;
    }
    const Path &path = *(const Path *)shape.ptr;
    if (path.thickness == nullptr || info.base_point_id < 0) {
        return shape.stroke_width;
    }
    auto point_id = info.point_id;
    auto t = info.t_root;
    switch (path.num_control_points[info.base_point_id]) {
        case 0: {
            auto i0 = point_id;
            auto i1 = (point_id + 1) % path.num_points;
            return (1 - t) * path.thickness[i0] + t * path.thickness[i1];
        }
        case 1: {
            auto i0 = point_id;
            auto i1 = point_id + 1;
            auto i2 = (point_id + 2) % path.num_points;
            return square(1 - t) * path.thickness[i0] +
                   (2 * (1 - t) * t) * path.thickness[i1] +
                   (t * t) * path.thickness[i2];
        }
        case 2: {
            auto i0 = point_id;
            auto i1 = point_id + 1;
            auto i2 = point_id + 2;
            auto i3 = (point_id + 3) % path.num_points;
            return cubic(1 - t) * path.thickness[i0] +
                   3 * square(1 - t) * t * path.thickness[i1] +
                   3 * (1 - t) * t * t * path.thickness[i2] +
                   t * t * t * path.thickness[i3];
        }
        default:
            assert(false);
    }
    return shape.stroke_width;
}

// Backward of prefilter_stroke_radius with the closest-point parameter t held
// fixed. The dependence of the interpolated radius on t is handled separately:
// the caller passes d_radius * prefilter_stroke_radius_dt as d_t_root to
// d_compute_distance, which applies dt/dparams (implicit function theorem).
DEVICE
inline
void d_prefilter_stroke_radius(const Shape &shape, const ClosestPointPathInfo &info,
                               float d_radius, Shape &d_shape) {
    if (shape.type == ShapeType::Path) {
        const Path &path = *(const Path *)shape.ptr;
        if (path.thickness != nullptr && info.base_point_id >= 0) {
            Path *d_p = (Path*)d_shape.ptr;
            auto point_id = info.point_id;
            auto t = info.t_root;
            switch (path.num_control_points[info.base_point_id]) {
                case 0: {
                    auto i0 = point_id;
                    auto i1 = (point_id + 1) % path.num_points;
                    atomic_add(&d_p->thickness[i0], (1 - t) * d_radius);
                    atomic_add(&d_p->thickness[i1], (    t) * d_radius);
                    break;
                }
                case 1: {
                    auto i0 = point_id;
                    auto i1 = point_id + 1;
                    auto i2 = (point_id + 2) % path.num_points;
                    atomic_add(&d_p->thickness[i0], square(1 - t) * d_radius);
                    atomic_add(&d_p->thickness[i1], (2 * (1 - t) * t) * d_radius);
                    atomic_add(&d_p->thickness[i2], (t * t) * d_radius);
                    break;
                }
                case 2: {
                    auto i0 = point_id;
                    auto i1 = point_id + 1;
                    auto i2 = point_id + 2;
                    auto i3 = (point_id + 3) % path.num_points;
                    atomic_add(&d_p->thickness[i0], cubic(1 - t) * d_radius);
                    atomic_add(&d_p->thickness[i1], 3 * square(1 - t) * t * d_radius);
                    atomic_add(&d_p->thickness[i2], 3 * (1 - t) * t * t * d_radius);
                    atomic_add(&d_p->thickness[i3], t * t * t * d_radius);
                    break;
                }
                default:
                    assert(false);
            }
            return;
        }
    }
    atomic_add(&d_shape.stroke_width, d_radius);
}

// d prefilter_stroke_radius / d t at the closest point (0 unless the shape is a
// path with per-point thickness).
DEVICE
inline
float prefilter_stroke_radius_dt(const Shape &shape, const ClosestPointPathInfo &info) {
    if (shape.type != ShapeType::Path) {
        return 0;
    }
    const Path &path = *(const Path *)shape.ptr;
    if (path.thickness == nullptr || info.base_point_id < 0) {
        return 0;
    }
    auto point_id = info.point_id;
    auto t = info.t_root;
    auto tt = 1 - t;
    const float *r = path.thickness;
    switch (path.num_control_points[info.base_point_id]) {
        case 0: {
            auto i0 = point_id;
            auto i1 = (point_id + 1) % path.num_points;
            return r[i1] - r[i0];
        }
        case 1: {
            auto i0 = point_id;
            auto i1 = point_id + 1;
            auto i2 = (point_id + 2) % path.num_points;
            return 2 * tt * (r[i1] - r[i0]) + 2 * t * (r[i2] - r[i1]);
        }
        case 2: {
            auto i0 = point_id;
            auto i1 = point_id + 1;
            auto i2 = point_id + 2;
            auto i3 = (point_id + 3) % path.num_points;
            return 3 * tt * tt * (r[i1] - r[i0]) + 6 * tt * t * (r[i2] - r[i1]) +
                   3 * t * t * (r[i3] - r[i2]);
        }
        default:
            assert(false);
    }
    return 0;
}

DEVICE
Vector4f sample_color_prefiltered(const SceneData &scene,
                                  const Vector4f *background_color,
                                  const Vector2f &screen_pt,
                                  const Vector4f *d_color = nullptr,
                                  Vector4f *d_background_color = nullptr,
                                  float *d_translation = nullptr) {
    // screen_pt is in screen space ([0, 1), [0, 1)),
    // need to transform to canvas space
    auto pt = screen_pt;
    pt.x *= scene.canvas_width;
    pt.y *= scene.canvas_height;
    constexpr auto max_hit_shapes = 64;
    constexpr auto max_bvh_stack_size = 64;
    PrefilterFragment fragments[max_hit_shapes];
    int bvh_stack[max_bvh_stack_size];
    auto stack_size = 0;
    auto num_fragments = 0;
    if (scene.num_shape_groups > 0) {
        bvh_stack[stack_size++] = 2 * scene.num_shape_groups - 2;
    }
    while (stack_size > 0) {
        const BVHNode &node = scene.bvh_nodes[bvh_stack[--stack_size]];
        if (node.child1 < 0) {
            // leaf
            auto group_id = node.child0;
            const ShapeGroup &shape_group = scene.shape_groups[group_id];
            if (shape_group.stroke_color != nullptr) {
                auto min_shape_id = -1;
                auto closest_pt = Vector2f{0, 0};
                auto local_path_info = ClosestPointPathInfo{-1, -1, 0};
                auto d = infinity<float>();
                compute_distance(scene, group_id, pt, infinity<float>(),
                                 &min_shape_id, &closest_pt, &local_path_info, &d);
                // No closest point (e.g. only degenerate geometry): no stroke
                // fragment, instead of asserting.
                auto w = 0.f;
                if (min_shape_id != -1) {
                    auto stroke_radius = prefilter_stroke_radius(scene.shapes[min_shape_id],
                                                                 local_path_info);
                    w = smoothstep(fabs(d) + stroke_radius) -
                        smoothstep(fabs(d) - stroke_radius);
                }
                if (w > 0) {
                    auto color_alpha = sample_color(shape_group.stroke_color_type,
                                                    shape_group.stroke_color,
                                                    pt);
                    color_alpha[3] *= w;

                    PrefilterFragment f;
                    f.color = Vector3f{color_alpha[0], color_alpha[1], color_alpha[2]};
                    f.alpha = color_alpha[3];
                    f.group_id = group_id;
                    f.shape_id = min_shape_id;
                    f.distance = d;
                    f.closest_pt = closest_pt;
                    f.is_stroke = true;
                    f.path_info = local_path_info;
                    f.within_distance = true;
                    assert(num_fragments < max_hit_shapes);
                    fragments[num_fragments++] = f;
                }
            }
            if (shape_group.fill_color != nullptr) {
                auto min_shape_id = -1;
                auto closest_pt = Vector2f{0, 0};
                auto local_path_info = ClosestPointPathInfo{-1, -1, 0};
                auto d = infinity<float>();
                auto found = compute_distance(scene,
                                              group_id,
                                              pt,
                                              1.f,
                                              &min_shape_id,
                                              &closest_pt,
                                              &local_path_info,
                                              &d);
                auto inside = is_inside(scene, group_id, pt, nullptr);
                if (found || inside) {
                    if (!inside) {
                        d = -d;
                    }
                    auto w = smoothstep(d);
                    if (w > 0) {
                        auto color_alpha = sample_color(shape_group.fill_color_type,
                                                        shape_group.fill_color,
                                                        pt);
                        color_alpha[3] *= w;

                        PrefilterFragment f;
                        f.color = Vector3f{color_alpha[0], color_alpha[1], color_alpha[2]};
                        f.alpha = color_alpha[3];
                        f.group_id = group_id;
                        f.shape_id = min_shape_id;
                        f.distance = d;
                        f.closest_pt = closest_pt;
                        f.is_stroke = false;
                        f.path_info = local_path_info;
                        f.within_distance = found;
                        assert(num_fragments < max_hit_shapes);
                        fragments[num_fragments++] = f;
                    }
                }
            }
        } else {
            assert(node.child0 >= 0 && node.child1 >= 0);
            const AABB &b0 = scene.bvh_nodes[node.child0].box;
            if (inside(b0, pt, scene.bvh_nodes[node.child0].max_radius)) {
                bvh_stack[stack_size++] = node.child0;
            }
            const AABB &b1 = scene.bvh_nodes[node.child1].box;
            if (inside(b1, pt, scene.bvh_nodes[node.child1].max_radius)) {
                bvh_stack[stack_size++] = node.child1;
            }
            assert(stack_size <= max_bvh_stack_size);
        }
    }
    if (num_fragments <= 0) {
        if (background_color != nullptr) {
            if (d_background_color != nullptr) {
                // Several samples (and threads) share one background pixel:
                // accumulate atomically instead of overwriting.
                atomic_add(*d_background_color, *d_color);
            }
            return *background_color;
        }
        return Vector4f{0, 0, 0, 0};
    }
    // Sort the fragments from back to front (i.e. increasing order of group id)
    // https://github.com/frigaut/yorick-imutil/blob/master/insort.c#L37
    for (int i = 1; i < num_fragments; i++) {
        auto j = i;
        auto temp = fragments[j];
        while (j > 0 && fragments[j - 1].group_id > temp.group_id) {
            fragments[j] = fragments[j - 1];
            j--;
        }
        fragments[j] = temp;
    }
    // Blend the color
    Vector3f accum_color[max_hit_shapes];
    float accum_alpha[max_hit_shapes];
    auto first_alpha = 0.f;
    auto first_color = Vector3f{0, 0, 0};
    if (background_color != nullptr) {
        first_alpha = background_color->w;
        first_color = Vector3f{background_color->x,
                               background_color->y,
                               background_color->z};
    }
    for (int i = 0; i < num_fragments; i++) {
        const PrefilterFragment &fragment = fragments[i];
        auto new_color = fragment.color;
        auto new_alpha = fragment.alpha;
        auto prev_alpha = i > 0 ? accum_alpha[i - 1] : first_alpha;
        auto prev_color = i > 0 ? accum_color[i - 1] : first_color;
        // prev_color is alpha premultiplied, don't need to multiply with
        // prev_alpha
        accum_color[i] = prev_color * (1 - new_alpha) + new_alpha * new_color;
        accum_alpha[i] = prev_alpha * (1 - new_alpha) + new_alpha;
    }
    auto final_color = accum_color[num_fragments - 1];
    auto final_alpha = accum_alpha[num_fragments - 1];
    if (final_alpha > 1e-6f) {
        final_color /= final_alpha;
    }
    assert(isfinite(final_color));
    assert(isfinite(final_alpha));
    if (d_color != nullptr) {
        // Backward pass
        auto d_final_color = Vector3f{(*d_color)[0], (*d_color)[1], (*d_color)[2]};
        auto d_final_alpha = (*d_color)[3];
        auto d_curr_color = d_final_color;
        auto d_curr_alpha = d_final_alpha;
        if (final_alpha > 1e-6f) {
            // final_color = curr_color / final_alpha
            d_curr_color = d_final_color / final_alpha;
            d_curr_alpha -= sum(d_final_color * final_color) / final_alpha;
        }
        assert(isfinite(*d_color));
        assert(isfinite(d_curr_color));
        assert(isfinite(d_curr_alpha));
        for (int i = num_fragments - 1; i >= 0; i--) {
            // color[n] = prev_color * (1 - new_alpha) + new_alpha * new_color;
            // alpha[n] = prev_alpha * (1 - new_alpha) + new_alpha;
            auto prev_alpha = i > 0 ? accum_alpha[i - 1] : first_alpha;
            auto prev_color = i > 0 ? accum_color[i - 1] : first_color;
            auto d_prev_alpha = d_curr_alpha * (1.f - fragments[i].alpha);
            auto d_alpha_i = d_curr_alpha * (1.f - prev_alpha);
            d_alpha_i += sum(d_curr_color * (fragments[i].color - prev_color));
            auto d_prev_color = d_curr_color * (1 - fragments[i].alpha);
            auto d_color_i = d_curr_color * fragments[i].alpha;
            auto group_id = fragments[i].group_id;
            if (fragments[i].is_stroke) {
                const auto &shape = scene.shapes[fragments[i].shape_id];
                auto d = fragments[i].distance;
                auto stroke_radius = prefilter_stroke_radius(shape, fragments[i].path_info);
                auto abs_d_plus_width = fabs(d) + stroke_radius;
                auto abs_d_minus_width = fabs(d) - stroke_radius;
                auto w = smoothstep(abs_d_plus_width) -
                         smoothstep(abs_d_minus_width);
                if (w != 0) {
                    auto d_w = w > 0 ? (fragments[i].alpha / w) * d_alpha_i : 0.f;
                    d_alpha_i *= w;

                    // Backprop to color
                    d_sample_color(scene.shape_groups[group_id].stroke_color_type,
                                   scene.shape_groups[group_id].stroke_color,
                                   pt,
                                   Vector4f{d_color_i[0], d_color_i[1], d_color_i[2], d_alpha_i},
                                   scene.d_shape_groups[group_id].stroke_color,
                                   d_translation);

                    auto d_abs_d_plus_width = d_smoothstep(abs_d_plus_width, d_w);
                    auto d_abs_d_minus_width = -d_smoothstep(abs_d_minus_width, d_w);

                    auto d_d = d_abs_d_plus_width + d_abs_d_minus_width;
                    if (d < 0) {
                        d_d = -d_d;
                    }
                    auto d_stroke_width = d_abs_d_plus_width - d_abs_d_minus_width;

                    const auto &shape_group = scene.shape_groups[group_id];
                    ShapeGroup &d_shape_group = scene.d_shape_groups[group_id];
                    Shape &d_shape = scene.d_shapes[fragments[i].shape_id];
                    // the interpolated per-point thickness also depends on the
                    // closest-point parameter t (0 for uniform-width strokes)
                    auto d_t_root = d_stroke_width *
                        prefilter_stroke_radius_dt(shape, fragments[i].path_info);
                    if (fabs(d_d) > 1e-10f || d_t_root != 0) {
                        d_compute_distance(shape_group.canvas_to_shape,
                                           shape_group.shape_to_canvas,
                                           shape,
                                           pt,
                                           fragments[i].closest_pt,
                                           fragments[i].path_info,
                                           d_d,
                                           d_shape_group.shape_to_canvas,
                                           d_shape,
                                           d_translation,
                                           d_t_root);
                    }
                    d_prefilter_stroke_radius(shape, fragments[i].path_info,
                                              d_stroke_width, d_shape);
                }
            } else {
                const auto &shape = scene.shapes[fragments[i].shape_id];
                auto d = fragments[i].distance;
                auto w = smoothstep(d);
                if (w != 0) {
                    // color_alpha[3] = color_alpha[3] * w;
                    auto d_w = w > 0 ? (fragments[i].alpha / w) * d_alpha_i : 0.f;
                    d_alpha_i *= w;

                    d_sample_color(scene.shape_groups[group_id].fill_color_type,
                                   scene.shape_groups[group_id].fill_color,
                                   pt,
                                   Vector4f{d_color_i[0], d_color_i[1], d_color_i[2], d_alpha_i},
                                   scene.d_shape_groups[group_id].fill_color,
                                   d_translation);

                    // w = smoothstep(d)
                    auto d_d = d_smoothstep(d, d_w);
                    if (d < 0) {
                        d_d = -d_d;
                    }

                    const auto &shape_group = scene.shape_groups[group_id];
                    ShapeGroup &d_shape_group = scene.d_shape_groups[group_id];
                    Shape &d_shape = scene.d_shapes[fragments[i].shape_id];
                    if (fabs(d_d) > 1e-10f && fragments[i].within_distance) {
                        d_compute_distance(shape_group.canvas_to_shape,
                                           shape_group.shape_to_canvas,
                                           shape,
                                           pt,
                                           fragments[i].closest_pt,
                                           fragments[i].path_info,
                                           d_d,
                                           d_shape_group.shape_to_canvas,
                                           d_shape,
                                           d_translation);
                    }
                }
            }
            d_curr_color = d_prev_color;
            d_curr_alpha = d_prev_alpha;
        }
        if (d_background_color != nullptr) {
            atomic_add(d_background_color->x, d_curr_color.x);
            atomic_add(d_background_color->y, d_curr_color.y);
            atomic_add(d_background_color->z, d_curr_color.z);
            atomic_add(d_background_color->w, d_curr_alpha);
        }
    }
    return Vector4f{final_color[0], final_color[1], final_color[2], final_alpha};
}

struct weight_kernel {
    DEVICE void operator()(int idx) {
        auto rng_state = init_pcg32(idx, seed);
        // height * width * num_samples_y * num_samples_x
        auto sx = idx % num_samples_x;
        auto sy = (idx / num_samples_x) % num_samples_y;
        auto x = (idx / (num_samples_x * num_samples_y)) % width;
        auto y = (idx / (num_samples_x * num_samples_y * width));
        assert(y < height);
        auto rx = next_pcg32_float(&rng_state);
        auto ry = next_pcg32_float(&rng_state);
        if (use_prefiltering) {
            rx = ry = 0.5f;
        }
        auto pt = Vector2f{x + ((float)sx + rx) / num_samples_x,
                           y + ((float)sy + ry) / num_samples_y};
        auto radius = scene.filter->radius;
        assert(radius >= 0);
        auto ri = (int)ceil(radius);
        for (int dy = -ri; dy <= ri; dy++) {
            for (int dx = -ri; dx <= ri; dx++) {
                auto xx = x + dx;
                auto yy = y + dy;
                if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                    auto xc = xx + 0.5f;
                    auto yc = yy + 0.5f;
                    auto filter_weight = compute_filter_weight(*scene.filter,
                                                               xc - pt.x,
                                                               yc - pt.y);
                    atomic_add(weight_image[yy * width + xx], filter_weight);
                    if (d_weight_image != nullptr) {
                        atomic_add(d_weight_image[yy * width + xx],
                            filter_weight_d_radius(*scene.filter, xc - pt.x, yc - pt.y));
                    }
                }
            }
        }
    }

    SceneData scene;
    float *weight_image;
    int width;
    int height;
    int num_samples_x;
    int num_samples_y;
    uint64_t seed;
    bool use_prefiltering;
    // Optional: d(weight_image)/d(filter radius), for the radius gradient
    float *d_weight_image;
};

// We use a "mega kernel" for rendering
struct render_kernel {
    DEVICE void operator()(int idx) {
        // height * width * num_samples_y * num_samples_x
        auto pt = Vector2f{0, 0};
        auto x = 0;
        auto y = 0;
        if (eval_positions == nullptr) {
            auto rng_state = init_pcg32(idx, seed);
            auto sx = idx % num_samples_x;
            auto sy = (idx / num_samples_x) % num_samples_y;
            x = (idx / (num_samples_x * num_samples_y)) % width;
            y = (idx / (num_samples_x * num_samples_y * width));
            assert(x < width && y < height);
            auto rx = next_pcg32_float(&rng_state);
            auto ry = next_pcg32_float(&rng_state);
            if (use_prefiltering) {
                rx = ry = 0.5f;
            }
            pt = Vector2f{x + ((float)sx + rx) / num_samples_x,
                          y + ((float)sy + ry) / num_samples_y};
        } else {
            pt = Vector2f{eval_positions[2 * idx],
                          eval_positions[2 * idx + 1]};
            x = int(pt.x);
            y = int(pt.y);
        }

        // normalize pt to [0, 1]
        auto npt = pt;
        npt.x /= width;
        npt.y /= height;
        auto num_samples = num_samples_x * num_samples_y;
        if (render_image != nullptr || d_render_image != nullptr) {
            Vector4f d_color = Vector4f{0, 0, 0, 0};
            if (d_render_image != nullptr) {
                // Gather d_color from d_render_image inside the filter kernel
                // normalize using weight_image
                d_color = gather_d_color(*scene.filter,
                                         d_render_image,
                                         weight_image,
                                         width,
                                         height,
                                         pt);
            }
            auto color = Vector4f{0, 0, 0, 0};
            if (use_prefiltering) {
                color = sample_color_prefiltered(scene,
                    background_image != nullptr ? (const Vector4f*)&background_image[4 * ((y * width) + x)] : nullptr,
                    npt,
                    d_render_image != nullptr ? &d_color : nullptr,
                    d_background_image != nullptr ? (Vector4f*)&d_background_image[4 * ((y * width) + x)] : nullptr,
                    d_translation != nullptr ? &d_translation[2 * (y * width + x)] : nullptr);
            } else {
                color = sample_color(scene,
                    background_image != nullptr ? (const Vector4f*)&background_image[4 * ((y * width) + x)] : nullptr,
                    npt,
                    d_render_image != nullptr ? &d_color : nullptr,
                    nullptr,
                    d_background_image != nullptr ? (Vector4f*)&d_background_image[4 * ((y * width) + x)] : nullptr,
                    d_translation != nullptr ? &d_translation[2 * (y * width + x)] : nullptr);
            }
            assert(isfinite(color));
            // Splat color onto render_image
            auto radius = scene.filter->radius;
            assert(radius >= 0);
            auto ri = (int)ceil(radius);
            for (int dy = -ri; dy <= ri; dy++) {
                for (int dx = -ri; dx <= ri; dx++) {
                    auto xx = x + dx;
                    auto yy = y + dy;
                    if (xx >= 0 && xx < width && yy >= 0 && yy < height &&
                            weight_image[yy * width + xx] > 0) {
                        auto weight_sum = weight_image[yy * width + xx];
                        auto xc = xx + 0.5f;
                        auto yc = yy + 0.5f;
                        auto filter_weight = compute_filter_weight(*scene.filter,
                                                                   xc - pt.x,
                                                                   yc - pt.y);
                        auto weighted_color = filter_weight * color / weight_sum;
                        if (render_image != nullptr) {
                            atomic_add(render_image[4 * (yy * width + xx) + 0],
                                       weighted_color[0]);
                            atomic_add(render_image[4 * (yy * width + xx) + 1],
                                       weighted_color[1]);
                            atomic_add(render_image[4 * (yy * width + xx) + 2],
                                       weighted_color[2]);
                            atomic_add(render_image[4 * (yy * width + xx) + 3],
                                       weighted_color[3]);
                        }
                        if (d_render_image != nullptr) {
                            // Backprop to filter_weight
                            // pixel = \sum weight * color / \sum weight
                            auto d_pixel = Vector4f{
                                d_render_image[4 * (yy * width + xx) + 0],
                                d_render_image[4 * (yy * width + xx) + 1],
                                d_render_image[4 * (yy * width + xx) + 2],
                                d_render_image[4 * (yy * width + xx) + 3],
                            };
                            // pixel = \sum_i w_i c_i / W, W = \sum_i w_i, so
                            // d pixel / d r = \sum_i (c_i - pixel) w_i' / W.
                            // The -pixel term, \sum_i pixel w_i' / W = pixel W' / W
                            // with pixel = \sum_i w_i c_i / W, is distributed over
                            // the samples: each contributes c_i w_i W' / W^2, where
                            // W' = d_weight_image (d W / d r, from weight_kernel).
                            auto dpc = dot(d_pixel, color);
                            auto d_radius = dpc / weight_sum *
                                (filter_weight_d_radius(*scene.filter, xc - pt.x, yc - pt.y) -
                                 filter_weight * d_weight_image[yy * width + xx] / weight_sum);
                            atomic_add(scene.d_filter->radius, d_radius);
                        }
                    }
                }
            }
        }
        if (sdf_image != nullptr || d_sdf_image != nullptr) {
            float d_dist = 0.f;
            if (d_sdf_image != nullptr) {
                if (eval_positions == nullptr) {
                    d_dist = d_sdf_image[y * width + x];
                } else {
                    d_dist = d_sdf_image[idx];
                }
            }
            auto weight = eval_positions == nullptr ? 1.f / num_samples : 1.f;
            auto dist = sample_distance(scene, npt, weight,
                d_sdf_image != nullptr ? &d_dist : nullptr, 
                d_translation != nullptr ? &d_translation[2 * (y * width + x)] : nullptr);
            if (sdf_image != nullptr) {
                if (eval_positions == nullptr) {
                    atomic_add(sdf_image[y * width + x], dist);
                } else {
                    atomic_add(sdf_image[idx], dist);
                }
            }
        }
    }

    SceneData scene;
    float *background_image;
    float *render_image;
    float *weight_image;
    float *sdf_image;
    float *d_background_image;
    float *d_render_image;
    float *d_sdf_image;
    float *d_translation;
    int width;
    int height;
    int num_samples_x;
    int num_samples_y;
    uint64_t seed;
    bool use_prefiltering;
    float *eval_positions;
    // d(weight_image)/d(filter radius); required when d_render_image is given
    float *d_weight_image;
};

struct BoundarySample {
    Vector2f pt;
    Vector2f local_pt;
    Vector2f normal; // canvas space
    Vector2f local_normal; // shape space
    // Scale that maps (local velocity . local normal) to
    // (canvas velocity . canvas normal): 1 / |canvas_to_shape^T local_normal|
    float local_velocity_scale;
    int shape_group_id;
    int shape_id;
    float t;
    BoundaryData data;
    float pdf;
};

struct sample_boundary_kernel {
    DEVICE void operator()(int idx) {
        boundary_samples[idx].pt = Vector2f{0, 0};
        boundary_samples[idx].shape_id = -1;
        boundary_ids[idx] = idx;
        morton_codes[idx] = 0;

        auto rng_state = init_pcg32(idx, seed);
        auto u = next_pcg32_float(&rng_state);
        // Sample a shape
        auto sample_id = sample(scene.sample_shapes_cdf,
                                scene.num_total_shapes,
                                u);
        assert(sample_id >= 0 && sample_id < scene.num_total_shapes);
        auto shape_id = scene.sample_shape_id[sample_id];
        assert(shape_id >= 0 && shape_id < scene.num_shapes);
        auto shape_group_id = scene.sample_group_id[sample_id];
        assert(shape_group_id >= 0 && shape_group_id < scene.num_shape_groups);
        // sample_shapes_pmf is indexed by sample id (group membership order),
        // not by shape id -- see build_shape_cdfs.
        auto shape_pmf = scene.sample_shapes_pmf[sample_id];
        if (shape_pmf <= 0) {
            return;
        }
        // Sample a point on the boundary of the shape
        auto boundary_pdf = 0.f;
        auto normal = Vector2f{0, 0};
        auto t = next_pcg32_float(&rng_state);
        BoundaryData boundary_data;
        const ShapeGroup &shape_group = scene.shape_groups[shape_group_id];
        auto local_boundary_pt = sample_boundary(
            scene, shape_group_id, shape_id,
            t, normal, boundary_pdf, boundary_data);
        if (boundary_pdf <= 0) {
            return;
        }

        // local_boundary_pt & normal are in shape's local space,
        // transform them to canvas space
        auto boundary_pt = xform_pt(shape_group.shape_to_canvas, local_boundary_pt);
        auto local_normal = normal;
        // The boundary pdf above is with respect to the *local* arc length,
        // but the boundary integral (Reynolds transport) is over the canvas
        // boundary. Convert the pdf to canvas arc length using the Jacobian
        // |d(canvas pt)/d(local pt) * local tangent|.
        const auto &m = shape_group.shape_to_canvas;
        auto local_tangent = Vector2f{-local_normal.y, local_normal.x};
        auto canvas_tangent = Vector2f{
            m(0, 0) * local_tangent.x + m(0, 1) * local_tangent.y,
            m(1, 0) * local_tangent.x + m(1, 1) * local_tangent.y};
        const auto &mi = shape_group.canvas_to_shape;
        // unnormalized canvas normal = canvas_to_shape^T * local_normal
        auto canvas_normal_unnormalized = Vector2f{
            mi(0, 0) * local_normal.x + mi(1, 0) * local_normal.y,
            mi(0, 1) * local_normal.x + mi(1, 1) * local_normal.y};
        auto arc_length_jacobian = length(canvas_tangent);
        auto canvas_normal_length = length(canvas_normal_unnormalized);
        if (!(arc_length_jacobian > 0) || !(canvas_normal_length > 0)) {
            return;
        }
        boundary_pdf /= arc_length_jacobian;
        normal = canvas_normal_unnormalized / canvas_normal_length;
        // Normalize boundary_pt to [0, 1)
        boundary_pt.x /= scene.canvas_width;
        boundary_pt.y /= scene.canvas_height;

        boundary_samples[idx].pt = boundary_pt;
        boundary_samples[idx].local_pt = local_boundary_pt;
        boundary_samples[idx].normal = normal;
        boundary_samples[idx].local_normal = local_normal;
        boundary_samples[idx].local_velocity_scale = 1 / canvas_normal_length;
        boundary_samples[idx].shape_group_id = shape_group_id;
        boundary_samples[idx].shape_id = shape_id;
        boundary_samples[idx].t = t;
        boundary_samples[idx].data = boundary_data;
        boundary_samples[idx].pdf = shape_pmf * boundary_pdf;
        TVector2<uint32_t> p_i{boundary_pt.x * 1023, boundary_pt.y * 1023};
        morton_codes[idx] = (expand_bits(p_i.x) << 1u) |
                            (expand_bits(p_i.y) << 0u);
    }

    SceneData scene;
    uint64_t seed;
    BoundarySample *boundary_samples;
    int *boundary_ids;
    uint32_t *morton_codes;
};

struct render_edge_kernel {
    DEVICE void operator()(int idx) {
        auto bid = boundary_ids[idx];
        if (boundary_samples[bid].shape_id == -1) {
            return;
        }
        auto boundary_pt = boundary_samples[bid].pt;
        auto local_boundary_pt = boundary_samples[bid].local_pt;
        auto normal = boundary_samples[bid].normal;
        auto local_normal = boundary_samples[bid].local_normal;
        auto local_velocity_scale = boundary_samples[bid].local_velocity_scale;
        auto shape_group_id = boundary_samples[bid].shape_group_id;
        auto shape_id = boundary_samples[bid].shape_id;
        auto t = boundary_samples[bid].t;
        auto boundary_data = boundary_samples[bid].data;
        auto pdf = boundary_samples[bid].pdf;

        const ShapeGroup &shape_group = scene.shape_groups[shape_group_id];

        auto bx = int(boundary_pt.x * width);
        auto by = int(boundary_pt.y * height);
        if (bx < 0 || bx >= width || by < 0 || by >= height) {
            return;
        }

        // Sample the two sides of the boundary
        auto inside_query = EdgeQuery{shape_group_id, shape_id, false};
        auto outside_query = EdgeQuery{shape_group_id, shape_id, false};
        auto color_inside = sample_color(scene,
            background_image != nullptr ? (const Vector4f *)&background_image[4 * ((by * width) + bx)] : nullptr,
            boundary_pt - 1e-4f * normal,
            nullptr, &inside_query);
        auto color_outside = sample_color(scene,
            background_image != nullptr ? (const Vector4f *)&background_image[4 * ((by * width) + bx)] : nullptr,
            boundary_pt + 1e-4f * normal,
            nullptr, &outside_query);
        if (!inside_query.hit && !outside_query.hit) {
            // occluded
            return;
        }
        if (!inside_query.hit) {
            normal = -normal;
            local_normal = -local_normal;
            swap_(inside_query, outside_query);
            swap_(color_inside, color_outside);
        }
        // Boundary point in screen space
        auto sboundary_pt = boundary_pt;
        sboundary_pt.x *= width;
        sboundary_pt.y *= height;
        auto d_color = gather_d_color(*scene.filter,
                                      d_render_image,
                                      weight_image,
                                      width,
                                      height,
                                      sboundary_pt);
        // Normalization factor
        d_color /= float(scene.canvas_width * scene.canvas_height);
        
        assert(isfinite(d_color));
        assert(isfinite(pdf) && pdf > 0);
        auto contrib = dot(color_inside - color_outside, d_color) / pdf;
        ShapeGroup &d_shape_group = scene.d_shape_groups[shape_group_id];
        accumulate_boundary_gradient(scene.shapes[shape_id],
            contrib, t, local_normal, boundary_data, scene.d_shapes[shape_id],
            shape_group.shape_to_canvas, local_boundary_pt, d_shape_group.shape_to_canvas,
            normal, local_velocity_scale);
        // Don't need to backprop to filter weights:
        // \int f'(x) g(x) dx doesn't contain discontinuities
        // if f is continuous, even if g is discontinuous
        if (d_translation != nullptr) {
            // According to Reynold transport theorem,
            // the Jacobian of the boundary integral is dot(velocity, normal)
            // The velocity of the object translating x is (1, 0)
            // The velocity of the object translating y is (0, 1)
            atomic_add(&d_translation[2 * (by * width + bx) + 0], normal.x * contrib);
            atomic_add(&d_translation[2 * (by * width + bx) + 1], normal.y * contrib);
        }
    }

    SceneData scene;
    const float *background_image;
    const BoundarySample *boundary_samples;
    const int *boundary_ids;
    float *weight_image;
    float *d_render_image;
    float *d_translation;
    int width;
    int height;
    int num_samples_x;
    int num_samples_y;
};

void render(std::shared_ptr<Scene> scene,
            ptr<float> background_image,
            ptr<float> render_image,
            ptr<float> render_sdf,
            int width,
            int height,
            int num_samples_x,
            int num_samples_y,
            uint64_t seed,
            ptr<float> d_background_image,
            ptr<float> d_render_image,
            ptr<float> d_render_sdf,
            ptr<float> d_translation,
            bool use_prefiltering,
            ptr<float> eval_positions,
            int num_eval_positions) {
#ifdef __NVCC__
    int old_device_id = -1;
    if (scene->use_gpu) {
        checkCuda(cudaGetDevice(&old_device_id));
        if (scene->gpu_index != -1) {
            checkCuda(cudaSetDevice(scene->gpu_index));
        }
    }
#endif
    parallel_init();

    float *weight_image = nullptr;
    // Allocate and zero the weight image
    if (scene->use_gpu) {
#ifdef __CUDACC__
        if (eval_positions.get() == nullptr) {
            checkCuda(cudaMallocManaged(&weight_image, width * height * sizeof(float)));
            cudaMemset(weight_image, 0, width * height * sizeof(float));
        }
#else
        assert(false);
#endif
    } else {
        if (eval_positions.get() == nullptr) {
            weight_image = (float*)malloc(width * height * sizeof(float));
            memset(weight_image, 0, width * height * sizeof(float));
        }
    }
    // d(weight_image)/d(filter radius), needed by the filter radius gradient
    float *d_weight_image = nullptr;
    if (weight_image != nullptr && d_render_image.get() != nullptr) {
        if (scene->use_gpu) {
#ifdef __CUDACC__
            checkCuda(cudaMallocManaged(&d_weight_image, width * height * sizeof(float)));
            cudaMemset(d_weight_image, 0, width * height * sizeof(float));
#else
            assert(false);
#endif
        } else {
            d_weight_image = (float*)malloc(width * height * sizeof(float));
            memset(d_weight_image, 0, width * height * sizeof(float));
        }
    }

    if (render_image.get() != nullptr || d_render_image.get() != nullptr ||
        render_sdf.get() != nullptr || d_render_sdf.get() != nullptr) {
        if (weight_image != nullptr) {
            parallel_for(weight_kernel{
                get_scene_data(*scene.get()),
                weight_image,
                width,
                height,
                num_samples_x,
                num_samples_y,
                seed,
                // sample positions must match render_kernel (pixel centres
                // when prefiltering)
                use_prefiltering,
                d_weight_image
            }, width * height * num_samples_x * num_samples_y, scene->use_gpu);
        }

        auto num_samples = eval_positions.get() == nullptr ?
            width * height * num_samples_x * num_samples_y : num_eval_positions;
        parallel_for(render_kernel{
            get_scene_data(*scene.get()),
            background_image.get(),
            render_image.get(),
            weight_image,
            render_sdf.get(),
            d_background_image.get(),
            d_render_image.get(),
            d_render_sdf.get(),
            d_translation.get(),
            width,
            height,
            num_samples_x,
            num_samples_y,
            seed,
            use_prefiltering,
            eval_positions.get(),
            d_weight_image
        }, num_samples, scene->use_gpu);
    }

    // Boundary sampling
    // (no edge sampling for empty scenes or scenes whose boundaries have zero
    // total length: build_shape_cdfs leaves the sampling tables at zero)
    if (!use_prefiltering && d_render_image.get() != nullptr &&
            scene->num_total_shapes > 0 &&
            scene->sample_shapes_cdf[scene->num_total_shapes - 1] > 0) {
        auto num_samples = width * height * num_samples_x * num_samples_y;
        BoundarySample *boundary_samples = nullptr;
        int *boundary_ids = nullptr; // for sorting
        uint32_t *morton_codes = nullptr; // for sorting
        // Allocate boundary samples
        if (scene->use_gpu) {
#ifdef __CUDACC__
            checkCuda(cudaMallocManaged(&boundary_samples,
                num_samples * sizeof(BoundarySample)));
            checkCuda(cudaMallocManaged(&boundary_ids,
                num_samples * sizeof(int)));
            checkCuda(cudaMallocManaged(&morton_codes,
                num_samples * sizeof(uint32_t)));
#else
            assert(false);
    #endif
        } else {
            boundary_samples = (BoundarySample*)malloc(
                num_samples * sizeof(BoundarySample));
            boundary_ids = (int*)malloc(
                num_samples * sizeof(int));
            morton_codes = (uint32_t*)malloc(
                num_samples * sizeof(uint32_t));
        }
        
        // Edge sampling
        // We sort the boundary samples for better thread coherency
        parallel_for(sample_boundary_kernel{
            get_scene_data(*scene.get()),
            seed,
            boundary_samples,
            boundary_ids,
            morton_codes
        }, num_samples, scene->use_gpu);
        if (scene->use_gpu) {
            thrust::sort_by_key(thrust::device, morton_codes, morton_codes + num_samples, boundary_ids);
        } else {
            // Don't need to sort for CPU, we are not using SIMD hardware anyway.
            // thrust::sort_by_key(thrust::host, morton_codes, morton_codes + num_samples, boundary_ids);
        }
        parallel_for(render_edge_kernel{
            get_scene_data(*scene.get()),
            background_image.get(),
            boundary_samples,
            boundary_ids,
            weight_image,
            d_render_image.get(),
            d_translation.get(),
            width,
            height,
            num_samples_x,
            num_samples_y
        }, num_samples, scene->use_gpu);
        if (scene->use_gpu) {
#ifdef __CUDACC__
            checkCuda(cudaFree(boundary_samples));
            checkCuda(cudaFree(boundary_ids));
            checkCuda(cudaFree(morton_codes));
#else
            assert(false);
#endif
        } else {
            free(boundary_samples);
            free(boundary_ids);
            free(morton_codes);
        }
    }

    // Clean up weight image
    if (scene->use_gpu) {
#ifdef __CUDACC__
        checkCuda(cudaFree(weight_image));
        checkCuda(cudaFree(d_weight_image));
#else
        assert(false);
#endif
    } else {
        free(weight_image);
        free(d_weight_image);
    }

    if (scene->use_gpu) {
        cuda_synchronize();
    }

    parallel_cleanup();
#ifdef __NVCC__
    if (old_device_id != -1) {
        checkCuda(cudaSetDevice(old_device_id));
    }
#endif
}

PYBIND11_MODULE(diffvg, m) {
    m.doc() = "Differential Vector Graphics";

    py::class_<ptr<void>>(m, "void_ptr")
        .def(py::init<std::size_t>())
        .def("as_size_t", &ptr<void>::as_size_t);
    py::class_<ptr<float>>(m, "float_ptr")
        .def(py::init<std::size_t>());
    py::class_<ptr<int>>(m, "int_ptr")
        .def(py::init<std::size_t>());

    py::class_<Vector2f>(m, "Vector2f")
        .def(py::init<float, float>())
        .def_readwrite("x", &Vector2f::x)
        .def_readwrite("y", &Vector2f::y);

    py::class_<Vector3f>(m, "Vector3f")
        .def(py::init<float, float, float>())
        .def_readwrite("x", &Vector3f::x)
        .def_readwrite("y", &Vector3f::y)
        .def_readwrite("z", &Vector3f::z);

    py::class_<Vector4f>(m, "Vector4f")
        .def(py::init<float, float, float, float>())
        .def_readwrite("x", &Vector4f::x)
        .def_readwrite("y", &Vector4f::y)
        .def_readwrite("z", &Vector4f::z)
        .def_readwrite("w", &Vector4f::w);

    py::enum_<ShapeType>(m, "ShapeType")
        .value("circle", ShapeType::Circle)
        .value("ellipse", ShapeType::Ellipse)
        .value("path", ShapeType::Path)
        .value("rect", ShapeType::Rect);

    py::class_<Circle>(m, "Circle")
        .def(py::init<float, Vector2f>())
        .def("get_ptr", &Circle::get_ptr)
        .def_readonly("radius", &Circle::radius)
        .def_readonly("center", &Circle::center);

    py::class_<Ellipse>(m, "Ellipse")
        .def(py::init<Vector2f, Vector2f>())
        .def("get_ptr", &Ellipse::get_ptr)
        .def_readonly("radius", &Ellipse::radius)
        .def_readonly("center", &Ellipse::center);

    py::class_<Path>(m, "Path")
        .def(py::init<ptr<int>, ptr<float>, ptr<float>, int, int, bool, bool>())
        .def("get_ptr", &Path::get_ptr)
        .def("has_thickness", &Path::has_thickness)
        .def("copy_to", &Path::copy_to)
        .def_readonly("num_points", &Path::num_points);

    py::class_<Rect>(m, "Rect")
        .def(py::init<Vector2f, Vector2f>())
        .def("get_ptr", &Rect::get_ptr)
        .def_readonly("p_min", &Rect::p_min)
        .def_readonly("p_max", &Rect::p_max);

    py::enum_<ColorType>(m, "ColorType")
        .value("constant", ColorType::Constant)
        .value("linear_gradient", ColorType::LinearGradient)
        .value("radial_gradient", ColorType::RadialGradient);

    py::class_<Constant>(m, "Constant")
        .def(py::init<Vector4f>())
        .def("get_ptr", &Constant::get_ptr)
        .def_readonly("color", &Constant::color);

    py::class_<LinearGradient>(m, "LinearGradient")
        .def(py::init<Vector2f, Vector2f, int, ptr<float>, ptr<float>>())
        .def("get_ptr", &LinearGradient::get_ptr)
        .def("copy_to", &LinearGradient::copy_to)
        .def_readonly("begin", &LinearGradient::begin)
        .def_readonly("end", &LinearGradient::end)
        .def_readonly("num_stops", &LinearGradient::num_stops);

    py::class_<RadialGradient>(m, "RadialGradient")
        .def(py::init<Vector2f, Vector2f, int, ptr<float>, ptr<float>>())
        .def("get_ptr", &RadialGradient::get_ptr)
        .def("copy_to", &RadialGradient::copy_to)
        .def_readonly("center", &RadialGradient::center)
        .def_readonly("radius", &RadialGradient::radius)
        .def_readonly("num_stops", &RadialGradient::num_stops);

    py::class_<Shape>(m, "Shape")
        .def(py::init<ShapeType, ptr<void>, float>())
        .def("as_circle", &Shape::as_circle)
        .def("as_ellipse", &Shape::as_ellipse)
        .def("as_path", &Shape::as_path)
        .def("as_rect", &Shape::as_rect)
        .def_readonly("type", &Shape::type)
        .def_readonly("stroke_width", &Shape::stroke_width);

    py::class_<ShapeGroup>(m, "ShapeGroup")
        .def(py::init<ptr<int>,
                      int,
                      ColorType,
                      ptr<void>,
                      ColorType,
                      ptr<void>,
                      bool,
                      ptr<float>>())
        .def("fill_color_as_constant", &ShapeGroup::fill_color_as_constant)
        .def("fill_color_as_linear_gradient", &ShapeGroup::fill_color_as_linear_gradient)
        .def("fill_color_as_radial_gradient", &ShapeGroup::fill_color_as_radial_gradient)
        .def("stroke_color_as_constant", &ShapeGroup::stroke_color_as_constant)
        .def("stroke_color_as_linear_gradient", &ShapeGroup::stroke_color_as_linear_gradient)
        .def("stroke_color_as_radial_gradient", &ShapeGroup::stroke_color_as_radial_gradient)
        .def("has_fill_color", &ShapeGroup::has_fill_color)
        .def("has_stroke_color", &ShapeGroup::has_stroke_color)
        .def("copy_to", &ShapeGroup::copy_to)
        .def_readonly("fill_color_type", &ShapeGroup::fill_color_type)
        .def_readonly("stroke_color_type", &ShapeGroup::stroke_color_type);

    py::enum_<FilterType>(m, "FilterType")
        .value("box", FilterType::Box)
        .value("tent", FilterType::Tent)
        .value("parabolic", FilterType::RadialParabolic)
        .value("hann", FilterType::Hann);

    py::class_<Filter>(m, "Filter")
        .def(py::init<FilterType,
                      float>());

    py::class_<Scene, std::shared_ptr<Scene>>(m, "Scene")
        .def(py::init<int,
                      int,
                      const std::vector<const Shape*> &,
                      const std::vector<const ShapeGroup*> &,
                      const Filter &,
                      bool,
                      int>())
        .def("get_d_shape", &Scene::get_d_shape)
        .def("get_d_shape_group", &Scene::get_d_shape_group)
        .def("get_d_filter_radius", &Scene::get_d_filter_radius)
        .def("export_flat", [](const Scene &scene) {
            std::vector<int> ip, ints;
            std::vector<float> floats;
            scene.export_flat(ip, ints, floats);
            py::array_t<int32_t> ip_arr((py::ssize_t)ip.size());
            py::array_t<int32_t> ints_arr((py::ssize_t)ints.size());
            py::array_t<float> floats_arr((py::ssize_t)floats.size());
            std::memcpy(ip_arr.mutable_data(), ip.data(), ip.size() * sizeof(int32_t));
            std::memcpy(ints_arr.mutable_data(), ints.data(), ints.size() * sizeof(int32_t));
            std::memcpy(floats_arr.mutable_data(), floats.data(), floats.size() * sizeof(float));
            return py::make_tuple(ip_arr, ints_arr, floats_arr);
        }, "Flattened scene pools for the Metal kernels: (ip int32[64], ints int32[>=64], floats float32[>=64])")
        .def_readonly("num_shapes", &Scene::num_shapes)
        .def_readonly("num_shape_groups", &Scene::num_shape_groups);

    m.def("render", &render, "");
}
