"""
    Flattened scene pools for the Metal kernels.

    diffvg.Scene.export_flat() copies an already built C++ scene (shapes,
    groups, BVHs, CDFs, filter) into three arrays whose layout is defined in
    pydiffvg/metal/common.metal. This module wraps that call and maps pool
    indices back to the serialized scene arguments, so that a gradient pool
    with the same layout as `floats` can be turned into per-argument
    gradients.
"""
import numpy as np
import mlx.core as mx
import diffvg

# ip slots (must match pydiffvg/metal/common.metal)
IP_W = 0
IP_H = 1
IP_NSX = 2
IP_NSY = 3
IP_CANVAS_W = 4
IP_CANVAS_H = 5
IP_NUM_SHAPES = 6
IP_NUM_GROUPS = 7
IP_NUM_TOTAL_SHAPES = 8
IP_FILTER_TYPE = 9
IP_USE_PREFILTERING = 10
IP_HAS_BACKGROUND = 11
IP_NUM_EVAL = 12
IP_SEED_LO = 13
IP_SEED_HI = 14
IP_SCENE_BVH_BASE = 15
IP_SHAPES_I_OFF = 16
IP_GROUPS_I_OFF = 17
IP_BVH_I_OFF = 18
IP_SAMPLE_SHAPE_ID_OFF = 19
IP_SAMPLE_GROUP_ID_OFF = 20
IP_BVH_F_OFF = 21
IP_SAMPLE_CDF_OFF = 22
IP_SAMPLE_PMF_OFF = 23
IP_FILTER_RADIUS_OFF = 24
IP_NUM_FLOATS = 25
IP_NUM_INTS = 26
IP_NUM_BVH_NODES = 27
IP_USE_EVAL_POSITIONS = 28

SHAPE_I_STRIDE = 12
GROUP_I_STRIDE = 12
BVH_F_STRIDE = 5
BVH_I_STRIDE = 2

SHAPE_CIRCLE = 0
SHAPE_ELLIPSE = 1
SHAPE_PATH = 2
SHAPE_RECT = 3

COLOR_NONE = -1
COLOR_CONSTANT = 0
COLOR_LINEAR = 1
COLOR_RADIAL = 2


def flatten_scene(scene):
    """
        Returns (ip int32[64], ints int32[>= 64], floats float32[>= 64]) as
        numpy arrays. The pools are zero-padded to at least 64 entries;
        ip[IP_NUM_INTS] and ip[IP_NUM_FLOATS] hold the unpadded sizes.
        ip[0..3] default to (canvas_width, canvas_height, 1, 1).
    """
    ip, ints, floats = scene.export_flat()
    return ip, ints, floats


def _range(off, n):
    return np.arange(off, off + n, dtype = np.int64)


def primal_offsets(ip, ints, args):
    """
        For the serialized scene `args` (RenderFunction.serialize_scene, without
        the background image) returns a list aligned with `args`: for every
        float mx.array argument that lives in the `floats` pool, an int64 array
        of pool indices in C order, so that
            d_floats[idx].reshape(args[k].shape)
        is the gradient of args[k]. Entries are None for everything else:
        non-arrays, int arrays, eval_positions, and the dummy stroke_width
        (0.0) of a path with per-point thickness, whose gradient is zero.
    """
    ip = np.asarray(ip)
    ints = np.asarray(ints)
    out = [None] * len(args)
    k = 0
    k += 2  # canvas_width, canvas_height
    num_shapes = args[k]
    num_groups = args[k + 1]
    k += 4  # num_shapes, num_groups, output_type, use_prefiltering
    k += 1  # eval_positions (not in the pool)
    if num_shapes > 0:
        assert int(ip[IP_NUM_SHAPES]) == num_shapes
        assert int(ip[IP_NUM_GROUPS]) == num_groups

    def size_of(a):
        return int(a.size) if isinstance(a, mx.array) else 1

    for sid in range(num_shapes):
        rec = int(ip[IP_SHAPES_I_OFF]) + SHAPE_I_STRIDE * sid
        f_off = int(ints[rec + 1])
        shape_type = args[k]
        k += 1
        use_thickness = False
        if shape_type == diffvg.ShapeType.circle:
            out[k] = _range(f_off + 1, 1)          # radius
            out[k + 1] = _range(f_off + 2, 2)      # center
            k += 2
        elif shape_type == diffvg.ShapeType.ellipse:
            out[k] = _range(f_off + 1, 2)          # radius
            out[k + 1] = _range(f_off + 3, 2)      # center
            k += 2
        elif shape_type == diffvg.ShapeType.rect:
            out[k] = _range(f_off + 1, 2)          # p_min
            out[k + 1] = _range(f_off + 3, 2)      # p_max
            k += 2
        elif shape_type == diffvg.ShapeType.path:
            num_points = int(ints[rec + 3])
            k += 1                                 # num_control_points (int)
            out[k] = _range(int(ints[rec + 2]), 2 * num_points)
            k += 1
            if args[k] is not None:
                use_thickness = True
                out[k] = _range(int(ints[rec + 6]), num_points)
            k += 1
            k += 2                                 # is_closed, use_distance_approx
        else:
            assert False
        if not use_thickness and isinstance(args[k], mx.array):
            out[k] = _range(f_off, 1)              # stroke_width
        k += 1

    def color(k, ctype, off, nstops):
        if ctype == diffvg.ColorType.constant:
            out[k] = _range(off, 4)
            return k + 1
        # linear: begin, end; radial: center, radius
        out[k] = _range(off, 2)
        out[k + 1] = _range(off + 2, 2)
        out[k + 2] = _range(off + 4, nstops)
        out[k + 3] = _range(off + 4 + nstops, 4 * nstops)
        return k + 4

    for gid in range(num_groups):
        rec = int(ip[IP_GROUPS_I_OFF]) + GROUP_I_STRIDE * gid
        k += 1  # shape_ids
        fill_type = args[k]
        k += 1
        if fill_type is not None:
            k = color(k, fill_type, int(ints[rec + 3]), int(ints[rec + 4]))
        stroke_type = args[k]
        k += 1
        if stroke_type is not None:
            k = color(k, stroke_type, int(ints[rec + 6]), int(ints[rec + 7]))
        k += 1  # use_even_odd_rule
        out[k] = _range(int(ints[rec + 10]), 9)  # shape_to_canvas
        k += 1
    k += 1  # filter type
    if isinstance(args[k], mx.array):
        out[k] = _range(int(ip[IP_FILTER_RADIUS_OFF]), 1)
    k += 1
    assert k == len(args)
    # An empty scene exports only the filter radius (floats[IP_FILTER_RADIUS_OFF]),
    # which the loop above already mapped; nothing else was visited.
    return out
