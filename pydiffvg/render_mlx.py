import mlx.core as mx
import numpy as np
import diffvg
import pydiffvg
import time
import gc
import contextlib
import weakref
from enum import IntEnum
import warnings

print_timing = False

@contextlib.contextmanager
def _no_gc():
    """
        Building/unpacking a scene creates hundreds of thousands of small
        Python objects for large SVGs; the cyclic GC repeatedly rescans them.
        None of these objects form cycles, so pause the collector meanwhile.
    """
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()

def _as_int32(x):
    return x if x.dtype == mx.int32 else x.astype(mx.int32)

class _PendingGrad:
    """ Placeholder for a gradient that will be sliced out of one flat buffer. """
    __slots__ = ('index',)
    def __init__(self, index):
        self.index = index

def set_print_timing(val):
    global print_timing
    print_timing=val

class OutputType(IntEnum):
    color = 1
    sdf = 2

def _is_finite(x):
    return mx.all(mx.isfinite(x)).item()

def _np(x, dtype = np.float32):
    """
        Contiguous numpy array (a zero-copy view when possible), safe to hand
        to C++ as a raw pointer while it is kept alive. diffvg.Scene copies
        all shape data, so the arrays only need to outlive scene construction.
    """
    a = np.asarray(x)
    if a.dtype != dtype or not a.flags.c_contiguous:
        a = np.ascontiguousarray(a, dtype = dtype)
    return a

def _floats(x):
    """
        Flat list of Python floats from an (evaluated) mx.array, numpy array
        or scalar. For the tiny per-shape arrays (colours, radii, widths) this
        is much cheaper than a numpy round trip.
    """
    if isinstance(x, mx.array):
        if x.dtype != mx.float32:
            x = x.astype(mx.float32)
        v = x.tolist()
        if not isinstance(v, list):
            return [v]
        if len(v) > 0 and isinstance(v[0], list):
            return np.asarray(v, dtype = np.float32).reshape(-1).tolist()
        return v
    return np.asarray(x, dtype = np.float32).reshape(-1).tolist()

def _fptr(x):
    return diffvg.float_ptr(x.ctypes.data if x is not None else 0)

def _eval_ptr(eval_positions):
    """
        diffvg treats any non-null eval_positions pointer as "evaluate only at
        these positions" (and then skips the weight image). A zero-length numpy
        array still has a non-null data pointer, unlike torch's empty data_ptr(),
        so pass null explicitly when there are no positions.
    """
    return _fptr(eval_positions if eval_positions.shape[0] > 0 else None)

def _iptr(x):
    return diffvg.int_ptr(x.ctypes.data)

def serialize_scene(canvas_width,
                    canvas_height,
                    shapes,
                    shape_groups,
                    filter = None,
                    output_type = OutputType.color,
                    use_prefiltering = False,
                    eval_positions = None):
    """
        Given a list of shapes, convert them to a linear list of argument,
        so that we can use it in MLX.
    """
    if filter is None:
        filter = pydiffvg.PixelFilter(type = diffvg.FilterType.box,
                                      radius = mx.array(0.5))
    if eval_positions is None:
        eval_positions = mx.zeros((0, 2))
    num_shapes = len(shapes)
    num_shape_groups = len(shape_groups)
    args = []
    args.append(canvas_width)
    args.append(canvas_height)
    args.append(num_shapes)
    args.append(num_shape_groups)
    args.append(output_type)
    args.append(use_prefiltering)
    args.append(eval_positions)
    for shape in shapes:
        use_thickness = False
        if isinstance(shape, pydiffvg.Circle):
            args.append(diffvg.ShapeType.circle)
            args.append(shape.radius)
            args.append(shape.center)
        elif isinstance(shape, pydiffvg.Ellipse):
            args.append(diffvg.ShapeType.ellipse)
            args.append(shape.radius)
            args.append(shape.center)
        elif isinstance(shape, pydiffvg.Path):
            assert(shape.points.shape[1] == 2)
            # (finiteness of points/thickness is checked in _build_scene, in
            # numpy, to avoid one MLX evaluation per path here)
            args.append(diffvg.ShapeType.path)
            args.append(_as_int32(shape.num_control_points))
            args.append(shape.points)
            if len(shape.stroke_width.shape) > 0 and shape.stroke_width.shape[0] > 1:
                use_thickness = True
                args.append(shape.stroke_width)
            else:
                args.append(None)
            args.append(shape.is_closed)
            args.append(shape.use_distance_approx)
        elif isinstance(shape, pydiffvg.Polygon):
            assert(shape.points.shape[1] == 2)
            args.append(diffvg.ShapeType.path)
            if shape.is_closed:
                args.append(mx.zeros(shape.points.shape[0], dtype = mx.int32))
            else:
                args.append(mx.zeros(shape.points.shape[0] - 1, dtype = mx.int32))
            args.append(shape.points)
            args.append(None)
            args.append(shape.is_closed)
            args.append(False) # use_distance_approx
        elif isinstance(shape, pydiffvg.Rect):
            args.append(diffvg.ShapeType.rect)
            args.append(shape.p_min)
            args.append(shape.p_max)
        else:
            assert(False)
        if use_thickness:
            args.append(mx.array(0.0))
        else:
            args.append(shape.stroke_width)

    open_path_ids = None
    for shape_group in shape_groups:
        args.append(_as_int32(shape_group.shape_ids))
        # Fill color
        if shape_group.fill_color is None:
            args.append(None)
        elif isinstance(shape_group.fill_color, mx.array):
            args.append(diffvg.ColorType.constant)
            args.append(shape_group.fill_color)
        elif isinstance(shape_group.fill_color, pydiffvg.LinearGradient):
            args.append(diffvg.ColorType.linear_gradient)
            args.append(shape_group.fill_color.begin)
            args.append(shape_group.fill_color.end)
            args.append(shape_group.fill_color.offsets)
            args.append(shape_group.fill_color.stop_colors)
        elif isinstance(shape_group.fill_color, pydiffvg.RadialGradient):
            args.append(diffvg.ColorType.radial_gradient)
            args.append(shape_group.fill_color.center)
            args.append(shape_group.fill_color.radius)
            args.append(shape_group.fill_color.offsets)
            args.append(shape_group.fill_color.stop_colors)

        if shape_group.fill_color is not None:
            # go through the underlying shapes and check if they are all closed
            if open_path_ids is None:
                open_path_ids = {i for i, s in enumerate(shapes)
                                 if isinstance(s, pydiffvg.Path) and not s.is_closed}
            if open_path_ids:
                for shape_id in np.asarray(shape_group.shape_ids).reshape(-1).tolist():
                    if shape_id in open_path_ids:
                        warnings.warn("Detected non-closed paths with fill color. This might causes unexpected results.", Warning)

        # Stroke color
        if shape_group.stroke_color is None:
            args.append(None)
        elif isinstance(shape_group.stroke_color, mx.array):
            args.append(diffvg.ColorType.constant)
            args.append(shape_group.stroke_color)
        elif isinstance(shape_group.stroke_color, pydiffvg.LinearGradient):
            args.append(diffvg.ColorType.linear_gradient)
            args.append(shape_group.stroke_color.begin)
            args.append(shape_group.stroke_color.end)
            args.append(shape_group.stroke_color.offsets)
            args.append(shape_group.stroke_color.stop_colors)
        elif isinstance(shape_group.stroke_color, pydiffvg.RadialGradient):
            args.append(diffvg.ColorType.radial_gradient)
            args.append(shape_group.stroke_color.center)
            args.append(shape_group.stroke_color.radius)
            args.append(shape_group.stroke_color.offsets)
            args.append(shape_group.stroke_color.stop_colors)
        args.append(shape_group.use_even_odd_rule)
        # Transformation
        args.append(shape_group.shape_to_canvas)
    args.append(filter.type)
    args.append(filter.radius)
    return args

def _unpack_color(args, current_index, color_type, keep_alive):
    """
        Build a diffvg color from the serialized arguments.
        Returns (color, new_index).
    """
    def next_arg():
        nonlocal current_index
        a = args[current_index]
        current_index += 1
        return a

    if color_type == diffvg.ColorType.constant:
        # scalars only: read them directly, no numpy conversion needed
        c = _floats(next_arg())
        color = diffvg.Constant(diffvg.Vector4f(c[0], c[1], c[2], c[3]))
    elif color_type in (diffvg.ColorType.linear_gradient, diffvg.ColorType.radial_gradient):
        a = _np(next_arg())
        b = _np(next_arg())
        offsets = _np(next_arg())
        stop_colors = _np(next_arg())
        assert(offsets.shape[0] == stop_colors.shape[0])
        assert(np.isfinite(stop_colors).all())
        keep_alive += [offsets, stop_colors]
        cls = diffvg.LinearGradient \
            if color_type == diffvg.ColorType.linear_gradient else diffvg.RadialGradient
        color = cls(diffvg.Vector2f(a[0], a[1]),
                    diffvg.Vector2f(b[0], b[1]),
                    offsets.shape[0],
                    _fptr(offsets),
                    _fptr(stop_colors))
    elif color_type is None:
        color = None
    else:
        assert(False)
    return color, current_index

def _build_scene(args):
    """
        Unpack serialized scene arguments into a diffvg.Scene.
        Returns (scene, output_type, use_prefiltering, eval_positions, keep_alive).
        keep_alive holds the buffers and objects diffvg points into;
        it must outlive every use of the scene.
    """
    # Evaluate every (possibly lazy) input in a single MLX call instead of
    # one evaluation per tiny array during conversion.
    mx.eval([a for a in args if isinstance(a, mx.array)])
    with _no_gc():
        return _build_scene_impl(args)

def _build_scene_impl(args):
    keep_alive = []
    current_index = 0
    canvas_width = args[current_index]
    current_index += 1
    canvas_height = args[current_index]
    current_index += 1
    num_shapes = args[current_index]
    current_index += 1
    num_shape_groups = args[current_index]
    current_index += 1
    output_type = args[current_index]
    current_index += 1
    use_prefiltering = args[current_index]
    current_index += 1
    eval_positions = _np(args[current_index])
    current_index += 1
    keep_alive.append(eval_positions)
    shapes = []
    shape_groups = []
    for shape_id in range(num_shapes):
        shape_type = args[current_index]
        current_index += 1
        if shape_type == diffvg.ShapeType.circle:
            radius = _floats(args[current_index])[0]
            current_index += 1
            center = _floats(args[current_index])
            current_index += 1
            shape = diffvg.Circle(radius, diffvg.Vector2f(center[0], center[1]))
        elif shape_type == diffvg.ShapeType.ellipse:
            radius = _floats(args[current_index])
            current_index += 1
            center = _floats(args[current_index])
            current_index += 1
            shape = diffvg.Ellipse(diffvg.Vector2f(radius[0], radius[1]),
                                   diffvg.Vector2f(center[0], center[1]))
        elif shape_type == diffvg.ShapeType.path:
            num_control_points = _np(args[current_index], np.int32)
            current_index += 1
            points = _np(args[current_index])
            assert(np.isfinite(points).all())
            current_index += 1
            thickness = args[current_index]
            if thickness is not None:
                thickness = _np(thickness)
                assert(np.isfinite(thickness).all())
            current_index += 1
            is_closed = args[current_index]
            current_index += 1
            use_distance_approx = args[current_index]
            current_index += 1
            keep_alive += [num_control_points, points, thickness]
            shape = diffvg.Path(_iptr(num_control_points),
                                _fptr(points),
                                _fptr(thickness),
                                num_control_points.shape[0],
                                points.shape[0],
                                is_closed,
                                use_distance_approx)
        elif shape_type == diffvg.ShapeType.rect:
            p_min = _floats(args[current_index])
            current_index += 1
            p_max = _floats(args[current_index])
            current_index += 1
            shape = diffvg.Rect(diffvg.Vector2f(p_min[0], p_min[1]),
                                diffvg.Vector2f(p_max[0], p_max[1]))
        else:
            assert(False)
        stroke_width = _floats(args[current_index])[0]
        current_index += 1
        shapes.append(diffvg.Shape(\
            shape_type, shape.get_ptr(), stroke_width))
        keep_alive.append(shape)

    for shape_group_id in range(num_shape_groups):
        shape_ids = _np(args[current_index], np.int32)
        current_index += 1
        if shape_ids.size == 0:
            # the C++ core hangs on these (BVH build of zero leaves)
            raise ValueError('shape group %d has no shapes: not supported by the C++ scene '
                             'construction (CPU backend, or set_gpu_scene_builder(\'cpu\')); '
                             'use the GPU backend with set_gpu_scene_builder(\'gpu\') or drop '
                             'the empty group' % shape_group_id)
        fill_color_type = args[current_index]
        current_index += 1
        fill_color, current_index = \
            _unpack_color(args, current_index, fill_color_type, keep_alive)
        stroke_color_type = args[current_index]
        current_index += 1
        stroke_color, current_index = \
            _unpack_color(args, current_index, stroke_color_type, keep_alive)
        use_even_odd_rule = args[current_index]
        current_index += 1
        shape_to_canvas = _np(args[current_index])
        current_index += 1

        keep_alive += [shape_ids, shape_to_canvas, fill_color, stroke_color]
        shape_groups.append(diffvg.ShapeGroup(\
            _iptr(shape_ids),
            shape_ids.shape[0],
            diffvg.ColorType.constant if fill_color_type is None else fill_color_type,
            diffvg.void_ptr(0) if fill_color is None else fill_color.get_ptr(),
            diffvg.ColorType.constant if stroke_color_type is None else stroke_color_type,
            diffvg.void_ptr(0) if stroke_color is None else stroke_color.get_ptr(),
            use_even_odd_rule,
            _fptr(shape_to_canvas)))

    filter_type = args[current_index]
    current_index += 1
    filter_radius = _floats(args[current_index])[0]
    current_index += 1
    filt = diffvg.Filter(filter_type, filter_radius)
    keep_alive.append(filt)

    start = time.time()
    scene = diffvg.Scene(canvas_width, canvas_height,
        shapes, shape_groups, filt, False, -1)
    time_elapsed = time.time() - start
    if print_timing:
        print('Scene construction, time: %.5f s' % time_elapsed)
    return scene, output_type, use_prefiltering, eval_positions, keep_alive

def _background(background_image):
    if background_image is None:
        return None
    if background_image.shape[2] == 3:
        raise NotImplementedError('Background image must have 4 channels, not 3. Add a fourth channel with all ones via mx.ones().')
    assert(background_image.shape[2] == 4)
    return _np(background_image)

def _forward(width, height, num_samples_x, num_samples_y, seed, background_image, args,
             scene_cache = None):
    """
        If scene_cache (a dict) is given, the built scene is stored in it under
        'scene' so that the backward pass can reuse it.
    """
    built = _build_scene(args)
    if scene_cache is not None:
        scene_cache['scene'] = built
    return _forward_built(width, height, num_samples_x, num_samples_y, seed,
                          background_image, built)

def _forward_built(width, height, num_samples_x, num_samples_y, seed, background_image, built):
    """ CPU forward pass on an already built scene (see _build_scene). """
    scene, output_type, use_prefiltering, eval_positions, keep_alive = built

    if output_type == OutputType.color:
        assert(eval_positions.shape[0] == 0)
        rendered_image = np.zeros((height, width, 4), dtype = np.float32)
    else:
        assert(output_type == OutputType.sdf)
        if eval_positions.shape[0] == 0:
            rendered_image = np.zeros((height, width, 1), dtype = np.float32)
        else:
            rendered_image = np.zeros((eval_positions.shape[0], 1), dtype = np.float32)

    background_image = _background(background_image)
    if background_image is not None:
        assert(background_image.shape[0] == rendered_image.shape[0])
        assert(background_image.shape[1] == rendered_image.shape[1])

    start = time.time()
    diffvg.render(scene,
                  _fptr(background_image),
                  _fptr(rendered_image if output_type == OutputType.color else None),
                  _fptr(rendered_image if output_type == OutputType.sdf else None),
                  width,
                  height,
                  num_samples_x,
                  num_samples_y,
                  seed,
                  _fptr(None), # d_background_image
                  _fptr(None), # d_render_image
                  _fptr(None), # d_render_sdf
                  _fptr(None), # d_translation
                  use_prefiltering,
                  _eval_ptr(eval_positions),
                  eval_positions.shape[0])
    assert(np.isfinite(rendered_image).all())
    time_elapsed = time.time() - start
    if print_timing:
        print('Forward pass, time: %.5f s' % time_elapsed)
    return mx.array(rendered_image)

def _gpu_mode(args):
    """
        Backend decision for serialized scene args on the GPU: 'color',
        'prefiltered' or 'sdf' (Metal kernels), or None (CPU core). Colour
        output at eval_positions is unsupported by the CPU core too, so it
        stays on the CPU (which asserts).
    """
    output_type, use_prefiltering, eval_positions = args[4], args[5], args[6]
    if output_type == OutputType.sdf:
        return 'sdf'
    if eval_positions is not None and eval_positions.shape[0] > 0:
        return None
    return 'prefiltered' if use_prefiltering else 'color'

def _eval_positions_arg(args):
    ev = args[6]
    return ev if ev is not None and ev.shape[0] > 0 else None

# ---------------------------------------------------------------------------
# GPU scene construction (Metal backend)
#
# 'gpu' (default): pydiffvg/scene_gpu.py builds the flat pools directly from
# the serialized args with MLX ops and GPU BVH builds; the structure-only part
# (topology) is cached across calls. 'cpu': the C++ core builds a diffvg.Scene
# (_build_scene) and export_flat copies it (the reference; also the fallback
# for scenes scene_gpu rejects).

_gpu_scene_builder = 'gpu'
_scene_refit = True

# BVH refit policy (builder 'gpu'). While only float parameters change between
# calls (same topology), the BVHs of the previous call are refit (same leaf
# order and tree, boxes recomputed bottom-up) instead of rebuilt. A refit BVH
# is always a valid BVH, so results do not depend on the policy; only the
# traversal cost can grow as leaves drift away from their Morton / y order.
# The BVHs are rebuilt
# - after REFIT_MAX_CALLS consecutive refits, and
# - when the tree quality drops: for each BVH kind (path, group, scene), the
#   ratio of the summed half-perimeters of the internal nodes to that of the
#   leaves (a surface-area-heuristic proxy for the expected number of visited
#   nodes, independent of uniform scaling) exceeds its value at the last
#   rebuild by more than REFIT_MAX_GROWTH. The ratio is read from the previous
#   call's pools, which are already evaluated then, so the check never blocks.
REFIT_MAX_CALLS = 20
REFIT_MAX_GROWTH = 0.10

def set_gpu_scene_builder(builder):
    """
        Scene construction on the GPU backend: 'gpu' (default; MLX/Metal,
        pydiffvg/scene_gpu.py) or 'cpu' (C++ diffvg.Scene + export_flat).
        Both produce the same pools; the CPU backend is unaffected.
    """
    global _gpu_scene_builder
    if builder not in ('gpu', 'cpu'):
        raise ValueError("set_gpu_scene_builder: expected 'gpu' or 'cpu', got %r" % (builder,))
    _gpu_scene_builder = builder

def get_gpu_scene_builder():
    return _gpu_scene_builder

def set_scene_refit(enabled):
    """
        Enables (default) or disables refitting the BVHs of the previous call
        when only float parameters change (GPU scene builder); see
        REFIT_MAX_CALLS / REFIT_MAX_GROWTH in render_mlx.py.
    """
    global _scene_refit
    _scene_refit = bool(enabled)
    _refit_states.clear()

def get_scene_refit():
    return _scene_refit

_scene_topology_trust = False

def set_scene_topology_trust(enabled):
    """
        UNSAFE opt-in (default False). When True, the GPU scene builder's
        topology cache trusts int arrays (num_control_points, shape_ids) that
        are the same Python objects as in a previous call without re-reading
        their contents (contour: cached lookup ~169 ms -> ~44 ms). Results are
        WRONG if such an array is modified in place (e.g. `ids[0] = 3`)
        between renders; assigning a new array is always safe.
    """
    global _scene_topology_trust
    _scene_topology_trust = bool(enabled)

def get_scene_topology_trust():
    return _scene_topology_trust

class _RefitState:
    __slots__ = ('pools', 'quality', 'base_quality', 'refits')

_refit_states = weakref.WeakKeyDictionary()   # SceneTopology -> _RefitState
_grad_gathers = weakref.WeakKeyDictionary()   # SceneTopology -> render_metal.GradGather

def _bvh_quality(pools):
    """ Lazy float32 (3,): internal / leaf half-perimeter sums of the path, group and scene BVHs. """
    q = []
    for b in pools['bvh']:
        if b is None:
            q.append(mx.array(0.0))
            continue
        nf, ni = b.nodes_f, b.nodes_i
        hp = (nf[:, 2] - nf[:, 0]) + (nf[:, 3] - nf[:, 1])
        internal = ni[:, 1] >= 0
        zero = mx.array(0.0)
        num = mx.sum(mx.where(internal, hp, zero))
        den = mx.sum(mx.where(internal, zero, hp))
        q.append(num / mx.maximum(den, mx.array(1e-20)))
    return mx.stack(q)

def _refit_source(topology):
    """ The previous pools to refit from, or None to rebuild (see the policy above). """
    if not _scene_refit:
        return None
    st = _refit_states.get(topology)
    if st is None or st.pools is None or st.refits >= REFIT_MAX_CALLS:
        return None
    if st.base_quality is None:
        st.base_quality = np.array(st.quality, dtype = np.float64)
        return st.pools
    q = np.array(st.quality, dtype = np.float64)
    if np.any(q > st.base_quality * (1.0 + REFIT_MAX_GROWTH) + 1e-12):
        return None
    return st.pools

def _record_pools(topology, pools, refit):
    if not _scene_refit or pools.get('bvh') is None:
        return
    st = _refit_states.get(topology)
    if st is None or not refit:
        st = _RefitState()
        st.refits = 0
        st.base_quality = None
        _refit_states[topology] = st
    else:
        st.refits += 1
    # only the BVH builds (build_pools reads refit_from['bvh']); the full dict
    # references the topology, which would keep the weak key alive
    st.pools = {'bvh': pools['bvh']}
    st.quality = _bvh_quality(pools)

def _scene_pools(args, width, height, num_samples_x, num_samples_y, seed, use_prefiltering,
                 eval_count, has_background):
    """
        Flat scene pools for the Metal kernels. Returns a dict:
        ip (numpy, per-call slots filled), ints / floats (mx.array for the
        'gpu' builder, numpy for 'cpu'), grads (render_metal.GradGather),
        builder, and 'scene' (the C++ scene for 'cpu').
    """
    from . import render_metal
    from . import scene_gpu
    if _gpu_scene_builder == 'gpu':
        mx.eval([a for a in args if isinstance(a, mx.array)])
        try:
            topology = scene_gpu.topology_from_args(args, trust_int_identity = _scene_topology_trust)
        except ValueError:
            topology = None   # structure not supported by scene_gpu: C++ scene
        if topology is not None:
            refit_from = _refit_source(topology)
            pools = scene_gpu.build_pools(topology, args, width, height, num_samples_x,
                                          num_samples_y, seed, use_prefiltering = use_prefiltering,
                                          eval_count = eval_count, has_background = has_background,
                                          refit_from = refit_from)
            _record_pools(topology, pools, refit_from is not None)
            grads = _grad_gathers.get(topology)
            if grads is None:
                grads = _grad_gathers[topology] = render_metal.GradGather(
                    topology.nargs, topology.grad_pos, topology.grad_start, topology.grad_len)
            return {'ip': pools['ip_np'], 'ints': pools['ints'], 'floats': pools['floats'],
                    'grads': grads, 'builder': 'gpu', 'scene': None}
    from . import render_metal_stage3 as r3
    from . import metal_scene
    ip, ints, floats, built = r3.flat_scene(args)
    grads = render_metal.GradGather(len(args), *render_metal.ranges_from_offsets(
        metal_scene.primal_offsets(ip, ints, args)))
    return {'ip': ip, 'ints': ints, 'floats': floats, 'grads': grads, 'builder': 'cpu',
            'scene': built}

def _prepare_gpu(mode, width, height, num_samples_x, num_samples_y, seed, background_image, args):
    """
        Scene pools plus, for 'color', the prepared kernel inputs and weight
        image ('flat'); for 'prefiltered', the weight image ('weight').
    """
    from . import render_metal
    from . import render_metal_stage3 as r3
    ev = _eval_positions_arg(args)
    use_pref = bool(args[5])
    prepared = _scene_pools(args, width, height, num_samples_x, num_samples_y, seed, use_pref,
                            0 if ev is None else int(ev.shape[0]), background_image is not None)
    ip, ints, floats = prepared['ip'], prepared['ints'], prepared['floats']
    if mode == 'color':
        if background_image is not None:
            _check_background_gpu(background_image, width, height)
        prepared['flat'] = render_metal.prepare_flat(ip, ints, floats, width, height, num_samples_x,
                                                     num_samples_y, seed, background_image)
    elif mode == 'prefiltered':
        prepared['weight'] = r3.prefiltered_weight_flat(ip, ints, floats, width, height,
                                                        num_samples_x, num_samples_y, seed)
    return prepared

def _check_background_gpu(background_image, width, height):
    if background_image.shape[2] == 3:
        raise NotImplementedError('Background image must have 4 channels, not 3. Add a fourth channel with all ones via mx.ones().')
    assert background_image.shape[0] == height and background_image.shape[1] == width

def _forward_gpu(mode, width, height, num_samples_x, num_samples_y, seed, background_image, args,
                 scene_cache):
    """ Metal forward pass; caches the pools (and weight image) in scene_cache for the vjp. """
    from . import render_metal
    from . import render_metal_stage3 as r3
    prepared = _prepare_gpu(mode, width, height, num_samples_x, num_samples_y, seed,
                            background_image, args)
    scene_cache['gpu'] = prepared
    if mode == 'color':
        return render_metal.render_color_prepared(prepared['flat'])
    ip, ints, floats = prepared['ip'], prepared['ints'], prepared['floats']
    if mode == 'sdf':
        return r3.sdf_forward_flat(ip, ints, floats, width, height, num_samples_x, num_samples_y,
                                   seed, _eval_positions_arg(args), bool(args[5]))
    return r3.prefiltered_forward_flat(ip, ints, floats, width, height, num_samples_x,
                                       num_samples_y, seed, background_image,
                                       weight_image = prepared['weight'])

def _backward_gpu(mode, width, height, num_samples_x, num_samples_y, seed, background_image, args,
                  grad_img, render_image, scene_cache):
    """
        Metal backward pass. Returns gradients aligned with
        [background_image] + args (None for non-differentiable entries;
        mapped gradients are flat and reshaped by the caller).
        render_image: the forward output (prefiltered filter-radius gradient).
        scene_cache: the forward pass's cache (rebuilt when empty).
    """
    from . import render_metal
    from . import render_metal_stage3 as r3
    prepared = scene_cache.get('gpu')
    if prepared is None:
        prepared = _prepare_gpu(mode, width, height, num_samples_x, num_samples_y, seed,
                                background_image, args)
    ip, ints, floats = prepared['ip'], prepared['ints'], prepared['floats']
    if mode == 'color':
        d_floats, d_bg, _ = render_metal.render_backward_prepared(prepared['flat'], grad_img)
    elif mode == 'sdf':
        d_floats, _ = r3.sdf_backward_flat(ip, ints, floats, width, height, num_samples_x,
                                           num_samples_y, seed, grad_img,
                                           _eval_positions_arg(args), False, bool(args[5]))
        d_bg = None
    else:
        d_floats, d_bg, _ = r3.prefiltered_backward_flat(
            ip, ints, floats, width, height, num_samples_x, num_samples_y, seed,
            background_image, grad_img, render_image = render_image,
            weight_image = prepared['weight'])
    return [d_bg] + prepared['grads'](d_floats)

def _render_grad_gpu(mode, grad_img, width, height, num_samples_x, num_samples_y, seed,
                     background_image, args):
    """ Screen-space translation gradient image (height, width, 2) on Metal. """
    from . import render_metal
    from . import render_metal_stage3 as r3
    prepared = _prepare_gpu(mode, width, height, num_samples_x, num_samples_y, seed,
                            background_image, args)
    ip, ints, floats = prepared['ip'], prepared['ints'], prepared['floats']
    if mode == 'color':
        _, _, d_tr = render_metal.render_backward_prepared(prepared['flat'], grad_img,
                                                           want_translation = True)
    elif mode == 'sdf':
        _, d_tr = r3.sdf_backward_flat(ip, ints, floats, width, height, num_samples_x,
                                       num_samples_y, seed, grad_img, _eval_positions_arg(args),
                                       True, bool(args[5]))
    else:
        _, _, d_tr = r3.prefiltered_backward_flat(ip, ints, floats, width, height, num_samples_x,
                                                  num_samples_y, seed, background_image, grad_img,
                                                  want_translation = True,
                                                  weight_image = prepared['weight'])
    mx.eval(d_tr)
    return d_tr

def _backward(width, height, num_samples_x, num_samples_y, seed, background_image, args, grad_img,
              built = None):
    """
        Returns gradients aligned with [background_image] + args
        (None for non-differentiable entries).
        built: optionally, the (unused) scene built by the forward pass for
        the same args. Its derivative buffers must still be zero.
    """
    if built is None:
        built = _build_scene(args)
    with _no_gc():
        return _backward_impl(width, height, num_samples_x, num_samples_y, seed,
                              background_image, grad_img, built)

def _backward_impl(width, height, num_samples_x, num_samples_y, seed, background_image, grad_img, built):
    scene, output_type, use_prefiltering, eval_positions, keep_alive = built
    grad_img = _np(grad_img)
    assert(np.isfinite(grad_img).all())

    background_image = _background(background_image)
    if background_image is not None:
        d_background_image = np.zeros_like(background_image)
    else:
        d_background_image = None

    start = time.time()
    diffvg.render(scene,
                  _fptr(background_image),
                  _fptr(None), # render_image
                  _fptr(None), # render_sdf
                  width,
                  height,
                  num_samples_x,
                  num_samples_y,
                  seed,
                  _fptr(d_background_image),
                  _fptr(grad_img if output_type == OutputType.color else None),
                  _fptr(grad_img if output_type == OutputType.sdf else None),
                  _fptr(None), # d_translation
                  use_prefiltering,
                  _eval_ptr(eval_positions),
                  eval_positions.shape[0])
    time_elapsed = time.time() - start
    if print_timing:
        print('Backward pass, time: %.5f s' % time_elapsed)

    # Gradients are collected into one flat buffer and handed back as views
    # of a single MLX array: creating one MLX array per tiny gradient is slow
    # and, for scenes with ~10^5 shapes, exceeds Metal's buffer-count limit.
    pending = []
    def arr(x):
        pending.append(np.asarray(x, dtype = np.float32))
        return _PendingGrad(len(pending) - 1)

    def color_grads(color_type, get_constant, get_linear, get_radial):
        if color_type == diffvg.ColorType.constant:
            c = get_constant().color
            return [arr((c.x, c.y, c.z, c.w))]
        if color_type == diffvg.ColorType.linear_gradient:
            g = get_linear()
            a, b = g.begin, g.end
        elif color_type == diffvg.ColorType.radial_gradient:
            g = get_radial()
            a, b = g.center, g.radius
        else:
            assert(False)
        offsets = np.zeros((g.num_stops), dtype = np.float32)
        stop_colors = np.zeros((g.num_stops, 4), dtype = np.float32)
        g.copy_to(_fptr(offsets), _fptr(stop_colors))
        return [arr((a.x, a.y)), arr((b.x, b.y)), arr(offsets), arr(stop_colors)]

    d_args = []
    d_args.append(None if d_background_image is None else mx.array(d_background_image))
    d_args.append(None) # canvas_width
    d_args.append(None) # canvas_height
    d_args.append(None) # num_shapes
    d_args.append(None) # num_shape_groups
    d_args.append(None) # output_type
    d_args.append(None) # use_prefiltering
    d_args.append(None) # eval_positions
    for shape_id in range(scene.num_shapes):
        d_args.append(None) # type
        d_shape = scene.get_d_shape(shape_id)
        use_thickness = False
        if d_shape.type == diffvg.ShapeType.circle:
            d_circle = d_shape.as_circle()
            d_args.append(arr(d_circle.radius))
            c = d_circle.center
            d_args.append(arr((c.x, c.y)))
        elif d_shape.type == diffvg.ShapeType.ellipse:
            d_ellipse = d_shape.as_ellipse()
            r = d_ellipse.radius
            d_args.append(arr((r.x, r.y)))
            c = d_ellipse.center
            d_args.append(arr((c.x, c.y)))
        elif d_shape.type == diffvg.ShapeType.path:
            d_path = d_shape.as_path()
            points = np.zeros((d_path.num_points, 2), dtype = np.float32)
            thickness = None
            if d_path.has_thickness():
                use_thickness = True
                thickness = np.zeros(d_path.num_points, dtype = np.float32)
            d_path.copy_to(_fptr(points), _fptr(thickness))
            d_args.append(None) # num_control_points
            d_args.append(arr(points))
            d_args.append(None if thickness is None else arr(thickness))
            d_args.append(None) # is_closed
            d_args.append(None) # use_distance_approx
        elif d_shape.type == diffvg.ShapeType.rect:
            d_rect = d_shape.as_rect()
            d_args.append(arr((d_rect.p_min.x, d_rect.p_min.y)))
            d_args.append(arr((d_rect.p_max.x, d_rect.p_max.y)))
        else:
            assert(False)
        if use_thickness:
            d_args.append(None)
        else:
            d_args.append(arr(d_shape.stroke_width))

    for group_id in range(scene.num_shape_groups):
        d_shape_group = scene.get_d_shape_group(group_id)
        d_args.append(None) # shape_ids
        d_args.append(None) # fill_color_type
        if d_shape_group.has_fill_color():
            d_args += color_grads(d_shape_group.fill_color_type,
                                  d_shape_group.fill_color_as_constant,
                                  d_shape_group.fill_color_as_linear_gradient,
                                  d_shape_group.fill_color_as_radial_gradient)
        d_args.append(None) # stroke_color_type
        if d_shape_group.has_stroke_color():
            d_args += color_grads(d_shape_group.stroke_color_type,
                                  d_shape_group.stroke_color_as_constant,
                                  d_shape_group.stroke_color_as_linear_gradient,
                                  d_shape_group.stroke_color_as_radial_gradient)
        d_args.append(None) # use_even_odd_rule
        d_shape_to_canvas = np.zeros((3, 3), dtype = np.float32)
        d_shape_group.copy_to(_fptr(d_shape_to_canvas))
        d_args.append(arr(d_shape_to_canvas))
    d_args.append(None) # filter_type
    d_args.append(arr(scene.get_d_filter_radius()))

    if len(pending) > 0:
        sizes = [p.size for p in pending]
        flat = np.concatenate([p.reshape(-1) for p in pending])
        assert(np.isfinite(flat).all())
        flat = mx.array(flat)
        offsets = np.concatenate([[0], np.cumsum(sizes)]).tolist()
        for k, d in enumerate(d_args):
            if isinstance(d, _PendingGrad):
                i = d.index
                d_args[k] = flat[offsets[i]:offsets[i + 1]].reshape(pending[i].shape)
    return d_args

def render(width,
           height,
           num_samples_x,
           num_samples_y,
           seed,
           background_image,
           *args):
    """
        Differentiable rendering. args is the output of serialize_scene.
        Gradients flow to every mx.array in background_image and args.
    """
    all_args = [background_image] + list(args)
    array_ids = [i for i, a in enumerate(all_args) if isinstance(a, mx.array)]

    def substitute(arrays):
        # mx.custom_function passes new array objects. Integer arrays
        # (num_control_points, shape_ids) are structure, not differentiable:
        # keep the caller's objects so that the GPU builder's topology cache
        # can recognise them (set_scene_topology_trust). Float arrays must be
        # the primals (tracers under transformations).
        full = list(all_args)
        for i, a in zip(array_ids, arrays):
            if not (a.dtype == mx.int32 or a.dtype == mx.int64):
                full[i] = a
        return full

    # The scene built by the forward pass is reused (once) by the backward
    # pass for the same inputs, instead of unpacking the arguments again.
    # GPU passes cache the scene pools, BVH builds, weight image and gradient
    # map under 'gpu' (see _prepare_gpu), and the backend decision ('mode'),
    # so forward and vjp stay consistent.
    scene_cache = {}

    @mx.custom_function
    def _render(*arrays):
        full = substitute(arrays)
        scene_cache.clear()
        mode = _gpu_mode(full[1:]) if pydiffvg.get_use_gpu() else None
        scene_cache['mode'] = mode
        if mode is None:
            return _forward(width, height, num_samples_x, num_samples_y, seed,
                            full[0], full[1:], scene_cache)
        return _forward_gpu(mode, width, height, num_samples_x, num_samples_y, seed,
                            full[0], full[1:], scene_cache)

    @_render.vjp
    def _render_vjp(primals, cotangent, output):
        if isinstance(cotangent, (list, tuple)):
            cotangent = cotangent[0]
        if isinstance(output, (list, tuple)):
            output = output[0]
        full = substitute(primals)
        if 'mode' in scene_cache:
            mode = scene_cache.pop('mode')
        else:
            # No cached forward (e.g. the vjp runs twice): current setting.
            mode = _gpu_mode(full[1:]) if pydiffvg.get_use_gpu() else None
        if mode is None:
            built = scene_cache.pop('scene', None)
            scene_cache.clear()
            d_full = _backward(width, height, num_samples_x, num_samples_y, seed,
                               full[0], full[1:], cotangent, built)
        else:
            d_full = _backward_gpu(mode, width, height, num_samples_x, num_samples_y, seed,
                                   full[0], full[1:], cotangent, output, scene_cache)
            scene_cache.clear()
        grads = []
        # Share one zero array per (shape, dtype) for non-differentiable inputs.
        zeros = {}
        for i, p in zip(array_ids, primals):
            d = d_full[i]
            if d is None:
                key = (tuple(p.shape), p.dtype)
                z = zeros.get(key)
                if z is None:
                    z = zeros[key] = mx.zeros(p.shape, dtype = p.dtype)
                grads.append(z)
            else:
                d = d.reshape(p.shape)
                grads.append(d if d.dtype == p.dtype else d.astype(p.dtype))
        return tuple(grads)

    return _render(*[all_args[i] for i in array_ids])

def render_grad(grad_img,
                width,
                height,
                num_samples_x,
                num_samples_y,
                seed,
                background_image,
                *args):
    """
        Returns the screen-space translation gradient image (height x width x 2).
    """
    mode = _gpu_mode(args) if pydiffvg.get_use_gpu() else None
    if mode is not None:
        bg = background_image
        if bg is not None and bg.shape[2] == 3:
            bg = mx.concatenate([bg, mx.ones((bg.shape[0], bg.shape[1], 1))], axis = 2)
        return _render_grad_gpu(mode, grad_img, width, height, num_samples_x, num_samples_y,
                                seed, bg, args)
    scene, output_type, use_prefiltering, eval_positions, keep_alive = _build_scene(args)
    grad_img = _np(grad_img)
    assert(np.isfinite(grad_img).all())
    if output_type == OutputType.color:
        assert(grad_img.shape[2] == 4)
    else:
        assert(grad_img.shape[2] == 1)

    if background_image is not None:
        if background_image.shape[2] == 3:
            background_image = mx.concatenate([\
                background_image,
                mx.ones((background_image.shape[0], background_image.shape[1], 1))], axis = 2)
        background_image = _background(background_image)

    translation_grad_image = np.zeros((height, width, 2), dtype = np.float32)
    start = time.time()
    diffvg.render(scene,
                  _fptr(background_image),
                  _fptr(None), # render_image
                  _fptr(None), # render_sdf
                  width,
                  height,
                  num_samples_x,
                  num_samples_y,
                  seed,
                  _fptr(None), # d_background_image
                  _fptr(grad_img if output_type == OutputType.color else None),
                  _fptr(grad_img if output_type == OutputType.sdf else None),
                  _fptr(translation_grad_image),
                  use_prefiltering,
                  _eval_ptr(eval_positions),
                  eval_positions.shape[0])
    time_elapsed = time.time() - start
    if print_timing:
        print('Gradient pass, time: %.5f s' % time_elapsed)
    assert(np.isfinite(translation_grad_image).all())
    return mx.array(translation_grad_image)

class RenderFunction:
    """
        The MLX interface of diffvg. Kept for compatibility with the
        PyTorch-style call pattern: RenderFunction.apply(...).
    """
    serialize_scene = staticmethod(serialize_scene)
    apply = staticmethod(render)
    render_grad = staticmethod(render_grad)
