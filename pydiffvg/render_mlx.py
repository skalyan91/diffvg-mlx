import mlx.core as mx
import numpy as np
import diffvg
import pydiffvg
import time
import gc
import contextlib
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

def _prepare_gpu(mode, width, height, num_samples_x, num_samples_y, seed, background_image, args,
                 built = None):
    """
        Builds the C++ scene (unless `built` is given) and exports its pools.
        Returns {'scene': built, 'flat': prepare_flat dict} for 'color', or
        {'scene': built, 'pools': (ip, ints, floats)} for 'sdf' / 'prefiltered'.
    """
    from . import render_metal
    from . import render_metal_stage3 as r3
    if built is None:
        built = _build_scene(args)
    if mode == 'color':
        built, flat = render_metal.prepare_scene_gpu(args, width, height, num_samples_x,
                                                     num_samples_y, seed, background_image, built)
        assert flat is not None
        return {'scene': built, 'flat': flat}
    ip, ints, floats, built = r3.flat_scene(args, built)
    return {'scene': built, 'pools': (ip, ints, floats)}

def _forward_gpu(mode, width, height, num_samples_x, num_samples_y, seed, background_image, args,
                 scene_cache):
    """ Metal forward pass; caches the scene and pools in scene_cache for the vjp. """
    from . import render_metal
    from . import render_metal_stage3 as r3
    prepared = _prepare_gpu(mode, width, height, num_samples_x, num_samples_y, seed,
                            background_image, args)
    scene_cache.update(prepared)
    if mode == 'color':
        return render_metal.render_color_prepared(prepared['flat'])
    ip, ints, floats = prepared['pools']
    if mode == 'sdf':
        return r3.sdf_forward_flat(ip, ints, floats, width, height, num_samples_x, num_samples_y,
                                   seed, _eval_positions_arg(args), bool(args[5]))
    return r3.prefiltered_forward_flat(ip, ints, floats, width, height, num_samples_x,
                                       num_samples_y, seed, background_image)

def _backward_gpu(mode, width, height, num_samples_x, num_samples_y, seed, background_image, args,
                  grad_img, render_image, scene_cache):
    """
        Metal backward pass. Returns gradients aligned with
        [background_image] + args (None for non-differentiable entries).
        render_image: the forward output (prefiltered filter-radius gradient).
        scene_cache: the forward pass's cache (rebuilt when empty).
    """
    from . import render_metal
    from . import render_metal_stage3 as r3
    key = 'flat' if mode == 'color' else 'pools'
    prepared = scene_cache if key in scene_cache else \
        _prepare_gpu(mode, width, height, num_samples_x, num_samples_y, seed, background_image,
                     args, scene_cache.get('scene'))
    if mode == 'color':
        flat = prepared['flat']
        d_floats, d_bg, _ = render_metal.render_backward_prepared(flat, grad_img)
        ip, ints = flat['ip'], flat['ints']
    else:
        ip, ints, floats = prepared['pools']
        if mode == 'sdf':
            d_floats, _ = r3.sdf_backward_flat(ip, ints, floats, width, height, num_samples_x,
                                               num_samples_y, seed, grad_img,
                                               _eval_positions_arg(args), False, bool(args[5]))
            d_bg = None
        else:
            d_floats, d_bg, _ = r3.prefiltered_backward_flat(
                ip, ints, floats, width, height, num_samples_x, num_samples_y, seed,
                background_image, grad_img, render_image = render_image)
    return [d_bg] + render_metal.grads_from_d_floats({'ip': ip, 'ints': ints}, args, d_floats)

def _render_grad_gpu(mode, grad_img, width, height, num_samples_x, num_samples_y, seed,
                     background_image, args):
    """ Screen-space translation gradient image (height, width, 2) on Metal. """
    from . import render_metal
    from . import render_metal_stage3 as r3
    if mode == 'color':
        return render_metal.render_grad_gpu(grad_img, args, width, height, num_samples_x,
                                            num_samples_y, seed, background_image)
    ip, ints, floats, _ = r3.flat_scene(args)
    if mode == 'sdf':
        _, d_tr = r3.sdf_backward_flat(ip, ints, floats, width, height, num_samples_x,
                                       num_samples_y, seed, grad_img, _eval_positions_arg(args),
                                       True, bool(args[5]))
    else:
        _, _, d_tr = r3.prefiltered_backward_flat(ip, ints, floats, width, height, num_samples_x,
                                                  num_samples_y, seed, background_image, grad_img,
                                                  want_translation = True)
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
        full = list(all_args)
        for i, a in zip(array_ids, arrays):
            full[i] = a
        return full

    # The scene built by the forward pass is reused (once) by the backward
    # pass for the same inputs, instead of unpacking the arguments again.
    # GPU passes also cache the exported pools ('pools' / 'flat') and the
    # backend decision ('mode') so forward and vjp stay consistent.
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
