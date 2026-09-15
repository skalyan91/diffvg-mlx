"""
    Packed-parameter scenes.

    The per-shape API keeps every parameter in its own small mx.array (points,
    stroke width, colours, transform ... of each shape and group). For large
    scenes, differentiating with respect to hundreds of thousands of arrays is
    dominated by MLX tracing and per-array Python work. A packed scene splits
    a scene into

    - `SceneStructure` (immutable, hashed by identity): shape kinds, path point
      / segment counts, num_control_points, is_closed, use_distance_approx,
      thickness flags, group shape ids, colour types and stop counts,
      use_even_odd_rule, filter type, canvas size, the index maps below, and
      the GPU scene topology (built once, cached on the structure);
    - `params`: a dict of seven float32 arrays (PARAM_KEYS), laid out like the
      blocks of the GPU float pool, so that building the pools is a handful of
      concatenations and one gather, and the gradients are one gather per
      array:

      points          (Npts, 2)  points of all paths and polygons, shape order.
                                 Rows of shape s: structure.path_points(s).
      thickness       (Nth,)     per-point stroke widths of the paths that have
                                 them (a Path whose stroke_width has more than
                                 one element), shape order.
                                 Entries of shape s: structure.path_thickness(s).
      stroke_width    (S,)       stroke width (half the SVG stroke-width) of each
                                 shape; 0 and unused for thickness paths.
      shape_params    (S, 4)     [a b c d] as the pool shape record:
                                 circle [radius cx cy 0], ellipse [rx ry cx cy],
                                 rect [pmin.x pmin.y pmax.x pmax.y],
                                 path / polygon [0 0 0 0]; unused entries are
                                 ignored and get zero gradients.
      shape_to_canvas (G, 3, 3)  transform of each group.
      colors          (C,)       for each group, its fill block then its stroke
                                 block (absent colours take no space):
                                 constant [r g b a];
                                 linear gradient [begin.x begin.y end.x end.y,
                                 offsets (n), stop colours (4n, rgba per stop)];
                                 radial gradient [center.x center.y radius.x
                                 radius.y, offsets (n), stop colours (4n)].
                                 Blocks: structure.group_fill_slice(g) /
                                 group_stroke_slice(g); fields:
                                 structure.color_fields(g, 'fill' | 'stroke').
      filter_radius   ()         pixel filter radius.

    API: pack_scene, unpack_scene, svg_to_packed_scene, render_packed,
    render_grad_packed. render_packed is differentiable with respect to the
    params dict (mx.value_and_grad(loss)(params)) and the background image.
    On the GPU backend (with the default 'gpu' scene builder) the pools are
    built from the params arrays with no per-shape Python work; otherwise the
    scene is unpacked into ordinary shapes and rendered by render_mlx.render
    (gradients reach params through MLX slicing).
"""
import numpy as np
import mlx.core as mx
import diffvg

from .device import get_use_gpu
from . import render_mlx
from .render_mlx import OutputType
from .shape import Circle, Ellipse, Path, Polygon, Rect, ShapeGroup
from .color import LinearGradient, RadialGradient
from .pixel_filter import PixelFilter

__all__ = ['SceneStructure', 'PackedScene', 'PARAM_KEYS', 'pack_scene', 'unpack_scene',
           'svg_to_packed_scene', 'render_packed', 'render_grad_packed']

PARAM_KEYS = ('points', 'thickness', 'stroke_width', 'shape_params', 'shape_to_canvas', 'colors',
              'filter_radius')

# shape kinds (SceneStructure.shape_kind); 0-3 are the pool shape types
KIND_CIRCLE, KIND_ELLIPSE, KIND_PATH, KIND_RECT, KIND_POLYGON = 0, 1, 2, 3, 4
# colour types (fill_type / stroke_type)
COLOR_NONE, COLOR_CONSTANT, COLOR_LINEAR, COLOR_RADIAL = -1, 0, 1, 2


def _ro(a, dtype):
    a = np.array(a, dtype = dtype).reshape(-1)
    a.flags.writeable = False
    return a


def _starts(counts):
    """ (n + 1,) int64 prefix sums (read-only). """
    c = np.asarray(counts, np.int64).reshape(-1)
    out = np.zeros(c.size + 1, np.int64)
    np.cumsum(c, out = out[1:])
    out.flags.writeable = False
    return out


class SceneStructure:
    """
        Static structure of a packed scene (see the module doc). Immutable;
        equality and hashing are by identity. Array members are read-only
        numpy arrays:

        shape_kind (S,) int8                 KIND_* (4 = polygon)
        shape_path_index (S,)                index into the per-path arrays, -1
        path_shape_ids (P,)                  shape id of each path / polygon
        path_num_points, path_num_segments (P,)
        path_is_closed, path_use_distance_approx, path_has_thickness (P,) bool
        path_point_start (P + 1,)            rows of `points`
        path_segment_start (P + 1,)          entries of num_control_points
        path_thickness_start (P,)            entry of `thickness`, -1 if none
        num_control_points (Nseg,) int32
        group_shape_ids (T,) int32, group_shape_start (G + 1,)
        fill_type, stroke_type (G,)          COLOR_* (-1 none)
        fill_num_stops, stroke_num_stops (G,)
        fill_color_start, stroke_color_start (G,)  offset in `colors`, -1 if none
        color_start (G + 1,)                 colour region of each group
        use_even_odd_rule (G,) bool
        filter_type                          diffvg.FilterType
        shape_ids_attr, group_ids_attr       tuples of the `id` strings
        topology                             scene_gpu.SceneTopology
    """

    def __init__(self, **fields):
        for k, v in fields.items():
            object.__setattr__(self, k, v)
        object.__setattr__(self, '_cache', {})

    def __setattr__(self, name, value):
        raise AttributeError('SceneStructure is immutable')

    __hash__ = object.__hash__

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        return 'SceneStructure(%dx%d, %d shapes, %d groups, %d points)' % (
            self.canvas_width, self.canvas_height, self.num_shapes, self.num_groups, self.num_points)

    # ---- sizes / layout
    def param_shapes(self):
        """ dict key -> expected shape of params[key] """
        return {'points': (self.num_points, 2), 'thickness': (self.num_thickness,),
                'stroke_width': (self.num_shapes,), 'shape_params': (self.num_shapes, 4),
                'shape_to_canvas': (self.num_groups, 3, 3), 'colors': (self.num_colors,),
                'filter_radius': ()}

    def _path(self, shape_id):
        p = int(self.shape_path_index[shape_id])
        if p < 0:
            raise ValueError('shape %d is not a path or polygon' % shape_id)
        return p

    def path_points(self, shape_id):
        """ slice of `points` rows of a path / polygon """
        p = self._path(shape_id)
        return slice(int(self.path_point_start[p]), int(self.path_point_start[p + 1]))

    def path_thickness(self, shape_id):
        """ slice of `thickness` of a path, or None if it has no per-point widths """
        p = self._path(shape_id)
        s = int(self.path_thickness_start[p])
        if s < 0:
            return None
        return slice(s, s + int(self.path_num_points[p]))

    def path_segments(self, shape_id):
        """ slice of `num_control_points` of a path / polygon """
        p = self._path(shape_id)
        return slice(int(self.path_segment_start[p]), int(self.path_segment_start[p + 1]))

    def group_shapes(self, group_id):
        """ slice of `group_shape_ids` """
        return slice(int(self.group_shape_start[group_id]), int(self.group_shape_start[group_id + 1]))

    def _color_slice(self, start, ctype, nstops):
        if ctype < 0:
            return None
        n = 4 if ctype == COLOR_CONSTANT else 4 + 5 * int(nstops)
        return slice(int(start), int(start) + n)

    def group_fill_slice(self, group_id):
        """ slice of the fill colour block in `colors`, or None """
        return self._color_slice(self.fill_color_start[group_id], int(self.fill_type[group_id]),
                                 self.fill_num_stops[group_id])

    def group_stroke_slice(self, group_id):
        """ slice of the stroke colour block in `colors`, or None """
        return self._color_slice(self.stroke_color_start[group_id], int(self.stroke_type[group_id]),
                                 self.stroke_num_stops[group_id])

    def color_fields(self, group_id, which = 'fill'):
        """
            dict field -> slice of `colors` for a group's fill or stroke colour
            (None if absent): constant {'color'}; linear {'begin', 'end',
            'offsets', 'stop_colors'}; radial {'center', 'radius', 'offsets',
            'stop_colors'} (stop_colors: 4n values, rgba per stop).
        """
        if which == 'fill':
            ct, st, n = int(self.fill_type[group_id]), int(self.fill_color_start[group_id]), \
                int(self.fill_num_stops[group_id])
        elif which == 'stroke':
            ct, st, n = int(self.stroke_type[group_id]), int(self.stroke_color_start[group_id]), \
                int(self.stroke_num_stops[group_id])
        else:
            raise ValueError("which must be 'fill' or 'stroke'")
        if ct < 0:
            return None
        if ct == COLOR_CONSTANT:
            return {'color': slice(st, st + 4)}
        a, b = ('begin', 'end') if ct == COLOR_LINEAR else ('center', 'radius')
        return {a: slice(st, st + 2), b: slice(st + 2, st + 4), 'offsets': slice(st + 4, st + 4 + n),
                'stop_colors': slice(st + 4 + n, st + 4 + 5 * n)}

    # ---- GPU gradient map
    def _grad_indices(self):
        """
            Per param key: mx.int32 index into [d_floats, 0] (the appended
            zero, index pool_size, stands for parameters without pool storage).
        """
        g = self._cache.get('grad_index')
        if g is not None:
            return g
        from . import scene_gpu as sg
        t = self.topology
        L = t.lay
        pool = max(sg.MIN_POOL, t.num_floats)
        Z = pool
        S, G = self.num_shapes, self.num_groups
        idx = {}
        if S == 0:
            for k, shp in self.param_shapes().items():
                idx[k] = np.full(int(np.prod(shp, dtype = np.int64)), Z, np.int64)
            idx['filter_radius'] = np.zeros(1, np.int64)
        else:
            f_off, npts, thick = L['f_off'], L['npts'], L['thick']
            idx['points'] = sg._ranges(L['points_off'], 2 * npts)
            idx['thickness'] = sg._ranges(L['th_off'][thick], npts[thick])
            sw = f_off.copy()
            sw[L['p_shape'][thick]] = Z
            idx['stroke_width'] = sw
            sp = f_off[:, None] + 1 + np.arange(4)
            sp[L['stype'] == sg.SHAPE_PATH] = Z
            sp[L['circ'], 3] = Z
            idx['shape_params'] = sp.reshape(-1)
            idx['shape_to_canvas'] = sg._ranges(L['gf_off'], np.full(G, 9))
            coff = np.stack([L['fill_off'], L['stroke_off']], axis = 1).reshape(-1)
            csz = np.stack([L['fsz'], L['ssz']], axis = 1).reshape(-1)
            idx['colors'] = sg._ranges(np.where(coff < 0, 0, coff), csz)
            idx['filter_radius'] = np.zeros(1, np.int64)
        g = {'pool': pool,
             'index': {k: (mx.array(v.astype(np.int32)) if v.size else None) for k, v in idx.items()}}
        self._cache['grad_index'] = g
        return g

    def grads_from_d_floats(self, d_floats):
        """ d_floats pool (length max(64, NF)) -> dict of param gradients (float32). """
        g = self._grad_indices()
        d = d_floats.reshape(-1)
        if d.size != g['pool']:
            raise ValueError('d_floats has %d entries, expected %d' % (d.size, g['pool']))
        dz = mx.concatenate([d, mx.zeros((1,), mx.float32)])
        out = {}
        for k, shp in self.param_shapes().items():
            i = g['index'][k]
            out[k] = mx.zeros(shp, mx.float32) if i is None else dz[i].reshape(shp)
        return out


class PackedScene:
    """
        A SceneStructure with its parameters: `structure`, `params` (dict of
        PARAM_KEYS arrays). Attribute lookups not found here (path_points,
        group_fill_slice, canvas_width, ...) go to the structure.
    """

    def __init__(self, structure, params):
        self.structure = structure
        self.params = params

    def __getattr__(self, name):
        if name in ('structure', 'params'):
            raise AttributeError(name)
        return getattr(self.structure, name)

    def unpack(self, params = None, differentiable = False):
        return unpack_scene(self.structure, self.params if params is None else params,
                            differentiable = differentiable)

    def __repr__(self):
        return 'PackedScene(%r)' % (self.structure,)


# ---------------------------------------------------------------------------
# pack / unpack

def pack_scene(canvas_width, canvas_height, shapes, shape_groups, filter = None):
    """
        Packs an ordinary scene (as for serialize_scene) into a PackedScene.
        The params arrays are new arrays (the shapes are not referenced).
    """
    from . import scene_gpu as sg
    args = render_mlx.serialize_scene(canvas_width, canvas_height, shapes, shape_groups, filter)
    mx.eval([a for a in args if isinstance(a, mx.array)])
    w = sg._walk(args)
    st, pos = sg._read_structure(args, w)
    t = sg._build_core(st)
    src = sg._sources(args, pos, t)
    S, G, P = t.num_shapes, t.num_groups, t.num_paths

    kind = np.asarray(st['stype'], np.int8).copy()
    for s, shape in enumerate(shapes):
        if isinstance(shape, Polygon):
            kind[s] = KIND_POLYGON
    p_shape = st['p_shape']
    path_index = np.full(S, -1, np.int64)
    path_index[p_shape] = np.arange(P)
    npts, nbp, thick = st['npts'], st['nbp'], st['thick']
    th_counts = np.where(thick, npts, 0)
    th_start = np.where(thick, np.cumsum(th_counts) - th_counts, -1)
    ftype, fn, sctype, sn = st['ftype'], st['fn'], st['sctype'], st['sn']
    fsz = np.where(ftype < 0, 0, np.where(ftype == 0, 4, 4 + 5 * fn))
    ssz = np.where(sctype < 0, 0, np.where(sctype == 0, 4, 4 + 5 * sn))
    gsz = fsz + ssz
    gstart = np.cumsum(gsz) - gsz
    structure = SceneStructure(
        canvas_width = int(st['W']), canvas_height = int(st['H']),
        num_shapes = S, num_groups = G, num_paths = P, num_points = t.num_points,
        num_thickness = t.num_thickness, num_segments = t.num_segments,
        num_total_group_shapes = t.num_total_shapes, num_colors = t.num_col,
        shape_kind = _ro(kind, np.int8), shape_path_index = _ro(path_index, np.int64),
        path_shape_ids = _ro(p_shape, np.int64), path_num_points = _ro(npts, np.int64),
        path_num_segments = _ro(nbp, np.int64), path_is_closed = _ro(st['closed'], bool),
        path_use_distance_approx = _ro(st['approx'], bool), path_has_thickness = _ro(thick, bool),
        path_point_start = _starts(npts), path_segment_start = _starts(nbp),
        path_thickness_start = _ro(th_start, np.int64),
        num_control_points = _ro(st['ctrl'], np.int32),
        group_shape_ids = _ro(st['sids'], np.int32), group_shape_start = _starts(st['nsh']),
        fill_type = _ro(ftype, np.int64), fill_num_stops = _ro(fn, np.int64),
        stroke_type = _ro(sctype, np.int64), stroke_num_stops = _ro(sn, np.int64),
        fill_color_start = _ro(np.where(ftype < 0, -1, gstart), np.int64),
        stroke_color_start = _ro(np.where(sctype < 0, -1, gstart + fsz), np.int64),
        color_start = _starts(gsz), use_even_odd_rule = _ro(st['eo'], bool),
        filter_type = args[-2],
        shape_ids_attr = tuple(getattr(s, 'id', '') for s in shapes),
        group_ids_attr = tuple(getattr(g, 'id', '') for g in shape_groups),
        topology = t)

    def get(k, shp):
        a = src.get(k)
        if a is None:
            return mx.zeros(shp, mx.float32)
        return mx.array(a.reshape(shp))
    shp = structure.param_shapes()
    params = {'points': get('points', shp['points']),
              'thickness': get('thickness', shp['thickness']),
              'stroke_width': get('stroke_width', shp['stroke_width']),
              'shape_params': get('shape_params', shp['shape_params']),
              'shape_to_canvas': get('shape_to_canvas', shp['shape_to_canvas']),
              'colors': get('colors', shp['colors']),
              'filter_radius': get('filter_radius', ())}
    mx.eval(list(params.values()))
    return PackedScene(structure, params)


def svg_to_packed_scene(filename):
    """ svg_to_scene + pack_scene. """
    from .parse_svg import svg_to_scene
    w, h, shapes, groups = svg_to_scene(filename)
    return pack_scene(w, h, shapes, groups)


def _structure_of(scene):
    if isinstance(scene, PackedScene):
        return scene.structure
    if isinstance(scene, SceneStructure):
        return scene
    raise TypeError('expected a PackedScene or SceneStructure, got %r' % (type(scene),))


def unpack_scene(packed_or_structure, params = None, differentiable = False):
    """
        Ordinary (shapes, shape_groups) from a packed scene (for save_svg,
        inspection, the CPU backend). params defaults to the PackedScene's.
        differentiable=False: independent mx.arrays copied from the params.
        differentiable=True: MLX slices of the params arrays (gradients flow
        back into params; one lazy op per piece).
    """
    s = _structure_of(packed_or_structure)
    if params is None:
        if not isinstance(packed_or_structure, PackedScene):
            raise ValueError('unpack_scene: params are required with a SceneStructure')
        params = packed_or_structure.params
    # Every piece is a slice of one array per parameter: an evaluated slice
    # is a view sharing that array's buffer. One mx.array (= one Metal
    # buffer) per piece would exceed MLX's buffer-count limit (~499k) for
    # scenes with ~10^5 shapes.
    made = []
    if differentiable:
        P = {k: (v if isinstance(v, mx.array) else mx.array(v)) for k, v in params.items()}
        P = {k: (v if v.dtype == mx.float32 else v.astype(mx.float32)) for k, v in P.items()}
        take = lambda a: a
    else:
        # independent copies of the current values (later edits of params do not leak in)
        P = {k: mx.array(np.array(params[k], dtype = np.float32)) for k in PARAM_KEYS}
        def take(a):
            made.append(a)
            return a
    ctrl_m = mx.array(np.asarray(s.num_control_points, np.int32))
    sids_m = mx.array(np.asarray(s.group_shape_ids, np.int32))
    pts, th, sw, sp = P['points'], P['thickness'], P['stroke_width'], P['shape_params']
    s2c, col = P['shape_to_canvas'], P['colors']
    shapes = []
    kinds = s.shape_kind.tolist()
    ppos = s.shape_path_index.tolist()
    pstart = s.path_point_start.tolist()
    sstart = s.path_segment_start.tolist()
    tstart = s.path_thickness_start.tolist()
    closed = s.path_is_closed.tolist()
    approx = s.path_use_distance_approx.tolist()
    ctrl = s.num_control_points
    ids = s.shape_ids_attr
    for sid, k in enumerate(kinds):
        if k == KIND_CIRCLE:
            shape = Circle(radius = take(sp[sid, 0]), center = take(sp[sid, 1:3]),
                           stroke_width = take(sw[sid]), id = ids[sid])
        elif k == KIND_ELLIPSE:
            shape = Ellipse(radius = take(sp[sid, 0:2]), center = take(sp[sid, 2:4]),
                            stroke_width = take(sw[sid]), id = ids[sid])
        elif k == KIND_RECT:
            shape = Rect(p_min = take(sp[sid, 0:2]), p_max = take(sp[sid, 2:4]),
                         stroke_width = take(sw[sid]), id = ids[sid])
        else:
            p = ppos[sid]
            points = take(pts[pstart[p]:pstart[p + 1]])
            if tstart[p] >= 0:
                width = take(th[tstart[p]:tstart[p] + (pstart[p + 1] - pstart[p])])
            else:
                width = take(sw[sid])
            if k == KIND_POLYGON:
                shape = Polygon(points = points, is_closed = closed[p], stroke_width = width,
                                id = ids[sid])
            else:
                ncp = ctrl_m[sstart[p]:sstart[p + 1]]
                made.append(ncp)
                shape = Path(num_control_points = ncp,
                             points = points, is_closed = closed[p], stroke_width = width,
                             id = ids[sid], use_distance_approx = approx[p])
        shapes.append(shape)

    def color(ct, st, n):
        if ct < 0:
            return None
        if ct == COLOR_CONSTANT:
            return take(col[st:st + 4])
        a, b = take(col[st:st + 2]), take(col[st + 2:st + 4])
        offsets = take(col[st + 4:st + 4 + n])
        stops = take(col[st + 4 + n:st + 4 + 5 * n].reshape(n, 4))
        if ct == COLOR_LINEAR:
            return LinearGradient(begin = a, end = b, offsets = offsets, stop_colors = stops)
        return RadialGradient(center = a, radius = b, offsets = offsets, stop_colors = stops)

    groups = []
    gstart = s.group_shape_start.tolist()
    ft, fs, fn = s.fill_type.tolist(), s.fill_color_start.tolist(), s.fill_num_stops.tolist()
    stt, ss, sn = s.stroke_type.tolist(), s.stroke_color_start.tolist(), s.stroke_num_stops.tolist()
    eo = s.use_even_odd_rule.tolist()
    for g in range(s.num_groups):
        ids_g = sids_m[gstart[g]:gstart[g + 1]]
        made.append(ids_g)
        groups.append(ShapeGroup(shape_ids = ids_g,
                                 fill_color = color(ft[g], fs[g], fn[g]),
                                 use_even_odd_rule = eo[g],
                                 stroke_color = color(stt[g], ss[g], sn[g]),
                                 shape_to_canvas = take(s2c[g]),
                                 id = s.group_ids_attr[g]))
    # evaluate the views once (float pieces only when not differentiable:
    # differentiable pieces may be tracers and are evaluated by the render)
    if made:
        mx.eval(made)
    return shapes, groups


# ---------------------------------------------------------------------------
# rendering

def _param_list(s, params):
    shp = s.param_shapes()
    out = []
    for k in PARAM_KEYS:
        if k not in params:
            raise KeyError('params is missing %r' % k)
        a = params[k]
        if not isinstance(a, mx.array):
            a = mx.array(np.asarray(a, np.float32))
        if a.size != int(np.prod(shp[k], dtype = np.int64)):
            raise ValueError('params[%r] has shape %s, expected %s' % (k, tuple(a.shape), shp[k]))
        out.append(a)
    return out


def _mode(output_type, use_prefiltering, eval_positions):
    if not get_use_gpu() or render_mlx.get_gpu_scene_builder() != 'gpu':
        return None
    return render_mlx._gpu_mode_of(output_type, use_prefiltering, eval_positions)


def _eval_or_none(eval_positions):
    return eval_positions if eval_positions is not None and eval_positions.shape[0] > 0 else None


def _prepare(s, mode, arrays, width, height, nsx, nsy, seed, background_image, use_prefiltering,
             eval_positions):
    """ Pools straight from the params arrays (GPU scene builder), plus kernel inputs. """
    from . import scene_gpu as sg
    t = s.topology
    f32 = lambda a: a if a.dtype == mx.float32 else a.astype(mx.float32)
    src = {k: f32(a).reshape(-1) for k, a in zip(PARAM_KEYS, arrays)}
    src['filter_radius'] = src['filter_radius'].reshape(1)
    refit_from = render_mlx._refit_source(t)
    pools = sg.build_pools_from_sources(t, src, width, height, nsx, nsy, seed,
                                        use_prefiltering = use_prefiltering,
                                        eval_count = 0 if eval_positions is None else int(eval_positions.shape[0]),
                                        has_background = background_image is not None,
                                        refit_from = refit_from)
    render_mlx._record_pools(t, pools, refit_from is not None)
    prepared = {'ip': pools['ip_np'], 'ints': pools['ints'], 'floats': pools['floats'],
                'builder': 'gpu', 'scene': None}
    return render_mlx._prepare_kernels(mode, prepared, width, height, nsx, nsy, seed, background_image)


def _unpacked_args(s, params, output_type, use_prefiltering, eval_positions):
    shapes, groups = unpack_scene(s, params, differentiable = True)
    radius = params['filter_radius']
    if not isinstance(radius, mx.array):
        radius = mx.array(radius, dtype = mx.float32)
    filt = PixelFilter(type = s.filter_type, radius = radius.reshape(()))
    return render_mlx.serialize_scene(s.canvas_width, s.canvas_height, shapes, groups, filt,
                                      output_type = output_type, use_prefiltering = use_prefiltering,
                                      eval_positions = eval_positions)


def render_packed(scene, params, width, height, num_samples_x, num_samples_y, seed,
                  background_image = None, output_type = OutputType.color, use_prefiltering = False,
                  eval_positions = None):
    """
        Differentiable rendering of a packed scene (same semantics as
        RenderFunction.apply on the unpacked scene). scene: PackedScene or
        SceneStructure; params: dict of PARAM_KEYS arrays (None: the
        PackedScene's). Gradients flow to the params arrays and to
        background_image.
    """
    s = _structure_of(scene)
    if params is None:
        params = scene.params
    ev = _eval_or_none(eval_positions)
    mode = _mode(output_type, use_prefiltering, ev)
    if mode is None:
        args = _unpacked_args(s, params, output_type, use_prefiltering, eval_positions)
        return render_mlx.render(width, height, num_samples_x, num_samples_y, seed,
                                 background_image, *args)
    arrays = _param_list(s, params)
    n = len(arrays)
    has_bg = background_image is not None
    inputs = arrays + ([background_image] if has_bg else [])
    cache = {}

    @mx.custom_function
    def _render(*xs):
        cache.clear()
        bg = xs[n] if has_bg else None
        prepared = _prepare(s, mode, xs[:n], width, height, num_samples_x, num_samples_y, seed, bg,
                            use_prefiltering, ev)
        cache['gpu'] = prepared
        return render_mlx._forward_gpu_prepared(mode, prepared, width, height, num_samples_x,
                                                num_samples_y, seed, bg, ev, use_prefiltering)

    @_render.vjp
    def _render_vjp(primals, cotangent, output):
        if isinstance(cotangent, (list, tuple)):
            cotangent = cotangent[0]
        if isinstance(output, (list, tuple)):
            output = output[0]
        bg = primals[n] if has_bg else None
        prepared = cache.pop('gpu', None)
        cache.clear()
        if prepared is None:
            prepared = _prepare(s, mode, primals[:n], width, height, num_samples_x, num_samples_y,
                                seed, bg, use_prefiltering, ev)
        d_floats, d_bg = render_mlx._backward_gpu_prepared(
            mode, prepared, width, height, num_samples_x, num_samples_y, seed, bg, ev,
            use_prefiltering, cotangent, output)
        d = s.grads_from_d_floats(d_floats)
        grads = []
        for k, p in zip(PARAM_KEYS, primals[:n]):
            g = d[k].reshape(p.shape)
            grads.append(g if g.dtype == p.dtype else g.astype(p.dtype))
        if has_bg:
            p = primals[n]
            if d_bg is None:
                grads.append(mx.zeros(p.shape, p.dtype))
            else:
                d_bg = d_bg.reshape(p.shape)
                grads.append(d_bg if d_bg.dtype == p.dtype else d_bg.astype(p.dtype))
        return tuple(grads)

    return _render(*inputs)


def render_grad_packed(scene, params, grad_img, width, height, num_samples_x, num_samples_y, seed,
                       background_image = None, output_type = OutputType.color,
                       use_prefiltering = False, eval_positions = None):
    """ RenderFunction.render_grad for a packed scene: translation gradient image (height, width, 2). """
    s = _structure_of(scene)
    if params is None:
        params = scene.params
    ev = _eval_or_none(eval_positions)
    mode = _mode(output_type, use_prefiltering, ev)
    if mode is None:
        args = _unpacked_args(s, params, output_type, use_prefiltering, eval_positions)
        return render_mlx.render_grad(grad_img, width, height, num_samples_x, num_samples_y, seed,
                                      background_image, *args)
    bg = background_image
    if bg is not None and bg.shape[2] == 3:
        bg = mx.concatenate([bg, mx.ones((bg.shape[0], bg.shape[1], 1))], axis = 2)
    prepared = _prepare(s, mode, _param_list(s, params), width, height, num_samples_x,
                        num_samples_y, seed, bg, use_prefiltering, ev)
    return render_mlx._render_grad_gpu_prepared(mode, prepared, grad_img, width, height,
                                                num_samples_x, num_samples_y, seed, bg, ev,
                                                use_prefiltering)
