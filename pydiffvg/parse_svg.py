import mlx.core as mx
import xml.etree.ElementTree as etree
import numpy as np
import os
import pydiffvg
import svgpathtools
import re
import warnings
import cssutils
import logging
from .color import parse_color_string
cssutils.log.setLevel(logging.ERROR)

# Everything in this module is computed with numpy; mx.arrays are only created
# once per shape / shape group at the very end. (Per-element mx updates such as
# `color[3] = alpha` turned into slice_update graphs that made parsing
# contour.svg take minutes.)
#
# Geometry and shape_to_canvas matrices use float32 composition, exactly like
# the original torch implementation (so the parsed arrays are unchanged).
# Gradients are resolved in float64 and cast to float32 at the end.

_NUMBER_RE = re.compile(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?')
_LENGTH_RE = re.compile(r'^\s*([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)\s*([a-zA-Z%]*)\s*$')
_UNIT_SCALE = {'': 1.0, 'px': 1.0, 'pt': 4.0 / 3.0, 'pc': 16.0, 'mm': 96.0 / 25.4,
               'cm': 96.0 / 2.54, 'in': 96.0, 'em': 16.0, 'ex': 8.0}

# Properties that are inherited from ancestors (subset that diffvg can use).
_INHERITED_PROPS = ('fill', 'fill-opacity', 'fill-rule', 'stroke', 'stroke-width',
                    'stroke-opacity', 'color')
# Presentation attributes / style properties we read.
_PROPS = _INHERITED_PROPS + ('opacity', 'display', 'transform', 'stop-color', 'stop-opacity', 'filter')

_FLOAT_IDENTITY = np.identity(3, dtype=np.float32)

def remove_namespaces(s):
    """
        {...} ... -> ...
    """
    return re.sub('{.*}', '', s)

def parse_style(s, defs = None):
    """
        'a: b; c: d' -> {'a': 'b', 'c': 'd'} (values are kept as strings).
    """
    style_dict = {}
    for e in s.split(';'):
        key_value = e.split(':', 1)
        if len(key_value) == 2:
            key = key_value[0].strip()
            value = key_value[1].strip()
            if key:
                style_dict[key] = value
    return style_dict

def parse_hex(s):
    """
        Hex to (r, g, b) mx.array
    """
    return mx.array(parse_color_string('#' + s.lstrip('#'))[:3], dtype=mx.float32)

def parse_int(s):
    """
        trim alphabets
    """
    return int(float(''.join(i for i in s if (not i.isalpha()) and i != '%')))

def parse_length(s, ref = None, default = 0.0):
    """
        Parse an SVG length ('10', '10px', '2mm', '50%') into user units.
        Percentages are relative to `ref` (or to 1 if `ref` is None).
    """
    if s is None:
        return default
    if not isinstance(s, str):
        return float(s)
    m = _LENGTH_RE.match(s)
    if m is None:
        warnings.warn('Cannot parse length: {}'.format(s))
        return default
    value = float(m.group(1))
    unit = m.group(2).lower()
    if unit == '%':
        return value / 100.0 * (1.0 if ref is None else ref)
    if unit not in _UNIT_SCALE:
        warnings.warn('Unknown length unit: {}'.format(s))
        return value
    return value * _UNIT_SCALE[unit]

def parse_numbers(s):
    return [float(x) for x in _NUMBER_RE.findall(s)]

def _to_mx_color(rgba):
    return mx.array([float(v) for v in rgba], dtype=mx.float32)

def parse_color(s, defs):
    """
        Parse a paint string into an mx.array RGBA color, None, or a gradient
        (gradients are resolved in user space with no object bounding box).
    """
    if s is None:
        return None
    if isinstance(s, mx.array) or isinstance(s, (pydiffvg.LinearGradient, pydiffvg.RadialGradient)):
        return s
    ctx = _Context(defs, 0, 0)
    return _make_paint(s, {}, np.identity(3), None, 1.0, 1.0, ctx)

# https://github.com/mathandy/svgpathtools/blob/7ebc56a831357379ff22216bec07e2c12e8c5bc6/svgpathtools/parser.py
def _parse_transform_substr(transform_substr):
    type_str, value_str = transform_substr.split('(')
    type_str = type_str.strip().lstrip(',').strip()
    values = parse_numbers(value_str)

    transform = np.identity(3)
    if type_str == 'matrix':
        transform[0:2, 0:3] = np.array([values[0:6:2], values[1:6:2]])
    elif type_str == 'translate':
        transform[0, 2] = values[0]
        if len(values) > 1:
            transform[1, 2] = values[1]
    elif type_str == 'scale':
        x_scale = values[0]
        y_scale = values[1] if (len(values) > 1) else x_scale
        transform[0, 0] = x_scale
        transform[1, 1] = y_scale
    elif type_str == 'rotate':
        angle = values[0] * np.pi / 180.0
        if len(values) == 3:
            offset = values[1:3]
        else:
            offset = (0, 0)
        tf_offset = np.identity(3)
        tf_offset[0:2, 2:3] = np.array([[offset[0]], [offset[1]]])
        tf_rotate = np.identity(3)
        tf_rotate[0:2, 0:2] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        tf_offset_neg = np.identity(3)
        tf_offset_neg[0:2, 2:3] = np.array([[-offset[0]], [-offset[1]]])

        transform = tf_offset.dot(tf_rotate).dot(tf_offset_neg)
    elif type_str == 'skewX':
        transform[0, 1] = np.tan(values[0] * np.pi / 180.0)
    elif type_str == 'skewY':
        transform[1, 0] = np.tan(values[0] * np.pi / 180.0)
    else:
        # Return an identity matrix if the type of transform is unknown, and warn the user
        warnings.warn('Unknown SVG transform type: {0}'.format(type_str))
    return transform

def _parse_transform64(transform_str):
    total_transform = np.identity(3)
    if not transform_str:
        return total_transform
    transform_substrs = transform_str.split(')')[:-1]  # Skip the last element, because it should be empty
    for substr in transform_substrs:
        if substr.strip().strip(',').strip() == '':
            continue
        total_transform = total_transform.dot(_parse_transform_substr(substr))
    return total_transform

def parse_transform(transform_str):
    """
        Converts a valid SVG transformation string into a 3x3 matrix
        (float32 mx.array). If the string is empty or null, this returns a
        3x3 identity matrix.
    """
    if not transform_str:
        return mx.eye(3)
    elif not isinstance(transform_str, str):
        raise TypeError('Must provide a string to parse')
    return mx.array(_parse_transform64(transform_str).astype(np.float32))

def _compose(transform32, transform_str):
    """ float32 composition, as in the original implementation """
    return np.matmul(transform32, _parse_transform64(transform_str).astype(np.float32))

class _GradientDef:
    """
        An unresolved <linearGradient>/<radialGradient>. Resolution against a
        referencing shape (its transform and bounding box) happens in
        _make_gradient, so a gradient used by several shapes, or inherited via
        href, is never modified in place.
    """
    def __init__(self, node):
        self.tag = remove_namespaces(node.tag)
        self.attrs = {}
        for key, value in node.attrib.items():
            self.attrs[remove_namespaces(key)] = value
        if 'style' in self.attrs:
            # Styles on the gradient element itself (rare)
            for k, v in parse_style(self.attrs['style']).items():
                self.attrs.setdefault(k, v)
        self.stops = []
        for child in node:
            if remove_namespaces(child.tag) != 'stop':
                continue
            props = {}
            for k in ('offset', 'stop-color', 'stop-opacity', 'color'):
                if k in child.attrib:
                    props[k] = child.attrib[k]
            if 'style' in child.attrib:
                props.update(parse_style(child.attrib['style']))
            offset = props.get('offset', '0').strip()
            if offset.endswith('%'):
                offset = float(offset[:-1]) / 100.0
            else:
                offset = float(offset)
            color = [0.0, 0.0, 0.0, 1.0]
            if 'stop-color' in props:
                try:
                    c = parse_color_string(props['stop-color'], props.get('color'))
                except ValueError:
                    warnings.warn('Unknown stop color: ' + props['stop-color'])
                    c = (0.0, 0.0, 0.0, 1.0)
                if c is None:
                    c = (0.0, 0.0, 0.0, 0.0)
                color = list(c)
            if 'stop-opacity' in props:
                color[3] = color[3] * min(max(parse_length(props['stop-opacity']), 0.0), 1.0)
            self.stops.append((offset, color))

class _Context:
    def __init__(self, defs, width, height):
        self.defs = defs
        self.width = float(width)
        self.height = float(height)
        self.cache = {}
        self.warned = set()

    def warn_once(self, msg):
        if msg not in self.warned:
            self.warned.add(msg)
            warnings.warn(msg)

def _resolve_gradient(ctx, gid):
    """
        Follow the href chain. Returns (tag, attrs, stops) where attrs/stops
        are fresh copies: the child's own attributes take precedence, stops
        come from the nearest gradient in the chain that has any.
    """
    chain = []
    seen = set()
    g = ctx.defs.get(gid)
    while isinstance(g, _GradientDef) and id(g) not in seen:
        seen.add(id(g))
        chain.append(g)
        href = g.attrs.get('href')
        if href is None or not href.strip().startswith('#'):
            break
        g = ctx.defs.get(href.strip()[1:])
    if not chain:
        return None
    tag = chain[0].tag
    geometric = {'x1', 'y1', 'x2', 'y2', 'cx', 'cy', 'r', 'fx', 'fy', 'fr'}
    common = {'gradientUnits', 'gradientTransform', 'spreadMethod'}
    attrs = {}
    stops = None
    for g in chain:
        for k, v in g.attrs.items():
            if k in attrs:
                continue
            if k in common or (k in geometric and g.tag == tag):
                attrs[k] = v
        if stops is None and len(g.stops) > 0:
            stops = list(g.stops)
    return tag, attrs, (stops if stops is not None else [])

def _make_gradient(gid, ctm32, bbox_fn, opacity, ctx):
    resolved = _resolve_gradient(ctx, gid)
    if resolved is None:
        return 'missing'
    tag, attrs, stops = resolved
    if len(stops) == 0:
        return None # No stops: painted as 'none'
    units = attrs.get('gradientUnits', 'objectBoundingBox').strip()
    bbox = None
    if units != 'userSpaceOnUse':
        bbox = bbox_fn() if bbox_fn is not None else None
        if bbox is None:
            ctx.warn_once('objectBoundingBox gradient used without a bounding box; using user space')
        else:
            xmin, ymin, xmax, ymax = bbox
            if not (xmax - xmin > 0 and ymax - ymin > 0):
                return None # Zero-area bounding box: the element is not painted
    key = (gid, np.asarray(ctm32, dtype=np.float32).tobytes(), bbox, float(opacity))
    if key in ctx.cache:
        return ctx.cache[key]

    offsets = []
    colors = []
    prev = 0.0
    for offset, color in stops:
        offset = min(max(offset, prev, 0.0), 1.0)
        prev = offset
        offsets.append(offset)
        colors.append([color[0], color[1], color[2], color[3] * opacity])
    if len(stops) == 1:
        ret = _to_mx_color(colors[0])
        ctx.cache[key] = ret
        return ret

    spread = attrs.get('spreadMethod', 'pad').strip()
    if spread != 'pad':
        ctx.warn_once('spreadMethod="{}" is not supported by diffvg; using pad'.format(spread))

    M = np.asarray(ctm32, dtype=np.float64)
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        M = M @ np.array([[xmax - xmin, 0.0, xmin], [0.0, ymax - ymin, ymin], [0.0, 0.0, 1.0]])
        def coord(name, default, axis):
            v = attrs.get(name, default).strip()
            return float(v[:-1]) / 100.0 if v.endswith('%') else float(v)
    else:
        diag = np.sqrt((ctx.width ** 2 + ctx.height ** 2) / 2.0)
        def coord(name, default, axis):
            ref = ctx.width if axis == 0 else (ctx.height if axis == 1 else diag)
            return parse_length(attrs.get(name, default), ref)
    M = M @ _parse_transform64(attrs.get('gradientTransform'))
    A = M[:2, :2]

    if tag == 'linearGradient':
        b = np.array([coord('x1', '0%', 0), coord('y1', '0%', 1), 1.0])
        e = np.array([coord('x2', '100%', 0), coord('y2', '0%', 1), 1.0])
        d = e[:2] - b[:2]
        dd = float(d @ d)
        if dd == 0.0 or abs(np.linalg.det(A)) < 1e-12:
            # Degenerate vector: painted with the last stop color
            ret = _to_mx_color(colors[-1])
            ctx.cache[key] = ret
            return ret
        B = (M @ b)[:2]
        # t(p) = <A^-1 (p - B), d> / |d|^2 = <p - B, g>, g = A^-T d / |d|^2.
        # diffvg evaluates t(p) = <p - B, E - B> / |E - B|^2, so choosing
        # E = B + g / |g|^2 is exact for any affine transform (including
        # non-uniform bounding-box scaling and skews, where simply mapping
        # the end point would tilt the isolines).
        g = np.linalg.solve(A.T, d) / dd
        E = B + g / float(g @ g)
        ret = pydiffvg.LinearGradient(begin = mx.array(B.astype(np.float32)),
                                      end = mx.array(E.astype(np.float32)),
                                      offsets = mx.array(np.array(offsets, dtype=np.float32)),
                                      stop_colors = mx.array(np.array(colors, dtype=np.float32)))
    else:
        # diffvg radial gradients are axis-aligned ellipses (center, (rx, ry))
        # without a focal point: fx, fy and fr are ignored.
        c = np.array([coord('cx', '50%', 0), coord('cy', '50%', 1), 1.0])
        r = coord('r', '50%', 2)
        if r <= 0.0:
            ret = _to_mx_color(colors[-1])
            ctx.cache[key] = ret
            return ret
        if 'fx' in attrs or 'fy' in attrs or 'fr' in attrs:
            ctx.warn_once('radialGradient focal point (fx, fy, fr) is not supported by diffvg; ignored')
        C = (M @ c)[:2]
        # The iso-contour |A^-1 (p - C)| = r is the ellipse with
        # (p - C)^T S^-1 (p - C) = 1, S = r^2 A A^T.
        S = (r * r) * (A @ A.T)
        rx = np.sqrt(S[0, 0])
        ry = np.sqrt(S[1, 1])
        if abs(S[0, 1]) > 1e-6 * rx * ry:
            ctx.warn_once('radialGradient with a rotated/skewed transform is approximated by an axis-aligned ellipse')
        ret = pydiffvg.RadialGradient(center = mx.array(C.astype(np.float32)),
                                      radius = mx.array(np.array([rx, ry], dtype=np.float32)),
                                      offsets = mx.array(np.array(offsets, dtype=np.float32)),
                                      stop_colors = mx.array(np.array(colors, dtype=np.float32)))
    ctx.cache[key] = ret
    return ret

def _make_paint(value, props, ctm32, bbox_fn, paint_opacity, opacity, ctx):
    """
        Resolve a fill/stroke value into None, an RGBA mx.array or a gradient.
    """
    if value is None:
        return None
    value = value.strip()
    if value.startswith('url('):
        close = value.find(')')
        ref = value[4:close].strip().strip('\'"').lstrip('#')
        fallback = value[close + 1:].strip()
        g = _make_gradient(ref, ctm32, bbox_fn, paint_opacity * opacity, ctx)
        if not (isinstance(g, str) and g == 'missing'):
            return g
        if fallback:
            value = fallback
        else:
            ctx.warn_once('Unknown paint server: ' + ref)
            return None
    try:
        c = parse_color_string(value, props.get('color'))
    except ValueError:
        warnings.warn('Unknown color command ' + value)
        c = (0.0, 0.0, 0.0, 1.0)
    if c is None:
        return None
    return _to_mx_color([c[0], c[1], c[2], c[3] * paint_opacity * opacity])

def _opacity(s):
    return min(max(parse_length(s, 1.0, 1.0), 0.0), 1.0)

def parse_stylesheet(node, transform, defs):
    # collect CSS classes (stored as '.name' so they cannot clash with ids)
    if not node.text:
        return defs
    sheet = cssutils.parseString(node.text)
    for rule in sheet:
        if hasattr(rule, 'selectorList') and hasattr(rule, 'style'):
            style = parse_style(rule.style.getCssText())
            for selector in rule.selectorList:
                name = selector.selectorText.strip()
                if len(name) >= 2 and name[0] == '.' and re.fullmatch(r'\.[-\w]+', name):
                    defs.setdefault(name, {}).update(style)
    return defs

def parse_defs(node, transform, defs):
    for child in node.iter():
        tag = remove_namespaces(child.tag)
        if tag in ('linearGradient', 'radialGradient'):
            if 'id' in child.attrib:
                defs[child.attrib['id']] = _GradientDef(child)
        elif tag == 'style':
            defs = parse_stylesheet(child, transform, defs)
    return defs

def _cascade(node, inherited, defs):
    """
        Computes the properties of `node`: inherited properties from the
        parent, then presentation attributes, then CSS classes, then the
        style attribute (later ones take precedence).
    """
    props = {k: v for k, v in inherited.items() if k in _INHERITED_PROPS}
    local = {}
    for k in _PROPS:
        if k in node.attrib:
            local[k] = node.attrib[k]
    if 'class' in node.attrib:
        for cls in node.attrib['class'].split():
            if '.' + cls in defs:
                local.update(defs['.' + cls])
    if 'style' in node.attrib:
        local.update(parse_style(node.attrib['style']))
    for k, v in local.items():
        if isinstance(v, str) and v.strip() == 'inherit':
            continue
        props[k] = v
    return props

def parse_common_attrib(node, transform, inherited, defs, ctx = None, bbox_fn = None):
    """
        Returns (props, new_transform, fill_color, stroke_color, stroke_width,
        use_even_odd_rule). `inherited` is the dict of properties of the
        parent element.
    """
    if ctx is None:
        ctx = _Context(defs, 0, 0)
    props = _cascade(node, inherited, defs)

    new_transform = transform
    if 'transform' in props:
        new_transform = _compose(transform, props['transform'])

    opacity = _opacity(props['opacity']) if 'opacity' in props else 1.0
    opacity *= inherited.get('__group_opacity__', 1.0)

    fill_color = _make_paint(props.get('fill', 'black'), props, new_transform, bbox_fn,
                             _opacity(props['fill-opacity']) if 'fill-opacity' in props else 1.0,
                             opacity, ctx)
    stroke_color = _make_paint(props.get('stroke', 'none'), props, new_transform, bbox_fn,
                               _opacity(props['stroke-opacity']) if 'stroke-opacity' in props else 1.0,
                               opacity, ctx)

    stroke_width = 0.5
    if 'stroke-width' in props:
        diag = np.sqrt((ctx.width ** 2 + ctx.height ** 2) / 2.0)
        stroke_width = parse_length(props['stroke-width'], diag, 1.0) / 2.0

    use_even_odd_rule = False
    if 'fill-rule' in props:
        rule = props['fill-rule'].strip()
        if rule == "evenodd":
            use_even_odd_rule = True
        elif rule == "nonzero":
            use_even_odd_rule = False
        else:
            warnings.warn('Unknown fill-rule: {}'.format(rule))

    if 'filter' in props and props['filter'].strip() != 'none':
        print('*** WARNING ***: Ignoring filter for path with id "{}"'.format(node.attrib.get('id', '')))

    return props, new_transform, fill_color, stroke_color, stroke_width, use_even_odd_rule

def is_shape(tag):
    return tag in ('path', 'polygon', 'polyline', 'line', 'circle', 'ellipse', 'rect')

def _rounded_rect_path(x, y, w, h, rx, ry):
    return ('M {x0} {y} H {x1} A {rx} {ry} 0 0 1 {xw} {y0} V {y1} A {rx} {ry} 0 0 1 {x1} {yh} '
            'H {x0} A {rx} {ry} 0 0 1 {x} {y1} V {y0} A {rx} {ry} 0 0 1 {x0} {y} Z').format(
        x = x, y = y, rx = rx, ry = ry, x0 = x + rx, x1 = x + w - rx, y0 = y + ry, y1 = y + h - ry,
        xw = x + w, yh = y + h)

def _add_paths(paths, name, props, new_transform, fill_color, stroke_color, stroke_width,
               use_even_odd_rule, shapes, shape_groups):
    if len(paths) == 0:
        return shapes
    sw = mx.array(stroke_width, dtype=mx.float32)
    for idx, path in enumerate(paths):
        assert(path.points.shape[1] == 2)
        path.stroke_width = sw
        path.source_id = name
        path.id = "{}-{}".format(name, idx) if len(paths) > 1 else name
    prev_shapes_size = len(shapes)
    shapes.extend(paths) # in place: `shapes = shapes + paths` was quadratic in the number of shapes
    shape_ids = mx.array(list(range(prev_shapes_size, len(shapes))), dtype=mx.int32)
    shape_groups.append(pydiffvg.ShapeGroup(\
        shape_ids = shape_ids,
        fill_color = fill_color,
        stroke_color = stroke_color,
        use_even_odd_rule = use_even_odd_rule,
        id = name))
    return shapes

def parse_shape(node, transform, inherited, shapes, shape_groups, defs, ctx = None):
    if ctx is None:
        ctx = _Context(defs, 0, 0)
    tag = remove_namespaces(node.tag)
    attrib = node.attrib
    name = attrib.get('id', '')
    W, H = ctx.width, ctx.height
    diag = np.sqrt((W ** 2 + H ** 2) / 2.0)
    def length(key, ref, default = 0.0):
        return parse_length(attrib.get(key), ref, default)

    # Geometry (user space) and its bounding box, used by objectBoundingBox gradients
    bbox_fn = None
    if tag == 'path':
        d = attrib.get('d', '')
        if d.strip() == '':
            return shapes, shape_groups
        def bbox_fn():
            xmin, xmax, ymin, ymax = svgpathtools.parse_path(d).bbox()
            return (xmin, ymin, xmax, ymax)
    elif tag in ('polygon', 'polyline'):
        pts = np.array(parse_numbers(attrib.get('points', '')), dtype=np.float64)
        pts = pts[:(len(pts) // 2) * 2].reshape(-1, 2)
        if pts.shape[0] == 0:
            return shapes, shape_groups
        bbox_fn = lambda: (float(pts[:, 0].min()), float(pts[:, 1].min()),
                           float(pts[:, 0].max()), float(pts[:, 1].max()))
    elif tag == 'line':
        x1, y1 = length('x1', W), length('y1', H)
        x2, y2 = length('x2', W), length('y2', H)
        bbox_fn = lambda: (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    elif tag == 'circle':
        r = length('r', diag)
        cx, cy = length('cx', W), length('cy', H)
        if r <= 0:
            return shapes, shape_groups
        bbox_fn = lambda: (cx - r, cy - r, cx + r, cy + r)
    elif tag == 'ellipse':
        rx_s = attrib.get('rx', 'auto').strip()
        ry_s = attrib.get('ry', 'auto').strip()
        rx = None if rx_s == 'auto' else parse_length(rx_s, W)
        ry = None if ry_s == 'auto' else parse_length(ry_s, H)
        # SVG 2: an 'auto' (or missing) radius takes the value of the other one
        if rx is None:
            rx = ry
        if ry is None:
            ry = rx
        cx, cy = length('cx', W), length('cy', H)
        if rx is None or rx <= 0 or ry <= 0:
            return shapes, shape_groups
        bbox_fn = lambda: (cx - rx, cy - ry, cx + rx, cy + ry)
    elif tag == 'rect':
        x, y = length('x', W), length('y', H)
        w, h = length('width', W), length('height', H)
        if w <= 0 or h <= 0:
            return shapes, shape_groups
        bbox_fn = lambda: (x, y, x + w, y + h)

    props, new_transform, fill_color, stroke_color, stroke_width, use_even_odd_rule = \
        parse_common_attrib(node, transform, inherited, defs, ctx, bbox_fn)
    if props.get('display', '').strip() == 'none':
        return shapes, shape_groups
    if tag == 'line':
        fill_color = None # lines are never filled
    sw = mx.array(stroke_width, dtype=mx.float32)

    def add_single(shape):
        nonlocal shapes
        shape.stroke_width = sw
        shape_ids = mx.array([len(shapes)], dtype=mx.int32)
        shapes.append(shape)
        shape_groups.append(pydiffvg.ShapeGroup(\
            shape_ids = shape_ids,
            fill_color = fill_color,
            stroke_color = stroke_color,
            use_even_odd_rule = use_even_odd_rule,
            shape_to_canvas = mx.array(new_transform),
            id = name))

    if tag == 'path':
        force_closing = fill_color is not None
        paths = pydiffvg.from_svg_path(d, new_transform, force_closing)
        shapes = _add_paths(paths, name, props, new_transform, fill_color, stroke_color,
                            stroke_width, use_even_odd_rule, shapes, shape_groups)
    elif tag in ('polygon', 'polyline'):
        # A polygon is always closed; a polyline is only closed for filling
        # (diffvg cannot close the fill but not the stroke).
        is_closed = True if tag == 'polygon' else fill_color is not None
        add_single(pydiffvg.Polygon(mx.array(pts.astype(np.float32)), is_closed))
    elif tag == 'line':
        points = mx.array(np.array([[x1, y1], [x2, y2]], dtype=np.float32))
        add_single(pydiffvg.Polygon(points, False))
    elif tag == 'circle':
        add_single(pydiffvg.Circle(radius = mx.array(r, dtype=mx.float32),
                                   center = mx.array(np.array([cx, cy], dtype=np.float32))))
    elif tag == 'ellipse':
        add_single(pydiffvg.Ellipse(radius = mx.array(np.array([rx, ry], dtype=np.float32)),
                                    center = mx.array(np.array([cx, cy], dtype=np.float32))))
    elif tag == 'rect':
        rx_s = attrib.get('rx', 'auto').strip()
        ry_s = attrib.get('ry', 'auto').strip()
        rx = None if rx_s == 'auto' else parse_length(rx_s, W)
        ry = None if ry_s == 'auto' else parse_length(ry_s, H)
        if rx is None:
            rx = ry
        if ry is None:
            ry = rx
        if rx is not None and rx > 0 and ry > 0:
            # diffvg's Rect has no rounded corners: convert to a path
            rx = min(rx, w / 2.0)
            ry = min(ry, h / 2.0)
            paths = pydiffvg.from_svg_path(_rounded_rect_path(x, y, w, h, rx, ry), new_transform, True)
            shapes = _add_paths(paths, name, props, new_transform, fill_color, stroke_color,
                                stroke_width, use_even_odd_rule, shapes, shape_groups)
        else:
            add_single(pydiffvg.Rect(p_min = mx.array(np.array([x, y], dtype=np.float32)),
                                     p_max = mx.array(np.array([x + w, y + h], dtype=np.float32))))
    return shapes, shape_groups

def parse_group(node, transform, inherited, shapes, shape_groups, defs, ctx = None):
    if ctx is None:
        ctx = _Context(defs, 0, 0)
    props = _cascade(node, inherited, defs)
    if props.get('display', '').strip() == 'none':
        return shapes, shape_groups
    if 'transform' in props:
        transform = _compose(transform, props['transform'])
    # Group opacity is approximated by multiplying it into the children
    props['__group_opacity__'] = inherited.get('__group_opacity__', 1.0) * \
        (_opacity(props['opacity']) if 'opacity' in props else 1.0)
    return _parse_children(node, transform, props, shapes, shape_groups, defs, ctx)

def _parse_children(node, transform, props, shapes, shape_groups, defs, ctx):
    for child in node:
        tag = remove_namespaces(child.tag)
        if is_shape(tag):
            shapes, shape_groups = parse_shape(\
                child, transform, props, shapes, shape_groups, defs, ctx)
        elif tag in ('g', 'a', 'switch'):
            shapes, shape_groups = parse_group(\
                child, transform, props, shapes, shape_groups, defs, ctx)
    return shapes, shape_groups

def parse_scene(node):
    canvas_width = -1
    canvas_height = -1
    defs = {}
    shapes = []
    shape_groups = []
    transform = _FLOAT_IDENTITY
    if 'viewBox' in node.attrib:
        view_box = parse_numbers(node.attrib['viewBox'])
        canvas_width = int(view_box[2])
        canvas_height = int(view_box[3])
        if view_box[0] != 0 or view_box[1] != 0:
            transform = _parse_transform64('translate({}, {})'.format(
                -view_box[0], -view_box[1])).astype(np.float32)
    else:
        if 'width' in node.attrib:
            canvas_width = parse_int(node.attrib['width'])
        else:
            print('Warning: Can\'t find canvas width.')
        if 'height' in node.attrib:
            canvas_height = parse_int(node.attrib['height'])
        else:
            print('Warning: Can\'t find canvas height.')
    # Collect gradients and stylesheets from the whole document first, so that
    # references to definitions appearing later in the file work.
    defs = parse_defs(node, transform, defs)
    ctx = _Context(defs, canvas_width, canvas_height)
    # The root <svg> element can carry presentation attributes too
    props = _cascade(node, {}, defs)
    props.pop('transform', None)
    props['__group_opacity__'] = _opacity(props['opacity']) if 'opacity' in props else 1.0
    shapes, shape_groups = _parse_children(node, transform, props, shapes, shape_groups, defs, ctx)
    return canvas_width, canvas_height, shapes, shape_groups

def svg_to_scene(filename):
    """
        Load from a SVG file and convert to MLX arrays.
    """

    tree = etree.parse(filename)
    root = tree.getroot()
    cwd = os.getcwd()
    if (os.path.dirname(filename) != ''):
        os.chdir(os.path.dirname(filename))
    try:
        # Transforms are composed with numpy (exact); mx.arrays are only
        # created from numpy data, so no mx computation happens here.
        ret = parse_scene(root)
    finally:
        os.chdir(cwd)
    return ret
