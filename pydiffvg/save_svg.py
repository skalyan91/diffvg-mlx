import numpy as np
import pydiffvg
import xml.etree.ElementTree as etree
from xml.dom import minidom


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = etree.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def _to_byte(x):
    return int(round(255 * float(x)))


def _num(x):
    """Shortest string that parses back to the same float32 value."""
    return str(np.float32(x))


def _color_str(c):
    """
    rgb() string for the first 3 channels of c. Uses integer bytes when that
    is exact, otherwise float percentages so parsing it back is lossless.
    """
    c = [float(np.float32(v)) for v in c[:3]]
    if all(np.float32(_to_byte(v) / 255.0) == np.float32(v) for v in c):
        return "rgb({}, {}, {})".format(*[_to_byte(v) for v in c])
    return "rgb({}%, {}%, {}%)".format(*[repr(v * 100.0) for v in c])


def _matrix_str(m):
    m = np.asarray(m, dtype=np.float64)
    return "matrix({} {} {} {} {} {})".format(
        *[repr(float(v)) for v in (m[0, 0], m[1, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2])]
    )


def _is_identity(m):
    return np.array_equal(np.asarray(m, dtype=np.float32), np.eye(3, dtype=np.float32))


def save_svg(filename, width, height, shapes, shape_groups, use_gamma=False):
    root = etree.Element("svg")
    root.set("version", "1.1")
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("width", str(width))
    root.set("height", str(height))
    defs = etree.SubElement(root, "defs")
    g = etree.SubElement(root, "g")
    if use_gamma:
        f = etree.SubElement(defs, "filter")
        f.set("id", "gamma")
        f.set("x", "0")
        f.set("y", "0")
        f.set("width", "100%")
        f.set("height", "100%")
        gamma = etree.SubElement(f, "feComponentTransfer")
        gamma.set("color-interpolation-filters", "sRGB")
        feFuncR = etree.SubElement(gamma, "feFuncR")
        feFuncR.set("type", "gamma")
        feFuncR.set("amplitude", str(1))
        feFuncR.set("exponent", str(1 / 2.2))
        feFuncG = etree.SubElement(gamma, "feFuncG")
        feFuncG.set("type", "gamma")
        feFuncG.set("amplitude", str(1))
        feFuncG.set("exponent", str(1 / 2.2))
        feFuncB = etree.SubElement(gamma, "feFuncB")
        feFuncB.set("type", "gamma")
        feFuncB.set("amplitude", str(1))
        feFuncB.set("exponent", str(1 / 2.2))
        feFuncA = etree.SubElement(gamma, "feFuncA")
        feFuncA.set("type", "gamma")
        feFuncA.set("amplitude", str(1))
        feFuncA.set("exponent", str(1 / 2.2))
        g.set("style", "filter:url(#gamma)")

    def is_gradient(c):
        return isinstance(c, (pydiffvg.LinearGradient, pydiffvg.RadialGradient))

    # Store color
    for i, shape_group in enumerate(shape_groups):
        # diffvg gradients live in canvas space. When the shape carries a
        # transform attribute, userSpaceOnUse coordinates would be interpreted
        # in the transformed space, so undo it with a gradientTransform.
        to_canvas = np.asarray(shape_group.shape_to_canvas, dtype=np.float64)
        inv_transform = None if _is_identity(to_canvas) else np.linalg.inv(to_canvas)

        def add_stops(color, grad):
            offsets = np.array(grad.offsets).reshape(-1)
            stop_colors = np.array(grad.stop_colors).reshape(-1, 4)
            for j in range(offsets.shape[0]):
                stop = etree.SubElement(color, "stop")
                stop.set("offset", _num(offsets[j]))
                c = stop_colors[j]
                stop.set("stop-color", _color_str(c))
                stop.set("stop-opacity", _num(c[3]))

        def add_color(shape_color, name):
            if isinstance(shape_color, pydiffvg.LinearGradient):
                lg = shape_color
                color = etree.SubElement(defs, "linearGradient")
                color.set("id", name)
                color.set("x1", _num(lg.begin[0].item()))
                color.set("y1", _num(lg.begin[1].item()))
                color.set("x2", _num(lg.end[0].item()))
                color.set("y2", _num(lg.end[1].item()))
                color.set("gradientUnits", "userSpaceOnUse")
                if inv_transform is not None:
                    color.set("gradientTransform", _matrix_str(inv_transform))
                add_stops(color, lg)
            elif isinstance(shape_color, pydiffvg.RadialGradient):
                rg = shape_color
                color = etree.SubElement(defs, "radialGradient")
                color.set("id", name)
                cx, cy = [float(v) for v in np.array(rg.center).reshape(-1)[:2]]
                radius = np.array(rg.radius).reshape(-1)
                rx = float(radius[0])
                ry = float(radius[1]) if radius.shape[0] > 1 else rx
                color.set("gradientUnits", "userSpaceOnUse")
                if rx == ry and inv_transform is None:
                    color.set("cx", _num(cx))
                    color.set("cy", _num(cy))
                    color.set("r", _num(rx))
                else:
                    # Elliptical gradient: unit circle mapped by
                    # matrix(rx 0 0 ry cx cy)
                    color.set("cx", "0")
                    color.set("cy", "0")
                    color.set("r", "1")
                    m = np.array([[rx, 0.0, cx], [0.0, ry, cy], [0.0, 0.0, 1.0]])
                    if inv_transform is not None:
                        m = inv_transform @ m
                    color.set("gradientTransform", _matrix_str(m))
                add_stops(color, rg)

        if shape_group.fill_color is not None:
            add_color(shape_group.fill_color, "shape_{}_fill".format(i))
        if shape_group.stroke_color is not None:
            add_color(shape_group.stroke_color, "shape_{}_stroke".format(i))

    for i, shape_group in enumerate(shape_groups):
        shape_ids = np.array(shape_group.shape_ids).reshape(-1).tolist()
        shape = shapes[shape_ids[0]]
        group_shapes = [shapes[sid] for sid in shape_ids]
        if isinstance(shape, pydiffvg.Circle):
            shape_node = etree.SubElement(g, "circle")
            shape_node.set("r", _num(np.array(shape.radius).reshape(-1)[0]))
            shape_node.set("cx", _num(shape.center[0].item()))
            shape_node.set("cy", _num(shape.center[1].item()))
        elif isinstance(shape, pydiffvg.Polygon):
            # A closed polygon is a <polygon>, an open one a <polyline>
            shape_node = etree.SubElement(g, "polygon" if shape.is_closed else "polyline")
            points = np.array(shape.points).reshape(-1, 2)
            shape_node.set(
                "points", " ".join("{},{}".format(_num(p[0]), _num(p[1])) for p in points)
            )
        elif isinstance(shape, pydiffvg.Path):
            shape_node = etree.SubElement(g, "path")
            path_str = ""
            for path in group_shapes:
                num_segments = path.num_control_points.shape[0]
                num_control_points = np.array(path.num_control_points)
                points = np.array(path.points)
                num_points = path.points.shape[0]
                path_str += "M {} {}".format(points[0, 0], points[0, 1])
                point_id = 1
                for j in range(0, num_segments):
                    if num_control_points[j] == 0:
                        p = point_id % num_points
                        path_str += " L {} {}".format(points[p, 0], points[p, 1])
                        point_id += 1
                    elif num_control_points[j] == 1:
                        p1 = (point_id + 1) % num_points
                        path_str += " Q {} {} {} {}".format(
                            points[point_id, 0],
                            points[point_id, 1],
                            points[p1, 0],
                            points[p1, 1],
                        )
                        point_id += 2
                    elif num_control_points[j] == 2:
                        p2 = (point_id + 2) % num_points
                        path_str += " C {} {} {} {} {} {}".format(
                            points[point_id, 0],
                            points[point_id, 1],
                            points[point_id + 1, 0],
                            points[point_id + 1, 1],
                            points[p2, 0],
                            points[p2, 1],
                        )
                        point_id += 3
                if path.is_closed:
                    path_str += " z"
                path_str += " "
            shape_node.set("d", path_str.strip())
        elif isinstance(shape, pydiffvg.Rect):
            shape_node = etree.SubElement(g, "rect")
            p_min = np.array(shape.p_min).reshape(-1)
            p_max = np.array(shape.p_max).reshape(-1)
            shape_node.set("x", _num(p_min[0]))
            shape_node.set("y", _num(p_min[1]))
            shape_node.set("width", _num(p_max[0] - p_min[0]))
            shape_node.set("height", _num(p_max[1] - p_min[1]))
        elif isinstance(shape, pydiffvg.Ellipse):
            shape_node = etree.SubElement(g, "ellipse")
            radius = np.array(shape.radius).reshape(-1)
            shape_node.set("cx", _num(shape.center[0].item()))
            shape_node.set("cy", _num(shape.center[1].item()))
            shape_node.set("rx", _num(radius[0]))
            shape_node.set("ry", _num(radius[1] if radius.shape[0] > 1 else radius[0]))
        else:
            assert False

        if not _is_identity(shape_group.shape_to_canvas):
            shape_node.set("transform", _matrix_str(shape_group.shape_to_canvas))

        stroke_width = np.array(shape.stroke_width).reshape(-1)[0]
        shape_node.set("stroke-width", _num(2 * stroke_width))
        if shape_group.fill_color is not None:
            if is_gradient(shape_group.fill_color):
                shape_node.set("fill", "url(#shape_{}_fill)".format(i))
            else:
                c = np.array(shape_group.fill_color).reshape(-1)
                shape_node.set("fill", _color_str(c))
                # fill-opacity (not opacity, which would also fade the stroke)
                shape_node.set("fill-opacity", _num(c[3]))
            shape_node.set(
                "fill-rule", "evenodd" if shape_group.use_even_odd_rule else "nonzero"
            )
        else:
            shape_node.set("fill", "none")
        if shape_group.stroke_color is not None:
            if is_gradient(shape_group.stroke_color):
                shape_node.set("stroke", "url(#shape_{}_stroke)".format(i))
            else:
                c = np.array(shape_group.stroke_color).reshape(-1)
                shape_node.set("stroke", _color_str(c))
                shape_node.set("stroke-opacity", _num(c[3]))
            shape_node.set("stroke-linecap", "round")
            shape_node.set("stroke-linejoin", "round")

    with open(filename, "w") as f:
        f.write(prettify(root))
