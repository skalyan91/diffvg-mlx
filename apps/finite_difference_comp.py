# python finite_difference_comp.py imgs/tiger.svg
# python finite_difference_comp.py --use_prefiltering True imgs/tiger.svg
# python finite_difference_comp.py imgs/boston.svg
# python finite_difference_comp.py --use_prefiltering True imgs/boston.svg
# python finite_difference_comp.py imgs/contour.svg
# python finite_difference_comp.py --use_prefiltering True imgs/contour.svg
# python finite_difference_comp.py --size_scale 0.5 --clamping_factor 0.05 imgs/hawaii.svg
# python finite_difference_comp.py --size_scale 0.5 --clamping_factor 0.05 --use_prefiltering True imgs/hawaii.svg
# python finite_difference_comp.py imgs/mcseem2.svg
# python finite_difference_comp.py --use_prefiltering True imgs/mcseem2.svg
# python finite_difference_comp.py imgs/reschart.svg
# python finite_difference_comp.py --use_prefiltering True imgs/reschart.svg

import pydiffvg
import diffvg
import argparse
import mlx.core as mx
import numpy as np

try:
    from matplotlib import cm
    _viridis = cm.viridis
except ImportError:
    # Coarse viridis lookup table, used when matplotlib is not installed.
    _VIRIDIS = np.array([[0.267, 0.005, 0.329], [0.283, 0.141, 0.458],
                         [0.254, 0.265, 0.530], [0.207, 0.372, 0.553],
                         [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
                         [0.135, 0.659, 0.518], [0.267, 0.749, 0.441],
                         [0.478, 0.821, 0.318], [0.741, 0.873, 0.150],
                         [0.993, 0.906, 0.144]], dtype = np.float32)
    def _viridis(x):
        x = np.clip(np.asarray(x, dtype = np.float32), 0.0, 1.0)
        pos = x * (len(_VIRIDIS) - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, len(_VIRIDIS) - 1)
        w = (pos - lo)[..., None]
        rgb = _VIRIDIS[lo] * (1 - w) + _VIRIDIS[hi] * w
        return np.concatenate([rgb, np.ones_like(rgb[..., :1])], axis = -1)

def viridis(x):
    """ Map values in [0, 1] to RGBA colors (matplotlib viridis if available). """
    return _viridis(np.asarray(x))

def normalize(x, min_, max_):
    range = max(abs(min_), abs(max_))
    return (x + range) / (2 * range)

def correlation(a, b):
    a = a.reshape(-1) - a.mean()
    b = b.reshape(-1) - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 0 else float('nan')

def main(args):
    pydiffvg.set_print_timing(True)

    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg_file)

    w = int(canvas_width * args.size_scale)
    h = int(canvas_height * args.size_scale)

    print(w, h)
    curve_counts = 0
    for s in shapes:
        if isinstance(s, pydiffvg.Circle):
            curve_counts += 1
        elif isinstance(s, pydiffvg.Ellipse):
            curve_counts += 1
        elif isinstance(s, pydiffvg.Path):
            curve_counts += len(s.num_control_points)
        elif isinstance(s, pydiffvg.Polygon):
            curve_counts += len(s.points) - 1
            if s.is_closed:
                curve_counts += 1
        elif isinstance(s, pydiffvg.Rect):
            curve_counts += 1
    print('curve_counts:', curve_counts)

    pfilter = pydiffvg.PixelFilter(type = diffvg.FilterType.box,
                                   radius = mx.array(0.5))

    use_prefiltering = args.use_prefiltering
    print('use_prefiltering:', use_prefiltering)

    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)

    num_samples_x = args.num_spp
    num_samples_y = args.num_spp
    if (use_prefiltering):
        num_samples_x = 1
        num_samples_y = 1

    render = pydiffvg.RenderFunction.apply
    img = render(w, # width
                 h, # height
                 num_samples_x, # num_samples_x
                 num_samples_y, # num_samples_y
                 0, # seed
                 None, # background_image
                 *scene_args)
    pydiffvg.imwrite(img, 'results/finite_difference_comp/img.png', gamma=1.0)

    epsilon = 0.1
    def perturb_scene(axis, epsilon):
        # MLX arrays are immutable values: build the offset and reassign.
        offset = np.zeros(2, dtype = np.float32)
        offset[axis] = epsilon
        offset = mx.array(offset)
        for s in shapes:
            if isinstance(s, (pydiffvg.Circle, pydiffvg.Ellipse)):
                s.center = s.center + offset
            elif isinstance(s, (pydiffvg.Path, pydiffvg.Polygon)):
                s.points = s.points + offset
            elif isinstance(s, pydiffvg.Rect):
                s.p_min = s.p_min + offset
                s.p_max = s.p_max + offset
        for s in shape_groups:
            if isinstance(s.fill_color, pydiffvg.LinearGradient):
                s.fill_color.begin = s.fill_color.begin + offset
                s.fill_color.end = s.fill_color.end + offset

    def render_scene():
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups,
            filter = pfilter,
            use_prefiltering = use_prefiltering)
        return np.array(render(w, # width
                               h, # height
                               num_samples_x,   # num_samples_x
                               num_samples_y,   # num_samples_y
                               0,   # seed
                               None, # background_image
                               *scene_args))

    perturb_scene(0, epsilon)
    img0 = render_scene()
    perturb_scene(0, -2 * epsilon)
    img1 = render_scene()
    x_diff = (img0 - img1) / (2 * epsilon)
    x_diff = x_diff.sum(axis = 2)
    x_diff_max = x_diff.max() * args.clamping_factor
    x_diff_min = x_diff.min() * args.clamping_factor
    print(x_diff.max())
    print(x_diff.min())
    pydiffvg.imwrite(viridis(normalize(x_diff, x_diff_min, x_diff_max)),
                     'results/finite_difference_comp/finite_x_diff.png', gamma=1.0)
    perturb_scene(0, epsilon)

    perturb_scene(1, epsilon)
    img0 = render_scene()
    perturb_scene(1, -2 * epsilon)
    img1 = render_scene()
    y_diff = (img0 - img1) / (2 * epsilon)
    y_diff = y_diff.sum(axis = 2)
    y_diff_max = y_diff.max() * args.clamping_factor
    y_diff_min = y_diff.min() * args.clamping_factor
    pydiffvg.imwrite(viridis(normalize(y_diff, y_diff_min, y_diff_max)),
                     'results/finite_difference_comp/finite_y_diff.png', gamma=1.0)
    perturb_scene(1, epsilon)

    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)
    render_grad = pydiffvg.RenderFunction.render_grad
    img_grad = render_grad(mx.ones((h, w, 4)),
                           w, # width
                           h, # height
                           num_samples_x, # num_samples_x
                           num_samples_y, # num_samples_y
                           0, # seed
                           None, # background_image
                           *scene_args)
    img_grad = np.array(img_grad)
    print(img_grad[:, :, 0].max())
    print(img_grad[:, :, 0].min())
    # Agreement between the analytic and the finite-difference gradients.
    print('correlation (x): %.4f' % correlation(img_grad[:, :, 0], x_diff))
    print('correlation (y): %.4f' % correlation(img_grad[:, :, 1], y_diff))
    pydiffvg.imwrite(viridis(normalize(img_grad[:, :, 0], x_diff_min, x_diff_max)),
                     'results/finite_difference_comp/ours_x_diff.png', gamma=1.0)
    pydiffvg.imwrite(viridis(normalize(img_grad[:, :, 1], y_diff_min, y_diff_max)),
                     'results/finite_difference_comp/ours_y_diff.png', gamma=1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("svg_file", help="source SVG path")
    parser.add_argument("--size_scale", type=float, default=1.0)
    parser.add_argument("--clamping_factor", type=float, default=0.1)
    parser.add_argument("--num_spp", type=int, default=4)
    parser.add_argument("--use_prefiltering", type=bool, default=False)
    args = parser.parse_args()
    main(args)
