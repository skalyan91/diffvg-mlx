import pydiffvg
import diffvg
import argparse
import mlx.core as mx
import numpy as np
from finite_difference_comp import viridis, normalize

def main(args):
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg_file)

    w = int(canvas_width * args.size_scale)
    h = int(canvas_height * args.size_scale)

    pfilter = pydiffvg.PixelFilter(type = diffvg.FilterType.box,
                                   radius = mx.array(0.5))

    use_prefiltering = False
    num_samples_x = 16
    num_samples_y = 16
    render = pydiffvg.RenderFunction.apply

    def render_scene():
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups,
            filter = pfilter,
            use_prefiltering = use_prefiltering)
        return np.array(render(w, # width
                               h, # height
                               num_samples_x, # num_samples_x
                               num_samples_y, # num_samples_y
                               0, # seed
                               None, # background_image
                               *scene_args))

    img = render_scene()
    pydiffvg.imwrite(img, 'results/finite_difference_comp/img.png', gamma=1.0)

    epsilon = 0.1
    def perturb_scene(axis, epsilon):
        offset = np.zeros(2, dtype = np.float32)
        offset[axis] = epsilon
        shapes[2].points = shapes[2].points + mx.array(offset)

    perturb_scene(0, epsilon)
    img0 = render_scene()

    forward_diff = (img0 - img) / (epsilon)
    forward_diff = forward_diff.sum(axis = 2)
    x_diff_max = 1.5
    x_diff_min = -1.5
    print(forward_diff.max())
    print(forward_diff.min())
    pydiffvg.imwrite(viridis(normalize(forward_diff, x_diff_min, x_diff_max)),
                     'results/finite_difference_comp/shared_edge_forward_diff.png', gamma=1.0)

    perturb_scene(0, -2 * epsilon)
    img1 = render_scene()
    backward_diff = (img - img1) / (epsilon)
    backward_diff = backward_diff.sum(axis = 2)
    print(backward_diff.max())
    print(backward_diff.min())
    pydiffvg.imwrite(viridis(normalize(backward_diff, x_diff_min, x_diff_max)),
                     'results/finite_difference_comp/shared_edge_backward_diff.png', gamma=1.0)
    perturb_scene(0, epsilon)

    num_samples_x = 4
    num_samples_y = 4
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
    pydiffvg.imwrite(viridis(normalize(img_grad[:, :, 0], x_diff_min, x_diff_max)),
                     'results/finite_difference_comp/ours_x_diff.png', gamma=1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("svg_file", help="source SVG path")
    parser.add_argument("--size_scale", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
