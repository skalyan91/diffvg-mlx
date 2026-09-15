import pydiffvg
import argparse
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import skimage.io

gamma = 1.0

def main(args):
    target = np.asarray(skimage.io.imread(args.target), dtype = np.float32) / 255.0
    if target.ndim == 2:
        target = np.stack([target] * 3, axis = 2)
    target = mx.array(target[:, :, :3]) ** gamma

    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)

    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None, # bg
                 *scene_args)
    # The output image is in linear RGB space. Do Gamma correction before saving the image.
    pydiffvg.imwrite(img, 'results/refine_svg/init.png', gamma=gamma)

    # Collect the variables. Several shape groups can share one fill color
    # array, so deduplicate the colors by object identity.
    point_shapes = [s for s in shapes if hasattr(s, 'points')]
    color_ids = {}
    group_color_index = []
    colors = []
    for group in shape_groups:
        c = group.fill_color
        if isinstance(c, mx.array):
            if id(c) not in color_ids:
                color_ids[id(c)] = len(colors)
                colors.append(c)
            group_color_index.append(color_ids[id(c)])
        else:
            group_color_index.append(None)
    points_params = {'points': [s.points for s in point_shapes]}
    color_params = {'colors': colors}

    def set_params(points_params, color_params):
        for s, p in zip(point_shapes, points_params['points']):
            s.points = p
        for group, i in zip(shape_groups, group_color_index):
            if i is not None:
                group.fill_color = color_params['colors'][i]

    def loss_fn(points_params, color_params):
        set_params(points_params, color_params)
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, # width
                     canvas_height, # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     0,   # seed
                     None, # bg
                     *scene_args)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + \
            mx.ones((img.shape[0], img.shape[1], 3)) * (1 - img[:, :, 3:4])
        loss = mx.mean((img - target) ** 2)
        return loss, img

    loss_and_grad = mx.value_and_grad(loss_fn, argnums = (0, 1))

    # Optimize
    # bias_correction matches torch.optim.Adam; without it the first steps are ~3x larger
    points_optim = optim.Adam(learning_rate=1.0, bias_correction=True)
    color_optim = optim.Adam(learning_rate=0.01, bias_correction=True)

    # Adam iterations.
    for t in range(args.num_iter):
        print('iteration:', t)
        (loss, img), (points_grad, color_grad) = \
            loss_and_grad(points_params, color_params)
        # Save the intermediate render.
        pydiffvg.imwrite(img, 'results/refine_svg/iter_{}.png'.format(t), gamma=gamma)
        print('render loss:', loss.item())

        # Take a gradient descent step.
        points_optim.update(points_params, points_grad)
        color_optim.update(color_params, color_grad)
        color_params['colors'] = [mx.clip(c, 0.0, 1.0) for c in color_params['colors']]
        mx.eval(points_params, color_params, points_optim.state, color_optim.state)
        set_params(points_params, color_params)

        if t % 10 == 0 or t == args.num_iter - 1:
            pydiffvg.save_svg('results/refine_svg/iter_{}.svg'.format(t),
                              canvas_width, canvas_height, shapes, shape_groups)

    # Render the final result.
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None, # bg
                 *scene_args)
    # Save the final render.
    pydiffvg.imwrite(img, 'results/refine_svg/final.png', gamma=gamma)
    # Convert the intermediate renderings to a video.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        "results/refine_svg/iter_%d.png", "-vb", "20M",
        "results/refine_svg/out.mp4"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", help="source SVG path")
    parser.add_argument("target", help="target image path")
    parser.add_argument("--num_iter", type=int, default=250)
    args = parser.parse_args()
    main(args)
