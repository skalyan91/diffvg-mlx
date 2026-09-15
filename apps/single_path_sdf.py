import pydiffvg
import mlx.core as mx
import mlx.optimizers as optim
import skimage

canvas_width, canvas_height = 510, 510
# https://www.flaticon.com/free-icon/black-plane_61212#term=airplane&page=1&position=8
shapes = pydiffvg.from_svg_path('M510,255c0-20.4-17.85-38.25-38.25-38.25H331.5L204,12.75h-51l63.75,204H76.5l-38.25-51H0L25.5,255L0,344.25h38.25l38.25-51h140.25l-63.75,204h51l127.5-204h140.25C492.15,293.25,510,275.4,510,255z')
path_group = pydiffvg.ShapeGroup(shape_ids = mx.array([0]),
                                 fill_color = mx.array([0.3, 0.6, 0.3, 1.0]))
shape_groups = [path_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)

render = pydiffvg.RenderFunction.apply
img = render(510, # width
             510, # height
             1,   # num_samples_x
             1,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
img = img / 510 # Normalize SDF to [0, 1]
pydiffvg.imwrite(img, 'results/single_path_sdf/target.png', gamma=1.0)
target = mx.stop_gradient(img)

# Move the path to produce initial guess
# normalize points for easier learning rate
noise = mx.random.uniform(0.0, 1.0, shapes[0].points.shape)
params = {
    'points_n': (shapes[0].points + (noise * 60 - 30)) / 510.0,
    'color': mx.array([0.3, 0.2, 0.5, 1.0]),
}

def set_params(params):
    shapes[0].points = params['points_n'] * 510
    path_group.fill_color = params['color']

set_params(params)
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)
img = render(510, # width
             510, # height
             1,   # num_samples_x
             1,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
img = img / 510 # Normalize SDF to [0, 1]
pydiffvg.imwrite(img, 'results/single_path_sdf/init.png', gamma=1.0)

def loss_fn(params, t):
    # Forward pass: render the image.
    set_params(params)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        output_type = pydiffvg.OutputType.sdf)
    img = render(510,   # width
                 510,   # height
                 1,     # num_samples_x
                 1,     # num_samples_y
                 t+1,   # seed
                 None, # background_image
                 *scene_args)
    img = img / 510 # Normalize SDF to [0, 1]
    # Compute the loss function. Here it is L2.
    loss = mx.sum((img - target) ** 2)
    return loss, img

loss_and_grad = mx.value_and_grad(loss_fn)

# Optimize
optimizer = optim.Adam(learning_rate=1e-2, bias_correction=True)
# Run 100 Adam iterations.
for t in range(100):
    print('iteration:', t)
    (loss, img), grads = loss_and_grad(params, t)
    # Save the intermediate render.
    pydiffvg.imwrite(img, 'results/single_path_sdf/iter_{}.png'.format(t), gamma=1.0)
    print('loss:', loss.item())

    # Print the gradients
    print('points_n.grad:', grads['points_n'])
    print('color.grad:', grads['color'])

    # Take a gradient descent step.
    optimizer.update(params, grads)
    mx.eval(params, optimizer.state)
    set_params(params)
    # Print the current params.
    print('points:', shapes[0].points)
    print('color:', path_group.fill_color)

# Render the final result.
set_params(params)
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)
img = render(510,   # width
             510,   # height
             1,     # num_samples_x
             1,     # num_samples_y
             102,    # seed
             None, # background_image
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img, 'results/single_path_sdf/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_path_sdf/iter_%d.png", "-vb", "20M",
    "results/single_path_sdf/out.mp4"])
