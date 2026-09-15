import pydiffvg
import mlx.core as mx
import mlx.optimizers as optim
import skimage
import numpy as np

canvas_width = 256
canvas_height = 256
circle = pydiffvg.Circle(radius = mx.array(40.0),
                         center = mx.array([128.0, 128.0]))
shapes = [circle]
circle_group = pydiffvg.ShapeGroup(shape_ids = mx.array([0]),
    fill_color = mx.array([0.3, 0.6, 0.3, 1.0]))
shape_groups = [circle_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)

render = pydiffvg.RenderFunction.apply
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None,
             *scene_args)
img = img / 256 # Normalize SDF to [0, 1]
pydiffvg.imwrite(img, 'results/single_circle_sdf/target.png')
target = mx.stop_gradient(img)

# Move the circle to produce initial guess
# normalize radius & center for easier learning rate
params = {
    'radius_n': mx.array(20.0 / 256.0),
    'center_n': mx.array([108.0 / 256.0, 138.0 / 256.0]),
    'color': mx.array([0.3, 0.2, 0.8, 1.0]),
}

def set_params(params):
    circle.radius = params['radius_n'] * 256
    circle.center = params['center_n'] * 256
    circle_group.fill_color = params['color']

set_params(params)
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None,
             *scene_args)
img = img / 256 # Normalize SDF to [0, 1]
pydiffvg.imwrite(img, 'results/single_circle_sdf/init.png')

def loss_fn(params, t):
    # Forward pass: render the image.
    set_params(params)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        output_type = pydiffvg.OutputType.sdf)
    img = render(256,   # width
                 256,   # height
                 2,     # num_samples_x
                 2,     # num_samples_y
                 t+1,   # seed
                 None,
                 *scene_args)
    img = img / 256 # Normalize SDF to [0, 1]
    # Compute the loss function. Here it is L2.
    loss = mx.sum((img - target) ** 2)
    return loss, img

loss_and_grad = mx.value_and_grad(loss_fn)

# Optimize for radius & center
optimizer = optim.Adam(learning_rate=1e-2, bias_correction=True)
# Run 100 Adam iterations.
for t in range(100):
    print('iteration:', t)
    (loss, img), grads = loss_and_grad(params, t)
    # Save the intermediate render.
    pydiffvg.imwrite(img, 'results/single_circle_sdf/iter_{}.png'.format(t), gamma=2.2)
    print('loss:', loss.item())

    # Print the gradients
    print('radius.grad:', grads['radius_n'])
    print('center.grad:', grads['center_n'])
    print('color.grad:', grads['color'])

    # Take a gradient descent step.
    optimizer.update(params, grads)
    mx.eval(params, optimizer.state)
    set_params(params)
    # Print the current params.
    print('radius:', circle.radius)
    print('center:', circle.center)
    print('color:', circle_group.fill_color)

# Render the final result.
set_params(params)
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)
img = render(256,   # width
             256,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             102,    # seed
             None,
             *scene_args)
img = img / 256 # Normalize SDF to [0, 1]
# Save the images and differences.
pydiffvg.imwrite(img, 'results/single_circle_sdf/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_circle_sdf/iter_%d.png", "-vb", "20M",
    "results/single_circle_sdf/out.mp4"])
