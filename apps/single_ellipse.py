import pydiffvg
import mlx.core as mx
import mlx.optimizers as optim
import skimage
import numpy as np

canvas_width, canvas_height = 256, 256
ellipse = pydiffvg.Ellipse(radius = mx.array([60.0, 30.0]),
                           center = mx.array([128.0, 128.0]))
shapes = [ellipse]
ellipse_group = pydiffvg.ShapeGroup(shape_ids = mx.array([0]),
                                    fill_color = mx.array([0.3, 0.6, 0.3, 1.0]))
shape_groups = [ellipse_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

render = pydiffvg.RenderFunction.apply
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img, 'results/single_ellipse/target.png', gamma=2.2)
target = mx.stop_gradient(img)

# Move the ellipse to produce initial guess
# normalize radius & center for easier learning rate
params = {
    'radius_n': mx.array([20.0 / 256.0, 40.0 / 256.0]),
    'center_n': mx.array([108.0 / 256.0, 138.0 / 256.0]),
    'color': mx.array([0.3, 0.2, 0.8, 1.0]),
}

def set_params(params):
    ellipse.radius = params['radius_n'] * 256
    ellipse.center = params['center_n'] * 256
    ellipse_group.fill_color = params['color']

set_params(params)
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img, 'results/single_ellipse/init.png', gamma=2.2)

def loss_fn(params, t):
    # Forward pass: render the image.
    set_params(params)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(256,   # width
                 256,   # height
                 2,     # num_samples_x
                 2,     # num_samples_y
                 t+1,   # seed
                 None, # background_image
                 *scene_args)
    # Compute the loss function. Here it is L2.
    loss = mx.sum((img - target) ** 2)
    return loss, img

loss_and_grad = mx.value_and_grad(loss_fn)

# Optimize for radius & center
optimizer = optim.Adam(learning_rate=1e-2, bias_correction=True)
# Run 50 Adam iterations.
for t in range(50):
    print('iteration:', t)
    (loss, img), grads = loss_and_grad(params, t)
    # Save the intermediate render.
    pydiffvg.imwrite(img, 'results/single_ellipse/iter_{}.png'.format(t), gamma=2.2)
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
    print('radius:', ellipse.radius)
    print('center:', ellipse.center)
    print('color:', ellipse_group.fill_color)

# Render the final result.
set_params(params)
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256,   # width
             256,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             52,    # seed
             None, # background_image
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img, 'results/single_ellipse/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_ellipse/iter_%d.png", "-vb", "20M",
    "results/single_ellipse/out.mp4"])
