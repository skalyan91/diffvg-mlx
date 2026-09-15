import pydiffvg
import mlx.core as mx
import mlx.optimizers as optim
import skimage
import numpy as np

canvas_width, canvas_height = 256, 256
ellipse = pydiffvg.Ellipse(radius = mx.array([60.0, 30.0]),
                           center = mx.array([128.0, 128.0]))
shapes = [ellipse]
ellipse_group = pydiffvg.ShapeGroup(\
    shape_ids = mx.array([0]),
    fill_color = mx.array([0.3, 0.6, 0.3, 1.0]),
    shape_to_canvas = mx.eye(3))
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
pydiffvg.imwrite(img, 'results/single_ellipse_transform/target.png', gamma=2.2)
target = mx.stop_gradient(img)

# Affine transform the ellipse to produce initial guess
params = {
    'color': mx.array([0.3, 0.2, 0.8, 1.0]),
    'affine': mx.array([[1.3, 0.2, 0.1],
                        [0.2, 0.6, 0.3]]),
}

def set_params(params):
    ellipse_group.fill_color = params['color']
    ellipse_group.shape_to_canvas = mx.concatenate(\
        [params['affine'], mx.array([[0.0, 0.0, 1.0]])], axis=0)

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
pydiffvg.imwrite(img, 'results/single_ellipse_transform/init.png', gamma=2.2)

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
# Run 150 Adam iterations.
for t in range(150):
    print('iteration:', t)
    (loss, img), grads = loss_and_grad(params, t)
    # Save the intermediate render.
    pydiffvg.imwrite(img, 'results/single_ellipse_transform/iter_{}.png'.format(t), gamma=2.2)
    print('loss:', loss.item())

    # Print the gradients
    print('color.grad:', grads['color'])
    print('affine.grad:', grads['affine'])

    # Take a gradient descent step.
    optimizer.update(params, grads)
    mx.eval(params, optimizer.state)
    set_params(params)
    # Print the current params.
    print('color:', ellipse_group.fill_color)
    print('affine:', params['affine'])

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
pydiffvg.imwrite(img, 'results/single_ellipse_transform/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_ellipse_transform/iter_%d.png", "-vb", "20M",
    "results/single_ellipse_transform/out.mp4"])
