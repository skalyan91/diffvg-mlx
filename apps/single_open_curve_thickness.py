import pydiffvg
import mlx.core as mx
import mlx.optimizers as optim
import skimage

pydiffvg.set_print_timing(True)

canvas_width, canvas_height = 256, 256
num_control_points = mx.array([2])
points = mx.array([[120.0,  30.0], # base
                   [150.0,  60.0], # control point
                   [ 90.0, 198.0], # control point
                   [ 60.0, 218.0]]) # base
thickness = mx.array([10.0, 5.0, 4.0, 20.0])
path = pydiffvg.Path(num_control_points = num_control_points,
                     points = points,
                     is_closed = False,
                     stroke_width = thickness)
shapes = [path]
path_group = pydiffvg.ShapeGroup(shape_ids = mx.array([0]),
                                 fill_color = None,
                                 stroke_color = mx.array([0.6, 0.3, 0.6, 0.8]))
shape_groups = [path_group]
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
pydiffvg.imwrite(img, 'results/single_open_curve_thickness/target.png', gamma=2.2)
target = mx.stop_gradient(img)

# Move the path to produce initial guess
# normalize points for easier learning rate
params = {
    'points_n': mx.array([[100.0/256.0,  40.0/256.0], # base
                          [155.0/256.0,  65.0/256.0], # control point
                          [100.0/256.0, 180.0/256.0], # control point
                          [ 65.0/256.0, 238.0/256.0]]), # base
    'thickness_n': mx.array([10.0 / 100.0, 10.0 / 100.0, 10.0 / 100.0, 10.0 / 100.0]),
    'stroke_color': mx.array([0.4, 0.7, 0.5, 0.5]),
}

def set_params(params):
    path.points = params['points_n'] * 256
    path.stroke_width = params['thickness_n'] * 100
    path_group.stroke_color = params['stroke_color']

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
pydiffvg.imwrite(img, 'results/single_open_curve_thickness/init.png', gamma=2.2)

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

# Optimize
optimizer = optim.Adam(learning_rate=1e-2, bias_correction=True)
# Run 200 Adam iterations.
for t in range(200):
    print('iteration:', t)
    (loss, img), grads = loss_and_grad(params, t)
    # Save the intermediate render.
    pydiffvg.imwrite(img, 'results/single_open_curve_thickness/iter_{}.png'.format(t), gamma=2.2)
    print('loss:', loss.item())

    # Print the gradients
    print('points_n.grad:', grads['points_n'])
    print('thickness_n.grad:', grads['thickness_n'])
    print('stroke_color.grad:', grads['stroke_color'])

    # Take a gradient descent step.
    optimizer.update(params, grads)
    mx.eval(params, optimizer.state)
    set_params(params)
    # Print the current params.
    print('points:', path.points)
    print('thickness:', path.stroke_width)
    print('stroke_color:', path_group.stroke_color)

# Render the final result.
set_params(params)
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256,   # width
             256,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             202,    # seed
             None, # background_image
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img, 'results/single_open_curve_thickness/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_open_curve_thickness/iter_%d.png", "-vb", "20M",
    "results/single_open_curve_thickness/out.mp4"])
