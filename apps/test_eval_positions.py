import pydiffvg
import mlx.core as mx
import mlx.optimizers as optim

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
             None, # background_image
             *scene_args)
img = img / 256 # Normalize SDF to [0, 1]
pydiffvg.imwrite(img, 'results/test_eval_positions/target.png')
target = img

# Move the circle to produce initial guess
# normalize radius & center for easier learning rate
params = {'radius_n': mx.array(20.0 / 256.0),
          'center_n': mx.array([108.0 / 256.0, 138.0 / 256.0]),
          'color': mx.array([0.3, 0.2, 0.8, 1.0])}

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
             None, # background_image
             *scene_args)
img = img / 256 # Normalize SDF to [0, 1]
pydiffvg.imwrite(img, 'results/test_eval_positions/init.png')

def loss_fn(params, eval_positions, seed):
    set_params(params)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        output_type = pydiffvg.OutputType.sdf,
        eval_positions = eval_positions)
    samples = render(256,   # width
                     256,   # height
                     0,     # num_samples_x
                     0,     # num_samples_y
                     seed,  # seed
                     None, # background_image
                     *scene_args)
    samples = samples / 256 # Normalize SDF to [0, 1]
    # Nearest-neighbour lookup of the target at the positions
    # (what grid_sample(mode='nearest') did in the PyTorch version).
    ij = mx.clip(mx.floor(eval_positions).astype(mx.int32), 0, 255)
    target_sampled = target[ij[:, 1], ij[:, 0]]
    return mx.mean((samples - target_sampled) ** 2)

loss_and_grad = mx.value_and_grad(loss_fn)

# Optimize for radius & center
optimizer = optim.Adam(learning_rate=1e-2, bias_correction=True)
# Run 200 Adam iterations.
for t in range(200):
    print('iteration:', t)
    # Evaluate 1000 positions
    eval_positions = mx.random.uniform(shape = (1000, 2)) * 256
    loss, grads = loss_and_grad(params, eval_positions, t + 1)
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
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)
img = render(256,   # width
             256,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             102,    # seed
             None, # background_image
             *scene_args)
img = img / 256 # Normalize SDF to [0, 1]
# Save the images and differences.
pydiffvg.imwrite(img, 'results/test_eval_positions/final.png')
