import pydiffvg
import mlx.core as mx
import mlx.optimizers as optim
import skimage
import numpy as np

canvas_width, canvas_height = 256, 256
num_control_points = mx.array([2])
# points = mx.array([[120.0,  30.0], # base
#                    [150.0,  60.0], # control point
#                    [ 90.0, 198.0], # control point
#                    [ 60.0, 218.0], # base
#                    [ 90.0, 180.0], # control point
#                    [200.0,  65.0], # control point
#                    [210.0,  98.0], # base
#                    [220.0,  70.0], # control point
#                    [130.0,  55.0]]) # control point
points = mx.array([[ 20.0, 128.0], # base
                   [ 50.0, 128.0], # control point
                   [170.0, 128.0], # control point
                   [200.0, 128.0]]) # base
path = pydiffvg.Path(num_control_points = num_control_points,
                     points = points,
                     is_closed = False,
                     stroke_width = mx.array(10.0))
shapes = [path]
path_group = pydiffvg.ShapeGroup(shape_ids = mx.array([0]),
                                 fill_color = None,
                                 stroke_color = mx.array([0.3, 0.6, 0.3, 1.0]))
shape_groups = [path_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)

render = pydiffvg.RenderFunction.apply
img = render(256, # width
             256, # height
             1,   # num_samples_x
             1,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)

path.points[:, 1] += 1e-3
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)
img2 = render(256, # width
              256, # height
              1,   # num_samples_x
              1,   # num_samples_y
              0,   # seed
              None, # background_image
              *scene_args)

# diff = img2 - img
# diff = diff[:, :, 0] / 1e-3
# import matplotlib.pyplot as plt
# plt.imshow(diff)
# plt.show()

# # The output image is in linear RGB space. Do Gamma correction before saving the image.
# pydiffvg.imwrite(img, 'results/single_curve_sdf/target.png', gamma=1.0)
# target = mx.stop_gradient(img)

render_grad = pydiffvg.RenderFunction.render_grad
img = render_grad(mx.ones((256, 256, 1)), # grad_img
                  256, # width
                  256, # height
                  1,   # num_samples_x
                  1,   # num_samples_y
                  0,   # seed
                  None, # background_image
                  *scene_args)
img = img[:, :, 0]
import matplotlib.pyplot as plt
plt.imshow(np.array(img))
plt.show()

# # Move the path to produce initial guess
# # normalize points for easier learning rate
# # points_n = mx.array([[100.0/256.0,  40.0/256.0], # base
# #                      [155.0/256.0,  65.0/256.0], # control point
# #                      [100.0/256.0, 180.0/256.0], # control point
# #                      [ 65.0/256.0, 238.0/256.0], # base
# #                      [100.0/256.0, 200.0/256.0], # control point
# #                      [170.0/256.0,  55.0/256.0], # control point
# #                      [220.0/256.0, 100.0/256.0], # base
# #                      [210.0/256.0,  80.0/256.0], # control point
# #                      [140.0/256.0,  60.0/256.0]]) # control point
# params = {
#     'points_n': mx.array([[118.4274/256.0,  32.0159/256.0],
#                           [174.9657/256.0,  28.1877/256.0],
#                           [ 87.6629/256.0, 175.1049/256.0],
#                           [ 57.8093/256.0, 232.8987/256.0],
#                           [ 80.1829/256.0, 165.4280/256.0],
#                           [197.3640/256.0,  83.4058/256.0],
#                           [209.3676/256.0,  97.9176/256.0],
#                           [219.1048/256.0,  72.0000/256.0],
#                           [143.1226/256.0,  57.0636/256.0]]),
#     'color': mx.array([0.3, 0.2, 0.5, 1.0]),
# }
#
# def set_params(params):
#     path.points = params['points_n'] * 256
#     path_group.fill_color = params['color']
#
# set_params(params)
# scene_args = pydiffvg.RenderFunction.serialize_scene(\
#     canvas_width, canvas_height, shapes, shape_groups,
#     output_type = pydiffvg.OutputType.sdf)
# img = render(256, # width
#              256, # height
#              1,   # num_samples_x
#              1,   # num_samples_y
#              1,   # seed
#              None, # background_image
#              *scene_args)
# img /= 256.0
# pydiffvg.imwrite(img, 'results/single_curve_sdf/init.png', gamma=1.0)
#
# def loss_fn(params, t):
#     # Forward pass: render the image.
#     set_params(params)
#     scene_args = pydiffvg.RenderFunction.serialize_scene(\
#         canvas_width, canvas_height, shapes, shape_groups,
#         output_type = pydiffvg.OutputType.sdf)
#     img = render(256,   # width
#                  256,   # height
#                  1,     # num_samples_x
#                  1,     # num_samples_y
#                  t+1,   # seed
#                  None, # background_image
#                  *scene_args)
#     img = img / 256.0
#     # Compute the loss function. Here it is L2.
#     loss = mx.sum((img - target) ** 2)
#     return loss, img
#
# loss_and_grad = mx.value_and_grad(loss_fn)
#
# # Optimize
# optimizer = optim.Adam(learning_rate=1e-3, bias_correction=True)
# # Run 100 Adam iterations.
# for t in range(2):
#     print('iteration:', t)
#     (loss, img), grads = loss_and_grad(params, t)
#     # Save the intermediate render.
#     pydiffvg.imwrite(img, 'results/single_curve_sdf/iter_{}.png'.format(t), gamma=1.0)
#     print('loss:', loss.item())
#
#     # Print the gradients
#     print('points_n.grad:', grads['points_n'])
#     print('color.grad:', grads['color'])
#
#     # Take a gradient descent step.
#     optimizer.update(params, grads)
#     mx.eval(params, optimizer.state)
#     set_params(params)
#     # Print the current params.
#     print('points:', path.points)
#     print('color:', path_group.fill_color)
# exit()
#
# # Render the final result.
# scene_args = pydiffvg.RenderFunction.serialize_scene(\
#     canvas_width, canvas_height, shapes, shape_groups,
#     output_type = pydiffvg.OutputType.sdf)
# img = render(256,   # width
#              256,   # height
#              1,     # num_samples_x
#              1,     # num_samples_y
#              102,    # seed
#              None, # background_image
#              *scene_args)
# img /= 256.0
# # Save the images and differences.
# pydiffvg.imwrite(img, 'results/single_curve_sdf/final.png', gamma=1.0)
#
# # Convert the intermediate renderings to a video.
# from subprocess import call
# call(["ffmpeg", "-framerate", "24", "-i",
#     "results/single_curve_sdf/iter_%d.png", "-vb", "20M",
#     "results/single_curve_sdf/out.mp4"])
