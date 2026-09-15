import pydiffvg
import mlx.core as mx
import numpy as np
from finite_difference_comp import viridis

canvas_width, canvas_height = 256, 256
num_control_points = mx.array([1])
points = mx.array([[ 50.0,  30.0], # base
                   [125.0, 400.0], # control point
                   [170.0,  30.0]]) # base
path = pydiffvg.Path(num_control_points = num_control_points,
                     points = points,
                     stroke_width = mx.array([30.0]),
                     is_closed = False,
                     use_distance_approx = False)
shapes = [path]
path_group = pydiffvg.ShapeGroup(shape_ids = mx.array([0]),
                                 fill_color = None,
                                 stroke_color = mx.array([0.5, 0.5, 0.5, 0.5]))
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
img = img / 256.0
img = viridis(np.array(img).squeeze())
pydiffvg.imwrite(img, 'results/quadratic_distance_approx/ref_sdf.png')

scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img, 'results/quadratic_distance_approx/ref_color.png')

shapes[0].use_distance_approx = True
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)
img = render(256, # width
             256, # height
             1,   # num_samples_x
             1,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
img = img / 256.0
img = viridis(np.array(img).squeeze())
pydiffvg.imwrite(img, 'results/quadratic_distance_approx/approx_sdf.png')

scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img, 'results/quadratic_distance_approx/approx_color.png')
