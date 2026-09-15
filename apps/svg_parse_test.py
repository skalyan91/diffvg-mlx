import pydiffvg
import numpy as np
import mlx.core as mx

from pydiffvg import OptimizableSvg

"""
for x in range(100000):
    inmat=np.eye(3)
    inmat[0:2,:]=(np.random.rand(2,3)-0.5)*2
    decomp=OptimizableSvg.TransformTools.decompose(inmat)
    outmat=np.array(OptimizableSvg.TransformTools.recompose(mx.array(decomp[0],dtype=mx.float32),mx.array(decomp[1],dtype=mx.float32),mx.array(decomp[2],dtype=mx.float32),mx.array(decomp[3],dtype=mx.float32)))
    dif=np.linalg.norm(inmat-outmat)
    if dif > 1e-3:
        print(dif)
        print(inmat)
        print(outmat)
        print(decomp)"""


infile='./imgs/note_small.svg'


canvas_width, canvas_height, shapes, shape_groups = \
	pydiffvg.svg_to_scene(infile)
scene_args = pydiffvg.RenderFunction.serialize_scene(\
	canvas_width, canvas_height, shapes, shape_groups)
render = pydiffvg.RenderFunction.apply
img = render(canvas_width, # width
             canvas_height, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img, 'test_old.png', gamma=1.0)

#optim=OptimizableSvg('linux.svg',verbose=True)
optim=OptimizableSvg(infile,verbose=True)

scene=optim.build_scene()
scene_args = pydiffvg.RenderFunction.serialize_scene(*scene)
render = pydiffvg.RenderFunction.apply
img = render(scene[0], # width
             scene[1], # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)



with open("resaved.svg","w") as f:
    f.write(optim.write_xml())

# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img, 'test_new.png', gamma=1.0)

print("Done!")
