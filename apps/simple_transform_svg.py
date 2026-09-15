import pydiffvg
import mlx.core as mx
from PIL import Image
import numpy as np
import skimage.transform
import math

def inv_exp(a,x,xpow=1):
    return pow(a,pow(1.-x,xpow))

# visdom is optional: plot the size and the loss when it is installed.
try:
    import visdom
    vis=visdom.Visdom(port=8080)
except ImportError:
    vis=None

def resize(img, size):
    """ Bilinear resize of an HxWxC image to size=(H, W). """
    img=np.asarray(img, dtype=np.float32)
    if tuple(img.shape[0:2])==tuple(size):
        return img
    return skimage.transform.resize(img, (size[0], size[1], img.shape[2]), order=1,
                                    mode='edge', anti_aliasing=False).astype(np.float32)

settings=pydiffvg.SvgOptimizationSettings()
settings.global_override(["optimize_color"],False)
settings.global_override(["optimize_alpha"],False)
settings.global_override(["gradients","optimize_color"],False)
settings.global_override(["gradients","optimize_alpha"],False)
settings.global_override(["gradients","optimize_stops"],False)
settings.global_override(["gradients","optimize_location"],False)
settings.global_override(["optimizer"],"Adam")
settings.global_override(["paths","optimize_points"],False)
settings.global_override(["transforms","transform_lr"],1e-2)
settings.undefault("linearGradient3152")
settings.retrieve("linearGradient3152")[0]["transforms"]["optimize_transforms"]=False

#optim=pydiffvg.OptimizableSvg("note_small.svg",settings,verbose=True)
optim=pydiffvg.OptimizableSvg("heart_green.svg",settings,verbose=True)

#img=np.asarray(Image.open("note_transformed.png"),dtype=np.float32)/255.
img=np.asarray(Image.open("heart_green_90.png"),dtype=np.float32)/255.

name="heart_green_90"

pydiffvg.imwrite(img, 'results/simple_transform_svg/target.png')
target = img

img=optim.render()
pydiffvg.imwrite(img, 'results/simple_transform_svg/init.png')

def printimg(optim):
    img=optim.render()
    comp = mx.stop_gradient(img)
    bg = mx.array([[[1., 1., 1.]]])
    comprgb = comp[:, :, 0:3]
    compalpha = mx.expand_dims(comp[:, :, 3], 2)
    comp = comprgb * compalpha \
           + bg * (1 - compalpha)
    return comp

def comp_grad(img, tgt, it, sz):
    dif=np.array(img)-tgt

    cdif=np.abs(dif)
    cdif[:,:,3]=1.

    resdif=np.abs(resize(cdif,sz))
    pydiffvg.imwrite(resdif[:,:,0:4], 'results/simple_transform_svg/dif_{:04}.png'.format(it))

    padded=np.pad(dif,[(1,1),(1,1),(0,0)],mode='edge')
    #print(padded[:-2,:,:].shape)
    grad_x=(padded[:-2,:,:]-padded[2:,:,:])[:,1:-1,:]
    grad_y=(padded[:,:-2,:]-padded[:,2:,:])[1:-1,:,:]

    resshape=dif.shape
    resshape=(resshape[0],resshape[1],2)
    res=np.zeros(resshape)

    for x in range(resshape[0]):
        for y in range(resshape[1]):
            A=np.concatenate((grad_x[x,y,:][:,np.newaxis],grad_y[x,y,:][:,np.newaxis]),axis=1)
            b=-dif[x,y,:]
            v=np.linalg.lstsq(np.dot(A.T,A),np.dot(A.T,b),rcond=None)
            res[x,y,:]=v[0]

    return res

def print_gradimg(gradimg,it,shape=None):
    out=np.zeros((gradimg.shape[0],gradimg.shape[1],3),dtype=np.float32)
    for x in range(gradimg.shape[0]):
        for y in range(gradimg.shape[1]):
            vec=(gradimg[x,y,:].clip(min=-1,max=1)/2)+.5
            out[x,y,:]=[vec[0],vec[1],0]

    if shape is not None:
        out=resize(out,shape)
    pydiffvg.imwrite(out, 'results/simple_transform_svg/grad_{:04}.png'.format(it))

def loss_fn(optim, seed, aux):
    img = optim.render(seed=seed, scale=None)
    sz=img.shape[0:2]
    restgt=mx.array(resize(target,sz))
    aux['img']=mx.stop_gradient(img)
    aux['restgt']=restgt
    # Compute the loss function. Here it is L2.
    return mx.mean((img - restgt) ** 2)

loss_and_grad = optim.value_and_grad(loss_fn)

# Run 1000 Adam iterations.
for t in range(1000):
    print('iteration:', t)
    with open('results/simple_transform_svg/viter_{:04}.svg'.format(t),"w") as f:
        f.write(optim.write_xml())
    scale=inv_exp(1/16,math.pow(t/1000,1),0.5)
    #print(scale)
    aux={}
    loss, grads = loss_and_grad(t + 1, aux)
    img = aux['img']
    if vis is not None:
        vis.line(np.array([img.shape[0]]), X=np.array([t]), win=name + " size", update="append",
                 opts={"title": name + " size"})

    gradimg=comp_grad(img, np.array(aux['restgt']), t, target.shape[0:2])
    print_gradimg(gradimg,t,target.shape[0:2])
    print('loss:', loss.item())
    if vis is not None:
        vis.line(np.array([loss.item()]), X=np.array([t]), win=name+" loss", update="append",
                 opts={"title": name + " loss"})

    # Take a gradient descent step.
    optim.step(grads)

    # Save the intermediate render.
    comp=printimg(optim)
    pydiffvg.imwrite(comp, 'results/simple_transform_svg/iter_{:04}.png'.format(t))


# Render the final result.

img = optim.render()
# Save the images and differences.
pydiffvg.imwrite(img, 'results/simple_transform_svg/final.png')
with open('results/simple_transform_svg/final.svg', "w") as f:
    f.write(optim.write_xml())

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/simple_transform_svg/iter_%04d.png", "-vb", "20M",
    "results/simple_transform_svg/out.mp4"])

call(["ffmpeg", "-framerate", "24", "-i",
    "results/simple_transform_svg/grad_%04d.png", "-vb", "20M",
    "results/simple_transform_svg/out_grad.mp4"])
