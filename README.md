# diffvg-mlx
Differentiable Rasterizer for Vector Graphics, ported to [MLX](https://github.com/ml-explore/mlx)

diffvg is a differentiable rasterizer for 2D vector graphics: it renders circles, ellipses, rectangles, polygons and Bézier paths with solid or gradient fills, and computes gradients of the image with respect to the scene parameters. See the [project page](https://people.csail.mit.edu/tzumao/diffvg) for more information.

This repository is a fork of [BachiLi/diffvg](https://github.com/BachiLi/diffvg) by Tzu-Mao Li and colleagues. All credit for the rasterizer and the algorithms goes to the original authors. The fork **replaces the PyTorch and TensorFlow bindings** with MLX bindings for Apple Silicon. The C++ core renders on the CPU with its own thread pool; the MLX layer passes scene data to the core and exposes rendering as a differentiable MLX function.

![teaser](https://user-images.githubusercontent.com/951021/92184822-2a0bc500-ee20-11ea-81a6-f26af2d120f4.jpg)

![circle](https://user-images.githubusercontent.com/951021/63556018-0b2ddf80-c4f8-11e9-849c-b4ecfcb9a865.gif)
![ellipse](https://user-images.githubusercontent.com/951021/63556021-0ec16680-c4f8-11e9-8fc6-8b34de45b8be.gif)
![rect](https://user-images.githubusercontent.com/951021/63556028-12ed8400-c4f8-11e9-8072-81702c9193e1.gif)
![polygon](https://user-images.githubusercontent.com/951021/63980999-1e99f700-ca72-11e9-9786-1cba14d2d862.gif)
![curve](https://user-images.githubusercontent.com/951021/64042667-3d9e9480-cb17-11e9-88d8-2f7b9da8b8ab.gif)
![path](https://user-images.githubusercontent.com/951021/64070625-7a52b480-cc19-11e9-9380-eac02f56f693.gif)
![gradient](https://user-images.githubusercontent.com/951021/64898668-da475300-d63c-11e9-917a-825b94be0710.gif)
![circle_outline](https://user-images.githubusercontent.com/951021/65125594-84f7a280-d9aa-11e9-8bc4-669fd2eff2f4.gif)
![ellipse_transform](https://user-images.githubusercontent.com/951021/67149013-06b54700-f25b-11e9-91eb-a61171c6d4a4.gif)

# Install
The build needs CMake, a C++ compiler and [uv](https://docs.astral.sh/uv/). The thrust submodule is required, so clone with submodules:
```
git clone --recurse-submodules https://github.com/skalyan91/diffvg-mlx.git
cd diffvg-mlx
uv venv
uv pip install -e .
```
In an existing clone, fetch the submodules with `git submodule update --init --recursive`.

The package is built with scikit-build-core (see `pyproject.toml`) and installs `mlx`, `numpy`, `svgpathtools`, `scikit-image` and `cssutils`. Some apps need extra packages; see the notes below.

# Usage
Shapes and shape groups hold `mx.array` fields. To optimise them, write the render and the loss as one function of a parameter dictionary, and differentiate that function with `mx.value_and_grad`:
```python
import mlx.core as mx
import mlx.optimizers as optim
import pydiffvg

w, h = 256, 256
circle = pydiffvg.Circle(radius = mx.array(40.0), center = mx.array([128.0, 128.0]))
group = pydiffvg.ShapeGroup(shape_ids = mx.array([0]),
                            fill_color = mx.array([0.3, 0.6, 0.3, 1.0]))
shapes, shape_groups = [circle], [group]
render = pydiffvg.RenderFunction.apply

def draw():
    scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
    return render(w, h, 2, 2, 0, None, *scene_args)  # width, height, samples x/y, seed, background

target = draw()

params = {"radius": mx.array(20.0), "center": mx.array([108.0, 138.0])}
def loss_fn(params):
    circle.radius = params["radius"]
    circle.center = params["center"]
    return mx.mean((draw() - target) ** 2)

loss_and_grad = mx.value_and_grad(loss_fn)
optimizer = optim.Adam(learning_rate = 1.0, bias_correction = True)  # match torch.optim.Adam
for t in range(100):
    loss, grads = loss_and_grad(params)
    optimizer.update(params, grads)
    mx.eval(params, optimizer.state)

pydiffvg.imwrite(draw(), "circle.png", gamma = 2.2)
```

To give parameters different learning rates, keep one parameter dictionary and one optimizer per learning rate. Clamp values such as colors with `mx.clip` after each update.

# Run
```
cd apps
```

Optimizing a single circle to a target.
```
python single_circle.py
```

Rendering an SVG file to an image.
```
python render_svg.py imgs/tiger.svg tiger.png
```

Finite difference comparison. The script renders the finite-difference gradients and the gradients from diffvg as images, and prints the correlation between them.
```
finite_difference_comp.py [-h] [--size_scale SIZE_SCALE]
                               [--clamping_factor CLAMPING_FACTOR]
                               [--num_spp NUM_SPP]
                               [--use_prefiltering USE_PREFILTERING]
                               svg_file
```
e.g.,
```
python finite_difference_comp.py imgs/tiger.svg
```

Image vectorization (refining the points and fill colors of an SVG to match an image, with an MSE loss)
```
python refine_svg.py [-h] [--num_iter NUM_ITER] svg target
```
e.g.,
```
python refine_svg.py imgs/flower.svg imgs/flower.jpg
```

Interactive editor (needs `pygame` and `tkinter`)
```
python svg_brush.py
```

Results go to `apps/results/`. Scripts that assemble a video from the iterations call `ffmpeg`.

# Notes on the MLX port
- `pydiffvg.RenderFunction.apply` keeps the call pattern of the PyTorch version, so existing scripts port by swapping tensors for `mx.array`s. `RenderFunction.apply` is the same function as `pydiffvg.render`, a differentiable function built with `mx.custom_function`. `RenderFunction.render_grad` returns the screen-space translation gradient image.
- **Rendering runs on the CPU only.** The CUDA code path is not built, `pydiffvg.get_device()` returns `mx.cpu`, and `pydiffvg.set_use_gpu(True)` raises `NotImplementedError`.
- MLX arrays cannot be updated in place: instead of `x.data.clamp_()` or `x += eps`, compute a new array and assign it back to the shape.
- `refine_svg.py` supports only the MSE loss. The `--use_lpips_loss` flag of upstream relies on `ttools` (PyTorch), so the flag has been removed.
- These apps depend heavily on PyTorch models or libraries and have been removed: `painterly_rendering.py`, `sketch_gan.py`, `style_transfer.py`, `texture_synthesis.py`, `seam_carving.py`, `gaussian_blur.py` and `optimize_pixel_filter.py`. The TensorFlow bindings (`pydiffvg_tensorflow/`) and the `*_tf.py` apps have been removed with TensorFlow support. The vector VAE and GAN code in `apps/generative_models/` has been removed for the same reason.
- Extra packages for some apps: `matplotlib` for the colormaps of `image_compare.py` (the other comparison scripts fall back to a built-in colormap), `pygame` and `tkinter` for `svg_brush.py`, and `Pillow` for `simple_transform_svg.py`.
- `pydiffvg.save_svg` writes a shape group with several subpaths as a single `<path>` element (a compound path), with one `M` command per subpath. Each closed subpath ends with `z`, and the element carries the fill rule of the shape group.

# Upstream bugs fixed in this fork
Rendering and gradients (C++ core):
- **Boundary gradients under a transform** were wrong. Gradients with respect to `shape_to_canvas` were too large by the inverse of the scale (6.5 times for a scale of 0.152), and gradients with respect to points, radii and stroke widths by its square. Rotations and non-uniform scales also gave wrong signs. Boundary sampling now accounts for the arclength Jacobian and the normal of the shape in its own space.
- **Signed distance fields** of circles were all zeros, the closest point on a rectangle was wrong, and ellipses had no distance function. SDF gradients were also too large by the number of samples per pixel (4 times at 2×2 samples), because the backward pass omitted the sample weight, and `max_radius` was ignored for circles and rectangles.
- A gradient used as a **stroke colour** crashed scene construction, and gradients with respect to a radial stroke gradient crashed the backward pass. The stop colours of radial gradients received their gradient twice.
- **Stroked ellipses** failed an assertion.
- Stroke gradients now sample round joins and caps and use the speed of the stroke outline rather than that of the centreline. The gradients of stroked circles, ellipses and rectangles match finite differences to within a few percent; stroked paths with sharp joins can still deviate by 10–20% on small entries.
- Boundary samples on a pixel border are no longer counted in both pixels, which had doubled the gradients of pixel-aligned rectangles.

SVG parsing (`pydiffvg.svg_to_scene`) and saving (`pydiffvg.save_svg`):
- **Radial gradients** take their centre and radius from `cx`, `cy` and `r`, with support for percentages, `gradientUnits` and `gradientTransform`. Before, the centre was always zero and the radius came from the focal attributes. Focal points (`fx`, `fy`, `fr`) and `spreadMethod="reflect"`/`"repeat"` are still not supported; the parser warns when it meets them.
- **Linear gradients** use the SVG defaults and bounding-box units.
- A gradient that inherits from another gradient through `href` no longer modifies the gradient it inherits from.
- The parser reads `<ellipse>`, `<polyline>` and rounded `<rect>` elements. The `x` and `y` attributes of a rectangle were previously ignored.
- Colours accept `rgb()` and `rgba()` with integers, decimals or percentages, `#rgb`/`#rgba`/`#rrggbb`/`#rrggbbaa`, and all 147 named SVG colours without matplotlib. `pydiffvg.OptimizableSvg` uses the same colour parser.
- Groups pass fill, stroke, stroke width, opacity and fill rule on to their children, and `style` takes precedence over classes and presentation attributes, as in the SVG specification. Opacity no longer leaks from one shape to the next.
- Parsing is faster on large files, because arrays are built in NumPy and converted to MLX once per shape.
- `save_svg` writes radial gradients, the transform of each shape group, and `fill-opacity` (instead of `opacity`, which also faded the stroke).

If you use diffvg in your academic work, please cite

```
@article{Li:2020:DVG,
    title = {Differentiable Vector Graphics Rasterization for Editing and Learning},
    author = {Li, Tzu-Mao and Luk\'{a}\v{c}, Michal and Gharbi Micha\"{e}l and Jonathan Ragan-Kelley},
    journal = {ACM Trans. Graph. (Proc. SIGGRAPH Asia)},
    volume = {39},
    number = {6},
    pages = {193:1--193:15},
    year = {2020}
}
```
