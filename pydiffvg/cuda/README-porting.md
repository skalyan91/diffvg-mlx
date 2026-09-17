# diffvg CUDA port — porting notes

CUDA translations of `pydiffvg/metal/*.metal`, for a CUDA backend dispatching
through `mx.fast.cuda_kernel`. Same function names, same signatures (modulo
address-space qualifiers), same semantics.

| file | translated from | contents |
|---|---|---|
| `common.cu` | `common.metal` | vector types, scalar helpers, `SceneView`, ip constants, pool accessors, AABB/BVH, `Mat3` + `xform_pt`/`d_xform_pt`, `GradWrites`, PCG32, filters, `sample_position` |
| `geometry.cu` | `geometry.metal` | solvers, Bernstein root isolation, winding numbers, `is_inside`, `closest_point`, `compute_distance`, `within_distance` |
| `color.cu` | `color.metal` | `sample_color_param`, `collect_fragments`, `sample_color_scene(_eq)`, `BlendState`/`blend_forward`/`blend_backward_step` |
| `backward.cu` | `backward.metal` | `d_sample_color_param`, `gather_d_color`, `cdf_sample`, `sample_boundary` (all shapes), `generate_boundary_sample`, `accumulate_boundary_gradient` |
| `distance_grad.cu` | `distance_grad.metal` | `smoothstep`/`d_smoothstep`, `path_thickness_at(_dt)`, `d_closest_point`, `d_compute_distance` |
| `prefilter.cu` | `prefilter.metal` | `sdf_query`, `pf_collect_fragments`, `pf_blend`, `pf_d_fragment`, `pf_d_filter_weight_dr` |

**Header concatenation order is load-bearing and identical to Metal:**
`common.cu → geometry.cu → color.cu → backward.cu → distance_grad.cu → prefilter.cu`.
The files have no `#include`s of one another; concatenate them into the
`header=` string exactly as `render_metal.py` / `render_metal_stage3.py` do for
the `.metal` files.

---

## 1. Translation rules

The authoritative, detailed version is the header comment of `common.cu`.
Summary:

1. **Address spaces.** `device` / `constant` / `thread` qualifiers are dropped;
   `thread T&` → `T&`, `thread T*` → `T*`, `const device float*` → `const float*`.
   MLX passes every CUDA input as a plain `const T*`, so the Metal rule *"every
   buffer must have ≥ 8 elements or it arrives as `constant`"* **does not apply**
   to CUDA. The existing padding is harmless; nothing in the port depends on it.
2. **Vector types.** CUDA's built-in `float2/3/4` are bare PODs with no
   operators, no broadcast constructor, no swizzles and no `operator[]`.
   `common.cu` defines `dfloat2/3/4` with Metal's exact semantics and then
   aliases `#define float2 dfloat2` (etc.), so the remaining ~3800 lines stay
   textually identical to the reviewed Metal. The alias is emitted after MLX's
   includes and the generated signature only mentions `float*`/`int*`, so it
   cannot collide. Define `DIFFVG_CUDA_NO_VEC_ALIAS` to disable it.
   The only syntactic change this does not cover: **`v.xyz` → `v.xyz()`**.
3. **Math functions.** Named exactly, to remove nvrtc overload-resolution risk:
   `fabs→fabsf`, `sqrt→sqrtf`, `cos/sin→cosf/sinf`, `acos→acosf`,
   `atan2→atan2f`, `pow→powf`, `ceil→ceilf`; float `min/max→fminf/fmaxf`,
   int `min/max→imin/imax`; float `clamp→clampf`, int `clamp→iclamp`,
   int `abs→iabs`. `dot`/`length`/`normalize`/`distance`/`distance_squared`/
   `length_squared` are defined in `common.cu` with Metal's formulas.
4. **Constants.** `constant T x = …` → `static constexpr T x = …`
   (`constexpr`, not `static const`: a namespace-scope `static const` read from
   device code is an nvrtc hazard).
5. **Integers.** `uint32_t`/`uint64_t` → `dvg_u32`/`dvg_u64`
   (`unsigned int` / `unsigned long long`), avoiding a `<cstdint>` dependency.
6. **Atomics.** `mx.fast.cuda_kernel` has **no `atomic_outputs` parameter**, so
   outputs are plain pointers:
   `atomic_fetch_add_explicit(&d[i], v, memory_order_relaxed)` → `atomicAdd(&d[i], v)`;
   `atomic_store_explicit(&o[i], v, …)` → `o[i] = v`.
   `atomicAdd(float*)` is relaxed, which is what the Metal code requested; the
   accumulation order is nondeterministic on both backends.
7. **GradWrites kept.** The Metal rule *"never pass an output buffer into a
   helper"* works around a Metal **compiler crash** and is irrelevant to CUDA.
   The `GradWrites` buffer-and-flush pattern is nevertheless retained so both
   backends share one structure; it costs one small local array and no
   measurable time. Flush with:
   ```cuda
   for (int k = 0; k < g.n; k++) atomicAdd(&d_floats[g.idx[k]], g.val[k]);
   ```

---

## 2. Numerics: how CUDA differs from the Metal build

The Metal kernels are forced to `math_mode: "fast"` (the AGX backend
miscompiles them otherwise). `mx.fast.cuda_kernel` has **no compile-options
parameter**, so nvrtc defaults apply. Every difference moves CUDA *closer* to
the C++ CPU reference:

| | Metal (fast) | CUDA (nvrtc default) | C++ core |
|---|---|---|---|
| division | fast reciprocal approximation | `--prec-div=true`, IEEE | IEEE |
| `sqrt` | **not** correctly rounded (~30% 1 ulp off) | `--prec-sqrt=true`, correctly rounded | correctly rounded |
| `a*b+c` | contracts to fma | `--fmad=true`, contracts to fma | contracts (`-ffp-contract=on`) |
| inf / NaN | may be assumed absent | IEEE semantics | IEEE |

Consequences:

- **`sg_sqrt` is not needed on CUDA.** `scene_gpu.py` carries a corrected
  `sg_sqrt` because Metal's `sqrt` is off by 1 ulp; CUDA's `sqrtf` is correctly
  rounded, so a CUDA scene builder should call `sqrtf` directly. The explicit
  `fma()` calls in the scene builder should be **kept** — nvrtc contracts too,
  so they still match the C++ core.
- **The `DIFFVG_BIG` sentinel and every guarded division are retained
  verbatim.** They are correct under both regimes and keep the two backends on
  identical branches. Nothing in the port relies on fast-math-only behaviour.
- fma contraction is consistent across all three backends, but *which* products
  each compiler contracts is compiler-specific, so float parity is
  tolerance-based, not bit-exact. Integer code (PCG32, indices, BVH traversal,
  winding numbers) **is** bit-exact and was verified as such.

---

## 3. What the Python dispatch layer must do differently

```python
kernel = mx.fast.cuda_kernel(
    name=..., input_names=[...], output_names=[...],
    source=BODY, header=CONCATENATED_CU_FILES,
    ensure_row_contiguous=True,   # default; keep
    shared_memory=0)              # no dynamic shared memory needed
out = kernel(inputs=[...], grid=(n,1,1), threadgroup=(128,1,1),
             output_shapes=[...], output_dtypes=[...], init_value=0)
```

1. **No `atomic_outputs`, no `compile_options`.** Both exist only on
   `metal_kernel`. Drop them. There is no math-mode knob on CUDA.
2. **`init_value=0` IS available.** `metal_kernel` and `cuda_kernel` return the
   same `CustomKernelFunction` (verified in `mlx/include/mlx/fast.h`), whose
   call signature includes `std::optional<float> init_value`. Kernels may assume
   accumulating outputs start zeroed, exactly as the Metal ones do.
3. **Thread index.** `uint elem = thread_position_in_grid.x;` becomes
   ```cuda
   auto elem = cooperative_groups::this_grid().thread_rank();
   ```
4. **Every kernel body MUST start with a bounds check.** Metal's `grid=` is
   dispatched exactly (`dispatchThreads`), so no out-of-range threads exist.
   CUDA launches whole blocks, so the final block is padded. Write
   `if (elem >= n) return;` (or wrap the body in `if (elem < n) { … }`) with `n`
   passed in a params buffer. This is correct whether or not MLX rounds the grid
   up — see the open questions.
5. **Use 1-D grids.** `thread_rank()` is a single linearised rank, not a
   `uint3`. Metal probes that used `grid=(N,K,1)` with
   `thread_position_in_grid.x`/`.y` must be flattened: launch `N*K` threads and
   recover `i = elem % N; k = elem / N;`.
6. **Buffer-count limits are a Metal concern.** Metal's ~30-binding cap and the
   "never reference `*_shape`" rule are AGX-specific. Keep sizes in the `ip`
   params buffer anyway — it costs nothing and keeps one code path.
7. **Padding inputs to ≥ 8 elements is unnecessary** (see rule 1) but harmless.
8. **`template=[("T", mx.float32)]`** works the same way if any kernel is
   templated.

---

## 4. Local verification (this machine has no NVIDIA GPU)

Since the CUDA sources cannot be compiled or run here, they were verified by
compiling **the exact same text** as ordinary C++ and comparing against the
**live Metal kernels** on the Apple GPU.

Harness: `scratchpad/cuda/emulate/`
- `cuda_shim.h` — defines `__device__`/`__global__`/`__forceinline__` away and
  supplies a serial `atomicAdd`. Nothing else is stubbed, because the
  translated sources use no other CUDA builtin.
- `probe.cpp` — compiles all six `.cu` files and evaluates every function
  single-threaded over a supplied scene.
- `probe_metal.py` — the identical record layout through
  `mx.fast.metal_kernel` against the live `.metal` sources
  (`math_mode: "fast"`, i.e. the production Metal behaviour).
- `compare.py` — builds the scenes, runs both, prints the tables.

```sh
cd scratchpad/cuda/emulate
clang++ -std=c++17 -O2 -I. -I../../../pydiffvg/cuda -o probe probe.cpp
NPTS=30000 NIDX=30000 .venv/bin/python compare.py
```

**Scenes** (10): the 8 synthetic cases required by the brief — circle fill,
circle + radial/linear gradient stroke, ellipse under a non-identity transform
with stroke, rect with gradient fill + stroke, closed cubic path, open
quadratic path with per-point thickness, even-odd compound polygon, non-zero
overlapping alpha — plus `apps/imgs/note_small.svg` and `apps/imgs/tiger.svg`
(NF = 56 599 floats). 30 000 query points and 30 000 sample indices each
(300 000 points total), half uniform over an inflated canvas box and half
jittered onto stored geometry to stress boundaries.

### Exact-match results (integer / structural), all 10 scenes

| quantity | mismatches |
|---|---|
| **PCG32 stream** (4 raw `uint32` per index) | **0 / 1 200 000** |
| `sample_position` pixel `(x, y)` | **0 / 600 000** |
| boundary-sample ids & flags (`shape_id`, `group_id`, `base_point_id`, `point_id`, `is_stroke`, valid) | **0 / 1 800 000** |
| `GradWrites` counts (4 producers) | **5 / 1 200 000** (all note_small; all at ties — see below) |
| `is_inside` | **0 / 300 000** |
| `within_distance` | **0 / 300 000** |
| `compute_distance` (found, shape_id) | **0 / 600 000** |
| `pf_collect_fragments` (count) | **0 / 300 000** |
| `sdf_query` (group_id, shape_id) | 2 / 900 000 (one tiger point, Δdist 1.45e-5) |

The PCG32 requirement — bit-identical numbers — **is met exactly**, as are all
integer and branch-structural results.

### Float agreement (representative; full log in `full_sweep.log`)

Typical `max |diff|` over 30 000 points, and max relative diff excluding tie
rows and near-zero denominators:

| function | synthetic | note_small | tiger |
|---|---|---|---|
| `compute_distance` (distance) | 3.1e-5 | 1.2e-4 | 3.1e-5 |
| `compute_distance` (closest_pt) | 1.5e-5 … 1.4e-2 | 3.3e-2 | 4.6e-5 |
| `sample_color_scene` | 0 (9/10 scenes exact) | 0 | 0 |
| `pf_blend` | 1.1e-5 | 2.4e-5 | 6.3e-5 |
| `gather_d_color` | 7.6e-6 | 1.5e-5 | 1.9e-6 |
| `smoothstep` | 1.1e-5 | 2.4e-5 | 2.3e-6 |
| `path_thickness_at` | 0 (3.4e-4 on the thickness path) | 0 | 0 |
| `sample_boundary` (normals) | 1.8e-7 | 2.4e-7 | 6.4e-5 |
| `d_translation` | 3.3e-5 | 1.2e-3 | 4.0e-2 |

**p99 relative agreement is ≤ 1e-5 on every function and every scene** once tie
rows are excluded (tiger `d_translation` p99 = 1.2e-7). The remaining maxima are
fully explained:

1. **Ties (8 rows on note_small, 0 on tiger).** The closest point is a *vertex
   shared by two adjacent segments*. One side attributes it to segment *k* at
   `t = 0`, the other to segment *k−1* at `t ≈ 0.9994` — **identical distance**
   (Δ = 0.0 exactly in 2 of 5 cases) but a line (2 control points → 13 grad
   writes) versus a cubic (4 → 17). Both answers are the same geometric point;
   this is the standard subgradient ambiguity at a vertex, and it is the same
   class of tie DESIGN.md already documents for Metal-vs-CPU. These rows account
   for all 5 `GradWrites` count mismatches and for the large `d_compute_distance`
   checksum outliers (note_small max-rel drops 2.6e-1 → 6.1e-3 when excluded).
2. **Near-zero denominators.** The surviving max-relative values coincide
   exactly with points whose distance is ≈ 0 (on the boundary), where relative
   error is meaningless; the absolute difference there is 3e-5 px.
3. **Cancellation.** tiger's `sample_boundary` normal tail (median 1.5e-8,
   p99.9 1.7e-5, max 6.4e-5) comes from `tangent` being a difference of nearby
   control points at coordinates in the hundreds, amplified differently by
   Metal's approximate division versus CUDA's IEEE division.
4. **Atomic accumulation order.** The `d_floats[…]` pool rows compare a GPU
   atomic scatter (nondeterministic order) against a serial CPU one, so large
   absolute sums differ in the last bits (max rel 1.4e-4 … 7e-3 with medians of
   0). This is inherent, not a translation artefact.

**Conclusion: no logic, indexing or algorithmic divergence was found.** Every
difference is either a genuine geometric tie or last-bit float noise.

---

## 5. What could NOT be verified locally

- **Nothing was compiled by `nvcc`/nvrtc.** `nvcc` is absent and the installed
  MLX wheel has no CUDA backend (`mx.fast.cuda_kernel` raises
  `RuntimeError: [cuda_kernel] No CUDA back-end.`). The sources are verified as
  **C++17 via clang++** only. nvrtc-specific rejections remain possible.
- **No kernel bodies exist yet.** Only the header/leaf functions are translated;
  the kernels (`weight`, `render_color`, `render_color_backward`,
  `sample_boundary`, `render_edge`, `sdf_*`, `prefilter_*`) are the Python
  layer's job and are unwritten.
- **Grid/block mapping is unconfirmed.** Whether MLX rounds `grid` up to whole
  blocks (creating out-of-range threads) could not be checked — the wheel ships
  no CUDA codegen. The mandatory bounds check makes the kernels correct either
  way.
- **No performance data.** See the risks below.
- **The `#define float2 dfloat2` alias** is safe by construction but has not
  been exercised through MLX's real CUDA preamble.
- `scene_gpu.py` / `scene_gpu_bvh.py` inline Metal helper kernels (path lengths,
  CDFs, leaves, matrix inverse, transformed boxes) are **not** part of this port
  and still need translating by whoever owns that layer.

---

## 6. Open questions / checklist for the RunPod run (RTX 4090, CUDA 12.8)

1. **Does it compile under nvrtc at all?** Build one trivial kernel with the
   full six-file header first and read the `verbose=True` output before writing
   any real kernel. Most likely failure points, in order:
   - the `float2/3/4` macro alias colliding with MLX's preamble → define
     `DIFFVG_CUDA_NO_VEC_ALIAS` and rename, or drop the alias;
   - `static constexpr` namespace-scope constants in device code;
   - function-name collisions between our `dot`/`length`/`normalize`/`distance`
     and anything MLX or CUDA injects at global scope (rename with a `dvg_`
     prefix if so);
   - the `smoothstep` / `within_distance` / `inside` / `cdf_sample` /
     `d_closest_point` / `d_compute_distance` **overload sets** resolving
     differently.
2. **Grid padding.** Confirm whether out-of-range threads are launched, and
   keep the bounds check regardless.
3. **Local-memory pressure — the biggest performance risk.** Per thread:
   `Fragment fragments[256]` (~6 KB) in `sample_color_scene_eq`, `BlendState`
   (~4 KB) in the backward blend, `PrefilterFragment[64]`, and 64–128-entry BVH
   stacks. On Metal these are thread-private; in CUDA they spill to local
   memory. Expect low occupancy. Mitigations, in order of preference: small
   blocks (64–128 threads), then lowering `MAX_HIT_SHAPES`/`PF_MAX_FRAGMENTS`
   (the CPU asserts above 256 fragments, so a real scene rarely needs it), then
   moving the fragment array to dynamic shared memory via `shared_memory=`.
   Measure with `-Xptxas -v` equivalents / Nsight before optimising.
4. **Re-run the parity harness against the real CUDA kernels.** `probe.cpp`'s
   record layout is reusable: implement the same two probes as
   `mx.fast.cuda_kernel` kernels and diff against `probe_metal.py`'s stored
   output, or directly against the C++ core. **Assert PCG32 equality first** —
   if the RNG matches, sample positions and boundary sample selection follow.
5. **Verify the sign of the `sqrt`/division improvement.** CUDA should agree
   with the C++ CPU core *better* than Metal does. Comparing CUDA against the
   CPU core on the same scenes is the cleanest end-to-end check and should show
   tighter tolerances than DESIGN.md's Metal-vs-CPU figures.
6. **Watch the known tie points.** `note_small.svg` and `tiger.svg` have
   documented equidistant-vertex and equidistant-group ties; a handful of
   differing gradient attributions there is expected, not a regression.
