// Shared definitions for the diffvg CUDA kernels.
//
// CUDA translation of pydiffvg/metal/common.metal. This file is concatenated
// (with geometry.cu, color.cu, backward.cu, distance_grad.cu, prefilter.cu, in
// that order) into the `header` of every mx.fast.cuda_kernel. It mirrors the
// flattened scene produced by diffvg.Scene.export_flat() / pydiffvg.scene_gpu;
// see the layout tables in DESIGN.md. Everything here is float32 (no double).
//
// ===========================================================================
// TRANSLATION RULES (Metal -> CUDA). These govern every file in pydiffvg/cuda/.
// ===========================================================================
//
// 1. ADDRESS SPACES. Metal's `device` / `constant` / `threadgroup` / `thread`
//    qualifiers have no CUDA equivalent and are simply dropped:
//        const device float *f   ->  const float *f
//        thread float2 &out      ->  float2 &out
//        thread Fragment *frags  ->  Fragment *frags
//    Consequences:
//      * Metal's rule "every buffer must have >= 8 elements so it arrives as
//        `device` rather than `constant`" is a Metal-only artefact of MLX's
//        address-space selection. It does NOT apply to CUDA, where every input
//        is a plain `const T*`. The padding is harmless, so the Python layer
//        may keep it (the pools are padded to >= 64 anyway), but nothing here
//        depends on it.
//      * Helper templates that existed only to abstract over `device` vs
//        `constant` pointers collapse into ordinary functions.
//      * Plain stack arrays (BVH stacks, Fragment[256], BlendState) become CUDA
//        *local memory*, which is correct but spills; see README-porting.md.
//
// 2. FUNCTION QUALIFIERS. Metal's free `inline` functions become
//    `__device__ inline`. The CPU emulation harness
//    (scratchpad/cuda/emulate/) compiles these same files as ordinary C++ by
//    `#define`-ing `__device__` away, so no CUDA builtin may appear outside the
//    small shim list (`atomicAdd`, `cooperative_groups`).
//
// 3. VECTOR TYPES. CUDA's built-in float2/float3/float4 are bare PODs: no
//    operators (`a + b`), no broadcast constructor (`float2(0)`), no swizzles
//    (`.xyz`), no `operator[]`. Rather than rewrite ~3800 lines of arithmetic
//    (and risk transcription bugs), this file defines dfloat2/dfloat3/dfloat4
//    structs carrying exactly Metal's semantics and then aliases the names:
//        #define float2 dfloat2   (etc.)
//    The alias is emitted AFTER any CUDA headers MLX includes (the `header`
//    string is placed after the includes and before the generated kernel), and
//    the generated kernel signature only ever mentions `const float*`/`float*`,
//    so the macro can never collide. Define DIFFVG_CUDA_NO_VEC_ALIAS to skip it
//    and spell the types dfloat2/dfloat3/dfloat4 instead.
//    The ONE syntactic change this does not cover is the swizzle: Metal's
//    `v.xyz` is written `v.xyz()` here.
//
// 4. MATH FUNCTIONS. Metal's overload set (`fabs`, `sqrt`, `min`, `max`,
//    `clamp`, `abs`) resolves differently under nvrtc, where the availability of
//    `min(float,float)` etc. is not guaranteed. To remove all overload-
//    resolution risk, the translated sources name the exact function:
//        fabs  -> fabsf      sqrt -> sqrtf      cos/sin -> cosf/sinf
//        acos  -> acosf      atan2 -> atan2f    pow -> powf   ceil -> ceilf
//        min/max on floats -> fminf/fmaxf       on ints -> imin/imax
//        clamp on floats   -> clampf            on ints -> iclamp
//        abs   on ints     -> iabs
//    length/dot/normalize/distance/distance_squared/length_squared are defined
//    below with the same formulas Metal uses.
//
// 4b. CONSTANTS. Metal's `constant T x = ...` becomes `static constexpr T x =
//    ...`, deliberately constexpr rather than `static const`: a namespace-scope
//    `static const` read from device code is an nvrtc hazard, whereas a
//    constexpr constant is unambiguously usable there.
//    Fixed-width integers are spelled `dvg_u32` / `dvg_u64` (typedefs below)
//    rather than uint32_t / uint64_t, so nothing depends on <cstdint> being
//    available to nvrtc.
//
// 5. NUMERICS — HOW CUDA DIFFERS FROM THE METAL BUILD. The Metal kernels are
//    compiled with math_mode "fast" (forced: the AGX backend miscompiles them
//    otherwise). mx.fast.cuda_kernel has NO compile-options parameter, so the
//    CUDA build uses nvrtc's defaults. Differences, all of which move CUDA
//    *closer* to the C++ CPU reference, not further away:
//      * DIVISION: Metal fast-math uses a fast reciprocal approximation; CUDA
//        defaults to --prec-div=true (correctly rounded IEEE division).
//      * SQRT: Metal's sqrt is not correctly rounded (~30% of inputs are 1 ulp
//        off, e.g. sqrt(3136) = 56.000004, which is why scene_gpu.py needs its
//        `sg_sqrt` correction); CUDA defaults to --prec-sqrt=true, i.e.
//        correctly rounded. A CUDA port therefore does NOT need sg_sqrt.
//      * FMA CONTRACTION: nvrtc defaults to --fmad=true and contracts a*b+c,
//        exactly as Metal does (Metal contracts even in "safe" mode) and as the
//        C++ core does (clang -ffp-contract=on). Contraction is therefore
//        consistent across all three backends, but the *choice* of which
//        products to contract is compiler-specific, so parity stays
//        tolerance-based, not bit-exact, for float results.
//        `--fmad=false` would disable it; we do not request it.
//      * INF/NaN: Metal fast-math may assume they never occur (and may fold
//        away isfinite/isnan tests); CUDA keeps IEEE semantics. The DIFFVG_BIG
//        sentinel and the explicitly guarded divisions/sqrt/acos arguments are
//        correct under BOTH regimes, so they are kept verbatim. Nothing in this
//        port relies on fast-math-only behaviour. The one place where Metal and
//        CUDA genuinely diverge is a degenerate input that the CPU turns into
//        inf/NaN: Metal fast-math yields unspecified garbage, CUDA yields a
//        true inf/NaN. Every such site is already guarded (see the "C++ divides
//        by zero here" comments), with one documented exception: xform_normal
//        below performs Metal's unguarded normalize().
//    INTEGER code (PCG RNG, indices, BVH traversal, winding numbers) is exact
//    on both backends and must match bit-for-bit.
//
// 6. THREAD INDEXING. Metal's `thread_position_in_grid.x` becomes
//        auto elem = cooperative_groups::this_grid().thread_rank();
//    in the KERNEL BODIES (which live in the Python dispatch layer, not here).
//    Two semantic differences the Python layer must handle:
//      * Metal's `grid=` is a thread count dispatched exactly
//        (dispatchThreads), so no out-of-range threads exist. CUDA launches
//        whole blocks, so the last block is padded: EVERY CUDA kernel body must
//        begin with an explicit bounds check `if (elem >= n) return;`.
//      * `thread_rank()` is a single linearised rank over the whole grid, not a
//        3-component uint3. Multi-dimensional Metal dispatches (e.g. the
//        distance-gradient probe's grid=(N,K,1) using both
//        thread_position_in_grid.x and .y) must be flattened to a 1-D grid with
//        the row/column indices recovered by division and modulo.
//
// 7. ATOMICS AND OUTPUTS. mx.fast.cuda_kernel has NO `atomic_outputs`
//    parameter, so outputs are plain pointers, not atomic<T>:
//        atomic_fetch_add_explicit(&d[i], v, memory_order_relaxed) -> atomicAdd(&d[i], v)
//        atomic_store_explicit(&o[i], v, memory_order_relaxed)     -> o[i] = v
//    atomicAdd(float*) is relaxed (no ordering guarantees), which is exactly
//    what the Metal code asked for; the accumulation order is nondeterministic
//    on both backends, so gradient sums differ in the last bits run to run.
//    `init_value=0` IS available on the CUDA call path (metal_kernel and
//    cuda_kernel return the same CustomKernelFunction), so kernels may assume
//    outputs start zeroed, as the Metal ones do.
//    NOTE: the Metal rule "never pass an output buffer into a helper function"
//    is a workaround for a Metal *compiler* crash and does NOT apply to CUDA.
//    The GradWrites buffering is kept anyway so both backends share one
//    structure; it costs a small local-memory array and no measurable time.
//
// ===========================================================================

// ---------------------------------------------------------------------------
// Scalar helpers (see rule 4)
__device__ inline int imin(int a, int b) { return a < b ? a : b; }
__device__ inline int imax(int a, int b) { return a > b ? a : b; }
__device__ inline int iabs(int a) { return a < 0 ? -a : a; }
__device__ inline int iclamp(int x, int lo, int hi) { return imin(imax(x, lo), hi); }
__device__ inline float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

// Fixed-width integers without relying on <cstdint> being available to nvrtc.
typedef unsigned int dvg_u32;
typedef unsigned long long dvg_u64;

// ---------------------------------------------------------------------------
// Vector types (see rule 3): Metal's float2/float3/float4 semantics.
struct dfloat2 {
    float x, y;
    __device__ dfloat2() {}
    __device__ dfloat2(float s) : x(s), y(s) {}
    __device__ dfloat2(float x_, float y_) : x(x_), y(y_) {}
    __device__ float &operator[](int i) { return i == 0 ? x : y; }
    __device__ const float &operator[](int i) const { return i == 0 ? x : y; }
};

struct dfloat3 {
    float x, y, z;
    __device__ dfloat3() {}
    __device__ dfloat3(float s) : x(s), y(s), z(s) {}
    __device__ dfloat3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    __device__ float &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    __device__ const float &operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

struct dfloat4 {
    float x, y, z, w;
    __device__ dfloat4() {}
    __device__ dfloat4(float s) : x(s), y(s), z(s), w(s) {}
    __device__ dfloat4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    __device__ dfloat4(dfloat3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
    __device__ float &operator[](int i) { return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w)); }
    __device__ const float &operator[](int i) const { return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w)); }
    __device__ dfloat3 xyz() const { return dfloat3(x, y, z); }  // Metal: v.xyz
};

// -- dfloat2 operators
__device__ inline dfloat2 operator-(dfloat2 a) { return dfloat2(-a.x, -a.y); }
__device__ inline dfloat2 operator+(dfloat2 a, dfloat2 b) { return dfloat2(a.x + b.x, a.y + b.y); }
__device__ inline dfloat2 operator-(dfloat2 a, dfloat2 b) { return dfloat2(a.x - b.x, a.y - b.y); }
__device__ inline dfloat2 operator*(dfloat2 a, dfloat2 b) { return dfloat2(a.x * b.x, a.y * b.y); }
__device__ inline dfloat2 operator/(dfloat2 a, dfloat2 b) { return dfloat2(a.x / b.x, a.y / b.y); }
__device__ inline dfloat2 operator+(dfloat2 a, float s) { return dfloat2(a.x + s, a.y + s); }
__device__ inline dfloat2 operator-(dfloat2 a, float s) { return dfloat2(a.x - s, a.y - s); }
__device__ inline dfloat2 operator*(dfloat2 a, float s) { return dfloat2(a.x * s, a.y * s); }
__device__ inline dfloat2 operator/(dfloat2 a, float s) { return dfloat2(a.x / s, a.y / s); }
__device__ inline dfloat2 operator+(float s, dfloat2 a) { return dfloat2(s + a.x, s + a.y); }
__device__ inline dfloat2 operator-(float s, dfloat2 a) { return dfloat2(s - a.x, s - a.y); }
__device__ inline dfloat2 operator*(float s, dfloat2 a) { return dfloat2(s * a.x, s * a.y); }
__device__ inline dfloat2 operator/(float s, dfloat2 a) { return dfloat2(s / a.x, s / a.y); }
__device__ inline dfloat2 &operator+=(dfloat2 &a, dfloat2 b) { a.x += b.x; a.y += b.y; return a; }
__device__ inline dfloat2 &operator-=(dfloat2 &a, dfloat2 b) { a.x -= b.x; a.y -= b.y; return a; }
__device__ inline dfloat2 &operator*=(dfloat2 &a, dfloat2 b) { a.x *= b.x; a.y *= b.y; return a; }
__device__ inline dfloat2 &operator/=(dfloat2 &a, dfloat2 b) { a.x /= b.x; a.y /= b.y; return a; }
__device__ inline dfloat2 &operator+=(dfloat2 &a, float s) { a.x += s; a.y += s; return a; }
__device__ inline dfloat2 &operator-=(dfloat2 &a, float s) { a.x -= s; a.y -= s; return a; }
__device__ inline dfloat2 &operator*=(dfloat2 &a, float s) { a.x *= s; a.y *= s; return a; }
__device__ inline dfloat2 &operator/=(dfloat2 &a, float s) { a.x /= s; a.y /= s; return a; }

// -- dfloat3 operators
__device__ inline dfloat3 operator-(dfloat3 a) { return dfloat3(-a.x, -a.y, -a.z); }
__device__ inline dfloat3 operator+(dfloat3 a, dfloat3 b) { return dfloat3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline dfloat3 operator-(dfloat3 a, dfloat3 b) { return dfloat3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline dfloat3 operator*(dfloat3 a, dfloat3 b) { return dfloat3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ inline dfloat3 operator/(dfloat3 a, dfloat3 b) { return dfloat3(a.x / b.x, a.y / b.y, a.z / b.z); }
__device__ inline dfloat3 operator+(dfloat3 a, float s) { return dfloat3(a.x + s, a.y + s, a.z + s); }
__device__ inline dfloat3 operator-(dfloat3 a, float s) { return dfloat3(a.x - s, a.y - s, a.z - s); }
__device__ inline dfloat3 operator*(dfloat3 a, float s) { return dfloat3(a.x * s, a.y * s, a.z * s); }
__device__ inline dfloat3 operator/(dfloat3 a, float s) { return dfloat3(a.x / s, a.y / s, a.z / s); }
__device__ inline dfloat3 operator+(float s, dfloat3 a) { return dfloat3(s + a.x, s + a.y, s + a.z); }
__device__ inline dfloat3 operator-(float s, dfloat3 a) { return dfloat3(s - a.x, s - a.y, s - a.z); }
__device__ inline dfloat3 operator*(float s, dfloat3 a) { return dfloat3(s * a.x, s * a.y, s * a.z); }
__device__ inline dfloat3 operator/(float s, dfloat3 a) { return dfloat3(s / a.x, s / a.y, s / a.z); }
__device__ inline dfloat3 &operator+=(dfloat3 &a, dfloat3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
__device__ inline dfloat3 &operator-=(dfloat3 &a, dfloat3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
__device__ inline dfloat3 &operator*=(dfloat3 &a, dfloat3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
__device__ inline dfloat3 &operator/=(dfloat3 &a, dfloat3 b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; return a; }
__device__ inline dfloat3 &operator+=(dfloat3 &a, float s) { a.x += s; a.y += s; a.z += s; return a; }
__device__ inline dfloat3 &operator-=(dfloat3 &a, float s) { a.x -= s; a.y -= s; a.z -= s; return a; }
__device__ inline dfloat3 &operator*=(dfloat3 &a, float s) { a.x *= s; a.y *= s; a.z *= s; return a; }
__device__ inline dfloat3 &operator/=(dfloat3 &a, float s) { a.x /= s; a.y /= s; a.z /= s; return a; }

// -- dfloat4 operators
__device__ inline dfloat4 operator-(dfloat4 a) { return dfloat4(-a.x, -a.y, -a.z, -a.w); }
__device__ inline dfloat4 operator+(dfloat4 a, dfloat4 b) { return dfloat4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ inline dfloat4 operator-(dfloat4 a, dfloat4 b) { return dfloat4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
__device__ inline dfloat4 operator*(dfloat4 a, dfloat4 b) { return dfloat4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
__device__ inline dfloat4 operator/(dfloat4 a, dfloat4 b) { return dfloat4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }
__device__ inline dfloat4 operator+(dfloat4 a, float s) { return dfloat4(a.x + s, a.y + s, a.z + s, a.w + s); }
__device__ inline dfloat4 operator-(dfloat4 a, float s) { return dfloat4(a.x - s, a.y - s, a.z - s, a.w - s); }
__device__ inline dfloat4 operator*(dfloat4 a, float s) { return dfloat4(a.x * s, a.y * s, a.z * s, a.w * s); }
__device__ inline dfloat4 operator/(dfloat4 a, float s) { return dfloat4(a.x / s, a.y / s, a.z / s, a.w / s); }
__device__ inline dfloat4 operator+(float s, dfloat4 a) { return dfloat4(s + a.x, s + a.y, s + a.z, s + a.w); }
__device__ inline dfloat4 operator-(float s, dfloat4 a) { return dfloat4(s - a.x, s - a.y, s - a.z, s - a.w); }
__device__ inline dfloat4 operator*(float s, dfloat4 a) { return dfloat4(s * a.x, s * a.y, s * a.z, s * a.w); }
__device__ inline dfloat4 operator/(float s, dfloat4 a) { return dfloat4(s / a.x, s / a.y, s / a.z, s / a.w); }
__device__ inline dfloat4 &operator+=(dfloat4 &a, dfloat4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
__device__ inline dfloat4 &operator-=(dfloat4 &a, dfloat4 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
__device__ inline dfloat4 &operator*=(dfloat4 &a, dfloat4 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
__device__ inline dfloat4 &operator/=(dfloat4 &a, dfloat4 b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w; return a; }
__device__ inline dfloat4 &operator+=(dfloat4 &a, float s) { a.x += s; a.y += s; a.z += s; a.w += s; return a; }
__device__ inline dfloat4 &operator-=(dfloat4 &a, float s) { a.x -= s; a.y -= s; a.z -= s; a.w -= s; return a; }
__device__ inline dfloat4 &operator*=(dfloat4 &a, float s) { a.x *= s; a.y *= s; a.z *= s; a.w *= s; return a; }
__device__ inline dfloat4 &operator/=(dfloat4 &a, float s) { a.x /= s; a.y /= s; a.z /= s; a.w /= s; return a; }

// -- geometric functions (same formulas as Metal)
__device__ inline float dot(dfloat2 a, dfloat2 b) { return a.x * b.x + a.y * b.y; }
__device__ inline float dot(dfloat3 a, dfloat3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float dot(dfloat4 a, dfloat4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
__device__ inline float length_squared(dfloat2 a) { return dot(a, a); }
__device__ inline float length_squared(dfloat3 a) { return dot(a, a); }
__device__ inline float length(dfloat2 a) { return sqrtf(dot(a, a)); }
__device__ inline float length(dfloat3 a) { return sqrtf(dot(a, a)); }
__device__ inline float distance_squared(dfloat2 a, dfloat2 b) { return length_squared(a - b); }
__device__ inline float distance(dfloat2 a, dfloat2 b) { return length(a - b); }
// Metal's normalize() is unguarded (v / length(v)); kept verbatim. Under CUDA's
// IEEE semantics a zero vector yields NaN rather than fast-math garbage.
__device__ inline dfloat2 normalize(dfloat2 a) { return a / length(a); }

#ifndef DIFFVG_CUDA_NO_VEC_ALIAS
#define float2 dfloat2
#define float3 dfloat3
#define float4 dfloat4
#endif

// ---------------------------------------------------------------------------
// Parameter block (int32[32])
static constexpr int IP_W = 0;
static constexpr int IP_H = 1;
static constexpr int IP_NSX = 2;
static constexpr int IP_NSY = 3;
static constexpr int IP_CANVAS_W = 4;
static constexpr int IP_CANVAS_H = 5;
static constexpr int IP_NUM_SHAPES = 6;
static constexpr int IP_NUM_GROUPS = 7;
static constexpr int IP_NUM_TOTAL_SHAPES = 8;
static constexpr int IP_FILTER_TYPE = 9;
static constexpr int IP_USE_PREFILTERING = 10;
static constexpr int IP_HAS_BACKGROUND = 11;
static constexpr int IP_NUM_EVAL = 12;
static constexpr int IP_SEED_LO = 13;
static constexpr int IP_SEED_HI = 14;
static constexpr int IP_SCENE_BVH_BASE = 15;
static constexpr int IP_SHAPES_I_OFF = 16;
static constexpr int IP_GROUPS_I_OFF = 17;
static constexpr int IP_BVH_I_OFF = 18;
static constexpr int IP_SAMPLE_SHAPE_ID_OFF = 19;
static constexpr int IP_SAMPLE_GROUP_ID_OFF = 20;
static constexpr int IP_BVH_F_OFF = 21;
static constexpr int IP_SAMPLE_CDF_OFF = 22;
static constexpr int IP_SAMPLE_PMF_OFF = 23;
static constexpr int IP_FILTER_RADIUS_OFF = 24;
static constexpr int IP_NUM_FLOATS = 25;
static constexpr int IP_NUM_INTS = 26;
static constexpr int IP_NUM_BVH_NODES = 27;
static constexpr int IP_USE_EVAL_POSITIONS = 28;

static constexpr int SHAPE_I_STRIDE = 12;
static constexpr int GROUP_I_STRIDE = 12;
static constexpr int BVH_F_STRIDE = 5;
static constexpr int BVH_I_STRIDE = 2;

static constexpr int SHAPE_CIRCLE = 0;
static constexpr int SHAPE_ELLIPSE = 1;
static constexpr int SHAPE_PATH = 2;
static constexpr int SHAPE_RECT = 3;

static constexpr int COLOR_NONE = -1;
static constexpr int COLOR_CONSTANT = 0;
static constexpr int COLOR_LINEAR = 1;
static constexpr int COLOR_RADIAL = 2;

static constexpr int FILTER_BOX = 0;
static constexpr int FILTER_TENT = 1;
static constexpr int FILTER_RADIAL_PARABOLIC = 2;
static constexpr int FILTER_HANN = 3;

static constexpr float DIFFVG_PI = 3.14159265358979323846f;

// The Metal build is compiled with math_mode "fast", which may assume there are
// no infinities or NaNs. CUDA keeps IEEE semantics, but this finite sentinel is
// kept (and every division still guarded) so both backends take identical
// branches: where the C++ core uses infinity<float>(), both GPUs use DIFFVG_BIG.
static constexpr float DIFFVG_BIG = 1e30f;
__device__ inline float diffvg_infinity() { return DIFFVG_BIG; }
__device__ inline float square(float x) { return x * x; }
__device__ inline float cubic(float x) { return x * x * x; }

// ---------------------------------------------------------------------------
// View over the flattened scene.
//
// Unlike Metal (where MLX hands buffers of < 8 elements to the kernel in the
// `constant` address space, forcing every pool to be padded), CUDA inputs are
// always plain `const T*`, so SceneView is just three pointers.
struct SceneView {
    const float *f;  // floats pool
    const int *i;    // ints pool
    const int *ip;   // parameter block
};

__device__ inline SceneView make_scene_view(const float *f, const int *i, const int *ip) {
    SceneView s;
    s.f = f;
    s.i = i;
    s.ip = ip;
    return s;
}

// ---------------------------------------------------------------------------
// Axis-aligned bounding boxes (aabb.h)
struct AABB {
    float2 p_min;
    float2 p_max;
};

__device__ inline bool inside(AABB box, float2 p) {
    return p.x >= box.p_min.x && p.x <= box.p_max.x &&
           p.y >= box.p_min.y && p.y <= box.p_max.y;
}

__device__ inline bool inside(AABB box, float2 p, float radius) {
    return p.x >= box.p_min.x - radius && p.x <= box.p_max.x + radius &&
           p.y >= box.p_min.y - radius && p.y <= box.p_max.y + radius;
}

__device__ inline bool within_distance(AABB box, float2 pt, float r) {
    return pt.x >= box.p_min.x - r && pt.x <= box.p_max.x + r &&
           pt.y >= box.p_min.y - r && pt.y <= box.p_max.y + r;
}

// ---------------------------------------------------------------------------
// BVH nodes (absolute node index n)
__device__ inline AABB node_box(SceneView s, int n) {
    int o = s.ip[IP_BVH_F_OFF] + BVH_F_STRIDE * n;
    AABB b;
    b.p_min = float2(s.f[o], s.f[o + 1]);
    b.p_max = float2(s.f[o + 2], s.f[o + 3]);
    return b;
}
__device__ inline float node_max_radius(SceneView s, int n) { return s.f[s.ip[IP_BVH_F_OFF] + BVH_F_STRIDE * n + 4]; }
__device__ inline int node_child0(SceneView s, int n) { return s.i[s.ip[IP_BVH_I_OFF] + BVH_I_STRIDE * n]; }
__device__ inline int node_child1(SceneView s, int n) { return s.i[s.ip[IP_BVH_I_OFF] + BVH_I_STRIDE * n + 1]; }
__device__ inline int scene_bvh_root(SceneView s) { return s.ip[IP_SCENE_BVH_BASE] + 2 * s.ip[IP_NUM_GROUPS] - 2; }

// ---------------------------------------------------------------------------
// Shapes
__device__ inline int shape_rec(SceneView s, int sid) { return s.ip[IP_SHAPES_I_OFF] + SHAPE_I_STRIDE * sid; }
__device__ inline int shape_type(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 0]; }
__device__ inline int shape_f_off(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 1]; }
__device__ inline int path_points_off(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 2]; }
__device__ inline int path_num_points(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 3]; }
__device__ inline int path_ctrl_off(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 4]; }
__device__ inline int path_num_base_points(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 5]; }
__device__ inline int path_thickness_off(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 6]; }
__device__ inline bool path_has_thickness(SceneView s, int sid) { return path_thickness_off(s, sid) >= 0; }
__device__ inline int path_bvh_base(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 7]; }
__device__ inline int path_bvh_root(SceneView s, int sid) { return path_bvh_base(s, sid) + 2 * path_num_base_points(s, sid) - 2; }
__device__ inline int path_cdf_off(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 8]; }
__device__ inline int path_pmf_off(SceneView s, int sid) { return path_cdf_off(s, sid) + path_num_base_points(s, sid); }
__device__ inline int path_pid_off(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 9]; }
__device__ inline bool path_is_closed(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 10] != 0; }
__device__ inline bool path_use_distance_approx(SceneView s, int sid) { return s.i[shape_rec(s, sid) + 11] != 0; }

__device__ inline float shape_stroke_width(SceneView s, int sid) { return s.f[shape_f_off(s, sid)]; }
__device__ inline float shape_param(SceneView s, int sid, int k) { return s.f[shape_f_off(s, sid) + 1 + k]; }  // a, b, c, d

__device__ inline float circle_radius(SceneView s, int sid) { return shape_param(s, sid, 0); }
__device__ inline float2 circle_center(SceneView s, int sid) { return float2(shape_param(s, sid, 1), shape_param(s, sid, 2)); }
__device__ inline float2 ellipse_radius(SceneView s, int sid) { return float2(shape_param(s, sid, 0), shape_param(s, sid, 1)); }
__device__ inline float2 ellipse_center(SceneView s, int sid) { return float2(shape_param(s, sid, 2), shape_param(s, sid, 3)); }
__device__ inline float2 rect_p_min(SceneView s, int sid) { return float2(shape_param(s, sid, 0), shape_param(s, sid, 1)); }
__device__ inline float2 rect_p_max(SceneView s, int sid) { return float2(shape_param(s, sid, 2), shape_param(s, sid, 3)); }

__device__ inline float2 path_point(SceneView s, int sid, int k) {
    int o = path_points_off(s, sid) + 2 * k;
    return float2(s.f[o], s.f[o + 1]);
}
__device__ inline float path_thickness(SceneView s, int sid, int k) { return s.f[path_thickness_off(s, sid) + k]; }
__device__ inline int path_num_control_points(SceneView s, int sid, int j) { return s.i[path_ctrl_off(s, sid) + j]; }
__device__ inline float path_length_cdf(SceneView s, int sid, int j) { return s.f[path_cdf_off(s, sid) + j]; }
__device__ inline float path_length_pmf(SceneView s, int sid, int j) { return s.f[path_pmf_off(s, sid) + j]; }
__device__ inline int path_point_id_map(SceneView s, int sid, int j) { return s.i[path_pid_off(s, sid) + j]; }

// ---------------------------------------------------------------------------
// Shape groups
__device__ inline int group_rec(SceneView s, int g) { return s.ip[IP_GROUPS_I_OFF] + GROUP_I_STRIDE * g; }
__device__ inline int group_num_shapes(SceneView s, int g) { return s.i[group_rec(s, g) + 1]; }
__device__ inline int group_shape_id(SceneView s, int g, int k) { return s.i[s.i[group_rec(s, g) + 0] + k]; }
__device__ inline int group_fill_type(SceneView s, int g) { return s.i[group_rec(s, g) + 2]; }
__device__ inline int group_fill_off(SceneView s, int g) { return s.i[group_rec(s, g) + 3]; }
__device__ inline int group_fill_num_stops(SceneView s, int g) { return s.i[group_rec(s, g) + 4]; }
__device__ inline int group_stroke_type(SceneView s, int g) { return s.i[group_rec(s, g) + 5]; }
__device__ inline int group_stroke_off(SceneView s, int g) { return s.i[group_rec(s, g) + 6]; }
__device__ inline int group_stroke_num_stops(SceneView s, int g) { return s.i[group_rec(s, g) + 7]; }
__device__ inline bool group_has_fill(SceneView s, int g) { return group_fill_type(s, g) != COLOR_NONE; }
__device__ inline bool group_has_stroke(SceneView s, int g) { return group_stroke_type(s, g) != COLOR_NONE; }
__device__ inline bool group_use_even_odd_rule(SceneView s, int g) { return s.i[group_rec(s, g) + 8] != 0; }
__device__ inline int group_bvh_base(SceneView s, int g) { return s.i[group_rec(s, g) + 9]; }
__device__ inline int group_bvh_root(SceneView s, int g) { return group_bvh_base(s, g) + 2 * group_num_shapes(s, g) - 2; }
__device__ inline int group_gf_off(SceneView s, int g) { return s.i[group_rec(s, g) + 10]; }

// ---------------------------------------------------------------------------
// 3x3 matrices, row-major like matrix.h: m(i, j) == r[i][j]
struct Mat3 {
    float3 r0;
    float3 r1;
    float3 r2;
};

__device__ inline float mat_at(Mat3 m, int i, int j) {
    return i == 0 ? m.r0[j] : (i == 1 ? m.r1[j] : m.r2[j]);
}

__device__ inline Mat3 mat3_from_pool(const float *f, int off) {
    Mat3 m;
    m.r0 = float3(f[off], f[off + 1], f[off + 2]);
    m.r1 = float3(f[off + 3], f[off + 4], f[off + 5]);
    m.r2 = float3(f[off + 6], f[off + 7], f[off + 8]);
    return m;
}

__device__ inline Mat3 mat3_zero() {
    Mat3 m;
    m.r0 = float3(0);
    m.r1 = float3(0);
    m.r2 = float3(0);
    return m;
}

__device__ inline Mat3 group_shape_to_canvas(SceneView s, int g) { return mat3_from_pool(s.f, group_gf_off(s, g)); }
__device__ inline Mat3 group_canvas_to_shape(SceneView s, int g) { return mat3_from_pool(s.f, group_gf_off(s, g) + 9); }

// matrix.h xform_pt (same operation order)
__device__ inline float2 xform_pt(Mat3 m, float2 pt) {
    float t0 = m.r0[0] * pt[0] + m.r0[1] * pt[1] + m.r0[2];
    float t1 = m.r1[0] * pt[0] + m.r1[1] * pt[1] + m.r1[2];
    float t2 = m.r2[0] * pt[0] + m.r2[1] * pt[1] + m.r2[2];
    return float2(t0 / t2, t1 / t2);
}

// matrix.h d_xform_pt: accumulates into d_m and d_pt
__device__ inline void d_xform_pt(Mat3 m, float2 pt, float2 d_out, Mat3 &d_m, float2 &d_pt) {
    float t0 = m.r0[0] * pt[0] + m.r0[1] * pt[1] + m.r0[2];
    float t1 = m.r1[0] * pt[0] + m.r1[1] * pt[1] + m.r1[2];
    float t2 = m.r2[0] * pt[0] + m.r2[1] * pt[1] + m.r2[2];
    float2 out = float2(t0 / t2, t1 / t2);
    float3 d_t = float3(d_out[0] / t2,
                        d_out[1] / t2,
                        -(d_out[0] * out[0] + d_out[1] * out[1]) / t2);
    d_m.r0 += float3(d_t[0] * pt[0], d_t[0] * pt[1], d_t[0]);
    d_m.r1 += float3(d_t[1] * pt[0], d_t[1] * pt[1], d_t[1]);
    d_m.r2 += float3(d_t[2] * pt[0], d_t[2] * pt[1], d_t[2]);
    d_pt[0] += d_t[0] * m.r0[0] + d_t[1] * m.r1[0] + d_t[2] * m.r2[0];
    d_pt[1] += d_t[0] * m.r0[1] + d_t[1] * m.r1[1] + d_t[2] * m.r2[1];
}

// matrix.h xform_normal
__device__ inline float2 xform_normal(Mat3 m_inv, float2 n) {
    return normalize(float2(m_inv.r0[0] * n[0] + m_inv.r1[0] * n[1],
                            m_inv.r0[1] * n[0] + m_inv.r1[1] * n[1]));
}

// ---------------------------------------------------------------------------
// Gradient writes
//
// On Metal, passing a kernel's output buffer into a helper crashes the Metal
// compiler service, so helpers push (index, value) pairs here and the kernel
// body flushes them. CUDA has no such restriction, but the pattern is kept so
// both backends share one structure. The CUDA flush is:
//     for (int k = 0; k < g.n; k++)
//         atomicAdd(&d_floats[g.idx[k]], g.val[k]);
// Flush after each fragment / boundary sample to keep n small.
static constexpr int MAX_GRAD_WRITES = 64;

struct GradWrites {
    int idx[MAX_GRAD_WRITES];
    float val[MAX_GRAD_WRITES];
    int n;
};

__device__ inline void gw_clear(GradWrites &g) { g.n = 0; }

__device__ inline void gw_push(GradWrites &g, int k, float v) {
    if (g.n < MAX_GRAD_WRITES) {
        g.idx[g.n] = k;
        g.val[g.n] = v;
        g.n++;
    }
}

// Pushes a 3x3 gradient at `off` (row-major)
__device__ inline void gw_mat3(GradWrites &g, int off, Mat3 m) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            gw_push(g, off + 3 * i + j, mat_at(m, i, j));
        }
    }
}

// ---------------------------------------------------------------------------
// PCG32 (pcg.h). Bit-exact with the CPU implementation AND with common.metal:
// this is pure 64-bit integer arithmetic, unaffected by any float mode.
struct pcg32_state {
    dvg_u64 state;
    dvg_u64 inc;
};

__device__ inline dvg_u32 next_pcg32(pcg32_state &rng) {
    dvg_u64 oldstate = rng.state;
    rng.state = oldstate * 6364136223846793005ULL + (rng.inc | 1ULL);
    dvg_u32 xorshifted = (dvg_u32)(((oldstate >> 18u) ^ oldstate) >> 27u);
    dvg_u32 rot = (dvg_u32)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

__device__ inline float next_pcg32_float(pcg32_state &rng) {
    // CPU: the bits (r >> 9) | 0x3f800000 read as a float, minus 1.
    // That float is 1 + (r >> 9) * 2^-23, so this is exact.
    return (float)(next_pcg32(rng) >> 9) * (1.0f / 8388608.0f);
}

__device__ inline pcg32_state init_pcg32(int idx, dvg_u64 seed) {
    pcg32_state st;
    st.state = 0ULL;
    st.inc = (((dvg_u64)(idx) + 1ULL) << 1u) | 1ULL;
    next_pcg32(st);
    st.state += (0x853c49e6748fea9bULL + seed);
    next_pcg32(st);
    return st;
}

__device__ inline dvg_u64 scene_seed(SceneView s) {
    return ((dvg_u64)((dvg_u32)(s.ip[IP_SEED_HI])) << 32) | (dvg_u64)((dvg_u32)(s.ip[IP_SEED_LO]));
}

// ---------------------------------------------------------------------------
// Pixel filters (filter.h)
__device__ inline float filter_radius(SceneView s) { return s.f[s.ip[IP_FILTER_RADIUS_OFF]]; }

__device__ inline float compute_filter_weight(int type, float radius, float dx, float dy) {
    if (fabsf(dx) > radius || fabsf(dy) > radius) {
        return 0;
    }
    if (type == FILTER_BOX) {
        return 1.f / square(2 * radius);
    } else if (type == FILTER_TENT) {
        return (radius - fabsf(dx)) * (radius - fabsf(dy)) / square(square(radius));
    } else if (type == FILTER_RADIAL_PARABOLIC) {
        return (4.f / 3.f) * (1 - square(dx / radius)) *
               (4.f / 3.f) * (1 - square(dy / radius));
    } else {
        float ndx = (dx / (2 * radius)) + 0.5f;
        float ndy = (dy / (2 * radius)) + 0.5f;
        return 0.5f * (1.f - cosf(2 * DIFFVG_PI * ndx)) *
               0.5f * (1.f - cosf(2 * DIFFVG_PI * ndy)) /
               square(radius);
    }
}

// d compute_filter_weight / d radius (filter.h filter_weight_d_radius): zero
// outside the closed support; the box filter's jump at the border is not
// differentiated.
__device__ inline float filter_weight_d_radius(int type, float r, float dx, float dy) {
    if (fabsf(dx) > r || fabsf(dy) > r) {
        return 0;
    }
    if (type == FILTER_BOX) {
        return -2 / cubic(2 * r) * 2;
    } else if (type == FILTER_TENT) {
        float fx = r - fabsf(dx);
        float fy = r - fabsf(dy);
        float r4 = square(square(r));
        return (fx + fy) / r4 - 4 * fx * fy / (r4 * r);
    } else if (type == FILTER_RADIAL_PARABOLIC) {
        float gx = 1 - square(dx / r);
        float gy = 1 - square(dy / r);
        float r3 = r * r * r;
        float d_gx = 2 * square(dx) / r3;
        float d_gy = 2 * square(dy) / r3;
        return (16.f / 9.f) * (d_gx * gy + gx * d_gy);
    } else {
        float ndx = (dx / (2 * r)) + 0.5f;
        float ndy = (dy / (2 * r)) + 0.5f;
        float hx = 0.5f * (1.f - cosf(2 * DIFFVG_PI * ndx));
        float hy = 0.5f * (1.f - cosf(2 * DIFFVG_PI * ndy));
        float d_hx = 0.5f * sinf(2 * DIFFVG_PI * ndx) * (2 * DIFFVG_PI) * (-dx / (2 * r * r));
        float d_hy = 0.5f * sinf(2 * DIFFVG_PI * ndy) * (2 * DIFFVG_PI) * (-dy / (2 * r * r));
        return (d_hx * hy + hx * d_hy) / square(r) - 2 * hx * hy / cubic(r);
    }
}

// Returns d(radius) for filter.h d_compute_filter_weight
__device__ inline float d_compute_filter_weight(int type, float radius, float dx, float dy, float d_return) {
    return d_return * filter_weight_d_radius(type, radius, dx, dy);
}

// ---------------------------------------------------------------------------
// Sample positions (weight_kernel / render_kernel in diffvg.cpp)
// idx enumerates height * width * num_samples_y * num_samples_x.
__device__ inline float2 sample_position(SceneView s, int idx, int &x, int &y) {
    int nsx = s.ip[IP_NSX];
    int nsy = s.ip[IP_NSY];
    int width = s.ip[IP_W];
    pcg32_state rng = init_pcg32(idx, scene_seed(s));
    int sx = idx % nsx;
    int sy = (idx / nsx) % nsy;
    x = (idx / (nsx * nsy)) % width;
    y = (idx / (nsx * nsy * width));
    float rx = next_pcg32_float(rng);
    float ry = next_pcg32_float(rng);
    if (s.ip[IP_USE_PREFILTERING] != 0) {
        rx = ry = 0.5f;
    }
    return float2(x + ((float)sx + rx) / nsx,
                  y + ((float)sy + ry) / nsy);
}
