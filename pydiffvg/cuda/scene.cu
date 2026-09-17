// Scene-construction helpers for pydiffvg/scene_gpu.py (CUDA).
//
// CUDA translation of pydiffvg/metal/scene.metal: same names, same signatures,
// same rounding. Read that file's header comment first; the differences are:
//
//   1. Contraction. Metal switches it off with `#pragma METAL fp
//      contract(off)`; nvrtc has no equivalent pragma and mx.fast.cuda_kernel
//      exposes no compile options, so --fmad=true is in force and nvrtc WOULD
//      fuse a*b + c on its own. That would not match the C++ core, which
//      contracts only within a source statement and therefore does NOT
//      contract the products that vector.h / matrix.h compute inside operator
//      functions. Contraction is blocked here the only way that works without
//      a compiler flag: the round-to-nearest intrinsics __fmul_rn / __fadd_rn,
//      which are defined to produce a separately rounded result and are never
//      fused. Everything the C++ core really does contract stays an explicit
//      sg_fma (= fmaf).
//   2. sqrt. nvrtc defaults to --prec-sqrt=true, so sqrtf is correctly
//      rounded, exactly like the CPU's fsqrt: sg_sqrt is plain sqrtf and
//      Metal's integer correction loop is not needed (see
//      pydiffvg/cuda/README-porting.md section 2).
//   3. Vectors. CUDA's built-in float2 is a bare POD with no operators, so
//      sg_f2 is defined here with Metal's float2 semantics for the operations
//      the scene kernels use (construction, +, scalar *). Unlike common.cu
//      this file does NOT `#define float2 sg_f2`: the kernel bodies in
//      scene_gpu.py spell the type sg_f2, so nothing can collide with MLX's
//      preamble.
//   4. Naming. `fma` is the double overload in CUDA, so sg_fma is fmaf;
//      infinity comes from __int_as_float rather than numeric_limits.

__device__ inline float sg_inf() {
    return __int_as_float(0x7f800000);
}

// Unfused product / sum: nvrtc may not contract these into an fma.
__device__ inline float sg_mul(float a, float b) {
    return __fmul_rn(a, b);
}

__device__ inline float sg_add(float a, float b) {
    return __fadd_rn(a, b);
}

__device__ inline float sg_fma(float a, float b, float c) {
    return fmaf(a, b, c);
}

// nvrtc's sqrtf is correctly rounded (--prec-sqrt=true), as the CPU's fsqrt;
// the Metal twin needs an integer correction loop, this does not.
__device__ inline float sg_sqrt(float x) {
    return sqrtf(x);
}

struct sg_f2 {
    float x, y;
    __device__ sg_f2() {}
    __device__ sg_f2(float a, float b) : x(a), y(b) {}
};

__device__ inline sg_f2 operator+(sg_f2 a, sg_f2 b) {
    return sg_f2(sg_add(a.x, b.x), sg_add(a.y, b.y));
}

__device__ inline sg_f2 operator*(float s, sg_f2 a) {
    return sg_f2(sg_mul(s, a.x), sg_mul(s, a.y));
}

__device__ inline float sg_dist(sg_f2 a, sg_f2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    // vector.h length_squared = square(x) + square(y): function calls, so
    // clang does not contract it
    return sg_sqrt(sg_add(sg_mul(dx, dx), sg_mul(dy, dy)));
}

// matrix.h xform_pt with clang's contraction: m(0,0)*x + m(0,1)*y + m(0,2)
__device__ inline float sg_xf(float a, float x, float b, float y, float c) {
    return sg_add(sg_fma(a, x, sg_mul(b, y)), c);
}
