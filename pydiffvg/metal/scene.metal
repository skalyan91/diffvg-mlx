// Scene-construction helpers for pydiffvg/scene_gpu.py (Metal).
//
// The CUDA twin is pydiffvg/cuda/scene.cu; same names, same semantics, same
// rounding. Both are written so that the flat pools match the C++ core's
// diffvg.Scene.export_flat() bit for bit:
//
//   - no contraction happens by accident. The C++ core is built with clang's
//     -ffp-contract=on, which only contracts a*b+c *within one source
//     statement*, so the contractions that survive inlining of vector.h /
//     matrix.h operators are few and known. Metal's clang contracts a*b + c
//     within one expression even in safe math mode (measured), so contraction
//     is switched off here and the C++ contractions are written out as
//     explicit sg_fma(). (CUDA's nvrtc contracts too, and has no such pragma;
//     scene.cu blocks it with __fmul_rn / __fadd_rn instead, which is why the
//     bodies go through sg_mul / sg_add / the sg_f2 operators rather than
//     writing * and + directly where a fusion would change the result.)
//   - sqrt is correctly rounded. Metal's is not (measured: 30% of random
//     inputs off by 1 ulp, sqrt(3136) = 56.000004), hence the correction in
//     sg_sqrt below; CUDA's sqrtf is correctly rounded and needs none.
//
// scene_gpu.py compiles these kernels with math_mode 'safe' and non-atomic
// outputs, unlike the rasterisation kernels of render_metal.py.
#pragma METAL fp contract(off)

typedef float2 sg_f2;

inline float sg_inf() {
    return numeric_limits<float>::infinity();
}

// Unfused product / sum. Contraction is off in this translation unit, so these
// are plain operators here; the CUDA twin needs the _rn intrinsics.
inline float sg_mul(float a, float b) {
    return a * b;
}

inline float sg_add(float a, float b) {
    return a + b;
}

// float fma (CUDA spells it fmaf; `fma` there is the double overload).
inline float sg_fma(float a, float b, float c) {
    return fma(a, b, c);
}

// Correctly rounded sqrt for finite x >= 0, as the CPU's fsqrt.
// The approximate root s = M * 2^(e-24) (M a 24-bit integer) is corrected with
// exact 64-bit integer tests (2M - 1)^2 < 4x/u^2 < (2M + 1)^2, u = 2^(e-24);
// 4x/u^2 is an exact integer and midpoints cannot occur.
inline float sg_sqrt(float x) {
    float s = sqrt(x);
    if (!(x >= 1.2e-38f) || !(x <= 3.4028235e38f)) {
        return s;  // 0, subnormals, inf/NaN (not produced by finite scenes)
    }
    for (int it = 0; it < 2; it++) {
        int e = 0;
        float f = frexp(s, e);
        long M = long(ldexp(f, 24));
        ulong X = ulong(ldexp(x, 2 - 2 * (e - 24)));
        long lo = 2 * M - 1;
        long hi = 2 * M + 1;
        while (ulong(hi) * ulong(hi) <= X) { M++; hi += 2; lo += 2; }
        while (lo > 0 && ulong(lo) * ulong(lo) >= X) { M--; hi -= 2; lo -= 2; }
        s = ldexp(float(M), e - 24);
    }
    return s;
}

inline float sg_dist(sg_f2 a, sg_f2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    // vector.h length_squared = square(x) + square(y): function calls, so
    // clang does not contract it
    return sg_sqrt(sg_add(sg_mul(dx, dx), sg_mul(dy, dy)));
}

// matrix.h xform_pt with clang's contraction: m(0,0)*x + m(0,1)*y + m(0,2)
inline float sg_xf(float a, float x, float b, float y, float c) {
    return sg_add(sg_fma(a, x, sg_mul(b, y)), c);
}
