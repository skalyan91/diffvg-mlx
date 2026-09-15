#pragma once

#include "diffvg.h"
#include "atomic.h"

enum class FilterType {
    Box,
    Tent,
    RadialParabolic, // 4/3(1 - (d/r))
    Hann // https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
};

struct Filter {
    FilterType type;
    float radius;
};

struct DFilter {
    float radius;
};

DEVICE
inline
float compute_filter_weight(const Filter &filter,
                            float dx,
                            float dy) {
    if (fabs(dx) > filter.radius || fabs(dy) > filter.radius) {
        return 0;
    }
    if (filter.type == FilterType::Box) {
        return 1.f / square(2 * filter.radius);
    } else if (filter.type == FilterType::Tent) {
        return (filter.radius - fabs(dx)) * (filter.radius - fabs(dy)) /
               square(square(filter.radius));
    } else if (filter.type == FilterType::RadialParabolic) {
        return (4.f / 3.f) * (1 - square(dx / filter.radius)) *
               (4.f / 3.f) * (1 - square(dy / filter.radius));
    } else {
        assert(filter.type == FilterType::Hann);
        // normalize dx, dy to [0, 1]
        auto ndx = (dx / (2*filter.radius)) + 0.5f;
        auto ndy = (dy / (2*filter.radius)) + 0.5f;
        // the normalization factor is R^2
        return 0.5f * (1.f - cos(float(2 * M_PI) * ndx)) *
               0.5f * (1.f - cos(float(2 * M_PI) * ndy)) /
               square(filter.radius);
    }
}

// d compute_filter_weight / d radius (zero outside the closed support, where
// the weight is identically zero; the jump of the box filter at the support
// border is not differentiated).
DEVICE
inline
float filter_weight_d_radius(const Filter &filter,
                             float dx,
                             float dy) {
    auto r = filter.radius;
    if (fabs(dx) > r || fabs(dy) > r) {
        return 0;
    }
    if (filter.type == FilterType::Box) {
        // w = 1 / (2r)^2
        return -2 / cubic(2 * r) * 2;
    } else if (filter.type == FilterType::Tent) {
        // w = fx * fy / r^4, fx = r - |dx|, fy = r - |dy|
        auto fx = r - fabs(dx);
        auto fy = r - fabs(dy);
        auto r4 = square(square(r));
        return (fx + fy) / r4 - 4 * fx * fy / (r4 * r);
    } else if (filter.type == FilterType::RadialParabolic) {
        // w = (4/3)^2 * gx * gy, gx = 1 - (dx/r)^2
        auto gx = 1 - square(dx / r);
        auto gy = 1 - square(dy / r);
        auto r3 = r * r * r;
        auto d_gx = 2 * square(dx) / r3;
        auto d_gy = 2 * square(dy) / r3;
        return (16.f / 9.f) * (d_gx * gy + gx * d_gy);
    } else {
        assert(filter.type == FilterType::Hann);
        // w = hx * hy / r^2, hx = 0.5 * (1 - cos(2 pi ndx)), ndx = dx / (2r) + 0.5
        auto ndx = (dx / (2*r)) + 0.5f;
        auto ndy = (dy / (2*r)) + 0.5f;
        auto hx = 0.5f * (1.f - cos(float(2*M_PI) * ndx));
        auto hy = 0.5f * (1.f - cos(float(2*M_PI) * ndy));
        auto d_hx = 0.5f * sin(float(2*M_PI) * ndx) * float(2*M_PI) * (-dx / (2 * r * r));
        auto d_hy = 0.5f * sin(float(2*M_PI) * ndy) * float(2*M_PI) * (-dy / (2 * r * r));
        return (d_hx * hy + hx * d_hy) / square(r) - 2 * hx * hy / cubic(r);
    }
}

DEVICE
inline
void d_compute_filter_weight(const Filter &filter,
                             float dx,
                             float dy,
                             float d_return,
                             DFilter *d_filter) {
    atomic_add(d_filter->radius, d_return * filter_weight_d_radius(filter, dx, dy));
}
