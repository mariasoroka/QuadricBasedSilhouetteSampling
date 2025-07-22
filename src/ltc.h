#pragma once

#include "vector.h"
#include "matrix.h"
#include "intersection.h"

namespace ltc {

    extern const float *tabMcpu;
    extern const float *tabMgpu;
    extern const float *tabM;

}

void initialize_ltc_table(bool use_gpu);

DEVICE Matrix3x3 get_ltc_matrix(const SurfacePoint &surface_point,
                                const Vector3 &wi,
                                Real roughness,
                                const float *tabM);