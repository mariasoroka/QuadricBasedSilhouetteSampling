#include "ltc.h"
#include "ltc.inc"

const float *ltc::tabMcpu = &ltc::tabM_[0];
const float *ltc::tabMgpu = nullptr;
const float *ltc::tabM = nullptr;

void initialize_ltc_table(bool use_gpu) {
    
    
    ltc::tabM = use_gpu ? ltc::tabMgpu : ltc::tabMcpu;
    if (use_gpu && ltc::tabM == nullptr) {
#ifdef __CUDACC__
        checkCuda(cudaMallocManaged(&ltc::tabMgpu, sizeof(ltc::tabM_)));
        checkCuda(cudaMemcpy((void*)ltc::tabMgpu,
                             (void*)ltc::tabM_, sizeof(ltc::tabM_), cudaMemcpyHostToDevice));
        ltc::tabM = ltc::tabMgpu;
#else
        assert(false);
#endif 
    }
}

DEVICE Matrix3x3 get_ltc_matrix(const SurfacePoint &surface_point,
                                const Vector3 &wi,
                                Real roughness,
                                const float *tabM) {
    auto cos_theta = dot(wi, surface_point.shading_frame.n);
    auto theta = acos(cos_theta);
    // search lookup table
    int rid = clamp(int(roughness * (ltc::size - 1)), 0, ltc::size - 1);
    int tid = clamp(int((theta / (M_PI / 2.f) * (ltc::size - 1))), 0, ltc::size - 1);

    return Matrix3x3(&tabM[9 * (rid + tid * ltc::size)]);
    
    // TODO: linear interpolation?
    // auto rw = roughness * (ltc::size - 1) - rid;
    // auto tw = (theta / (M_PI / 2.f)) * (ltc::size - 1) - tid;

    // auto rid_1 = clamp(rid + 1, 0, ltc::size - 1);
    // auto tid_1 = clamp(tid + 1, 0, ltc::size - 1);

    // Matrix3x3 m0 = Matrix3x3(&tabM[9 * (rid + tid * ltc::size)]);
    // Matrix3x3 m1 = Matrix3x3(&tabM[9 * (rid_1 + tid * ltc::size)]);
    // Matrix3x3 m2 = Matrix3x3(&tabM[9 * (rid + tid_1 * ltc::size)]);
    // Matrix3x3 m3 = Matrix3x3(&tabM[9 * (rid_1 + tid_1 * ltc::size)]);

    // return m0 * (1 - rw) * (1 - tw) + m1 * rw * (1 - tw) + m2 * (1 - rw) * tw + m3 * rw * tw;
    
}