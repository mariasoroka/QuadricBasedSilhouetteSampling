#pragma once

#include "redner.h"
#include "matrix.h"
#include "vector.h"
#include "lp_solve.h"

namespace optsphere_normals {
    // 16 vectors uniformly distributed on the unit hemisphere
    constexpr int size = 16;
    constexpr float normals[size * 3] = {
        -0.00000000e+00, -0.00000000e+00,  1.00000000e+00,
         6.61437828e-01,  0.00000000e+00,  7.50000000e-01,
         4.05013859e-17,  6.61437828e-01,  7.50000000e-01,
        -6.61437828e-01,  8.10027719e-17,  7.50000000e-01,
        -1.21504158e-16, -6.61437828e-01,  7.50000000e-01,
         8.66025404e-01,  0.00000000e+00,  5.00000000e-01,
         5.30287619e-17,  8.66025404e-01,  5.00000000e-01,
        -8.66025404e-01,  1.06057524e-16,  5.00000000e-01,
        -1.59086286e-16, -8.66025404e-01,  5.00000000e-01,
         9.68245837e-01,  0.00000000e+00,  2.50000000e-01,
         5.92879582e-17,  9.68245837e-01,  2.50000000e-01,
        -9.68245837e-01,  1.18575916e-16,  2.50000000e-01,
        -1.77863875e-16, -9.68245837e-01,  2.50000000e-01,
         1.00000000e+00,  0.00000000e+00,  6.12323400e-17,
        -5.00000000e-01,  8.66025404e-01,  6.12323400e-17,
        -5.00000000e-01, -8.66025404e-01,  6.12323400e-17
    };
}

template <typename T>
struct TMinSphere3D {
    TVector3<T> center;
    T radius;
};

typedef TMinSphere3D<double> MinSphere3D;

// Find the optimal bounding sphere for a set of unit vectors in 3D.
// We use Welzl's algorithm with move-to-front heuristics.
// See "Fast and Robust Smallest Enclosing Balls" by Bernd Gärtner and 
// "Smallest Enclosing Disks (Balls and Ellipsoids)" by Emo Welzl.
template <typename T>
DEVICE
TMinSphere3D<T> find_optimal_bounding_sphere(bool use_gpu, const BufferView<TVector3<T>> &plane_normals) {
    int N_points = plane_normals.size();
    T safe_radius_mult = 1 + 1e-4;

    int dim = 3;

    int R[dim + 1];
    int nR = 0;
    int R_size[dim + 1];

    Buffer<int> P(use_gpu, N_points);
    for (int i = 0; i < N_points; i++) {
        P[i] = i;
    }
    int nP = N_points;
    int nP_best = 0;
    int nP_worst = N_points;
    int idx = 0;

    Buffer<int> stack(use_gpu, N_points);

    int stack_size = 0;

    TVector3<T> center;
    T radius = -1;
    bool terminate = false;

    int step = 0;

    while (!terminate){
        step += 1;        
        
        if (nP_worst == nP_best){
            if (nR == 4) {
                TMatrix3x3<T> A;
                TVector3<T> b;
                TVector3<T> qi;
                TVector3<T> qj;
                for (int i = 1; i < 4; i++) {
                    qi = plane_normals[R[i]] - plane_normals[R[0]];
                    b[i - 1] = dot(qi, qi);
                    for (int j = 1; j < 4; j++) {
                        qj = plane_normals[R[j]] - plane_normals[R[0]];
                        A(i - 1, j - 1) = 2 * dot(qi, qj);
                    }
                }
                
                TVector3<T> lams = inverse(A) * b;
                TVector3<T> C = lams[0] * (plane_normals[R[1]] - plane_normals[R[0]]) + 
                                lams[1] * (plane_normals[R[2]] - plane_normals[R[0]]) + 
                                lams[2] * (plane_normals[R[3]] - plane_normals[R[0]]);
                center = C + plane_normals[R[0]];
                radius = sqrt(dot(C, C)) * safe_radius_mult;
            }

            else if (nR == 3){
                T A[2][2];
                T b[2];
                TVector3<T> qi;
                TVector3<T> qj;

                for (int i = 1; i < 3; i++) {
                    qi = plane_normals[R[i]] - plane_normals[R[0]];
                    b[i - 1] = dot(qi, qi);
                    for (int j = 1; j < 3; j++) {
                        qj = plane_normals[R[j]] - plane_normals[R[0]];
                        A[i - 1][j - 1] = 2 * dot(qi, qj);
                    }
                }

                T lams[2];
                T A_inv[2][2];
                T det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
                A_inv[0][0] = A[1][1] / det;
                A_inv[0][1] = -A[0][1] / det;
                A_inv[1][0] = -A[1][0] / det;
                A_inv[1][1] = A[0][0] / det;

                lams[0] = A_inv[0][0] * b[0] + A_inv[0][1] * b[1];
                lams[1] = A_inv[1][0] * b[0] + A_inv[1][1] * b[1];

                TVector3<T> C = lams[0] * (plane_normals[R[1]] - plane_normals[R[0]]) + 
                                lams[1] * (plane_normals[R[2]] - plane_normals[R[0]]);

                center = C + plane_normals[R[0]];
                radius = sqrt(dot(C, C)) * safe_radius_mult;
            }

            else if (nR == 2){
                center = (plane_normals[R[0]] + plane_normals[R[1]]) / 2;
                radius = length(plane_normals[R[0]] - plane_normals[R[1]]) / 2;
                radius *= safe_radius_mult;
            }

            else if (nR == 1){
                center = plane_normals[R[0]];
                radius = 0;
            }
                
            else {
                radius = 0;
                center = TVector3<T>(0, 0, 0);
            }

        }

        if (radius != -1) {

            if (nP_best <= -nP) {
                nP_best += nP;
                nP_worst += nP;
            }

            if (nP_worst > nP) {
                nP_best -= nP;
                nP_worst -= nP;
            }
            
            for (int i = nR - 1; i >= 0; i--) {
                if (R_size[i] >= stack_size) {

                    P[(nP_best - 1 + 2 * nP) % nP] = R[i];
                    nP_best -= 1;
                    nR -= 1;
                }
            }
            
            if (stack_size == 0){
                terminate = true;
            }
            
            else {
                idx = stack[stack_size - 1];
                stack_size -= 1;
                T radius_check = length(center - plane_normals[idx]);

                if (radius_check <= radius){

                    P[(nP_worst + nP) % nP] = idx;
                    nP_worst += 1;
                }
                
                else {
                    bool duplicate = false;
                    for (int k = 0; k < nR; k++) {
                        if (plane_normals[R[k]] == plane_normals[idx]){
                            duplicate = true;
                            break;
                        }
                    }
                    
                    if (!duplicate){
                        R[nR] = idx;
                        R_size[nR] = stack_size;
                        nR += 1;
                        radius = -1;
                    }
                    
                }
            }

        }
        else {
            stack[stack_size] = P[(nP_worst - 1 + 2 * nP) % nP ];
            stack_size += 1;
            nP_worst -= 1;
        }
    }      
    return TMinSphere3D<T>{center, radius};  
}

// Find a close-to-optimal bounding sphere for the set of unit normals of the dual planes.
// We use a method proposed in "Fast and Tight Fitting Bounding Spheres" by Thomas Larsson.
template <typename T>
DEVICE
TMinSphere3D<T> find_approx_bounding_sphere(bool use_gpu, const BufferView<TVector4<T>> &dual_planes) {

    Buffer<TVector3<T>> plane_normals(use_gpu, dual_planes.size());
    for (int i = 0; i < dual_planes.size(); i++) {
        plane_normals[i] = Vector3(dual_planes[i].x, dual_planes[i].y, dual_planes[i].z);
    }

    T safe_radius_mult = 1 + 1e-4;
    int idxes[2 * optsphere_normals::size];
    int n_set = 0;

    for (int i = 0; i < optsphere_normals::size; i++) {
        T min = infinity<T>();
        T max = -infinity<T>();
        int min_idx = 0;
        int max_idx = 0;
        for (int j = 0; j < plane_normals.size(); j++) {
            T dot_prod = plane_normals[j][0] * optsphere_normals::normals[i * 3] + 
                         plane_normals[j][1] * optsphere_normals::normals[i * 3 + 1] + 
                         plane_normals[j][2] * optsphere_normals::normals[i * 3 + 2];

            if (dot_prod < min) {
                min = dot_prod;
                min_idx = j;
            }
            if (dot_prod > max) {
                max = dot_prod;
                max_idx = j;
            }
        }

        bool found_min = false;
        bool found_max = false;
        for (int j = 0; j < n_set; j++) {
            if (idxes[j] == min_idx) {
                found_min = true;
            }
            if (idxes[j] == max_idx) {
                found_max = true;
            }
        }

        if (!found_min) {
            idxes[n_set] = min_idx;
            n_set += 1;
        }
        if (!found_max) {
            idxes[n_set] = max_idx;
            n_set += 1;
        }
    }

    Buffer<TVector3<T>> new_points(use_gpu, n_set);
    for (int i = 0; i < n_set; i++) {
        new_points[i] = plane_normals[idxes[i]];
    }

    TMinSphere3D<T> sphere = find_optimal_bounding_sphere(use_gpu, new_points.view(0, n_set));

    for (int i = 0; i < plane_normals.size(); i++) {
        if (length(plane_normals[i] - sphere.center) > sphere.radius) {
            sphere.radius += length(plane_normals[i] - sphere.center);
            sphere.radius /= 2;
            sphere.center = plane_normals[i] + sphere.radius * (sphere.center - plane_normals[i]) / length(sphere.center - plane_normals[i]);
            sphere.radius *= safe_radius_mult;
        }
    }
    return sphere;
}

// Find the vector D to construct the offset quadric. 
// See Sec. 4.4 of "Quadric-Based Silhouette Sampling for Differentiable Rendering" for details.
template <typename T>
DEVICE
TVector4<T> find_offset_quadric_vector(bool use_gpu, const BufferView<TVector4<T>> &dual_planes) {
    Buffer<TVector4<T>> points(use_gpu, dual_planes.size());
    for (int i = 0; i < dual_planes.size(); i++) {
        Vector3 normal = Vector3(dual_planes[i].x, dual_planes[i].y, dual_planes[i].z);
        points[i] = dual_planes[i] / length(normal);
    }

    TMinSphere3D<T> sphere = find_approx_bounding_sphere(use_gpu, points.view(0, dual_planes.size()));
    if (sphere.radius >= 1 * (1 - 1e-5)){
        TVector4<T> lp_solve = solve_constrained_lp(points.view(0, points.size()));
        return lp_solve;
    }
    else {
        TVector3<T> center_dir = (sphere.center / length(sphere.center));
        return TVector4<T>(center_dir.x, center_dir.y, center_dir.z, T(0));
    }
}

template <typename T>
DEVICE
TMinSphere3D<T> test_find_approx_bounding_sphere(ptr<TVector4<T>> points, int size) {
    TMinSphere3D<T> result = find_approx_bounding_sphere(false, BufferView<TVector4<T>>(points.get(), size));
    return result;
}

template <typename T>
DEVICE
TVector4<T> test_find_offset_quadric_vector(ptr<TVector4<T>> dual_planes, int size) {
    TVector4<T> result = find_offset_quadric_vector(false, BufferView<TVector4<T>>(dual_planes.get(), size));
    return result;
}