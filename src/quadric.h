#pragma once

#include "redner.h"
#include "matrix.h"
#include "vector.h"
#include "ptr.h"
#include <fstream>
#include <string>
template <typename T>
struct TConic {
    DEVICE
    TConic() {
        matrix = TMatrix3x3<T>();
    }

    template <typename T2>
    DEVICE
    TConic(TMatrix3x3<T2> &matr) {
        matrix = TMatrix3x3<T>(matr);
    }
    TMatrix3x3<T> matrix;
};

template <typename T>
struct TQuadric {
    DEVICE
    TQuadric() {
        matrix = TMatrix4x4<T>();
    }

    template <typename T2>
    DEVICE
    TQuadric(TMatrix4x4<T2> &matr) {
        matrix = TMatrix4x4<T>(matr);
    }

    template <typename T2>
    DEVICE
    TQuadric(const TMatrix4x4<T2> &matr) {
        matrix = TMatrix4x4<T>(matr);
    }


    TMatrix4x4<T> matrix;
};

template <typename T>
struct TQuadricPair {
    DEVICE
    TQuadricPair() {
        quadric1 = TQuadric<T>();
        quadric2 = TQuadric<T>();
    }

    template <typename T2>
    DEVICE
    TQuadricPair(TQuadric<T2> &q1, TQuadric<T2> &q2) {
        quadric1 = TQuadric<T>(q1.matrix);
        quadric2 = TQuadric<T>(q2.matrix);
    }
    
    TQuadric<T> quadric1;
    TQuadric<T> quadric2;
};

template <typename T>
struct TConicIntersection {
    TVector3<T> p1;
    TVector3<T> p2;
};

template <typename T>
struct TQuadricIntersection {
    TVector4<T> p1;
    TVector4<T> p2;
};

using Conic = TConic<Real>;
using Quadric = TQuadric<Real>;
using QuadricPair = TQuadricPair<Real>;
using ConicIntersection = TConicIntersection<Real>;
using QuadricIntersection = TQuadricIntersection<Real>;

// Intersect a conic with a line. See Section 11.3 from "Perspectives on Projective Geometry"
// by Jürgen Richter-Gebert for details.
template <typename T>
DEVICE
TConicIntersection<T> intersect_with_line(TConic<T> &C, TVector3<T> &line) {

    TMatrix3x3<T> mu_h = TMatrix3x3<T>(T(0), line[2], -line[1],
                                        -line[2], T(0), line[0],
                                        line[1], -line[0], T(0));

    TMatrix3x3<T> B = transpose(mu_h) * C.matrix * mu_h;
    size_t pivot = argmax(abs(line));
    T scale = line[pivot];

    T minor = minor_det(B, pivot);

    if (minor > 0) {
        return TConicIntersection<T>{TVector3<T>(0, 0, 0), TVector3<T>(0, 0, 0)};
    }

    else {
        T alpha = sqrt(-minor) / scale;
        TMatrix3x3<T> C = B + mu_h * alpha;

        size_t idx = argmax(abs(TVector3<T>(C(0, 0), C(1, 1), C(2, 2))));
        TVector3<T> q = TVector3<T>(C(0, idx), C(1, idx), C(2, idx));
        TVector3<T> r = TVector3<T>(C(idx, 0), C(idx, 1), C(idx, 2));

        q = q / q[2];
        r = r / r[2];

        return TConicIntersection<T>{q, r};
    }
}

// Get a point on the conic defined by matrix C. See Appendix C of 
// "Quadrics-Based Silhouete Sampling for Differentiable Rendering" by Soroka et al. for details.
template <typename T>
DEVICE
TVector3<T> get_point_on_conic(TConic<T> &C) {
    TVector3<T> point = TVector3<T>(T(1), T(0), T(0));
    TVector3<T> line = TVector3<T>(T(0), T(1), T(0));
    TVector3<T> polar = C.matrix * point;
    normalize(polar);
    TConicIntersection<T> intersection = intersect_with_line(C, line);
    if (length(intersection.p1) > 0) {
        return intersection.p1;
    }
    else {
        intersection = intersect_with_line(C, polar);
        return intersection.p1;
    }
}

// Get a conic in the intersection of a quadric and a plane defined by two vectors u and v, and an offset o.
template <typename T>
DEVICE
TConic<T> get_conic_from_quadric(TQuadric<T> &quadric, TVector4<T> &o, TVector4<T> &u, TVector4<T> &v) {
    TMatrix3x3<T> matrix;
    matrix(0, 0) = dot(u, quadric.matrix * u);
    matrix(0, 1) = 0.5 * (dot(u, quadric.matrix * v) + dot(v, quadric.matrix * u));
    matrix(0, 2) = 0.5 * (dot(u, quadric.matrix * o) + dot(o, quadric.matrix * u));
    matrix(1, 0) = matrix(0, 1);
    matrix(1, 1) = dot(v, quadric.matrix * v);
    matrix(1, 2) = 0.5 * (dot(o, quadric.matrix * v) + dot(v, quadric.matrix * o));
    matrix(2, 0) = matrix(0, 2);
    matrix(2, 1) = matrix(1, 2);
    matrix(2, 2) = dot(o, quadric.matrix * o);
    return TConic<T>(matrix);
}

// Is there an intersection between a quadric and a segment defined by two endpoints v0 and v1?
template <typename T>
DEVICE
bool intersects_with_segment(TQuadric<T> &quadric, TVector3<T> &v0, TVector3<T> &v1) {
    TVector4<T> v0_hom = TVector4<T>(v0[0], v0[1], v0[2], T(1.0));
    TVector4<T> v1_hom = TVector4<T>(v1[0], v1[1], v1[2], T(1.0));

    T a2 = dot((v1_hom - v0_hom), quadric.matrix * (v1_hom - v0_hom));
    T a0 = dot(v0_hom, quadric.matrix * v0_hom) / a2;
    T a1 = 2 * dot(v0_hom, quadric.matrix * (v1_hom - v0_hom)) / a2;

    T discr = a1 * a1 - 4 * a0;
    if (discr < 0) {
        return false;
    }
    else if (discr < 1e-10) {
        T t = -0.5 * a1;
        if (t < 0 || t > 1) {
            return false;
        }
        else {
            return true;
        }
    }
    else {
        T t1 = (-a1 - sqrt(discr)) * 0.5;
        T t2 = (-a1 + sqrt(discr)) * 0.5;
        if (t1 < 0 || t1 > 1) {
            if (t2 < 0 || t2 > 1) {
                return false;
            }
            else {
                return true;
            }
        }
        else {
            return true;
        }
    }
}

// Intersect a quadric with a segment defined by two endpoints v0 and v1.
template <typename T>
DEVICE
TQuadricIntersection<T> intersect_with_segment(TQuadric<T> &quadric, TVector3<T> &v0, TVector3<T> &v1) {
    TVector4<T> v0_hom = TVector4<T>(v0[0], v0[1], v0[2], T(1.0));
    TVector4<T> v1_hom = TVector4<T>(v1[0], v1[1], v1[2], T(1.0));

    T a2 = dot((v1_hom - v0_hom), quadric.matrix * (v1_hom - v0_hom));
    T a0 = dot(v0_hom, quadric.matrix * v0_hom) / a2;
    T a1 = 2 * dot(v0_hom, quadric.matrix * (v1_hom - v0_hom)) / a2;

    T discr = a1 * a1 - 4 * a0;
    if (discr < 0) {
        return TQuadricIntersection<T>{TVector4<T>(T(0), T(0), T(0), T(0)), TVector4<T>(T(0), T(0), T(0), T(0))};
    }
    else if (discr < 1e-10) {
        T t = -0.5 * a1;
        if (t < 0 || t > 1) {
            return TQuadricIntersection<T>{TVector4<T>(T(0), T(0), T(0), T(0)), TVector4<T>(T(0), T(0), T(0), T(0))};
        }
        else {
            return TQuadricIntersection<T>{v0_hom + t * (v1_hom - v0_hom), TVector4<T>(T(0), T(0), T(0), T(0))};
        }
    }
    else {
        T t1 = (-a1 - sqrt(discr)) * 0.5;
        T t2 = (-a1 + sqrt(discr)) * 0.5;
        if (t1 < 0 || t1 > 1) {
            if (t2 < 0 || t2 > 1) {
                return TQuadricIntersection<T>{TVector4<T>(T(0), T(0), T(0), T(0)), TVector4<T>(T(0), T(0), T(0), T(0))};
            }
            else {
                return TQuadricIntersection<T>{v0_hom + t2 * (v1_hom - v0_hom), TVector4<T>(T(0), T(0), T(0), T(0))};
            }
        }
        else {
            if (t2 < 0 || t2 > 1) {
                return TQuadricIntersection<T>{v0_hom + t1 * (v1_hom - v0_hom), TVector4<T>(T(0), T(0), T(0), T(0))};
            }
            else {
                return TQuadricIntersection<T>{v0_hom + t1 * (v1_hom - v0_hom), v0_hom + t2 * (v1_hom - v0_hom)};
            }
        }
    }
}
// Compute LU decomposition of a matrix M and store the result in L, U and P.
template <typename T>
DEVICE
void compute_LU(T *M, T *L, T *U, int *P) {
    for (int i = 0; i < 10; i++) {
        P[i] = i;
        for (int j = 0; j < 10; j++) {
            U[i * 10 + j] = M[i * 10 + j];
            if (i == j) {
                L[i * 10 + j] = 1;
            }
            else {
                L[i * 10 + j] = 0;
            }
        }
    }
    for (int i = 0; i < 10; i++) {
        T max_val = -1;
        int max_idx = -1;
        for (int j = i; j < 10; j++) {
            if (fabs(U[j * 10 + i]) > max_val) {
                max_val = fabs(U[j * 10 + i]);
                max_idx = j;
            }
        }
        if (max_idx != i) {
            for (int j = i; j < 10; j++) {
                T tmp = U[i * 10 + j];
                U[i * 10 + j] = U[max_idx * 10 + j];
                U[max_idx * 10 + j] = tmp;
            }
            for (int j = 0; j < i; j++) {
                T tmp = L[i * 10 + j];
                L[i * 10 + j] = L[max_idx * 10 + j];
                L[max_idx * 10 + j] = tmp;
            }
            swap_(P[i], P[max_idx]);
        }
        if (U[i * 10 + i] != 0) {
            for (int j = i + 1; j < 10; j++) {
                L[j * 10 + i] = U[j * 10 + i] / U[i * 10 + i];
                for (int k = i; k < 10; k++) {
                    U[j * 10 + k] -= L[j * 10 + i] * U[i * 10 + k];
                }
            }
        }
    }
} 

// Solve the linear system Ax = b using LU decomposition, where A is decomposed into L and U matrices, and P is a permutation vector.
template <typename T>
DEVICE
void solve_LU(T *L, T *U, int* P, T* rhs, T* eigvect) {
    T rhs_tmp[10];
    T sum = 0;
    for (int i = 0; i < 10; i++) {
        rhs_tmp[i] = rhs[P[i]];
    }
    

    for (int i = 0; i < 10; i++) {
        sum = 0;
        for (int j = 0; j < i; j++) {
            sum += L[i * 10 + j] * eigvect[j];
        }
        eigvect[i] = (rhs_tmp[i] - sum) / L[i * 10 + i];
    }

    for (int i = 9; i >= 0; i--) {
        sum = 0;
        for (int j = i + 1; j < 10; j++) {
            sum += U[i * 10 + j] * eigvect[j];
        }
        if (U[i * 10 + i] == 0) {
            assert (eigvect[i] - sum == 0);
            eigvect[i] = 0;
        }
        else{
            eigvect[i] = (eigvect[i] - sum) / U[i * 10 + i];
        }
    }
}

// Solve generalized eigenvalue problem Ax = λBx
template <typename T>
DEVICE
void solve_gen_eig(T *A, T *B, T* eigvect, T* eigval) {
    T prev_eigval = T(0);
    T norm_eigvect_sq = T(0);
    for (int i = 0; i < 10; i++) {
        eigvect[i] = T(i);
        norm_eigvect_sq += eigvect[i] * eigvect[i];
    }
    T norm_eigvect = sqrt(norm_eigvect_sq);
    for (int i = 0; i < 10; i++) {
        eigvect[i] /= norm_eigvect;
    }

    T L[100];
    T U[100];
    int P[10];
    compute_LU(B, L, U, P);

    bool terminate1 = false;
    bool terminate2 = false;
    *eigval = T(1);
    T rhs[10];
    T error[10];
    T M[100];

    T norm = 0;
    T numerator = T(0);
    T denominator = T(0);
    T error_norm = T(0);
    int counter = 0;

    while ((!terminate1) || (!terminate2)) {
        counter++;
        numerator = T(0);
        denominator = T(0);
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                numerator += A[i * 10 + j] * eigvect[j] * eigvect[i];
                denominator += B[i * 10 + j] * eigvect[j] * eigvect[i];
            }
        }
        *eigval = numerator / denominator;

        for (int i = 0; i < 10; i++) {
            rhs[i] = 0;
            for (int j = 0; j < 10; j++) {
                if (!terminate1) {
                    rhs[i] += A[i * 10 + j] * eigvect[j];
                }
                else {
                    rhs[i] += B[i * 10 + j] * eigvect[j];   
                }
            }
        }
        if (!terminate1) {
            solve_LU(L, U, P, rhs, eigvect);
        }
        else {
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    M[i * 10 + j] = A[i * 10 + j] - *eigval * B[i * 10 + j];
                }
            }
            compute_LU(M, L, U, P);

            bool terminate = false;
            for (int i = 0; i < 10; i++) {
                if (U[i * 10 + i] == 0) {
                    terminate = true;
                }
            }
            if (terminate) {
                break;
            }
            solve_LU(L, U, P, rhs, eigvect);
        }

        norm = 0;

        for (int i = 0; i < 10; i++) {
            norm += eigvect[i] * eigvect[i];
        }

        norm = sqrt(norm);

        for (int i = 0; i < 10; i++) {
            eigvect[i] /= norm;
        }

        error_norm = fabs(prev_eigval - *eigval);
        prev_eigval = *eigval;

        terminate1 = terminate1 || (error_norm < 1e-3 || counter > 20);
        terminate2 = (error_norm < 1e-12 || counter > 30);
    }  
}

// Fit a quadric to a set of dual planes. For details, see "Quadrics-Based Silhouette Sampling for Differentiable Rendering" Sec. 4.3.
template <typename T>
DEVICE
TQuadric<T> fit_quadric(const BufferView<TVector4<T>> &dual_planes) {

    T M[100];
    T N[100];
    T l[10];
    T lx[10];

    for (int j = 0; j < 10; j++) {
        for (int k = 0; k < 10; k++) {
            M[j * 10 + k] = 0;
            N[j * 10 + k] = 0;
        }
    }
    for (int i = 0; i < dual_planes.size(); i++) {
        assert ((dual_planes[i].x * dual_planes[i].x + dual_planes[i].y * dual_planes[i].y + dual_planes[i].z * dual_planes[i].z - T(1)) < 1e-5);
        l[0] = dual_planes[i].x * dual_planes[i].x;
        l[1] = 2 * dual_planes[i].x * dual_planes[i].y;
        l[2] = 2 * dual_planes[i].x * dual_planes[i].z;
        l[3] = 2 * dual_planes[i].x * dual_planes[i].w;
        l[4] = dual_planes[i].y * dual_planes[i].y;
        l[5] = 2 * dual_planes[i].y * dual_planes[i].z;
        l[6] = 2 * dual_planes[i].y * dual_planes[i].w;
        l[7] = dual_planes[i].z * dual_planes[i].z;
        l[8] = 2 * dual_planes[i].z * dual_planes[i].w;
        l[9] = dual_planes[i].w * dual_planes[i].w;

        lx[0] = 0;
        lx[1] = 0;
        lx[2] = 0;
        lx[3] = dual_planes[i].x;
        lx[4] = 0;
        lx[5] = 0;
        lx[6] = dual_planes[i].y;
        lx[7] = 0;
        lx[8] = dual_planes[i].z;
        lx[9] = dual_planes[i].w;

        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                M[j * 10 + k] += l[j] * l[k] / dual_planes.size();
                N[j * 10 + k] += lx[j] * lx[k] / dual_planes.size();
            }
        }
    }

    T eigvect[10];
    T eigval;
    solve_gen_eig(N, M, eigvect, &eigval);
    
    TMatrix4x4<T> quadric_matrix;
    quadric_matrix(0, 0) = eigvect[0];
    quadric_matrix(0, 1) = eigvect[1];
    quadric_matrix(0, 2) = eigvect[2];
    quadric_matrix(0, 3) = eigvect[3];
    quadric_matrix(1, 0) = quadric_matrix(0, 1);
    quadric_matrix(1, 1) = eigvect[4];
    quadric_matrix(1, 2) = eigvect[5];
    quadric_matrix(1, 3) = eigvect[6];
    quadric_matrix(2, 0) = quadric_matrix(0, 2);
    quadric_matrix(2, 1) = quadric_matrix(1, 2);
    quadric_matrix(2, 2) = eigvect[7];
    quadric_matrix(2, 3) = eigvect[8];
    quadric_matrix(3, 0) = quadric_matrix(0, 3);
    quadric_matrix(3, 1) = quadric_matrix(1, 3);
    quadric_matrix(3, 2) = quadric_matrix(2, 3);
    quadric_matrix(3, 3) = eigvect[9];

    return TQuadric<T>(quadric_matrix);
};

template <typename T>
DEVICE
void solve_gen_eig_py(ptr<T> A, ptr<T> B, ptr<T> eigvect) {
    T eigval;
    solve_gen_eig(A.get_pointer(), B.get_pointer(), eigvect.get_pointer(), &eigval);
}

template<typename T>
DEVICE
void compute_LU_py(ptr<T> M, ptr<T> L, ptr<T> U, ptr<int> P) {
    compute_LU(M.get_pointer(), L.get_pointer(), U.get_pointer(), P.get_pointer());
}

template<typename T>
DEVICE
void solve_LU_py(ptr<T> L, ptr<T> U, ptr<int> P, ptr<T> rhs, ptr<T> eigvect) {
    solve_LU(L.get_pointer(), U.get_pointer(), P.get_pointer(), rhs.get_pointer(), eigvect.get_pointer());
}

template <typename T>
DEVICE
TQuadric<T> fit_quadric_file_input(std::string filename) {
    std::ifstream file(filename);
    TVector4<T>* buffer = new TVector4<T>[100];
    int i = 0;
    while (file >> buffer[i].x >> buffer[i].y >> buffer[i].z >> buffer[i].w) {
        i++;
    }
    BufferView<TVector4<T>> dual_planes(buffer, 100);
    TQuadric<T> quadric = fit_quadric(dual_planes);
    delete[] buffer;
    return quadric;
}
