#pragma once

#include "redner.h"
#include "matrix.h"
#include "vector.h"
#include "quadric.h"

// Rejection test for "Quadrics-Based Silhouette Sampling for Differentiable Rendering" paper.
// Given a pair of bounding quadrics, a matrix m that is the orthogonal basis A from Sec. 3.2 of the paper,
// the dual bounding box, the shading point p, determine whether the corresponding node can be skipped.
// inter_points and num_points are used to store the intersection points of the dual bounding box with the plane defined by A^T x,
// where x = (p.x, p.y, p.z, 1).
template <typename T>
DEVICE 
bool rejection_test(const TQuadricPair<T> &quadric_pair, 
                    const TMatrix4x4<T> &m, 
                    const AABB3 &dual_bounds, 
                    const TVector3<T> &p,
                    TVector3<T> *inter_points,
                    int &num_points,
                    const int *bbox_edges_idxs) {
    TVector4<T> p_3d = transpose(m) * TVector4<T>(p.x, p.y, p.z, T(1));
    
    int num_points_ = 0;
    intersect_plane(dual_bounds, p_3d, inter_points, &num_points_, bbox_edges_idxs);
    num_points = num_points_;
    
    // Does the plane intersect the dual bounding box?
    if (num_points == 0) {
        return false;
    }
    else {
        TMatrix4x4<T> Q1_3d = transpose(m) * (quadric_pair).quadric1.matrix * m;
        TMatrix4x4<T> Q2_3d = transpose(m) * (quadric_pair).quadric2.matrix * m;

        // Is any of the vertices covered by the family of quadrics? See Fig. 4 (b).
        for (int i = 0; i < num_points; i++) {
            TVector4<T> vertex(inter_points[i].x, inter_points[i].y, inter_points[i].z, T(1));
            T Q1_v = dot(vertex, Q1_3d * vertex);
            T Q2_v = dot(vertex, Q2_3d * vertex);

            T alpha = Q1_v / (Q1_v - Q2_v);

            if (alpha >= 0 && alpha <= 1) {
                return true;
            }
        }

        // Does any of the polygon edges intersect any of the bounding quadrics? See Fig. 4 (c).
        TQuadric<T> Q1_3d_quad(Q1_3d);
        TQuadric<T> Q2_3d_quad(Q2_3d);
        for (int i = 0; i < num_points; i++) {
            bool inter = intersects_with_segment(Q1_3d_quad, inter_points[i], inter_points[(i + 1) % num_points]);
            if (inter) {
                return true;
            }
            inter = intersects_with_segment(Q2_3d_quad, inter_points[i], inter_points[(i + 1) % num_points]);
            if (inter) {
                return true;
            }
        }

        // Compute the conics in the intersection of the quadrics and the plane defined by the shading point p.
        TVector3<T> tmp = TVector3<T>(p_3d.x, p_3d.y, p_3d.z);
        TMatrix3x3<T> basis_3d = find_basis(tmp / length(tmp));

        TVector3<T> u(basis_3d(0, 0), basis_3d(1, 0), basis_3d(2, 0));
        TVector3<T> v(basis_3d(0, 1), basis_3d(1, 1), basis_3d(2, 1));
        
        TVector4<T> u_4d(u.x, u.y, u.z, T(0));
        TVector4<T> v_4d(v.x, v.y, v.z, T(0));
        TVector4<T> o_4d(inter_points[0].x, inter_points[0].y, inter_points[0].z, T(1));

        TConic<T> C1 = get_conic_from_quadric(Q1_3d_quad, o_4d, u_4d, v_4d);
        TConic<T> C2 = get_conic_from_quadric(Q2_3d_quad, o_4d, u_4d, v_4d);

        // Is any of the conics contained in the polygon? See Fig. 4 (d).
        TVector3<T> point_C1 = get_point_on_conic(C1);
        TVector3<T> point_C2 = get_point_on_conic(C2);

        if (length(point_C1) > 0) {
            TVector3<T> point_C1_3d = point_C1.x * u + point_C1.y * v + inter_points[0];
            if (::inside(dual_bounds, point_C1_3d)) {
                return true;
            }
        }
        if (length(point_C2) > 0) {
            TVector3<T> point_C2_3d = point_C2.x * u + point_C2.y * v + inter_points[0];
            if (::inside(dual_bounds, point_C2_3d)) {
                return true;
            }
        }
        return false;
    }
}


template <typename T>
DEVICE 
bool rejection_test_py(const TQuadricPair<T> &quadric_pair, 
                       const TMatrix4x4<T> &m, 
                       const AABB3 &dual_bounds, 
                       const TVector3<T> &p) {
    Buffer<int> bbox_edges_idxs(false, 24);
    Buffer<Vector3> bbox_edges_normals(false, 24);
    init_bbox_edges_and_normals(bbox_edges_idxs.view(0, 24), bbox_edges_normals.view(0, 24));
    Vector3 inter_points[12];
    int num_points = 0;
    return rejection_test(quadric_pair, m, dual_bounds, p, inter_points, num_points, bbox_edges_idxs.begin());
}