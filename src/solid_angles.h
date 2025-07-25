#pragma once

#include "redner.h"
#include "vector.h"
#include "matrix.h"
#include "intersection.h"
#include "buffer.h"
#include "aabb.h"
#include "ptr.h"
#include <cmath>

// Piecewise polynomial approximation of atan2 
// See https://github.com/MomentsInGraphics/vulkan_renderer/blob/38b1dc04ec3e67a311dad2f8e863eb070d076135/src/shaders/polygon_sampling.glsl
DEVICE 
inline Real atan2_approx(Real y_, Real x_){
    Real tan = y_ / x_;
	Real rx;
	Real ry;
	Real rz;
	rx = (std::abs(tan) > 1.0f) ? (1.0f / std::abs(tan)) : std::abs(tan);
	ry = rx * rx;
	rz = std::fma(ry, 0.02083509974181652f, -0.08513300120830536);
	rz = std::fma(ry, rz, 0.18014100193977356f);
	rz = std::fma(ry, rz, -0.3302994966506958f);
	ry = std::fma(ry, rz, 0.9998660087585449f);
	rz = std::fma(-2.0f * ry, rx, M_PI / 2);
	rz = (std::abs(tan) > 1.0f) ? rz : 0.0f;
	rx = std::fma(rx, ry, rz);
    if (y_ >= 0) {
	    return (tan < 0.0f) ? (M_PI - rx) : rx;
    }
    else {
        return -M_PI + ((tan < 0.0f) ? (M_PI - rx) : rx);
    }
}

// Compute the contribution of a segment to the solid angle
DEVICE 
inline Real segment_contrib_to_solid_angle(const Vector3 &v0,
                                           const Vector3 &v1){
    Real res = 0;
    Vector3 cross_prod = cross(v0, v1);
    Real norm_v0 = length(v0);
    Real norm_v1 = length(v1);
    Real tmp = std::fma(v1.z, norm_v0, dot(v0, v1));
    Real denominator = std::fma((norm_v0 + v0.z), norm_v1, tmp);
    res += atan2_approx(cross_prod.z, denominator);
    return 2 * res;
}

// Compute solid angle of a bounding box
DEVICE inline Real bbox_solid_angle(const AABB3 &bounds,
                                    const Matrix3x3 &frame_inv,
                                    const SurfacePoint &p) {
    Vector3 corners[8];
    for(int i = 0; i < 8; i++){
        corners[i] = corner(bounds, i);
    }
    Real integral = 0;

    // Check if all corners are below the surface point
    bool all_below = true;
    for(int i = 0; i < 8; i++){
        if(dot(normalize(corners[i] - p.position), p.geom_normal) > 1e-15){
            all_below = false;
        }
    }
    if(all_below){
        return 0;
    }

    // If the surface point is outside of the bounding box, compute the solid angle
    if (!::inside(bounds, p.position)) {
        int closest_corner;
        Real min_dist = infinity<Real>();
        for(int i = 0; i < 8; i++){
            Real dist_tmp = distance(corners[i], p.position);
            if(dist_tmp < min_dist){
                min_dist = dist_tmp;
                closest_corner = i;
            }
        }

        Vector3 closest_corner_dir = corners[closest_corner];
        Vector3 closest_corner_dir_rel = corners[closest_corner] - p.position;
        Vector3 e1 = corners[closest_corner ^ 1] - closest_corner_dir;
        Vector3 e2 = corners[closest_corner ^ 2] - closest_corner_dir;
        Vector3 e3 = corners[closest_corner ^ 4] - closest_corner_dir;

        Vector3 v0;
        Vector3 v1;

        if(dot(closest_corner_dir_rel + (e1 + e2) / 2 , e3) > 0){
            int idx = closest_corner;
            int idx0, idx1;
            int counter = 1;
            for(int j = 0; j < 4; j++){
                idx0 = idx;
                idx1 = idx ^ counter;
                idx = idx1;
                v0 = frame_inv * (corners[idx0] - p.position);
                v1 = frame_inv * (corners[idx1] - p.position);
                integral += segment_contrib_to_solid_angle(v0, v1);
                counter *= 2;
                counter = counter % 3;
            }
        }

        if(dot(closest_corner_dir_rel + (e3 + e2) / 2 , e1) > 0){
            int idx = closest_corner;
            int idx0, idx1;
            int counter = 2;
            for(int j = 0; j < 4; j++){
                idx0 = idx;
                idx1 = idx ^ counter;
                idx = idx1;
                v0 = frame_inv * (corners[idx0] - p.position);
                v1 = frame_inv * (corners[idx1] - p.position);
                integral += segment_contrib_to_solid_angle(v0, v1);
                counter *= 2;
                counter = counter % 6;
            }
        }

        if(dot(closest_corner_dir_rel + (e1 + e3) / 2 , e2) > 0){
            int idx = closest_corner;
            int idx0, idx1;
            int counter = 4;
            for(int j = 0; j < 4; j++){
                idx0 = idx;
                idx1 = idx ^ counter;
                idx = idx1;
                v0 = frame_inv * (corners[idx0] - p.position);
                v1 = frame_inv * (corners[idx1] - p.position);
                integral += segment_contrib_to_solid_angle(v0, v1);
                counter *= 4;
                counter = counter % 15;
            }
        }
    }
    // If the surface point is inside the bounding box, return 2 * M_PI
    else {
        integral = 2 * M_PI;
    }
    return integral >= 0 ? integral : -integral;
}

// Solid angle of a polygon viewed from a surface point p with normal 'normal'.
DEVICE 
inline Real polygon_solid_angle(int num_vert, 
                                BufferView<Vector3> vertices, 
                                const Matrix3x3 &frame_inv,
                                const Vector3 &normal,
                                const Vector3 &p) {

    Real integral = 0;
    // Check if all vertices are below the surface point
    bool all_below = true;
    for(int i = 0; i < num_vert; i++){
        if(dot((vertices[i] - p), normal) > 0){
            all_below = false;
            break;
        }
    }
    if(all_below){
        return 0;
    }

    for(int i = 0; i < num_vert; i++){
        Vector3 v0 = frame_inv * (vertices[i] - p);
        Vector3 v1 = frame_inv * (vertices[(i + 1) % num_vert] - p);
        integral += segment_contrib_to_solid_angle(v0, v1);
    }
    return integral >= 0 ? integral : -integral;
}





// Compute the contribution of a segment to the LTC integral. Clip the segment to the horizon.
DEVICE 
inline Real segment_contrib_to_ltc(const Vector3 &corner0_transf,
                                   const Vector3 &corner1_transf){

    Vector3 n(0.0, 0.0, 1.0);
    Real res = 0;
    Vector3 v0 = corner0_transf;
    Vector3 v1 = corner1_transf;


    if(v0.z < 0 && v1.z > 0) {
        Vector3 v0_prime = v0 - n * v0.z;
        Real t = - v0.z / (v1.z - v0.z);
        v0 = v1 + (v0 - v1) * t;

        Vector3 cross_prod = cross(v0_prime, v0);
        Real length_cross_prod = length(cross_prod);
        if (length_cross_prod != 0) {
            res += atan2_approx(length_cross_prod, dot(v0, v0_prime)) * cross_prod.z / (length_cross_prod);
        }
    }
    else if(v0.z > 0 && v1.z < 0) {
        Vector3 v1_prime = v1 - n * v1.z;
        Real t = - v1.z / (v0.z - v1.z);
        v1 = v0 + (v1 - v0) * t;

        Vector3 cross_prod = cross(v1, v1_prime);
        Real length_cross_prod = length(cross_prod);
        if (length_cross_prod != 0) {
            res += atan2_approx(length_cross_prod, dot(v1, v1_prime)) * cross_prod.z / (length_cross_prod);
        }
    }

    else if(v0.z < 0 && v1.z < 0) {
        v1 = v1 - n * v1.z;
        v0 = v0 - n * v0.z;
    }

    Vector3 cross_prod = cross(v0, v1);
    Real length_cross_prod = length(cross_prod);
    if (length_cross_prod != 0) {
        res += atan2_approx(length_cross_prod, dot(v0, v1)) * cross_prod.z / (length_cross_prod);
    }

    return res / (2 * M_PI);
}

DEVICE inline Real bbox_ltc(const AABB3 &bounds,
                            const SurfacePoint &p,
                            const Matrix3x3 &m,
                            const Matrix3x3 &m_inv,
                            const Matrix3x3 &iso_frame) {

    Real integral = 0;
    // Check if all corners are below the surface point in the original coordinate frame
    bool all_below = true;
    for(int i = 0; i < 8; i++){
        if(dot(normalize(corner(bounds, i) - p.position), p.geom_normal) > 1e-15){
            all_below = false;
            break;
        }
    }
    if(all_below){
        return 0;
    }

    // Check if all corners are below the surface point in the transformed coordinate frame
    all_below = true;
    for(int i = 0; i < 8; i++){
        if((m_inv * (corner(bounds, i) - p.position)).z > 0){
            all_below = false;
            break;
        }
    }
    if(all_below){
        return 0;
    }

    if (!::inside(bounds, p.position)) {
        int closest_corner;
        Real min_dist = infinity<Real>();
        for(int i = 0; i < 8; i++){
            Real dist_tmp = distance(corner(bounds, i), p.position);
            if(dist_tmp < min_dist){
                min_dist = dist_tmp;
                closest_corner = i;
            }
        }

        Vector3 closest_corner_dir = corner(bounds, closest_corner) - p.position;
        Vector3 e1 = corner(bounds, closest_corner ^ 1) - corner(bounds, closest_corner);
        Vector3 e2 = corner(bounds, closest_corner ^ 2) - corner(bounds, closest_corner);
        Vector3 e3 = corner(bounds, closest_corner ^ 4) - corner(bounds, closest_corner);


        if(dot(closest_corner_dir + (e1 + e2) / 2, e3) > 0){

            int idx = closest_corner;
            int idx0, idx1;
            int counter = 1;
            for(int j = 0; j < 4; j++){
                idx0 = idx;
                idx1 = idx ^ counter;
                idx = idx1;
                integral += segment_contrib_to_ltc(m_inv * (corner(bounds, idx0) - p.position), 
                                                   m_inv * (corner(bounds, idx1) - p.position));
                counter *= 2;
                counter = counter % 3;
            }
        }

        if(dot(closest_corner_dir + (e2 + e3) / 2, e1) > 0){
            int idx = closest_corner;
            int idx0, idx1;
            int counter = 2;
            for(int j = 0; j < 4; j++){
                idx0 = idx;
                idx1 = idx ^ counter;
                idx = idx1;
                integral += segment_contrib_to_ltc(m_inv * (corner(bounds, idx0) - p.position), 
                                                   m_inv * (corner(bounds, idx1) - p.position));
                counter *= 2;
                counter = counter % 6;
            }
        }

        if(dot(closest_corner_dir + (e1 + e3) / 2, e2) > 0){
            int idx = closest_corner;
            int idx0, idx1;
            int counter = 4;
            for(int j = 0; j < 4; j++){
                idx0 = idx;
                idx1 = idx ^ counter;
                idx = idx1;
                integral += segment_contrib_to_ltc(m_inv * (corner(bounds, idx0) - p.position), 
                                                   m_inv * (corner(bounds, idx1) - p.position));
                counter *= 4;
                counter = counter % 15;
            }
        }

    }
    else {
        integral = 1;
    }
    return integral >= 0 ? integral : -integral;
}

// Compute LTC integral over a polygon
DEVICE 
inline Real polygon_ltc(int num_vert, 
                        BufferView<Vector3> vertices,
                        const Matrix3x3 &m,
                        const Matrix3x3 &m_inv,
                        const Matrix3x3 &iso_frame,
                        const Vector3 &normal,
                        const Vector3 &p) {

    Real integral = 0;
    bool all_below = true;
    for(int i = 0; i < 8; i++){
        if(dot((vertices[i] - p), normal) > 0){
            all_below = false;
            break;
        }
    }
    if(all_below){
        return 0;
    }

    all_below = true;

    for(int i = 0; i < num_vert; i++){
        if((m_inv * (vertices[i] - p)).z > 0){
            all_below = false;
            break;
        }
    }
    if(all_below){
        return 0;
    }

    for(int i = 0; i < num_vert; i++){
        Vector3 v0 = vertices[i] - p;
        Vector3 v1 = vertices[(i + 1) % num_vert] - p;
        integral += segment_contrib_to_ltc(m_inv * v0, m_inv * v1);
    }
    return integral >= 0 ? integral : -integral;
}

DEVICE 
inline void get_bbox_silhouette(const Vector3 *corners,
                                       const Vector3 &p,
                                       const int *bbox_edges_idxs,
                                        const Vector3 *bbox_edges_normals,
                                        int *silhouette_edges,
                                        Vector3 &avg,
                                        int &n_silh){
    Real dot0 = 0;
    Real dot1 = 0;
    for (int i = 0; i < 12; i++) {
        dot0 = dot(p - corners[*(bbox_edges_idxs + i * 2)], *(bbox_edges_normals + i * 2));
        dot1 = dot(p - corners[*(bbox_edges_idxs + i * 2)], *(bbox_edges_normals + i * 2 + 1));
        if (dot0 * dot1 < 0) {
            silhouette_edges[n_silh] = *(bbox_edges_idxs + i * 2);
            silhouette_edges[n_silh + 1] = *(bbox_edges_idxs + i * 2 + 1);
            avg += normalize(corners[silhouette_edges[n_silh]] - p) + normalize(corners[silhouette_edges[n_silh + 1]] - p);
            n_silh += 2;
        }
    }
}

// Compute the average BSDF over a bounding box
DEVICE 
inline Real bbox_average_bsdf(const AABB3 &bounds,
                              const SurfacePoint &p,
                              const Matrix3x3 &m_inv,
                              const Matrix3x3 &frame_inv,
                              const int *bbox_edges_idxs,
                              const Vector3 *bbox_edges_normals) {

    Real integral = 0;
    Real ltc_integral = 0;

    Vector3 corners[8];
    for(int i = 0; i < 8; i++){
        corners[i] = corner(bounds, i);
    }

    bool all_below = true;
    for(int i = 0; i < 8; i++){
        if(dot(normalize(corners[i] - p.position), p.geom_normal) > 1e-10){
            all_below = false;
        }
    }
    if(all_below){
        return 0;
    }

    if (!::inside(bounds, p.position)) {
        int silhouette_edges[12];

        Vector3 frame_inv_corners[8];
        Vector3 m_inv_corners[8];
        for(int i = 0; i < 8; i++){
            frame_inv_corners[i] = frame_inv * (corners[i] - p.position);
            m_inv_corners[i] = m_inv * (corners[i] - p.position);
        }

        bool all_below_ltc = true;
        for(int i = 0; i < 8; i++){
            if((m_inv_corners[i]).z > 0){
                all_below_ltc = false;
            }
        }
        if(all_below_ltc){
            return 0;
        }

        
        Vector3 avg(0.0, 0.0, 0.0);
        int n_silh = 0;
        get_bbox_silhouette(corners, 
                            p.position, 
                            bbox_edges_idxs, 
                            bbox_edges_normals, 
                            silhouette_edges, 
                            avg, 
                            n_silh);


        int idx0 = 0;
        int idx1 = 0;
        for(int i = 0; i < n_silh; i += 2) {
            idx0 = silhouette_edges[i];
            idx1 = silhouette_edges[i + 1];
            if (dot(cross(corners[idx0] - p.position, corners[idx1] - p.position), avg) < 0) {
                idx0 = silhouette_edges[i + 1];
                idx1 = silhouette_edges[i];
            }
            integral += segment_contrib_to_solid_angle(frame_inv_corners[idx0], frame_inv_corners[idx1]);
            ltc_integral += segment_contrib_to_ltc(m_inv_corners[idx0], m_inv_corners[idx1]);
        }
    }
    else {
        integral = 2 * M_PI;
        ltc_integral = 1.0;
    }

    if (integral == 0) {
        return 0;
    }

    Real avg_bsdf = ltc_integral / integral;
    return avg_bsdf >= 0 ? avg_bsdf : -avg_bsdf;
}


DEVICE 
Real test_polygon_solid_angle(int num_vert, ptr<Vector3> vertices);


DEVICE
Real test_bbox_average_bsdf(const AABB3 &bounds,
                            const SurfacePoint &p,
                            const Matrix3x3 &m,
                            const Matrix3x3 &m_inv,
                            const Matrix3x3 &frame_inv);