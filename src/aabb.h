#pragma once

#include "redner.h"
#include "vector.h"
#include "matrix.h"
#include "cuda_utils.h"
#include "ray.h"
#include "ptr.h"
#include "buffer.h"

#include <iostream>

struct AABB3 {
    DEVICE AABB3(
        const Vector3 &p_min = Vector3{ infinity<Real>(),  infinity<Real>(),  infinity<Real>()},
        const Vector3 &p_max = Vector3{-infinity<Real>(), -infinity<Real>(), -infinity<Real>()})
            : p_min(p_min), p_max(p_max) {}

    DEVICE
    const Vector3& operator[](int i) const {
        return i == 0 ? p_min : p_max;
    }

    Vector3 p_min;
    Vector3 p_max;
};

struct AABB6 {
    DEVICE AABB6(
        const Vector3 &p_min = Vector3{ infinity<Real>(),  infinity<Real>(),  infinity<Real>()},
        const Vector3 &d_min = Vector3{ infinity<Real>(),  infinity<Real>(),  infinity<Real>()},
        const Vector3 &p_max = Vector3{-infinity<Real>(), -infinity<Real>(), -infinity<Real>()},
        const Vector3 &d_max = Vector3{-infinity<Real>(), -infinity<Real>(), -infinity<Real>()})
            : p_min(p_min), d_min(d_min), p_max(p_max), d_max(d_max) {}

    Vector3 p_min, d_min;
    Vector3 p_max, d_max;
};

template<typename T>
DEVICE
inline T convert_aabb(const AABB6 &b) {
    assert(false);
}

template<>
DEVICE
inline AABB3 convert_aabb(const AABB6 &b) {
    return AABB3{b.p_min, b.p_max};
}

template<>
DEVICE
inline AABB6 convert_aabb(const AABB6 &b) {
    return b;
}

struct Sphere {
    Vector3 center;
    Real radius;
};

DEVICE
inline Vector3 corner(const AABB3 &b, int i) {
    Vector3 ret;
    ret[0] = ((i & 1) == 0) ? b.p_min[0] : b.p_max[0];
    ret[1] = ((i & 2) == 0) ? b.p_min[1] : b.p_max[1];
    ret[2] = ((i & 4) == 0) ? b.p_min[2] : b.p_max[2];
    return ret;
}

DEVICE
inline AABB3 merge(const AABB3 &b, const Vector3 &p) {
    return AABB3{
        Vector3{
            min(b.p_min[0], p[0]),
            min(b.p_min[1], p[1]),
            min(b.p_min[2], p[2])},
        Vector3{
            max(b.p_max[0], p[0]),
            max(b.p_max[1], p[1]),
            max(b.p_max[2], p[2])}};
}

DEVICE
inline AABB3 merge(const AABB3 &b0, const AABB3 &b1) {
    return AABB3{
        Vector3{
            min(b0.p_min[0], b1.p_min[0]),
            min(b0.p_min[1], b1.p_min[1]),
            min(b0.p_min[2], b1.p_min[2])},
        Vector3{
            max(b0.p_max[0], b1.p_max[0]),
            max(b0.p_max[1], b1.p_max[1]),
            max(b0.p_max[2], b1.p_max[2])}};
}

DEVICE
inline AABB6 merge(const AABB6 &b0, const AABB6 &b1) {
    return AABB6{
        Vector3{
            min(b0.p_min[0], b1.p_min[0]),
            min(b0.p_min[1], b1.p_min[1]),
            min(b0.p_min[2], b1.p_min[2])},
        Vector3{
            min(b0.d_min[0], b1.d_min[0]),
            min(b0.d_min[1], b1.d_min[1]),
            min(b0.d_min[2], b1.d_min[2])},
        Vector3{
            max(b0.p_max[0], b1.p_max[0]),
            max(b0.p_max[1], b1.p_max[1]),
            max(b0.p_max[2], b1.p_max[2])},
        Vector3{
            max(b0.d_max[0], b1.d_max[0]),
            max(b0.d_max[1], b1.d_max[1]),
            max(b0.d_max[2], b1.d_max[2])}};
}

DEVICE
inline Sphere compute_bounding_sphere(const AABB3 &b) {
    auto c = 0.5f * (b.p_max + b.p_min);
    auto r = distance(c, b.p_max);
    return Sphere{c, r};
}

DEVICE
inline Sphere compute_bounding_sphere(const AABB6 &b) {
    auto c = 0.5f * (b.p_max + b.p_min);
    auto r = distance(c, b.p_max);
    return Sphere{c, r};
}

DEVICE
inline bool inside(const AABB3 &b, const Vector3 &p) {
    return p.x >= b.p_min.x && p.x <= b.p_max.x &&
           p.y >= b.p_min.y && p.y <= b.p_max.y &&
           p.z >= b.p_min.z && p.z <= b.p_max.z;
}

DEVICE
inline bool inside(const AABB6 &b, const Vector3 &p) {
    return p.x >= b.p_min.x && p.x <= b.p_max.x &&
           p.y >= b.p_min.y && p.y <= b.p_max.y &&
           p.z >= b.p_min.z && p.z <= b.p_max.z;
}

DEVICE
inline bool inside(const Sphere &b, const Vector3 &p) {
    return distance(p, b.center) <= b.radius;
}

DEVICE
inline Vector3 center(const AABB3 &b) {
    return 0.5f * (b.p_max + b.p_min);
}


DEVICE
inline bool intersect(const Sphere &s, const AABB3 &b) {
    // "A Simple Method for Box-Sphere Intersection Testing", Jim Arvo
    // https://github.com/erich666/GraphicsGems/blob/master/gems/BoxSphere.c
    auto d_min = Real(0);
    auto r2 = square(s.radius);
    for(int i = 0; i < 3; i++) {
        if (s.center[i] < b.p_min[i]) {
            d_min += square(s.center[i] - b.p_min[i]);
        } else if (s.center[i] > b.p_max[i]) {
            d_min += square(s.center[i] - b.p_max[i]);
        }
        if (d_min <= r2) {
            return true;
        }
    }
    return false;
}

DEVICE
inline bool intersect(const AABB3 &b, const Ray &r, Real expand_dist = 0) {
    // From https://github.com/mmp/pbrt-v3/blob/master/src/core/geometry.h
    auto t0 = r.tmin, t1 = r.tmax;
    for (int i = 0; i < 3; i++) {
        // Update interval for _i_th bounding box slab
        auto inv_ray_dir = 1 / r.dir[i];
        auto t_near = (b.p_min[i] - expand_dist - r.org[i]) * inv_ray_dir;
        auto t_far = (b.p_max[i] + expand_dist - r.org[i]) * inv_ray_dir;

        // Update parametric interval from slab intersection $t$ values
        if (t_near > t_far) {
            swap_(t_near, t_far);
        }

        // Update t_far to ensure robust ray bounds intersection
        t_far *= (1 + 1e-6f);
        t0 = t_near > t0 ? t_near : t0;
        t1 = t_far < t1 ? t_far : t1;
        if (t0 > t1) {
            return false;
        }
    }
    return true;
}

DEVICE
inline bool intersect(const AABB6 &b, const Ray &r, Real expand_dist = 0) {
    return intersect(convert_aabb<AABB3>(b), r, expand_dist);
}

DEVICE
inline bool intersect(const AABB3 &b, const Ray &r,
                      const Vector3 &inv_dir, const TVector3<bool> dir_is_neg,
                      Real expand_dist = 0) {
    // From https://github.com/mmp/pbrt-v3/blob/master/src/core/geometry.h
    auto t_min = (b[dir_is_neg[0]].x - r.org.x) * inv_dir.x;
    auto t_max = (b[1 - dir_is_neg[0]].x - r.org.x) * inv_dir.x;
    auto ty_min = (b[dir_is_neg[1]].y - r.org.y) * inv_dir.y;
    auto ty_max = (b[1 - dir_is_neg[1]].y - r.org.y) * inv_dir.y;
    // Update tMax and tyMax to ensure robust bounds intersection
    t_max *= (1 + 1e-6f);
    t_min *= (1 + 1e-6f);
    if (t_min > ty_max || ty_min > t_max) {
        return false;
    }
    if (ty_min > t_min) {
        t_min = ty_min;
    }
    if (ty_max < t_max) {
        t_max = ty_max;
    }
    // Check for ray intersection against z slab
    auto tz_min = (b[dir_is_neg[2]].z - r.org.z) * inv_dir.z;
    auto tz_max = (b[1 - dir_is_neg[2]].z - r.org.z) * inv_dir.z;
    // Update tzMax to ensure robust bounds intersection
    tz_max *= (1 + 1e-6f);
    if (t_min > tz_max || tz_min > t_max) return false;
    if (tz_min > t_min) t_min = tz_min;
    if (tz_max < t_max) t_max = tz_max;
    return (t_min < r.tmax) && (t_max > r.tmin);
}

DEVICE
inline bool intersect(const AABB6 &b, const Ray &r,
                      const Vector3 &inv_dir, const TVector3<bool> dir_is_neg,
                      Real expand_dist = 0) {
    return intersect(convert_aabb<AABB3>(b), r,
        inv_dir, dir_is_neg, expand_dist);
}

// Intersect a bounding box with a plane and return correctly ordered intersection points
DEVICE
inline void intersect_plane(const AABB3 &b, const Vector4 &plane, Vector3 *points, int *num_points, const int *bbox_edges_idxs) {
    *num_points = 0;
    Vector3 n = Vector3{plane[0], plane[1], plane[2]};
    Vector3 avg{0, 0, 0};
    Vector3 corners[8];
    for (int i = 0; i < 8; i++) {
        corners[i] = corner(b, i);
    }
    for (int i = 0; i < 24; i += 2) {
        auto c_0 = corners[bbox_edges_idxs[i]];
        auto c_1 = corners[bbox_edges_idxs[i + 1]];
        if(std::abs(dot(n, c_1 - c_0)) / (length(n) * length(c_0 - c_1)) > 1e-9){
            Real lam = (-plane[3] - dot(n, c_0)) / dot(n, c_1 - c_0);
            if (lam >= 0 && lam <= 1) {
                points[*num_points] = c_0 + lam * (c_1 - c_0);
                avg += points[*num_points];
                (*num_points)++;
            }
        }
    }
    if (*num_points == 0) {
        return;
    }
    avg /= *num_points;

    Matrix3x3 basis = find_basis(n / length(n));
    Vector3 u(basis(0, 0), basis(1, 0), basis(2, 0));
    Vector3 v(basis(0, 1), basis(1, 1), basis(2, 1));

    Real cosines[*num_points];
    Real sines[*num_points];
    for (int i = 0; i < *num_points; i++) {
        Vector3 p = points[i] - avg;
        p /= length(p);
        cosines[i] = dot(p, u);
        sines[i] = dot(p, v);
    }

    for (int i = 0; i < *num_points; i++) {
        int min_idx = i;
        for (int j = i; j < *num_points; j++) {
            if (sines[j] > 0 && sines[min_idx] < 0) {
                min_idx = j;
            }
            else if (sines[j] < 0 && sines[min_idx] < 0 && cosines[j] < cosines[min_idx]) {
                min_idx = j;
            }
            else if (sines[j] > 0 && sines[min_idx] > 0 && cosines[j] > cosines[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            swap_(cosines[i], cosines[min_idx]);
            swap_(sines[i], sines[min_idx]);
            swap_(points[i], points[min_idx]);
        }
    }
}

DEVICE
inline void init_bbox_edges_and_normals(BufferView<int> bbox_edges_idxs, BufferView<Vector3> bbox_edges_normals){
    bbox_edges_idxs[0] = 0;
    bbox_edges_idxs[1] = 1;
    bbox_edges_idxs[2] = 1;
    bbox_edges_idxs[3] = 3;
    bbox_edges_idxs[4] = 3;
    bbox_edges_idxs[5] = 2;
    bbox_edges_idxs[6] = 2;
    bbox_edges_idxs[7] = 0;

    bbox_edges_idxs[8] = 4;
    bbox_edges_idxs[9] = 5;
    bbox_edges_idxs[10] = 5;
    bbox_edges_idxs[11] = 7;
    bbox_edges_idxs[12] = 7;
    bbox_edges_idxs[13] = 6;
    bbox_edges_idxs[14] = 6;
    bbox_edges_idxs[15] = 4;

    bbox_edges_idxs[16] = 0;
    bbox_edges_idxs[17] = 4;
    bbox_edges_idxs[18] = 1;
    bbox_edges_idxs[19] = 5;
    bbox_edges_idxs[20] = 2;
    bbox_edges_idxs[21] = 6;
    bbox_edges_idxs[22] = 3;
    bbox_edges_idxs[23] = 7;

    bbox_edges_normals[0] = Vector3{0, -1, 0};
    bbox_edges_normals[1] = Vector3{0, 0, -1};
    bbox_edges_normals[2] = Vector3{1, 0, 0};
    bbox_edges_normals[3] = Vector3{0, 0, -1};
    bbox_edges_normals[4] = Vector3{0, 1, 0};
    bbox_edges_normals[5] = Vector3{0, 0, -1};
    bbox_edges_normals[6] = Vector3{-1, 0, 0};
    bbox_edges_normals[7] = Vector3{0, 0, -1};

    bbox_edges_normals[8] = Vector3{0, -1, 0};
    bbox_edges_normals[9] = Vector3{0, 0, 1};
    bbox_edges_normals[10] = Vector3{1, 0, 0};
    bbox_edges_normals[11] = Vector3{0, 0, 1};
    bbox_edges_normals[12] = Vector3{0, 1, 0};
    bbox_edges_normals[13] = Vector3{0, 0, 1};
    bbox_edges_normals[14] = Vector3{-1, 0, 0};
    bbox_edges_normals[15] = Vector3{0, 0, 1};

    bbox_edges_normals[16] = Vector3{-1, 0, 0};
    bbox_edges_normals[17] = Vector3{0, -1, 0};
    bbox_edges_normals[18] = Vector3{1, 0, 0};
    bbox_edges_normals[19] = Vector3{0, -1, 0};
    bbox_edges_normals[20] = Vector3{-1, 0, 0};
    bbox_edges_normals[21] = Vector3{0, 1, 0};
    bbox_edges_normals[22] = Vector3{1, 0, 0};
    bbox_edges_normals[23] = Vector3{0, 1, 0};
}

DEVICE
inline void test_aabb(const AABB3 &b, const Vector4 &plane, ptr<Vector3> points, ptr<int> num_points) {
    Buffer<int> bbox_edges_idxs(false, 24);
    Buffer<Vector3> bbox_edges_normals(false, 24);
    init_bbox_edges_and_normals(bbox_edges_idxs.view(0, 24), bbox_edges_normals.view(0, 24));
    intersect_plane(b, plane, points.get_pointer(), num_points.get_pointer(), bbox_edges_idxs.begin());
}

std::ostream& operator<<(std::ostream &os, const AABB3 &bounds);
std::ostream& operator<<(std::ostream &os, const AABB6 &bounds);
