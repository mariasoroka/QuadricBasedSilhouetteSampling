#pragma once

#include "redner.h"
#include "vector.h"
#include "buffer.h"
#include "aabb.h"
#include "quadric.h"
#include "solid_angles.h"
#include "rejection_test.h"
#include "polygon_clipping.h"

#include <iostream>

struct Camera;
struct Shape;
struct Edge;

struct BVHNode {
    AABB3 bounds;
    AABB3 dual_bounds;
    bool rejectable;
    Vector4 D;
    Matrix4x4 m;
    QuadricPair quadric_pair;
    Real total_length_weighted;
    BVHNode *children[4];
    bool is_leaf;
    int edge_ids[4];
    int num_children;
};

struct BVHNodePtr {
    DEVICE BVHNodePtr() {}
    DEVICE BVHNodePtr(const BVHNode *ptr) : ptr(ptr) {}
    const BVHNode *ptr;
};

DEVICE
inline bool is_leaf(const BVHNodePtr &node_ptr) {
    return node_ptr.ptr->is_leaf;
}

DEVICE
inline int get_edge_id(const BVHNodePtr &node_ptr, int i) {
    return node_ptr.ptr->edge_ids[i];
}

DEVICE
inline void get_children(const BVHNodePtr &node_ptr, BVHNodePtr children[4]) {
    children[0] = BVHNodePtr(node_ptr.ptr->children[0]);
    children[1] = BVHNodePtr(node_ptr.ptr->children[1]);
    children[2] = BVHNodePtr(node_ptr.ptr->children[2]);
    children[3] = BVHNodePtr(node_ptr.ptr->children[3]);
}

// Node importance function. See Sec. 3.5 of "Quadric-Based Silhouette Sampling for Differentiable Rendering"
DEVICE 
inline Real node_importance(const BVHNode *node,
                            const SurfacePoint &p,
                            const Matrix3x3 &m_inv,
                            const Matrix3x3 &isotropic_frame,
                            const int* bbox_edges_idxs,
                            const Vector3 *bbox_edges_normals) {
    if (node->rejectable) {
        Vector3 inter_points[12];
        int num_points = 0;
        bool has_silhouette = rejection_test(node->quadric_pair, node->m, 
                                            node->dual_bounds, 
                                            p.position, 
                                            inter_points, 
                                            num_points,
                                            bbox_edges_idxs);
        if (!has_silhouette) { 
            return 0;
        }
    }

    auto center = 0.5f * (node->bounds.p_min + node->bounds.p_max);
    Real dist_sq = length_squared(center - p.position);
    Real average_length = node->total_length_weighted;
    Real bsdf = bbox_average_bsdf(node->bounds, p, m_inv, 
                                  isotropic_frame, 
                                  bbox_edges_idxs, 
                                  bbox_edges_normals);

    return bsdf * average_length / dist_sq;
}


// Node importance function for polygon lights. See Sec. 3.6 of "Quadric-Based Silhouette Sampling for Differentiable Rendering"
DEVICE 
inline Real node_importance_poly_light(const BVHNode *node,
                                       const SurfacePoint &p,
                                       const Matrix3x3 &m,
                                       const Matrix3x3 &m_inv,
                                       const Matrix3x3 &isotropic_frame,
                                       const float *light_vertices,
                                       const int *light_silhouette,
                                       const int n_light_silhouette,
                                       const Real polygon_radiance,
                                       const Real envmap_radiance,
                                       const int *bbox_edges_idxs) {
    
    if (node->rejectable) {
        Vector3 inter_points[12];
        int num_points = 0;
        bool has_silhouette = rejection_test(node->quadric_pair, 
                                             node->m, 
                                             node->dual_bounds, 
                                             p.position, 
                                             inter_points, 
                                             num_points, 
                                             bbox_edges_idxs);
        if (!has_silhouette) { 
            return 0;
        }
    }
    
    Real integral = 0;
    Real solid_angle = 0;

    if (!::inside(node->bounds, p.position)) {

        Vector3 bbox_silhouette[7 + n_light_silhouette];
        BufferView<Vector3> bbox_silhouette_view(bbox_silhouette, 7 + n_light_silhouette);
        int n_bbox_silhouette = 0;
        get_bbox_silhouette(node->bounds, p.position, bbox_silhouette_view, n_bbox_silhouette);
        clip_silhouette_horizon(bbox_silhouette_view, n_bbox_silhouette, p.shading_frame.n, p.position);

        solid_angle = polygon_solid_angle(n_bbox_silhouette, 
                                          bbox_silhouette_view,
                                          isotropic_frame, 
                                          p.shading_frame.n,
                                          p.position);
        Real integral_ltc_envmap = polygon_ltc(n_bbox_silhouette, 
                                               bbox_silhouette_view,
                                               m, m_inv, isotropic_frame,
                                               p.shading_frame.n,
                                               p.position);

        clip_silhouette_polygon(bbox_silhouette_view, 
                                n_bbox_silhouette, 
                                light_vertices, 
                                light_silhouette, 
                                n_light_silhouette,
                                p.position);

        Real solid_angle_new = polygon_solid_angle(n_bbox_silhouette, 
                                                   bbox_silhouette_view,
                                                   isotropic_frame, 
                                                   p.shading_frame.n,
                                                   p.position);
        Real integral_ltc_light = polygon_ltc(n_bbox_silhouette, 
                                              bbox_silhouette_view,
                                              m, m_inv, isotropic_frame,
                                              p.shading_frame.n,
                                              p.position);

        integral = integral_ltc_light * polygon_radiance + integral_ltc_envmap * envmap_radiance;
        solid_angle = solid_angle;
    }
    else {
        Vector3 bbox_silhouette[7 + n_light_silhouette];
        BufferView<Vector3> bbox_silhouette_view(bbox_silhouette, 7 + n_light_silhouette);
        int n_bbox_silhouette = n_light_silhouette / 2;
        for (int i = 0; i < n_light_silhouette / 2; i++){
            int idx1 = light_silhouette[2 * i + 0];
            bbox_silhouette[i] = Vector3(light_vertices[3 * idx1 + 0], 
                                                light_vertices[3 * idx1 + 1], 
                                                light_vertices[3 * idx1 + 2]);
        }

        clip_silhouette_horizon(bbox_silhouette_view, n_bbox_silhouette, p.shading_frame.n, p.position);

        Real integral_ltc_light = polygon_ltc(n_bbox_silhouette, 
                                              bbox_silhouette_view,
                                              m, m_inv, isotropic_frame,
                                              p.shading_frame.n,
                                              p.position);
        Real integral_ltc_envmap = 1.0;
        integral = integral_ltc_light * polygon_radiance + integral_ltc_envmap * envmap_radiance;
        solid_angle = 2 * M_PI;
    }

    auto center = 0.5f * (node->bounds.p_min + node->bounds.p_max);
    Real dist_sq = length_squared(center - p.position);
    Real average_length = node->total_length_weighted;
    
    if (solid_angle == 0) {
        return 0;
    }
    Real bsdf = integral / solid_angle;
    return bsdf * average_length / dist_sq;
}


struct EdgeTree {
    EdgeTree(bool use_gpu,
             const Camera &camera,
             const BufferView<Shape> &shapes,
             const BufferView<Edge> &edges,
             int max_edges_per_leaf,
             int min_points_for_fitting);

    Buffer<BVHNode> non_manifold_bvh_nodes;
    Buffer<BVHNode> non_manifold_bvh_leaves;
    Buffer<BVHNode> manifold_bvh_nodes;
    Buffer<BVHNode> manifold_bvh_leaves;
    Real edge_bounds_expand;
    int n_non_manifold_bvh_roots = 0;
    int n_manifold_bvh_roots = 0;
    int non_manifold_size;
    int manifold_size;
    int max_edges_per_leaf;
    int min_points_for_fitting;
};

struct EdgeTreeRoots {
    const BVHNode *non_manifold_bvh_roots;
    const BVHNode *manifold_bvh_roots;
    int n_non_manifold_bvh_roots;
    int n_manifold_bvh_roots;
};

inline EdgeTreeRoots get_edge_tree_roots(const EdgeTree *edge_tree) {
    if (edge_tree == nullptr) {
        return EdgeTreeRoots{nullptr, nullptr, 0, 0};
    } else {
        return EdgeTreeRoots{
            edge_tree->non_manifold_bvh_nodes.begin(),
            edge_tree->manifold_bvh_nodes.begin(),
            edge_tree->n_non_manifold_bvh_roots,
            edge_tree->n_manifold_bvh_roots};
    }
}

void test_compute_stack_item(int n_edges, ptr<int> edge_data, ptr<double> output, Shape shape, bool use_gpu);
