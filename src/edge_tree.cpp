#include "edge_tree.h"
#include "vector.h"
#include "cuda_utils.h"
#include "atomic.h"
#include "edge.h"
#include "parallel.h"
#include "thrust_utils.h"
#include "offset_quadric.h"
#include "solid_angles.h"

#include <thrust/transform_reduce.h>
 #include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/partition.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>

struct edge_partitioner {
    DEVICE bool operator()(int edge_id) const {
        bool result = is_onesided(shapes, edges[edge_id]);
        return result;
    }

    const Shape *shapes;
    const Edge *edges;
};

struct edge_3d_bounds_computer {
    DEVICE void operator()(int idx) {
        const auto &edge = edges[idx];
        // Compute position bound
        auto v0 = get_v0(shapes, edge);
        auto v1 = get_v1(shapes, edge);
        auto p_min = Vector3{0, 0, 0};
        auto p_max = Vector3{0, 0, 0};
        for (int i = 0; i < 3; i++) {
            p_min[i] = min(v0[i], v1[i]);
            p_max[i] = max(v0[i], v1[i]);
        }
        edge_aabbs[idx].p_min = p_min;
        edge_aabbs[idx].p_max = p_max;
        assert(isfinite(p_min));
        assert(isfinite(p_max));

    }

    const Shape *shapes;
    const Edge *edges;
    const Vector3 cam_org;
    AABB3 *edge_aabbs;
};

void compute_edge_bounds(const Shape *shapes,
                         const BufferView<Edge> &edges,
                         const Vector3 cam_org,
                         BufferView<AABB3> edge_aabbs,
                         bool use_gpu) {
    parallel_for(edge_3d_bounds_computer{
                     shapes, edges.begin(), cam_org, edge_aabbs.begin()},
                 edges.size(),
                 use_gpu);
}

struct id_to_edge_pt_sum {
    DEVICE Vector3 operator()(int id) const {
        auto v0 = get_v0(shapes, edges[id]);
        auto v1 = get_v1(shapes, edges[id]);
        return v0 + v1;
    }

    const Shape *shapes;
    const Edge *edges;
};

struct id_to_edge_pt_abs {
    DEVICE Vector3 operator()(int id) const {
        auto v0 = get_v0(shapes, edges[id]);
        auto v1 = get_v1(shapes, edges[id]);
        auto v0_abs = Vector3{}, v1_abs = Vector3{};
        for (int i = 0; i < 3; i++) {
            v0_abs[i] = fabs(v0[i] - mean[i]);
            v1_abs[i] = fabs(v1[i] - mean[i]);
        }
        return v0_abs + v1_abs;
    }

    const Shape *shapes;
    const Edge *edges;
    Vector3 mean;
};

struct id_to_aabb3 {
    DEVICE AABB3 operator()(int id) const {
        auto b = bounds[id];
        return AABB3{b.p_min, b.p_max};
    }

    const AABB3 *bounds;
};


struct union_bounding_box {
    DEVICE AABB3 operator()(const AABB3 &b0, const AABB3 &b1) const {
        auto p_min = Vector3{min(b0.p_min[0], b1.p_min[0]),
                             min(b0.p_min[1], b1.p_min[1]),
                             min(b0.p_min[2], b1.p_min[2])};
        auto p_max = Vector3{max(b0.p_max[0], b1.p_max[0]),
                             max(b0.p_max[1], b1.p_max[1]),
                             max(b0.p_max[2], b1.p_max[2])};
        return AABB3{p_min, p_max};
    }
};

struct sum_vec3 {
    DEVICE Vector3 operator()(const Vector3 &v0, const Vector3 &v1) const {
        return v0 + v1;
    }
};

struct edge_center_is_inside_bbox {
    DEVICE inline bool operator()(int idx) {
        if(inside(bbox, center(edge_bounds[idx]))){
            return true;
        }
        return false;
    }
    AABB3 bbox;
    const AABB3 *edge_bounds;
};

struct StackItem{
    AABB3 bounds; // Bounding box of the node
    int node_idx;
    int start_idx;
    int end_idx;
    bool build_dual;
    bool rejectable; // If false, the node can not be rejected. See Appendix B from "Quadric-Based Silhouette Sampling for Differentiable Rendering"
    Vector4 D; // The offset quadric vector.
    AABB3 dual_bounds;
    QuadricPair quadric_pair;
    Matrix4x4 m; // Orthogonal basis A from Sec. 3.2.
    Matrix4x4 fitted_matrix; // Matrix for Qf
    int size(){
        return end_idx - start_idx;
    }
};

struct NodeSplit{
    StackItem child_0;
    StackItem child_1;
};

struct ShapeFacePair{
    int shape_id;
    int face_id;
};

struct ShapeVertexPair{
    int shape_id;
    int vertex_id;
};

struct dual_planes_idxs_assigner{
    DEVICE void operator()(int idx){
        auto edge = edges[edge_ids_local[idx + start_idx]];
        dual_planes_idxs[2 * idx] = ShapeFacePair{edge.shape_id, edge.f0};
        dual_planes_idxs[2 * idx + 1] = ShapeFacePair{edge.shape_id, edge.f1};
    }
    const Edge *edges;
    int start_idx;
    ShapeFacePair *dual_planes_idxs;
    int *edge_ids_local;
};

struct dual_planes_from_idx{
    DEVICE void operator()(int idx){
        const auto &shape = *(shapes + dual_planes_idxs[idx].shape_id);
        Vector3 normal = get_normal(shape, dual_planes_idxs[idx].face_id);
        auto indices = get_indices(shape, dual_planes_idxs[idx].face_id);
        Vector3 point = get_vertex(shape, indices[0]);
        dual_planes[idx] = Vector4{normal.x, normal.y, normal.z, -dot(normal, point)};
    }
    const Shape *shapes;
    const ShapeFacePair *dual_planes_idxs;
    Vector4 *dual_planes;
};

// Compute \lambda(q) for each q in the set of dual planes. See Sec. 4.2 from "Quadric-Based Silhouette Sampling for Differentiable Rendering"
struct compute_face_lambda{
    DEVICE void operator()(int idx){
        lambdas[idx] = - dot(dual_planes[idx], fitted_matrix * dual_planes[idx]) / dot(dual_planes[idx], offset_matrix * dual_planes[idx]);
    }
    Vector4 *dual_planes;
    Matrix4x4 offset_matrix;
    Matrix4x4 fitted_matrix;
    Real *lambdas;
};

// Compute \lambda^*(e) for each edge e in the set of edges. See Sec. 4.2 from "Quadric-Based Silhouette Sampling for Differentiable Rendering"
struct compute_edge_lambdas{
    DEVICE void operator()(int idx){
        const Edge edge = edges[edge_ids_local[idx + start_idx]];
        const Shape &shape = shapes[edge.shape_id];
        if (edge.f0 == -1 || edge.f1 == -1) {
            offset_lambdas[idx] = 0;
            return;
        }
        else {
            Vector3 normal_0 = get_normal(shape, edge.f0);
            Vector3 normal_1 = get_normal(shape, edge.f1);
            auto indices_0 = get_indices(shape, edge.f0);
            auto indices_1 = get_indices(shape, edge.f1);
            Vector3 vertex_0 = get_vertex(shape, indices_0[0]);
            Vector3 vertex_1 = get_vertex(shape, indices_1[0]);
            Vector4 plane_0{normal_0.x, normal_0.y, normal_0.z, -dot(normal_0, vertex_0)};
            Vector4 plane_1{normal_1.x, normal_1.y, normal_1.z, -dot(normal_1, vertex_1)};

            Real Qf01 = dot(plane_0, fitted_matrix * plane_1);
            Real Qf00 = dot(plane_0, fitted_matrix * plane_0);
            Real Qf11 = dot(plane_1, fitted_matrix * plane_1);
            Real Qo01 = dot(plane_0, offset_matrix * plane_1);
            Real Qo00 = dot(plane_0, offset_matrix * plane_0);
            Real Qo11 = dot(plane_1, offset_matrix * plane_1);

            Real numerator =  Qf01 * Qf01 - Qf00 * Qf11;

            if(std::abs(numerator) < 1e-15){
                offset_lambdas[idx] = 0;
                return;
            }

            Real denominator = 2 * Qf01 * Qo01 - Qf00 * Qo11 - Qf11 * Qo00;
            Real lam = - numerator / denominator;

            Real sign_check_1 = Qf01 + lam * Qo01;
            Real sign_check_2 = (Qf00 + Qf11 - 2 * Qf01) + lam * (Qo00 + Qo11 - 2 * Qo01);

            if(sign_check_1 * sign_check_2 < 0){
                offset_lambdas[idx] = lam;
            }
            else{
                offset_lambdas[idx] = 0;
            }
        }
    }
    const Shape *shapes;
    const Edge* edges;
    const int* edge_ids_local;
    int start_idx;
    Matrix4x4 offset_matrix;
    Matrix4x4 fitted_matrix;
    Real *offset_lambdas;
};

void compute_stack_item(StackItem &node,
                        const BufferView<Shape> &shapes,
                        const BufferView<Edge> &edges,
                        const Buffer<AABB3> &edge_bounds,
                        int start_idx,
                        int end_idx,
                        Buffer<int> &edge_ids_local,
                        bool build_dual,
                        int min_points_for_fitting,
                        const Matrix4x4 &parent_fitted_matrix,
                        bool use_gpu) {
    node = StackItem{AABB3(), 
                     0, 
                     start_idx, 
                     end_idx, 
                     build_dual, 
                     false, 
                     Vector4(), 
                     AABB3(), 
                     QuadricPair(), 
                     Matrix4x4(), 
                     Matrix4x4()};

    node.bounds = DISPATCH(use_gpu,
                        thrust::transform_reduce, edge_ids_local.begin() + start_idx,
                        edge_ids_local.begin() + end_idx,
                        id_to_aabb3{edge_bounds.begin()}, AABB3(), union_bounding_box{});  
    make_non_degenerate(node.bounds); 

    if (build_dual) {

        // Get shape-face pain for each edge
        Buffer<ShapeFacePair> dual_planes_idxs(use_gpu, 2 * (end_idx - start_idx));
        DISPATCH(use_gpu, thrust::fill, dual_planes_idxs.begin(), dual_planes_idxs.end(), ShapeFacePair{-1, -1});
        parallel_for(dual_planes_idxs_assigner{edges.begin(), start_idx, dual_planes_idxs.begin(), edge_ids_local.begin()},
                     end_idx - start_idx, use_gpu);

        // Compute dual planes for each edge
        Buffer<Vector4> dual_planes(use_gpu, 2 * (end_idx - start_idx));
        parallel_for(dual_planes_from_idx{shapes.begin(), dual_planes_idxs.begin(), dual_planes.begin()},
                    2 * (end_idx - start_idx), use_gpu);

        // Remove duplicate planes
        DISPATCH(use_gpu, thrust::sort, dual_planes.begin(), dual_planes.end(), [](const Vector4 &a, const Vector4 &b){
                                        if (a < b) {
                                            return true;
                                        }
                                        return false;
                                    });
        auto end_it = DISPATCH(use_gpu, thrust::unique, dual_planes.begin(), dual_planes.end(), [](const Vector4 &a, const Vector4 &b){
                                        return a == b;
                                    });

        int n_dual_planes = end_it - dual_planes.begin();

        Quadric fitted;
        if (n_dual_planes >= min_points_for_fitting) {
            fitted = fit_quadric(dual_planes.view(0, n_dual_planes));
        }
        // If there are not enough planes to fit a quadric, use the parent fitted quadric
        else {
            fitted = Quadric(parent_fitted_matrix);
        }
        node.fitted_matrix = fitted.matrix;

        // Find vector D for the offset quadric. See Sec. 4.4 of "Quadric-Based Silhouette Sampling for Differentiable Rendering".
        // The same vector is used as vector Z in Sec. 3.2.
        Vector4 D = find_offset_quadric_vector(use_gpu, dual_planes.view(0, n_dual_planes));

        if (length(D) > 0) {
            node.rejectable = true;
            node.D = D;

            // Compute orthogonal basis A from Sec. 3.2.
            Matrix4x4 basis = find_basis(D / length(D));
            node.m = basis;

            // Project dual planes to compute the set \mathcal{W}'.
            Buffer<Vector3> proj_dual_planes(use_gpu, n_dual_planes);
            DISPATCH(use_gpu, thrust::transform, dual_planes.begin(), end_it, 
                                proj_dual_planes.begin(), [&basis](const Vector4 &plane){
                                        Vector4 transformed_plane = transpose(basis) * plane;
                                        transformed_plane /= transformed_plane.w;
                                        return Vector3(transformed_plane.x, transformed_plane.y, transformed_plane.z);
                                    });

            // Compute the dual bounds of the node.
            AABB3 dual_bounds = DISPATCH(use_gpu, thrust::transform_reduce, proj_dual_planes.begin(), proj_dual_planes.end(),
                                        [](const Vector3 &v){return AABB3(v, v);}, AABB3(), union_bounding_box{});
            node.dual_bounds = dual_bounds;
            make_non_degenerate(node.dual_bounds);

            // Compute the offset quadric. Qo = D * D^T.
            Matrix4x4 offset_matrix = outer_product(D);

            // Compute the set \Lambda from Sec. 4.2 of "Quadric-Based Silhouette Sampling for Differentiable Rendering".
            int n_edges = end_idx - start_idx;
            int n_faces = n_dual_planes;
            Buffer<Real> lambdas(use_gpu, n_faces + n_edges);
            DISPATCH(use_gpu, thrust::fill, lambdas.begin(), lambdas.end(), 0);
            parallel_for(compute_face_lambda{dual_planes.begin(), 
                                             offset_matrix, 
                                             fitted.matrix, 
                                             lambdas.begin()}, 
                                             n_faces, use_gpu);
            parallel_for(compute_edge_lambdas{shapes.begin(), 
                                              edges.begin(), 
                                              edge_ids_local.begin(), 
                                              start_idx, offset_matrix, 
                                              fitted.matrix, 
                                              lambdas.begin() + n_faces}, 
                                              n_edges, use_gpu);
            auto min_lambda_itr = DISPATCH(use_gpu, thrust::min_element, lambdas.begin(), lambdas.end());
            auto max_lambda_itr = DISPATCH(use_gpu, thrust::max_element, lambdas.begin(), lambdas.end());

            // Compute the bounding quadrics
            Matrix4x4 m_1 = fitted.matrix + *(min_lambda_itr) * offset_matrix;
            Matrix4x4 m_2 = fitted.matrix + *(max_lambda_itr) * offset_matrix;
            node.quadric_pair.quadric1 = Quadric(m_1);
            node.quadric_pair.quadric2 = Quadric(m_2);
        }
        else{
            node.rejectable = false;
        }
    }            
}

// Compute SAH splitting costs
Buffer<Real> get_costs(StackItem &node_to_split,
                       int num_buckets,
                       int dim,
                       const BufferView<Shape> &shapes,
                       const BufferView<Edge> &edges,
                       const Buffer<AABB3> &edge_bounds,
                       Buffer<int> &edge_ids_local,
                       bool use_gpu){

    Buffer<Real> costs(use_gpu, num_buckets);
    auto d = node_to_split.bounds.p_max - node_to_split.bounds.p_min;

    for(int idx = 0; idx < num_buckets; idx++){
        auto child_box_0 = node_to_split.bounds;
        auto child_box_1 = node_to_split.bounds;
        child_box_0.p_max[dim] = node_to_split.bounds.p_min[dim] + (idx + 1) * d[dim] / (num_buckets + 1);
        child_box_1.p_min[dim] = node_to_split.bounds.p_min[dim] + (idx + 1) * d[dim] / (num_buckets + 1);

        int split_loc = DISPATCH(use_gpu, thrust::partition, edge_ids_local.begin() + node_to_split.start_idx, edge_ids_local.begin() + node_to_split.end_idx,
                            edge_center_is_inside_bbox{child_box_0, edge_bounds.begin()}) - edge_ids_local.begin();

        if(split_loc == node_to_split.start_idx || split_loc == node_to_split.end_idx) {
            costs[idx] = infinity<Real>(); 
        }
        else{
            AABB3 child_box_0_new = DISPATCH(use_gpu, thrust::transform_reduce, edge_ids_local.begin() + node_to_split.start_idx, edge_ids_local.begin() + split_loc,
                                            id_to_aabb3{edge_bounds.begin()}, AABB3(), union_bounding_box{});
            AABB3 child_box_1_new = DISPATCH(use_gpu, thrust::transform_reduce, edge_ids_local.begin() + split_loc, edge_ids_local.begin() + node_to_split.end_idx,
                                            id_to_aabb3{edge_bounds.begin()}, AABB3(), union_bounding_box{});
            costs[idx] = get_area(child_box_0_new) * (split_loc - node_to_split.start_idx) + get_area(child_box_1_new) * (node_to_split.end_idx - split_loc);
        }
    }
    return costs;
}

NodeSplit split(StackItem &node_to_split,
                const BufferView<Shape> &shapes,
                const BufferView<Edge> &edges,
                const Buffer<AABB3> &edge_bounds,
                Buffer<int> &edge_ids_local,
                int min_points_for_fitting,
                bool use_gpu){
    Real best_cost = infinity<Real>();

    int split_loc = 0;
    if (node_to_split.end_idx - node_to_split.start_idx <= 4) {
        split_loc = (node_to_split.start_idx + node_to_split.end_idx) / 2;
    }
    else {
        auto d = node_to_split.bounds.p_max - node_to_split.bounds.p_min;
        int dim = argmax(d);

        int num_buckets = 10;
        Buffer<Real> costs = get_costs(node_to_split, 
                                       num_buckets, 
                                       dim, 
                                       shapes, 
                                       edges, 
                                       edge_bounds, 
                                       edge_ids_local, 
                                       use_gpu);

        int best_cut = 0;
        for(int i = 0; i < num_buckets; i++){
            if(costs[i] < best_cost){
                best_cost = costs[i];
                best_cut = i;
            }
        }

        auto child_box_0 = node_to_split.bounds;
        auto child_box_1 = node_to_split.bounds;

        child_box_0.p_max[dim] = node_to_split.bounds.p_min[dim] + (best_cut + 1) * d[dim] / (num_buckets + 1);
        child_box_1.p_min[dim] = node_to_split.bounds.p_min[dim] + (best_cut + 1) * d[dim] / (num_buckets + 1);

        split_loc = DISPATCH(use_gpu, thrust::partition, edge_ids_local.begin() + node_to_split.start_idx, 
                             edge_ids_local.begin() + node_to_split.end_idx,
                             edge_center_is_inside_bbox{child_box_0, edge_bounds.begin()}) - edge_ids_local.begin();

        if(split_loc == node_to_split.start_idx || split_loc == node_to_split.end_idx) {
            split_loc = (node_to_split.start_idx + node_to_split.end_idx) / 2;
        }
    }

    NodeSplit result;
    compute_stack_item(result.child_0, shapes, edges, edge_bounds, node_to_split.start_idx, split_loc, edge_ids_local, 
                        node_to_split.build_dual, min_points_for_fitting, node_to_split.fitted_matrix, use_gpu);
    compute_stack_item(result.child_1, shapes, edges, edge_bounds, split_loc, node_to_split.end_idx, edge_ids_local, 
                        node_to_split.build_dual, min_points_for_fitting, node_to_split.fitted_matrix, use_gpu);
    return result;
}


int build_tree(const AABB3 &scene_bounds,
                const BufferView<Shape> &shapes,
                const BufferView<Edge> &edges,
                const Buffer<AABB3> &edge_bounds,
                const BufferView<int> &edge_ids,
                Buffer<BVHNode> &nodes,
                Buffer<BVHNode> &leaves,
                bool build_dual,
                int max_edges_per_leaf,
                int min_points_for_fitting,
                bool use_gpu) {

    int buffer_size = 200;
    StackItem buffer[buffer_size];
    StackItem *stack_ptr = &buffer[0];

    int nodes_counter = 0;
    int leaves_counter = 0;

    Buffer<int> edge_ids_local(use_gpu, edge_ids.size());
    DISPATCH(use_gpu, thrust::copy, edge_ids.begin(), edge_ids.end(), edge_ids_local.begin());
    DISPATCH(use_gpu, thrust::sort,  edge_ids_local.begin(),  edge_ids_local.end(), 
            [&](const int i, const int j) {return edges[i].shape_id < edges[j].shape_id;});

    Buffer<int> shape_ids_local(use_gpu, edge_ids.size());
    DISPATCH(use_gpu, thrust::transform, edge_ids_local.begin(),  edge_ids_local.end(), shape_ids_local.begin(),
                [&](const int i){return edges[i].shape_id; });

    // Build a separate hierarchy for each shape
    int n_roots = 0;
    for (int i = 0; i < shapes.size(); i++) {
        auto range = DISPATCH(use_gpu, thrust::equal_range, shape_ids_local.begin(), shape_ids_local.end(), i);
        if (range.first != range.second) {
            StackItem object_root;
            compute_stack_item(object_root, shapes, edges, edge_bounds, range.first - shape_ids_local.begin(), range.second - shape_ids_local.begin(), edge_ids_local, 
                    build_dual, min_points_for_fitting, Matrix4x4(), use_gpu);

            auto children = split(object_root, 
                                  shapes, 
                                  edges, 
                                  edge_bounds, 
                                  edge_ids_local, 
                                  min_points_for_fitting, 
                                  use_gpu);

            StackItem grandchildren[4];
            int n_grandchildren = 0;
            if (children.child_0.end_idx - children.child_0.start_idx <= max_edges_per_leaf && children.child_0.end_idx - children.child_0.start_idx > 0) {
                grandchildren[0] = children.child_0;
                n_grandchildren = 1;
            }
            else {
                auto grandchildren_0 = split(children.child_0, shapes, edges, edge_bounds, edge_ids_local, min_points_for_fitting, use_gpu);
                grandchildren[0] = grandchildren_0.child_0;
                grandchildren[1] = grandchildren_0.child_1;
                n_grandchildren = 2;
            }

            if (children.child_1.end_idx - children.child_1.start_idx <= max_edges_per_leaf && children.child_1.end_idx - children.child_1.start_idx > 0) {
                grandchildren[n_grandchildren] = children.child_1;
                n_grandchildren++;
            }
            else {
                auto grandchildren_1 = split(children.child_1, shapes, edges, edge_bounds, edge_ids_local, min_points_for_fitting, use_gpu);
                grandchildren[n_grandchildren + 0] = grandchildren_1.child_0;
                grandchildren[n_grandchildren + 1] = grandchildren_1.child_1;
                n_grandchildren += 2;
            }

            for(int j = 0; j < n_grandchildren; j++){
                auto child = grandchildren[j];
                nodes[n_roots].bounds = child.bounds;
                nodes[n_roots].is_leaf = false;
                
                if (build_dual) {
                    nodes[n_roots].rejectable = child.rejectable;
                    nodes[n_roots].D = child.D;
                    nodes[n_roots].dual_bounds = child.dual_bounds;
                    nodes[n_roots].m = child.m;
                    nodes[n_roots].quadric_pair = child.quadric_pair;
                }
                
                child.node_idx = n_roots;
                n_roots++;
                *stack_ptr++ = child;
            }
        }
    }
    nodes_counter = n_roots;

    while(stack_ptr != &buffer[0]){
        assert(stack_ptr > &buffer[0] && stack_ptr < &buffer[buffer_size]);

        StackItem node_to_split = *--stack_ptr;
        auto children = split(node_to_split, shapes, edges, edge_bounds, edge_ids_local, min_points_for_fitting, use_gpu);

        StackItem grandchildren[4];
        int n_grandchildren = 0;
        if (children.child_0.end_idx - children.child_0.start_idx <= max_edges_per_leaf && children.child_0.end_idx - children.child_0.start_idx > 0) {
            grandchildren[0] = children.child_0;
            n_grandchildren = 1;
        }
        else {
            auto grandchildren_0 = split(children.child_0, shapes, edges, edge_bounds, edge_ids_local, min_points_for_fitting, use_gpu);
            grandchildren[0] = grandchildren_0.child_0;
            grandchildren[1] = grandchildren_0.child_1;
            n_grandchildren = 2;
        }

        if (children.child_1.end_idx - children.child_1.start_idx <= max_edges_per_leaf && children.child_1.end_idx - children.child_1.start_idx > 0) {
            grandchildren[n_grandchildren] = children.child_1;
            n_grandchildren++;
        }
        else {
            auto grandchildren_1 = split(children.child_1, shapes, edges, edge_bounds, edge_ids_local, min_points_for_fitting, use_gpu);
            grandchildren[n_grandchildren + 0] = grandchildren_1.child_0;
            grandchildren[n_grandchildren + 1] = grandchildren_1.child_1;
            n_grandchildren += 2;
        }

        int children_counter = 0;
        for(int i = 0; i < n_grandchildren; i++){
            auto child = grandchildren[i];
            // If the node has more than max_edges_per_leaf edges, it is a non-leaf node
            if(child.end_idx - child.start_idx > max_edges_per_leaf){
                nodes[nodes_counter].bounds = child.bounds;
                nodes[nodes_counter].is_leaf = false;
                
                if (build_dual) {
                    nodes[nodes_counter].rejectable = child.rejectable;
                    nodes[nodes_counter].D = child.D;
                    nodes[nodes_counter].dual_bounds = child.dual_bounds;
                    nodes[nodes_counter].m = child.m;
                    nodes[nodes_counter].quadric_pair = child.quadric_pair;
                }
                
                child.node_idx = nodes_counter;
                (nodes.begin() + node_to_split.node_idx)->children[children_counter] = &nodes[nodes_counter];
                children_counter++;
                nodes_counter++;
                *stack_ptr++ = child;
            }
            // If the node has less than max_edges_per_leaf edges, it is a leaf node
            else if(child.end_idx - child.start_idx <= max_edges_per_leaf && child.end_idx - child.start_idx > 0) {
                leaves[leaves_counter].bounds = child.bounds;
                leaves[leaves_counter].is_leaf = true;
                leaves[leaves_counter].num_children = child.end_idx - child.start_idx;

                for(int itmp = child.start_idx; itmp < child.end_idx; itmp++){
                    leaves[leaves_counter].edge_ids[itmp - child.start_idx] = edge_ids_local[itmp];
                }

                if (build_dual) {
                    leaves[leaves_counter].rejectable = child.rejectable;
                    leaves[leaves_counter].D = child.D;
                    leaves[leaves_counter].dual_bounds = child.dual_bounds;
                    leaves[leaves_counter].m = child.m;
                    leaves[leaves_counter].quadric_pair = child.quadric_pair;
                }

                (nodes.begin() + node_to_split.node_idx)->children[children_counter] = &leaves[leaves_counter];
                children_counter++;
                leaves_counter++;
            }
        }
        (nodes.begin() + node_to_split.node_idx)->num_children = children_counter;
    }
    nodes.count = nodes_counter;
    leaves.count = leaves_counter;
    return n_roots;
}

void update_length(const BufferView<Shape> &shapes,
                   const BufferView<Edge> &edges,
                   Buffer<BVHNode> &nodes,
                   Buffer<BVHNode> &leaves,
                   bool use_gpu) {

    for(auto idx = leaves.begin(); idx != leaves.end(); idx++){
        idx->total_length_weighted = 0;
        for (int i = 0; i < idx->num_children; i++){
            Edge e = edges[idx->edge_ids[i]];
            Vector3 v0 = get_vertex(shapes[e.shape_id], e.v0);
            Vector3 v1 = get_vertex(shapes[e.shape_id], e.v1);
            idx->total_length_weighted += distance(v0, v1) * compute_exterior_dihedral_angle(shapes.data, e);
        }
    }

    for(auto idx = nodes.end() - 1; idx != nodes.begin() - 1; idx--){
        idx->total_length_weighted = 0;
        if(idx->num_children > 0){
            for(int itmp = 0; itmp < idx->num_children; itmp++){
                idx->total_length_weighted += idx->children[itmp]->total_length_weighted;
            }
        }
    }
}

EdgeTree::EdgeTree(bool use_gpu,
                   const Camera &camera,
                   const BufferView<Shape> &shapes,
                   const BufferView<Edge> &edges,
                   int max_edges_per_leaf,
                   int min_points_for_fitting) {
    if (edges.size() == 0) {
        return;
    }

    Buffer<int> edge_ids(use_gpu, edges.size());
    DISPATCH(use_gpu, thrust::sequence, edge_ids.begin(), edge_ids.end());
    auto cam_org = xfm_point(camera.cam_to_world, Vector3{0, 0, 0});
    auto partition_result = DISPATCH(use_gpu,
        thrust::stable_partition, edge_ids.begin(), edge_ids.end(),
        edge_partitioner{shapes.begin(), edges.begin()});
    // We call the set of edges in 1) "non_manifold_edges" and the set 2) "manifold_edges"
    BufferView<int> non_manifold_edge_ids(edge_ids.begin(), partition_result - edge_ids.begin());
    BufferView<int> manifold_edge_ids(partition_result, edge_ids.end() - partition_result);
    Buffer<AABB3> edge_bounds(use_gpu, edges.size());
    compute_edge_bounds(shapes.begin(),
                        edges,
                        cam_org,
                        edge_bounds.view(0, edge_ids.size()),
                        use_gpu);
    auto edge_pt_mean = DISPATCH(use_gpu,
        thrust::transform_reduce, edge_ids.begin(), edge_ids.end(),
        id_to_edge_pt_sum{shapes.begin(), edges.begin()},
        Vector3{0, 0, 0}, sum_vec3{});
    edge_pt_mean /= 2. * Real(edge_ids.size());
    auto edge_pt_mad = DISPATCH(use_gpu,
        thrust::transform_reduce, edge_ids.begin(), edge_ids.end(),
        id_to_edge_pt_abs{shapes.begin(), edges.begin(), edge_pt_mean},
        Vector3{0, 0, 0}, sum_vec3{});
    edge_pt_mad /= Real(edge_ids.size());
    edge_bounds_expand = 0.01f * length(edge_pt_mad);

    assert(non_manifold_edge_ids.size() == 0);

    // Initialize nodes
    BVHNode init_node{AABB3(), 
                        AABB3(),
                        false,
                        Vector4(),
                        Matrix4x4(),
                        QuadricPair(),
                        Real(0), 
                        {nullptr, nullptr, nullptr, nullptr}, 
                        false, 
                        {-1, -1, -1, -1}, 
                        0};

    // Warning: not tested
    if (non_manifold_edge_ids.size() > 0) {
        // Compute scene bounding box for BVH
        AABB3 non_manifold_scene_bounds = DISPATCH(use_gpu,
            thrust::transform_reduce, non_manifold_edge_ids.begin(), non_manifold_edge_ids.end(),
            id_to_aabb3{edge_bounds.begin()}, AABB3(), union_bounding_box{});
        assert(non_manifold_scene_bounds.p_max.x - non_manifold_scene_bounds.p_min.x >= 0.f &&
               non_manifold_scene_bounds.p_max.y - non_manifold_scene_bounds.p_min.y >= 0.f &&
               non_manifold_scene_bounds.p_max.z - non_manifold_scene_bounds.p_min.z >= 0.f);
  

        non_manifold_bvh_nodes = Buffer<BVHNode>(use_gpu, max(non_manifold_edge_ids.size() - 1, 1));
        non_manifold_bvh_leaves = Buffer<BVHNode>(use_gpu, non_manifold_edge_ids.size());

        DISPATCH(use_gpu, thrust::fill, non_manifold_bvh_nodes.begin(), non_manifold_bvh_nodes.end(), init_node);
        DISPATCH(use_gpu, thrust::fill, non_manifold_bvh_leaves.begin(), non_manifold_bvh_leaves.end(), init_node);
        // Build tree 
        n_non_manifold_bvh_roots = build_tree(non_manifold_scene_bounds,
                                              shapes,
                                              edges,
                                              edge_bounds,
                                              non_manifold_edge_ids,
                                              non_manifold_bvh_nodes,
                                              non_manifold_bvh_leaves,
                                              false,
                                              max_edges_per_leaf,
                                              min_points_for_fitting,
                                              use_gpu);

        update_length(shapes,
                      edges,
                      non_manifold_bvh_nodes,
                      non_manifold_bvh_leaves,
                      use_gpu);
        
    }

    // Build tree for manifold edges
    if (manifold_edge_ids.size() > 0) {
        // Compute scene bounding box for BVH
        AABB3 manifold_scene_bounds = DISPATCH(use_gpu,
            thrust::transform_reduce, manifold_edge_ids.begin(), manifold_edge_ids.end(),
            id_to_aabb3{edge_bounds.begin()}, AABB3(), union_bounding_box{});
        assert(manifold_scene_bounds.p_max.x - manifold_scene_bounds.p_min.x >= 0.f &&
               manifold_scene_bounds.p_max.y - manifold_scene_bounds.p_min.y >= 0.f &&
               manifold_scene_bounds.p_max.z - manifold_scene_bounds.p_min.z >= 0.f);

        manifold_bvh_nodes = Buffer<BVHNode>(use_gpu, max(manifold_edge_ids.size() - 1, 1));
        manifold_bvh_leaves = Buffer<BVHNode>(use_gpu, manifold_edge_ids.size());

        DISPATCH(use_gpu, thrust::fill, manifold_bvh_nodes.begin(), manifold_bvh_nodes.end(), init_node);
        DISPATCH(use_gpu, thrust::fill, manifold_bvh_leaves.begin(), manifold_bvh_leaves.end(), init_node);

        n_manifold_bvh_roots = build_tree(manifold_scene_bounds,
                    shapes,
                    edges,
                    edge_bounds,
                    manifold_edge_ids,
                    manifold_bvh_nodes,
                    manifold_bvh_leaves,
                    true,
                    max_edges_per_leaf,
                    min_points_for_fitting,
                    use_gpu);

        update_length(shapes,
                      edges,
                      manifold_bvh_nodes,
                      manifold_bvh_leaves,
                      use_gpu);
    }
    non_manifold_size = non_manifold_edge_ids.size();
    manifold_size = manifold_edge_ids.size();
}

void test_compute_stack_item(int n_edges, ptr<int> edge_data, ptr<double> output, Shape shape, bool use_gpu) {

    Buffer<Edge> edges(use_gpu, n_edges);
    for (int i = 0; i < n_edges; i++) {
        edges[i].shape_id = 0;
        edges[i].v0 = edge_data[4 * i];
        edges[i].v1 = edge_data[4 * i + 1];
        edges[i].f0 = edge_data[4 * i + 2];
        edges[i].f1 = edge_data[4 * i + 3];
    }

    int start_idx = 0;
    int end_idx = n_edges;
    Buffer<int> edge_ids_local(use_gpu, n_edges);
    for (int i = 0; i < n_edges; i++) {
        edge_ids_local[i] = i;
    }

    BufferView<Shape> shapes(&shape, 1);

    Buffer<AABB3> edge_bounds(use_gpu, n_edges);
    compute_edge_bounds(shapes.begin(),
                        edges.view(0, n_edges),
                        Vector3(0, 0, 0),
                        edge_bounds.view(0, n_edges),
                        use_gpu);

    StackItem node;
    Matrix4x4 m = Matrix4x4::identity();
    compute_stack_item(node, 
                       shapes, 
                       edges.view(0, n_edges), 
                       edge_bounds, 
                       start_idx, 
                       end_idx, 
                       edge_ids_local, 
                       true, 
                       15, 
                       m, 
                       use_gpu);

    output[0] = node.bounds.p_min.x;
    output[1] = node.bounds.p_min.y;
    output[2] = node.bounds.p_min.z;

    output[3] = node.bounds.p_max.x;
    output[4] = node.bounds.p_max.y;
    output[5] = node.bounds.p_max.z;

    output[6] = node.rejectable;

    output[7] = node.D.x;
    output[8] = node.D.y;
    output[9] = node.D.z;
    output[10] = node.D.w;

    output[11] = node.dual_bounds.p_min.x;
    output[12] = node.dual_bounds.p_min.y;
    output[13] = node.dual_bounds.p_min.z;

    output[14] = node.dual_bounds.p_max.x;
    output[15] = node.dual_bounds.p_max.y;
    output[16] = node.dual_bounds.p_max.z;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            output[17 + i * 4 + j] = node.quadric_pair.quadric1.matrix(i, j);
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            output[33 + i * 4 + j] = node.quadric_pair.quadric2.matrix(i, j);
        }
    }
}