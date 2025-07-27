#include "visualizer.h"
#include "scene.h"
#include "ltc.h"

int get_n_nodes(const EdgeSampler &edge_sampler) {
    return edge_sampler.edge_tree->manifold_bvh_nodes.size();
}
int get_n_edges(const EdgeSampler &edge_sampler) {
    return edge_sampler.edges.size();
}

bool intersect_one_ray(const Scene &scene,
                       const Vector3 &ray_origin, 
                       const Vector3 &ray_dir, 
                       SurfacePoint &sp,
                       Matrix3x3 &m, 
                       Matrix3x3 &m_inv, 
                       Matrix3x3 &isotropic_frame) {

    initialize_ltc_table(false);

    int active_pixels = 0;
    Ray ray(ray_origin, normalize(ray_dir));
    RayDifferential ray_differential;
    Intersection intersection;
    RayDifferential new_ray_differential;
    OptiXRay optix_ray;
    OptiXHit optix_hit;

    intersect(scene, BufferView<int>(&active_pixels, 1),
                BufferView<Ray>(&ray, 1),
                BufferView<RayDifferential>(&ray_differential, 1),
                BufferView<Intersection>(&intersection, 1),
                BufferView<SurfacePoint>(&sp, 1),
                BufferView<RayDifferential>(&new_ray_differential, 1),
                BufferView<OptiXRay>(&optix_ray, 1),
                BufferView<OptiXHit>(&optix_hit, 1));

    if (intersection.valid()){
        const Shape &shape = scene.shapes[intersection.shape_id];
        const Material &material = scene.materials[shape.material_id];

        Vector3 wi = -normalize(ray.dir);
        Vector3 n = sp.shading_frame.n;
        if (material.two_sided) {
            if (dot(wi, n) < 0.f) {
                n = -n;
            }
        }
        auto frame_x = normalize(wi - n * dot(wi, n));
        auto frame_y = cross(n, frame_x);
        isotropic_frame = Matrix3x3(Frame{frame_x, frame_y, n});
        auto roughness = max(get_roughness(material, sp), 10e-9);
        m_inv = inverse(get_ltc_matrix(sp, wi, roughness, ltc::tabM)) *
                        isotropic_frame;
        m = inverse(m_inv);
        return true;
    }
    else {
        return false;
    }
}

// This rejection test is the same as the one in rejection_test.h. The only difference is that it returns 
// 0 if the point was rejected by the bbox test and 0.5 if the point was rejected by the quadric test.
template <typename T>
DEVICE 
Real rejection_test_visualizer(const TQuadricPair<T> &quadric_pair, 
                               const TMatrix4x4<T> &m, 
                               const AABB3 &dual_bounds, 
                               const TVector3<T> &p,
                               TVector3<T> *inter_points,
                               int &num_points) {
    TVector4<T> p_3d = transpose(m) * TVector4<T>(p.x, p.y, p.z, T(1));
    
    int num_points_ = 0;
    Buffer<int> bbox_edges_idxs(false, 24);
    Buffer<Vector3> bbox_edges_normals(false, 24);
    init_bbox_edges_and_normals(bbox_edges_idxs.view(0, 24), bbox_edges_normals.view(0, 24));
    intersect_plane(dual_bounds, p_3d, inter_points, &num_points_, bbox_edges_idxs.begin());
    num_points = num_points_;
    
    if (num_points == 0) {
        return 0;
    }
    else {
        TMatrix4x4<T> Q1_3d = transpose(m) * (quadric_pair).quadric1.matrix * m;
        TMatrix4x4<T> Q2_3d = transpose(m) * (quadric_pair).quadric2.matrix * m;

        for (int i = 0; i < num_points; i++) {
            TVector4<T> vertex(inter_points[i].x, inter_points[i].y, inter_points[i].z, T(1));
            T Q1_v = dot(vertex, Q1_3d * vertex);
            T Q2_v = dot(vertex, Q2_3d * vertex);
            T alpha = Q1_v / (Q1_v - Q2_v);
            if (alpha >= 0 && alpha <= 1) {
                return 1;
            }
        }


        TQuadric<T> Q1_3d_quad(Q1_3d);
        TQuadric<T> Q2_3d_quad(Q2_3d);

        for (int i = 0; i < num_points; i++) {
            TQuadricIntersection<T> inter = intersect_with_segment(Q1_3d_quad, inter_points[i], inter_points[(i + 1) % num_points]);
            if (length(inter.p1) > 0 || length(inter.p2) > 0) {
                return 1;
            }
            inter = intersect_with_segment(Q2_3d_quad, inter_points[i], inter_points[(i + 1) % num_points]);
            if (length(inter.p1) > 0 || length(inter.p2) > 0) {
                return 1;
            }

        }


        TVector3<T> p_3d_short = TVector3<T>(p_3d.x, p_3d.y, p_3d.z);
        TMatrix3x3<T> basis_3d = find_basis(p_3d_short / length(p_3d_short));

        TVector3<T> u(basis_3d(0, 0), basis_3d(1, 0), basis_3d(2, 0));
        TVector3<T> v(basis_3d(0, 1), basis_3d(1, 1), basis_3d(2, 1));
        
        TVector4<T> u_4d(u.x, u.y, u.z, T(0));
        TVector4<T> v_4d(v.x, v.y, v.z, T(0));
        TVector4<T> o_4d(inter_points[0].x, inter_points[0].y, inter_points[0].z, T(1));


        TConic<T> C1 = get_conic_from_quadric(Q1_3d_quad, o_4d, u_4d, v_4d);
        TConic<T> C2 = get_conic_from_quadric(Q2_3d_quad, o_4d, u_4d, v_4d);

        TVector3<T> point_C1 = get_point_on_conic(C1);
        TVector3<T> point_C2 = get_point_on_conic(C2);

        if (length(point_C1) > 0) {
            TVector3<T> point_C1_3d = point_C1.x * u + point_C1.y * v + inter_points[0];
            if (::inside(dual_bounds, point_C1_3d)) {
                return 1;
            }
        }
        if (length(point_C2) > 0) {
            TVector3<T> point_C2_3d = point_C2.x * u + point_C2.y * v + inter_points[0];
            if (::inside(dual_bounds, point_C2_3d)) {
                return 1;
            }
        }
        return 0.5;
    }
}


void compute_rejection_pattern(ptr<double> xs, 
                               ptr<double> ys, 
                               ptr<double> zs,
                               ptr<double> res, 
                               int nx, 
                               int ny, 
                               int nz,
                               int n_node, 
                               const EdgeSampler &edge_sampler) {
    BVHNode node = edge_sampler.edge_tree->manifold_bvh_nodes[n_node];
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                double x = xs[i];
                double y = ys[j];
                double z = zs[k];
                Vector3 p = Vector3(x, y, z);
                if (node.rejectable) {
                    Vector3 inter_points[12];
                    int num_points = 0;
                    res[i * ny * nz + j * nz + k] = rejection_test_visualizer(node.quadric_pair, 
                                                                              node.m,
                                                                              node.dual_bounds, 
                                                                              p, 
                                                                              inter_points, 
                                                                              num_points);
                } else {
                    res[i * ny * nz + j * nz + k] = 1;
                }
            }
        }
    }

}

void compute_gt_rejection_pattern(ptr<double> xs, 
                                  ptr<double> ys, 
                                  ptr<double> zs,
                                  ptr<double> res, 
                                  int nx, 
                                  int ny, 
                                  int nz,
                                  int n_node, 
                                  const EdgeSampler &edge_sampler, 
                                  const Scene &scene) {
    const BVHNode &node = edge_sampler.edge_tree->manifold_bvh_nodes[n_node];
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                res[i * ny * nz + j * nz + k] = 0;
                double x = xs[i];
                double y = ys[j];
                double z = zs[k];
                Vector3 p = Vector3(x, y, z);
                constexpr auto buffer_size = 128;
                BVHStackItemH buffer[buffer_size];

                auto stack_ptr = &buffer[0];
                
                *stack_ptr++ = BVHStackItemH{
                    BVHNodePtr{&node}, 0, 0};
                
                while (stack_ptr != &buffer[0]) {
                    assert(stack_ptr > &buffer[0] && stack_ptr < &buffer[buffer_size]);
                    const auto stack_item = *--stack_ptr;
                    
                    if (is_leaf(stack_item.node_ptr)) {
                        for (int ni = 0; ni < (stack_item.node_ptr.ptr)->num_children; ni++) {
                            int selected_edge = (stack_item.node_ptr.ptr)->edge_ids[ni];
                            const Edge &edge = edge_sampler.edges[selected_edge];
                            if (is_silhouette(scene.shapes.begin(), p, edge)) {
                                res[i * ny * nz + j * nz + k] = 1;
                            }
                        }
                    } else {
                        BVHNodePtr children[4];
                        get_children(stack_item.node_ptr, children);
                        for (int ni = 0; ni < (stack_item.node_ptr.ptr)->num_children; ni++) {
                            *stack_ptr++ = BVHStackItemH{children[ni], 0, 0};
                        }
                    }
                }
            }
        }
    }
}

Real compute_edge_probs(const SurfacePoint &sp, 
                        const Matrix3x3 &m,
                        const Matrix3x3 &m_inv, 
                        const Matrix3x3 &isotropic_frame,
                        ptr<double> probs, 
                        ptr<Vector3> edge_starts, 
                        ptr<Vector3> edge_ends,
                        const EdgeSampler &edge_sampler, 
                        const Scene &scene) {
    Buffer<int> bbox_edges_idxs(false, 24);
    Buffer<Vector3> bbox_edges_normals(false, 24);
    init_bbox_edges_and_normals(bbox_edges_idxs.view(0, 24), bbox_edges_normals.view(0, 24));

    constexpr auto buffer_size = 128;
    BVHStackItemH buffer[buffer_size];

    auto stack_ptr = &buffer[0];

    const BVHNode *nodes_buffer;
    int n_roots;

    auto edge_tree_roots = get_edge_tree_roots(edge_sampler.edge_tree.get());

    nodes_buffer = edge_tree_roots.manifold_bvh_roots;
    n_roots = edge_tree_roots.n_manifold_bvh_roots;

    Real prob_roots[n_roots];
    Real prob_sum_split = 0;
    for (int i = 0; i < n_roots; i++) {
        prob_roots[i] = node_importance(nodes_buffer + i, sp, m_inv, 
                                            isotropic_frame, 
                                            bbox_edges_idxs.data, 
                                            bbox_edges_normals.data);
        prob_sum_split += prob_roots[i];
    }

    if (prob_sum_split > 0) {
        for (int i = 0; i < n_roots; i++) {
            prob_roots[i] /= prob_sum_split;
            *stack_ptr++ = BVHStackItemH{
                BVHNodePtr{nodes_buffer + i}, 0, prob_roots[i]};
        }
    }
    else {
        for (int i = 0; i < n_roots; i++) {
            *stack_ptr++ = BVHStackItemH{
                BVHNodePtr{nodes_buffer + i}, 0, 1.0 / n_roots};
        }
    }
    
    Real prob_sum = 0;
    int n_edges = 0;
    
    while (stack_ptr != &buffer[0]) {
        assert(stack_ptr > &buffer[0] && stack_ptr < &buffer[buffer_size]);
        const auto stack_item = *--stack_ptr;
        
        if (is_leaf(stack_item.node_ptr)) {
            Real sum = 0;
            n_edges += (stack_item.node_ptr.ptr)->num_children;
            for (int ni = 0; ni < (stack_item.node_ptr.ptr)->num_children; ni++) {
                int selected_edge = (stack_item.node_ptr.ptr)->edge_ids[ni];
                const Edge &edge = edge_sampler.edges[selected_edge];
                Real prob = edge_importance(edge, sp, m, m_inv, isotropic_frame, scene.shapes.data);
                sum += prob;
            }
            if (sum == 0) {
                for (int ni = 0; ni < (stack_item.node_ptr.ptr)->num_children; ni++) {
                    int selected_edge = (stack_item.node_ptr.ptr)->edge_ids[ni];
                    const Edge &edge = edge_sampler.edges[selected_edge];
                    probs[selected_edge] = stack_item.pmf / (stack_item.node_ptr.ptr)->num_children;
                    prob_sum += probs[selected_edge];
                    const Shape &shape = scene.shapes[edge.shape_id];
                    edge_starts[selected_edge] = get_vertex(shape, edge.v0);
                    edge_ends[selected_edge] = get_vertex(shape, edge.v1);
                }
            } else {
                for (int ni = 0; ni < (stack_item.node_ptr.ptr)->num_children; ni++) {
                    int selected_edge = (stack_item.node_ptr.ptr)->edge_ids[ni];
                    const Edge &edge = edge_sampler.edges[selected_edge];
                    Real prob = edge_importance(edge, sp, m, m_inv, isotropic_frame, scene.shapes.data);
                    probs[selected_edge] = (prob / sum) * stack_item.pmf;
                    prob_sum += probs[selected_edge];
                    const Shape &shape = scene.shapes[edge.shape_id];
                    edge_starts[selected_edge] = get_vertex(shape, edge.v0);
                    edge_ends[selected_edge] = get_vertex(shape, edge.v1);
                }
            }
        } else {
            BVHNodePtr children[4];
            get_children(stack_item.node_ptr, children);

            Real probs[4];
            for (int i = 0; i < 4; i++) {
                probs[i] = 0;
            }
            Real current_pmf = stack_item.pmf;

            for (int i = 0; i < stack_item.node_ptr.ptr->num_children; i++) {                        
                probs[i] = node_importance(children[i].ptr, sp, m_inv, 
                                               isotropic_frame, 
                                               bbox_edges_idxs.data, 
                                               bbox_edges_normals.data);
            }
            Real imp_total = probs[0] + probs[1] + probs[2] + probs[3];
            if (imp_total > 0) {
                for (int i = 0; i < stack_item.node_ptr.ptr->num_children; i++) {
                    probs[i] /= imp_total;
                    *stack_ptr++ = BVHStackItemH{
                                children[i], 0, current_pmf * probs[i]};
                }
            }
            else {
                for (int i = 0; i < stack_item.node_ptr.ptr->num_children; i++) {
                    *stack_ptr++ = BVHStackItemH{
                                children[i], 0, current_pmf / stack_item.node_ptr.ptr->num_children};
                }
            }
        }
    }
    Real max_prob = 0;
    for (int i = 0; i < edge_sampler.edges.size(); i++) {
        probs[i] /= prob_sum;
        if (probs[i] > max_prob) {
            max_prob = probs[i];
        }
    }
    assert (n_edges == edge_sampler.edges.size());
    return max_prob;
}


Real compute_gt_edge_probs(const SurfacePoint &sp, 
                           const Matrix3x3 &m,
                           const Matrix3x3 &m_inv, 
                           const Matrix3x3 &isotropic_frame,
                           ptr<double> probs,
                           ptr<Vector3> edge_starts, 
                           ptr<Vector3> edge_ends,
                           const EdgeSampler &edge_sampler, 
                           const Scene &scene) {
    Real sum = 0;
    for (int i = 0; i < edge_sampler.edges.size(); i++) {
        const Edge &edge = edge_sampler.edges[i];
        Real imp = edge_importance(edge, sp, m, m_inv, isotropic_frame, scene.shapes.data);
        sum += imp;
        probs[i] = imp;
        const Shape &shape = scene.shapes[edge.shape_id];
        edge_starts[i] = get_vertex(shape, edge.v0);
        edge_ends[i] = get_vertex(shape, edge.v1);
    }
    Real max_prob = 0;
    for (int i = 0; i < edge_sampler.edges.size(); i++) {
        probs[i] /= sum;
        if (probs[i] > max_prob) {
            max_prob = probs[i];
        }
    }
    return max_prob;
}