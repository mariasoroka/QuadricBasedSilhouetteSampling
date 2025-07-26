#include "edge.h"
#include "line_clip.h"
#include "scene.h"
#include "parallel.h"
#include "thrust_utils.h"
#include "dpdf.h"
#include "ltc.h"
#include "solid_angles.h"
#include <memory>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/remove.h>

#define MAX_EDGES_PER_LEAF 4
#define MIN_POINTS_FOR_FITTING 15

struct edge_collector {
    DEVICE inline void operator()(int idx) {
        const auto &shape = *shape_ptr;
        // For each triangle
        auto ind = get_indices(shape, idx / 3);
        if ((idx % 3) == 0) {
            edges[idx] = Edge{shape_id,
                              min(ind[0], ind[1]),
                              max(ind[0], ind[1]),
                              idx / 3, -1};
        } else if ((idx % 3) == 1) {
            edges[idx] = Edge{shape_id,
                              min(ind[1], ind[2]),
                              max(ind[1], ind[2]),
                              idx / 3, -1};
        } else {
            edges[idx] = Edge{shape_id,
                              min(ind[2], ind[0]),
                              max(ind[2], ind[0]),
                              idx / 3, -1};
        }
    }

    int shape_id;
    const Shape *shape_ptr;
    Edge *edges;
};

struct edge_less_comparer {
    DEVICE inline bool operator()(const Edge &e0, const Edge &e1) {
        if (e0.v0 == e1.v0) {
            return e0.v1 < e1.v1;
        }
        return e0.v0 < e1.v0;
    }
};

struct edge_equal_comparer {
    DEVICE inline bool operator()(const Edge &e0, const Edge &e1) {
        return e0.v0 == e1.v0 && e0.v1 == e1.v1;
    }
};

struct edge_merger {
    DEVICE inline Edge operator()(const Edge &e0, const Edge &e1) {
        return Edge{e0.shape_id, e0.v0, e0.v1, e0.f0, e1.f0};
    }
};

DEVICE inline bool less_than(const Vector3f &v0, const Vector3f &v1) {
    if (v0.x != v1.x) {
        return v0.x < v1.x;
    } else if (v0.y != v1.y) {
        return v0.y < v1.y;
    } else if (v0.z != v1.z) {
        return v0.z < v1.z;
    }
    return true;
}

struct edge_vertex_comparer {
    DEVICE inline bool operator()(const Edge &e0, const Edge &e1) {
        // First, locally sort v0 & v1 within e0 & e1
        auto v00 = get_vertex(*shape_ptr, e0.v0);
        auto v01 = get_vertex(*shape_ptr, e0.v1);
        if (less_than(v01, v00)) {
            swap_(v00, v01);
        }
        auto v10 = get_vertex(*shape_ptr, e1.v0);
        auto v11 = get_vertex(*shape_ptr, e1.v1);
        if (less_than(v11, v10)) {
            swap_(v10, v11);
        }
        // Next, compare and return
        if (v00 != v10) {
            return less_than(v00, v10);
        } else if (v01 != v11) {
            return less_than(v01, v11);
        }
        return true;
    }

    const Shape *shape_ptr;
};

struct edge_face_assigner {
    DEVICE void operator()(int idx) {
        auto &edge = edges[idx];
        if (edge.f1 != -1) {
            return;
        }
        auto v0 = get_vertex(*shape_ptr, edge.v0);
        auto v1 = get_vertex(*shape_ptr, edge.v1);
        if (less_than(v1, v0)) {
            swap_(v0, v1);
        }
        if (idx > 0) {
            const auto &cmp_edge = edges[idx - 1];
            auto cmp_v0 = get_vertex(*shape_ptr, cmp_edge.v0);
            auto cmp_v1 = get_vertex(*shape_ptr, cmp_edge.v1);
            if (less_than(cmp_v1, cmp_v0)) {
                swap_(cmp_v0, cmp_v1);
            }
            if (v0 == cmp_v0 && v1 == cmp_v1) {
                edge.f1 = cmp_edge.f0;
            }
        }
        if (idx < num_edges - 1) {
            const auto &cmp_edge = edges[idx + 1];
            auto cmp_v0 = get_vertex(*shape_ptr, cmp_edge.v0);
            auto cmp_v1 = get_vertex(*shape_ptr, cmp_edge.v1);
            if (less_than(cmp_v1, cmp_v0)) {
                swap_(cmp_v0, cmp_v1);
            }
            if (v0 == cmp_v0 && v1 == cmp_v1) {
                edge.f1 = cmp_edge.f0;
            }
        }
    }

    const Shape *shape_ptr;
    Edge *edges;
    int num_edges;
};

struct edge_remover {
    DEVICE inline bool operator()(const Edge &e) {
        if (e.f0 == -1 || e.f1 == -1) {
            // Only adjacent to one face
            return false;
        }
        auto v0 = Vector3{get_v0(shapes, e)};
        auto v1 = Vector3{get_v1(shapes, e)};
        auto ns_v0 = Vector3{get_non_shared_v0(shapes, e)};
        auto ns_v1 = Vector3{get_non_shared_v1(shapes, e)};
        auto n0 = normalize(cross(v0 - ns_v0, v1 - ns_v0));
        auto n1 = normalize(cross(v1 - ns_v1, v0 - ns_v1));
        auto mid_point_vec = normalize((ns_v0 + ns_v1) / 2 - v0);
        auto n0_geom = Vector3{get_normal(shapes[e.shape_id], e.f0)};
        auto n1_geom = Vector3{get_normal(shapes[e.shape_id], e.f1)};
        if (remove_concave) {
            return (dot(n0, n1) >= (1 - 1e-6f)) || (dot(mid_point_vec, n0_geom) > 0) || (dot(mid_point_vec, n1_geom) > 0);
        }
        else {
            return dot(n0, n1) >= (1 - 1e-6f);
        }
    }
    const Shape *shapes;
    bool remove_concave;
};

struct primary_edge_weighter {
    DEVICE void operator()(int idx) {
        const auto &edge = edges[idx];
        auto &primary_edge_weight = primary_edge_weights[idx];
        auto v0 = get_v0(shapes, edge);
        auto v1 = get_v1(shapes, edge);
        auto v0p = Vector2{};
        auto v1p = Vector2{};
        primary_edge_weight = 0;
        // Project to screen space
        if (project(camera, Vector3(v0), Vector3(v1), v0p, v1p)) {
            auto v0c = v0p;
            auto v1c = v1p;
            // Clip against screen boundaries
            if (clip_line(v0p, v1p, v0c, v1c)) {
                // Reject non-silhouette edges
                auto org = xfm_point(camera.cam_to_world, Vector3{0, 0, 0});
                if (is_silhouette(shapes, org, edge)) {
                    primary_edge_weight = distance(v0c, v1c);
                }
            }
        }
    }

    Camera camera;
    const Shape *shapes;
    const Edge *edges;
    Real *primary_edge_weights;
};

struct secondary_edge_weighter {
    DEVICE void operator()(int idx) {
        const auto &edge = edges[idx];
        // We use the length * (pi - dihedral angle) to sample the edges
        // If the dihedral angle is large, it's less likely that the edge would be a silhouette
        auto &secondary_edge_weight = secondary_edge_weights[idx];
        auto exterior_dihedral = compute_exterior_dihedral_angle(shapes, edge);
        auto v0 = get_v0(shapes, edge);
        auto v1 = get_v1(shapes, edge);
        secondary_edge_weight = distance(v0, v1) * exterior_dihedral;
    }

    const Shape *shapes;
    const Edge *edges;
    Real *secondary_edge_weights;
};

EdgeSampler::EdgeSampler(const Scene &scene,
                         bool use_primary_edge_sampling,
                         bool use_secondary_edge_sampling,
                         bool remove_concave) {
    if (!use_primary_edge_sampling && !use_secondary_edge_sampling) {
        // No need to collect edges
        return;
    }
    auto shapes_buffer = scene.shapes.view(0, (int)scene.shapes.count);

    // Conservatively allocate a big buffer for all edges
    auto num_total_triangles = 0;
    for (int shape_id = 0; shape_id < (int)scene.shapes.count; shape_id++) {
        if (shapes_buffer[shape_id].light_id == -1) {
            num_total_triangles += shapes_buffer[shape_id].num_triangles;
        }
    }
    // Collect the edges
    // TODO: this assumes each edge is only associated with two triangles
    //       which may be untrue for some pathological meshes.
    //       For edges associated to more than two triangles, 
    //       we should just ignore them
    edges = Buffer<Edge>(scene.use_gpu, 3 * num_total_triangles);
    auto edges_buffer = Buffer<Edge>(scene.use_gpu, 3 * num_total_triangles);
    auto current_num_edges = 0;
    for (int shape_id = 0; shape_id < (int)scene.shapes.count; shape_id++) {
        if (shapes_buffer[shape_id].light_id != -1) {
            continue;
        }
        parallel_for(edge_collector{
            shape_id,
            shapes_buffer.begin() + shape_id,
            edges.data + current_num_edges
        }, 3 * shapes_buffer[shape_id].num_triangles, scene.use_gpu);
        // Merge the edges
        auto edges_begin = edges.data + current_num_edges;
        DISPATCH(scene.use_gpu, thrust::sort,
                 edges_begin,
                 edges_begin + 3 * shapes_buffer[shape_id].num_triangles,
                 edge_less_comparer{});
        auto edges_buffer_begin = edges_buffer.data;
        auto new_end = DISPATCH(scene.use_gpu, thrust::reduce_by_key,
            edges_begin, // input keys
            edges_begin + 3 * shapes_buffer[shape_id].num_triangles,
            edges_begin, // input values
            edges_buffer_begin, // output keys
            edges_buffer_begin, // output values
            edge_equal_comparer{},
            edge_merger{}).first;
        auto num_edges = new_end - edges_buffer_begin;
        // Sometimes there are duplicated edges that don't get merged 
        // in the procedure above (e.g. UV seams), here we make sure these edges
        // are associated with two faces.
        // We do this by sorting the edges again based on vertex positions,
        // look at nearby edges and assign faces.
        DISPATCH(scene.use_gpu, thrust::sort,
                 edges_buffer_begin,
                 edges_buffer_begin + num_edges,
                 edge_vertex_comparer{shapes_buffer.begin() + shape_id});
        parallel_for(edge_face_assigner{
            shapes_buffer.begin() + shape_id,
            edges_buffer_begin,
            (int)num_edges
        }, num_edges, scene.use_gpu);

        DISPATCH(scene.use_gpu, thrust::copy, edges_buffer_begin, new_end, edges_begin);
        current_num_edges += (int)num_edges;
    }
    // Remove edges with 180 degree dihedral angles. Optionally, remove concave edges.
    auto edges_end = DISPATCH(scene.use_gpu, thrust::remove_if, edges.begin(),
        edges.begin() + current_num_edges, edge_remover{shapes_buffer.begin(), remove_concave});
    edges.count = edges_end - edges.begin();

    if (use_primary_edge_sampling) {
        // Primary edge sampler:
        primary_edges_pmf = Buffer<Real>(scene.use_gpu, edges.count);
        primary_edges_cdf = Buffer<Real>(scene.use_gpu, edges.count);
        // For each edge, if it is a silhouette, we project them on screen
        // and compute the screen-space length. We store the length in
        // primary_edges_pmf
        {
            parallel_for(primary_edge_weighter{
                scene.camera,
                scene.shapes.data,
                edges.begin(),
                primary_edges_pmf.begin()
            }, edges.size(), scene.use_gpu);
            // Compute PMF & CDF
            // First normalize primary_edges_pmf.
            auto total_length = DISPATCH(scene.use_gpu, thrust::reduce,
                primary_edges_pmf.begin(),
                primary_edges_pmf.end(),
                Real(0),
                thrust::plus<Real>());
            DISPATCH(scene.use_gpu, thrust::transform,
                primary_edges_pmf.begin(),
                primary_edges_pmf.end(),
                thrust::make_constant_iterator(total_length),
                primary_edges_pmf.begin(),
                thrust::divides<Real>());
            // Next we compute a prefix sum
            DISPATCH(scene.use_gpu, thrust::transform_exclusive_scan,
                primary_edges_pmf.begin(),
                primary_edges_pmf.end(),
                primary_edges_cdf.begin(),
                thrust::identity<Real>(), Real(0), thrust::plus<Real>());
        }
    }
    if (use_secondary_edge_sampling) {
        // Secondary edge sampler

        // Build a hierarchical data structure for edge sampling
        edge_tree = std::unique_ptr<EdgeTree>(
            new EdgeTree(scene.use_gpu,
                            scene.camera,
                            shapes_buffer,
                            edges.view(0, edges.size()),
                            MAX_EDGES_PER_LEAF,
                            MIN_POINTS_FOR_FITTING
                            ));
    }
}

struct primary_edge_sampler {
    DEVICE void operator()(int idx) {
        // Initialize output
        edge_records[idx] = PrimaryEdgeRecord{};
        throughputs[2 * idx + 0] = Vector3{0, 0, 0};
        throughputs[2 * idx + 1] = Vector3{0, 0, 0};
        auto nd = channel_info.num_total_dimensions;
        for (int d = 0; d < nd; d++) {
            channel_multipliers[2 * nd * idx + d] = 0;
            channel_multipliers[2 * nd * idx + d + nd] = 0;
        }
        rays[2 * idx + 0] = Ray(Vector3{0, 0, 0}, Vector3{0, 0, 0});
        rays[2 * idx + 1] = Ray(Vector3{0, 0, 0}, Vector3{0, 0, 0});

        // Sample an edge by binary search on cdf
        auto sample = samples[idx];
        const Real *edge_ptr = thrust::upper_bound(thrust::seq,
                edges_cdf, edges_cdf + num_edges,
                sample.edge_sel);
        auto edge_id = clamp((int)(edge_ptr - edges_cdf - 1),
                                   0, num_edges - 1);
        const auto &edge = edges[edge_id];
        // Sample a point on the edge
        auto v0 = Vector3{get_v0(shapes, edge)};
        auto v1 = Vector3{get_v1(shapes, edge)};
        // Project the edge onto screen space
        auto v0_ss = Vector2{0, 0};
        auto v1_ss = Vector2{0, 0};
        if (!project(camera, v0, v1, v0_ss, v1_ss)) {
            return;
        }
        if (edges_pmf[edge_id] <= 0.f) {
            return;
        }

        if ((camera.camera_type == CameraType::Perspective ||
                camera.camera_type == CameraType::Orthographic) &&
                !camera.distortion_params.defined) {
            // Perspective or Orthographic cameras

            // Uniform sample on the edge
            auto edge_pt = v0_ss + sample.t * (v1_ss - v0_ss);
            // Reject samples outside of image plane
            if (!in_screen(camera, edge_pt)) {
                return;
            }

            edge_records[idx].edge = edge;
            edge_records[idx].edge_pt = edge_pt;

            // Generate two rays at the two sides of the edge
            auto half_space_normal = get_normal(normalize(v0_ss - v1_ss));
            // Sample a primary ray through the sampled point on the edge
            auto ray_tmp = sample_primary(camera, edge_pt);
            // Compute the corresponding intersection point on the edge
            // For that solve for alpha and t s.t. v0 + alpha*(v1 - v0) = ray_tmp.org + t*ray_tmp.dir
            Real a1 = dot(ray_tmp.dir, v0 - ray_tmp.org);
            Real a2 = dot(v0 - ray_tmp.org, v0 - v1);
            Real b1 = length_squared(ray_tmp.dir);
            Real b2 = dot(v0 - v1, ray_tmp.dir);
            Real b4 = length_squared(v0 - v1);
            Real det = b1 * b4 - b2 * b2;
            Real alpha = (-b2 * a1 + b1 * a2) / det;
            auto on_edge_pt = v0 + alpha * (v1 - v0);

            // Set the two rays. One of them starts at the edge point, and the second one is ray_tmp
            rays[2 * idx + 0] = Ray(on_edge_pt, ray_tmp.dir);
            rays[2 * idx + 1] = ray_tmp;
            
            // It is unknown which face the first ray will intersect
            shading_isects[2 * idx + 0] = -1;
            // The second ray will intersect the triangle plane adjacent to the sampled edge
            int face_idx = dot(get_normal(scene.shapes[edge.shape_id], edge.f0), ray_tmp.dir) < 0 ? edge.f0 : edge.f1;
            shading_isects[2 * idx + 1] = Intersection(edge.shape_id, face_idx);
            shading_points[2 * idx + 1].position = on_edge_pt;

            Vector3 normal = cross(v0 - v1, ray_tmp.dir);
            Vector3 other_vert = get_non_shared_v0(shapes, edge);

            // Compute the corresponding backprop derivatives
            auto xi = clamp(int(edge_pt[0] * camera.width - camera.viewport_beg.x),
                            0, camera.viewport_end.x - camera.viewport_beg.x);
            auto yi = clamp(int(edge_pt[1] * camera.height - camera.viewport_beg.y),
                            0, camera.viewport_end.y - camera.viewport_beg.y);
            auto rd = channel_info.radiance_dimension;
            auto d_color = Vector3{0, 0, 0};
            if (rd != -1) {
                auto viewport_width = camera.viewport_end.x - camera.viewport_beg.x;
                auto viewport_height = camera.viewport_end.y - camera.viewport_beg.y;

                if(camera.filter_type == FilterType::Box) {
                    d_color = Vector3{
                        d_rendered_image[nd * (yi * viewport_width + xi) + rd + 0],
                        d_rendered_image[nd * (yi * viewport_width + xi) + rd + 1],
                        d_rendered_image[nd * (yi * viewport_width + xi) + rd + 2]
                    };
                } else if (camera.filter_type == FilterType::Gaussian) {
                    // Aggregate weighted sum of nearest 
                    auto normalization = Real(0);
                    for(int _kx = -3; _kx <= 3; _kx++) {
                        for(int _ky = -3; _ky <= 3; _ky++) {
                            auto kx = xi + _kx;
                            auto ky = yi + _ky;

                            if (kx < 0 || kx >= viewport_width)
                                continue;

                            if (ky < 0 || ky >= viewport_height)
                                continue;
                            
                            // Get pixel center.
                            Vector2 pixel_center;
                            local_to_screen_pos(camera, 
                                        (ky * viewport_width + kx),
                                        Vector2{0, 0},
                                        pixel_center);
                            
                            auto local_pos = Vector2{
                                (edge_pt[0] - pixel_center[0]) * viewport_width,
                                (edge_pt[1] - pixel_center[1]) * viewport_height
                            };

                            Vector2 jacobian;
                            Real value;
                            screen_filter_grad(camera, 
                                                idx,
                                                local_pos,
                                                jacobian,
                                                value);

                            d_color += value * Vector3{
                                d_rendered_image[nd * (ky * viewport_width + kx) + rd + 0],
                                d_rendered_image[nd * (ky * viewport_width + kx) + rd + 1],
                                d_rendered_image[nd * (ky * viewport_width + kx) + rd + 2]
                            };

                            normalization += value;
                        }
                    }

                    if (normalization > Real(1e-15)) {
                        d_color /= normalization;
                    }

                }
                else {
                    d_color = Vector3{
                        d_rendered_image[nd * (yi * viewport_width + xi) + rd + 0],
                        d_rendered_image[nd * (yi * viewport_width + xi) + rd + 1],
                        d_rendered_image[nd * (yi * viewport_width + xi) + rd + 2]
                    };
                }
            }
            // The weight is the length of edge divided by the probability
            // of selecting this edge, divided by the length of gradients
            // of the edge equation w.r.t. screen coordinate.
            // For perspective projection the length of edge and gradients
            // cancel each other out.
            // For fisheye & panorama we need to compute the Jacobians
            auto upper_weight = d_color / edges_pmf[edge_id];
            auto lower_weight = -d_color / edges_pmf[edge_id];

            assert(isfinite(d_color));
            assert(isfinite(upper_weight));

            // Make sure that throughputs are assigned correctly
            if (dot(normal, other_vert - v0) < 0) {
                throughputs[2 * idx + 1] = upper_weight;
                throughputs[2 * idx + 0] = lower_weight;
            }
            else {
                throughputs[2 * idx + 0] = upper_weight;
                throughputs[2 * idx + 1] = lower_weight;
            }

            for (int d = 0; d < nd; d++) {
                auto viewport_width = camera.viewport_end.x - camera.viewport_beg.x;
                auto d_channel = d_rendered_image[nd * (yi * viewport_width + xi) + d];
                channel_multipliers[2 * nd * idx + d] = d_channel / edges_pmf[edge_id];
                channel_multipliers[2 * nd * idx + d + nd] = -d_channel / edges_pmf[edge_id];
            }
        } else {
            assert(camera.camera_type == CameraType::Fisheye ||
                   camera.camera_type == CameraType::Panorama ||
                   camera.distortion_params.defined);
            // No support for Gaussian filter for non-persective lens.
            assert(camera.filter_type == FilterType::Box);

            // Fisheye or Panorama

            // In paper we focused on linear projection model.
            // However we also support non-linear models such as fisheye
            // projection.
            // To sample a point on the edge for non-linear models,
            // we need to sample in camera space instead of screen space,
            // since the edge is no longer a line segment in screen space.
            // Therefore we perform an "unprojection" to project the edge
            // from screen space to the film in camera space.
            // For perspective camera this is equivalent to sample in screen space:
            // we unproject (x, y) to (x', y', 1) where x', y' are just individual
            // affine transforms of x, y.
            // For fisheye camera we unproject from screen-space to the unit
            // sphere.
            // Therefore the following code also works for perspective camera,
            // but to make things more consistent to the paper we provide
            // two versions of code.
            auto v0_dir = screen_to_camera(camera, v0_ss);
            auto v1_dir = screen_to_camera(camera, v1_ss);
            // Uniform sample in camera space
            auto v_dir3 = v1_dir - v0_dir;
            auto edge_pt3 = v0_dir + sample.t * v_dir3;
            // Project back to screen space
            auto edge_pt = camera_to_screen(camera, edge_pt3);
            // Reject samples outside of image plane
            if (!in_screen(camera, edge_pt)) {
                // In theory this shouldn't happen since we clamp the edges
                return;
            }

            edge_records[idx].edge = edge;
            edge_records[idx].edge_pt = edge_pt;

            // The 3D edge equation for the fisheye & panorama camera is:
            // alpha(p) = dot(p, cross(v0_dir, v1_dir))
            // Thus the half-space normal is cross(v0_dir, v1_dir)
            // Generate two rays at the two sides of the edge
            // We choose the ray offset such that the longer the edge is from
            // the camera, the smaller the offset is.
            auto half_space_normal = normalize(cross(v0_dir, v1_dir));
            auto v0_local = xfm_point(camera.world_to_cam, v0);
            auto v1_local = xfm_point(camera.world_to_cam, v1);
            auto edge_local = v0_local + sample.t * v1_local;
            auto offset = 1e-5f / length(edge_local);
            auto upper_dir = normalize(edge_pt3 + offset * half_space_normal);
            auto upper_pt = camera_to_screen(camera, upper_dir);
            auto upper_ray = sample_primary(camera, upper_pt);
            auto lower_dir = normalize(edge_pt3 - offset * half_space_normal);
            auto lower_pt = camera_to_screen(camera, lower_dir);
            auto lower_ray = sample_primary(camera, lower_pt);
            rays[2 * idx + 0] = upper_ray;
            rays[2 * idx + 1] = lower_ray;

            // Compute the corresponding backprop derivatives
            auto xi = clamp(int(edge_pt[0] * camera.width - camera.viewport_beg.x),
                            0, camera.viewport_end.x - camera.viewport_beg.x);
            auto yi = clamp(int(edge_pt[1] * camera.height - camera.viewport_beg.y),
                            0, camera.viewport_end.y - camera.viewport_beg.y);
            auto rd = channel_info.radiance_dimension;
            auto d_color = Vector3{0, 0, 0};
            if (rd != -1) {
                auto viewport_width = camera.viewport_end.x - camera.viewport_beg.x;
                d_color = Vector3{
                    d_rendered_image[nd * (yi * viewport_width + xi) + rd + 0],
                    d_rendered_image[nd * (yi * viewport_width + xi) + rd + 1],
                    d_rendered_image[nd * (yi * viewport_width + xi) + rd + 2]
                };
            }
            // The weight is the length of edge divided by the probability
            // of selecting this edge, divided by the length of gradients
            // of the edge equation w.r.t. screen coordinate.
            // For perspective projection the length of edge and gradients
            // cancel each other out.
            // For fisheye & Panorama we need to compute the Jacobians
            auto upper_weight = d_color / edges_pmf[edge_id];
            auto lower_weight = -d_color / edges_pmf[edge_id];

            // alpha(p(x, y)) = dot(p(x, y), cross(v0_dir, v1_dir))
            // p = screen_to_camera(x, y)
            auto d_edge_pt = Vector2{0, 0};
            // dalpha/dx & dalpha/dy (d alpha / d p = cross(v0_dir, v1_dir))
            d_screen_to_camera(camera, edge_pt, cross(v0_dir, v1_dir), d_edge_pt);
            auto dirac_jacobian = 1.f / sqrt(square(d_edge_pt.x) + square(d_edge_pt.y));
            // We use finite difference to compute the Jacobian
            // for sampling on the line
            auto jac_offset = Real(1e-6);
            auto edge_pt3_delta = v0_dir + (sample.t + jac_offset) * v_dir3;
            auto edge_pt_delta = camera_to_screen(camera, edge_pt3_delta);
            auto line_jacobian = length((edge_pt_delta - edge_pt) / offset);
            auto jacobian = line_jacobian * dirac_jacobian;
            upper_weight *= jacobian;
            lower_weight *= jacobian;

            assert(isfinite(upper_weight));

            throughputs[2 * idx + 0] = upper_weight;
            throughputs[2 * idx + 1] = lower_weight;
            for (int d = 0; d < nd; d++) {
                auto viewport_width = camera.viewport_end.x - camera.viewport_beg.x;
                auto d_channel = d_rendered_image[nd * (yi * viewport_width + xi) + d];
                channel_multipliers[2 * nd * idx + d] =
                    d_channel * jacobian / edges_pmf[edge_id];
                channel_multipliers[2 * nd * idx + d + nd] =
                    -d_channel * jacobian / edges_pmf[edge_id];
            }
        }

        // Ray differential computation
        auto screen_pos = edge_records[idx].edge_pt;
        auto ray = sample_primary(camera, screen_pos);
        auto delta = Real(1e-3);
        auto screen_pos_dx = screen_pos + Vector2{delta, Real(0)};
        auto ray_dx = sample_primary(camera, screen_pos_dx);
        auto screen_pos_dy = screen_pos + Vector2{Real(0), delta};
        auto ray_dy = sample_primary(camera, screen_pos_dy);
        auto pixel_size_x = Real(0.5) / camera.width;
        auto pixel_size_y = Real(0.5) / camera.height;
        auto org_dx = pixel_size_x * (ray_dx.org - ray.org) / delta;
        auto org_dy = pixel_size_y * (ray_dy.org - ray.org) / delta;
        auto dir_dx = pixel_size_x * (ray_dx.dir - ray.dir) / delta;
        auto dir_dy = pixel_size_y * (ray_dy.dir - ray.dir) / delta;
        primary_ray_differentials[idx] = RayDifferential{org_dx, org_dy, dir_dx, dir_dy};
    }

    const FlattenScene scene;
    const Camera camera;
    const Shape *shapes;
    const Edge *edges;
    int num_edges;
    const Real *edges_pmf;
    const Real *edges_cdf;
    const PrimaryEdgeSample *samples;
    const float *d_rendered_image;
    const ChannelInfo channel_info;
    PrimaryEdgeRecord *edge_records;
    Ray *rays;
    RayDifferential *primary_ray_differentials;
    Vector3 *throughputs;
    Real *channel_multipliers;
    Intersection *shading_isects;
    SurfacePoint *shading_points;
};

void sample_primary_edges(const Scene &scene,
                          const BufferView<PrimaryEdgeSample> &samples,
                          const float *d_rendered_image,
                          const ChannelInfo &channel_info,
                          BufferView<PrimaryEdgeRecord> edge_records,
                          BufferView<Ray> rays,
                          BufferView<RayDifferential> primary_ray_differentials,
                          BufferView<Vector3> throughputs,
                          BufferView<Real> channel_multipliers,
                          BufferView<Intersection> &shading_isects,
                          BufferView<SurfacePoint> &shading_points) {
    parallel_for(primary_edge_sampler{
        get_flatten_scene(scene),
        scene.camera,
        scene.shapes.data,
        scene.edge_sampler.edges.begin(),
        (int)scene.edge_sampler.edges.size(),
        scene.edge_sampler.primary_edges_pmf.begin(),
        scene.edge_sampler.primary_edges_cdf.begin(),
        samples.begin(),
        d_rendered_image,
        channel_info,
        edge_records.begin(),
        rays.begin(),
        primary_ray_differentials.begin(),
        throughputs.begin(),
        channel_multipliers.begin(),
        shading_isects.begin(),
        shading_points.begin()
    }, samples.size(), scene.use_gpu);
}

struct primary_edge_weights_updater {
    DEVICE void operator()(int idx) {
        const auto &edge_record = edge_records[idx];
        auto isect_upper = shading_isects[2 * idx + 0];
        auto isect_lower = shading_isects[2 * idx + 1];
        auto &throughputs_upper = throughputs[2 * idx + 0];
        auto &throughputs_lower = throughputs[2 * idx + 1];
        // At least one of the intersections should be connected to the edge
        bool upper_connected = isect_upper.shape_id == edge_record.edge.shape_id &&
            (isect_upper.tri_id == edge_record.edge.f0 || isect_upper.tri_id == edge_record.edge.f1);
        bool lower_connected = isect_lower.shape_id == edge_record.edge.shape_id &&
            (isect_lower.tri_id == edge_record.edge.f0 || isect_lower.tri_id == edge_record.edge.f1);
        // Check that both intersections are valid
        bool valid_upper = isect_upper.valid() || (isect_upper.infinity() && scene.envmap != nullptr);
        bool valid_lower = isect_lower.valid() || (isect_lower.infinity() && scene.envmap != nullptr);
        if ((!upper_connected && !lower_connected) || !valid_upper || !valid_lower) {
            throughputs_upper = Vector3{0, 0, 0};
            throughputs_lower = Vector3{0, 0, 0};
            auto nd = channel_info.num_total_dimensions;
            for (int d = 0; d < nd; d++) {
                channel_multipliers[2 * nd * idx + d] = 0;
                channel_multipliers[2 * nd * idx + d + nd] = 0;
            }
        }
    }
    const FlattenScene scene;
    const PrimaryEdgeRecord *edge_records;
    const Intersection *shading_isects;
    const ChannelInfo channel_info;
    Vector3 *throughputs;
    Real *channel_multipliers;
};

void update_primary_edge_weights(const Scene &scene,
                                 const BufferView<PrimaryEdgeRecord> &edge_records,
                                 const BufferView<Intersection> &edge_isects,
                                 const ChannelInfo &channel_info,
                                 BufferView<Vector3> throughputs,
                                 BufferView<Real> channel_multipliers) {
    // XXX: Disable this at the moment. Not sure if this is more robust or not.
    // Removing invalid intersections
    parallel_for(primary_edge_weights_updater{
        get_flatten_scene(scene),
        edge_records.begin(),
        edge_isects.begin(),
        channel_info,
        throughputs.begin(),
        channel_multipliers.begin()
    }, edge_records.size(), scene.use_gpu);
}

struct primary_edge_derivatives_computer {
    DEVICE void operator()(int idx) {
        const auto &edge_record = edge_records[idx];
        auto edge_contrib_upper = edge_contribs[2 * idx + 0];
        auto edge_contrib_lower = edge_contribs[2 * idx + 1];
        auto edge_contrib = edge_contrib_upper + edge_contrib_lower;

        // Initialize derivatives
        if (edge_record.edge.shape_id < 0) {
            return;
        }

        auto v0 = Vector3{get_v0(shapes, edge_record.edge)};
        auto v1 = Vector3{get_v1(shapes, edge_record.edge)};
        auto v0_ss = Vector2{0, 0};
        auto v1_ss = Vector2{0, 0};
        if (!project(camera, v0, v1, v0_ss, v1_ss)) {
            return;
        }
        auto d_v0_ss = Vector2{0, 0};
        auto d_v1_ss = Vector2{0, 0};
        auto d_edge_pt = Vector2{0, 0};
        auto edge_pt = edge_record.edge_pt;
        if ((camera.camera_type == CameraType::Perspective ||
                camera.camera_type == CameraType::Orthographic) &&
                !camera.distortion_params.defined) {
            // Equation 8 in the paper
            d_v0_ss.x = v1_ss.y - edge_pt.y;
            d_v0_ss.y = edge_pt.x - v1_ss.x;
            d_v1_ss.x = edge_pt.y - v0_ss.y;
            d_v1_ss.y = v0_ss.x - edge_pt.x;
            d_edge_pt.x = v0_ss.y - v1_ss.y;
            d_edge_pt.y = v1_ss.x - v0_ss.x;
        } else {
            assert(camera.camera_type == CameraType::Fisheye ||
                   camera.camera_type == CameraType::Panorama ||
                   camera.distortion_params.defined);

            // This also works for perspective camera,
            // but for consistency we provide two versions.
            // alpha(p) = dot(p, cross(v0_dir, v1_dir))
            // v0_dir = screen_to_camera(v0_ss)
            // v1_dir = screen_to_camera(v1_ss)
            // d alpha / d v0_ss_x = dot(cross(v1_dir, p),
            //     d_screen_to_camera(v0_ss).x)
            auto v0_dir = screen_to_camera(camera, v0_ss);
            auto v1_dir = screen_to_camera(camera, v1_ss);
            auto edge_dir = screen_to_camera(camera, edge_pt);
            d_screen_to_camera(camera, v0_ss, cross(v1_dir, edge_dir), d_v0_ss);
            d_screen_to_camera(camera, v1_ss, cross(edge_dir, v0_dir), d_v1_ss);
            d_screen_to_camera(camera, v1_ss, cross(v0_dir, v1_dir), d_edge_pt);
        }
        d_v0_ss *= edge_contrib;
        d_v1_ss *= edge_contrib;
        d_edge_pt *= edge_contrib;

        // v0_ss, v1_ss = project(camera, v0, v1)
        auto d_v0 = Vector3{0, 0, 0};
        auto d_v1 = Vector3{0, 0, 0};
        d_project(camera, v0, v1,
            d_v0_ss.x, d_v0_ss.y,
            d_v1_ss.x, d_v1_ss.y,
            d_camera, d_v0, d_v1);

        atomic_add(&d_shapes[edge_record.edge.shape_id].vertices[3 * edge_record.edge.v0], d_v0);
        atomic_add(&d_shapes[edge_record.edge.shape_id].vertices[3 * edge_record.edge.v1], d_v1);
        if (screen_gradient_image != nullptr) {
            auto xi = clamp(int(edge_pt[0] * camera.width - camera.viewport_beg.x),
                            0, camera.viewport_end.x - camera.viewport_beg.x);
            auto yi = clamp(int(edge_pt[1] * camera.height - camera.viewport_beg.y),
                            0, camera.viewport_end.y - camera.viewport_beg.y);
            auto pixel_idx = yi * camera.width + xi;
            auto normalization = Real(0);
            for(int _kx = -3; _kx <= 3; _kx++){
                for(int _ky = -3; _ky <= 3; _ky++) {
                    auto kx = xi + _kx;
                    auto ky = yi + _ky;

                    if (kx < 0 || kx >= camera.width)
                        continue;

                    if (ky < 0 || ky >= camera.height)
                        continue;

                    // Get pixel center.
                    Vector2 pixel_center;
                    local_to_screen_pos(camera,
                                (ky * camera.width + kx),
                                Vector2{0, 0},
                                pixel_center);

                    auto local_pos = Vector2{
                        (edge_pt[0] - pixel_center[0]) * camera.width,
                        (edge_pt[1] - pixel_center[1]) * camera.height
                    };

                    Vector2 jacobian;
                    Real value;
                    screen_filter_grad(camera,
                                        idx,
                                        local_pos,
                                        jacobian,
                                        value);

                    normalization += value;
                }
            }

            for(int _kx = -3; _kx <= 3; _kx++) {
                for(int _ky = -3; _ky <= 3; _ky++) {
                    auto kx = xi + _kx;
                    auto ky = yi + _ky;

                    if (kx < 0 || kx >= camera.width)
                        continue;

                    if (ky < 0 || ky >= camera.height)
                        continue;

                    // Get pixel center.
                    Vector2 pixel_center;
                    local_to_screen_pos(camera,
                                (ky * camera.width + kx),
                                Vector2{0, 0},
                                pixel_center);

                    auto local_pos = Vector2{
                        (edge_pt[0] - pixel_center[0]) * camera.width,
                        (edge_pt[1] - pixel_center[1]) * camera.height
                    };

                    Vector2 jacobian;
                    Real value;
                    screen_filter_grad(camera,
                                        idx,
                                        local_pos,
                                        jacobian,
                                        value);
                    if (! TEASER) {
                        //atomic add?
                        if (edge_record.edge.shape_id == SHAPE_SELECT) {
                            screen_gradient_image[2 * (ky * camera.width + kx) + 0] += (d_v0[DIM_SELECT] + d_v1[DIM_SELECT]) * value / normalization;
                            screen_gradient_image[2 * (ky * camera.width + kx) + 1] += (d_v0[DIM_SELECT] + d_v1[DIM_SELECT]) * value / normalization;
                        }
                    }
                    else {
                        if (edge_record.edge.shape_id >= 1 && edge_record.edge.shape_id <= 4) {
                            screen_gradient_image[2 * (ky * camera.width + kx) + 0] += (d_v0[DIM_SELECT_TEASER] + d_v1[DIM_SELECT_TEASER]) * value / normalization;
                            screen_gradient_image[2 * (ky * camera.width + kx) + 1] += (d_v0[DIM_SELECT_TEASER] + d_v1[DIM_SELECT_TEASER]) * value / normalization;
                        }
                    }
                }
            }
        }

        /* TODO: Debug code. */
        if (debug_image != nullptr && edge_record.edge.shape_id == SHAPE_SELECT) {
            auto xi = clamp(int(edge_pt[0] * camera.width - camera.viewport_beg.x),
                            0, camera.viewport_end.x - camera.viewport_beg.x);
            auto yi = clamp(int(edge_pt[1] * camera.height - camera.viewport_beg.y),
                            0, camera.viewport_end.y - camera.viewport_beg.y);
            // Aggregate weighted sum of nearest 
            auto normalization = Real(0);
            for(int _kx = -3; _kx <= 3; _kx++)
                for(int _ky = -3; _ky <= 3; _ky++) {
                    auto kx = xi + _kx;
                    auto ky = yi + _ky;

                    if (kx < 0 || kx >= camera.width)
                        continue;

                    if (ky < 0 || ky >= camera.height)
                        continue;
                    
                    // Get pixel center.
                    Vector2 pixel_center;
                    local_to_screen_pos(camera, 
                                (ky * camera.width + kx),
                                Vector2{0, 0},
                                pixel_center);
                    
                    auto local_pos = Vector2{
                        (edge_pt[0] - pixel_center[0]) * camera.width,
                        (edge_pt[1] - pixel_center[1]) * camera.height
                    };

                    Vector2 jacobian;
                    Real value;
                    screen_filter_grad(camera, 
                                        idx,
                                        local_pos,
                                        jacobian,
                                        value);

                    normalization += value;
                }

            for(int _kx = -3; _kx <= 3; _kx++)
                for(int _ky = -3; _ky <= 3; _ky++) {
                    auto kx = xi + _kx;
                    auto ky = yi + _ky;

                    if (kx < 0 || kx >= camera.width)
                        continue;

                    if (ky < 0 || ky >= camera.height)
                        continue;
                    
                    // Get pixel center.
                    Vector2 pixel_center;
                    local_to_screen_pos(camera, 
                                (ky * camera.width + kx),
                                Vector2{0, 0},
                                pixel_center);
                    
                    auto local_pos = Vector2{
                        (edge_pt[0] - pixel_center[0]) * camera.width,
                        (edge_pt[1] - pixel_center[1]) * camera.height
                    };

                    Vector2 jacobian;
                    Real value;
                    screen_filter_grad(camera, 
                                        idx,
                                        local_pos,
                                        jacobian,
                                        value);
                    
                    debug_image[ky * camera.width + kx] += (d_v0[DIM_SELECT] + d_v1[DIM_SELECT]) * (value / normalization);
                }
        }
        /* End debug code */

    }

    const Camera camera;
    const Shape *shapes;
    const PrimaryEdgeRecord *edge_records;
    const Real *edge_contribs;
    DShape *d_shapes;
    DCamera d_camera;
    float *debug_image;
    float *screen_gradient_image;
    const Matrix4x4 &m_transf;
};

void compute_primary_edge_derivatives(const Scene &scene,
                                      const BufferView<PrimaryEdgeRecord> &edge_records,
                                      const BufferView<Real> &edge_contribs,
                                      BufferView<DShape> d_shapes,
                                      DCamera d_camera,
                                      float *debug_image,
                                      float *screen_gradient_image,
                                      const Matrix4x4 &m_transf) {
    parallel_for(primary_edge_derivatives_computer{
        scene.camera,
        scene.shapes.data,
        edge_records.begin(),
        edge_contribs.begin(),
        d_shapes.begin(),
        d_camera,
        debug_image,
        screen_gradient_image,
        m_transf
    }, edge_records.size(), scene.use_gpu);
}

struct secondary_edge_sampler {

    static constexpr auto num_h_samples = 1;

    DEVICE int sample_edge_h(const EdgeTreeRoots &edge_tree_roots,
                             const SurfacePoint &p,
                             const Matrix3x3 &m,
                             const Matrix3x3 &m_inv,
                             const Matrix3x3 &isotropic_frame,
                             const Ray &nee_ray,
                             Real sample,
                             Real resample_sample,
                             Real &sample_weight,
                             bool use_nee) {
        constexpr auto buffer_size = 3;
        BVHStackItemH buffer[buffer_size];
        auto selected_edge = -1;
        auto edge_weight = Real(0);
        auto wsum = Real(0);

        auto stack_ptr = &buffer[0];

        // randomly sample an edge using edge hierarchy
        // push both nodes into stack
        auto imp_non_manifold = Real(0);
        auto imp_manifold = Real(0);
        if (edge_tree_roots.non_manifold_bvh_roots != nullptr) {
            imp_non_manifold = non_manifold_size;
        }
        if (edge_tree_roots.manifold_bvh_roots != nullptr) {
            imp_manifold = manifold_size;
        }
        if (imp_non_manifold <= 0 && imp_manifold <= 0) {
            return -1;
        }
        auto prob_non_manifold = imp_non_manifold / (imp_non_manifold + imp_manifold);
        const BVHNode *nodes_buffer;
        int n_roots;

        if (sample < prob_non_manifold) {
            sample /= prob_non_manifold;
            nodes_buffer = edge_tree_roots.non_manifold_bvh_roots;
            n_roots = edge_tree_roots.n_non_manifold_bvh_roots;
        }
        else {
            sample = (1 - sample) / (1 - prob_non_manifold);
            nodes_buffer = edge_tree_roots.manifold_bvh_roots;
            n_roots = edge_tree_roots.n_manifold_bvh_roots;
        }

        Real prob_roots[n_roots];
        Real prob_sum = 0;

        if (use_nee) {
            int light_id = 0;
            int shape_light_id = scene.area_lights[light_id].shape_id;
            Real polygon_radiance = sum(scene.area_lights[light_id].intensity) / 3;
            Real envmap_radiance = 0;
            if (scene.envmap != nullptr) {
                envmap_radiance = scene.envmap->mean_intensity;
            }
            for (int i = 0; i < n_roots; i++) {
                prob_roots[i] = node_importance_poly_light(nodes_buffer + i, p, m, m_inv, isotropic_frame,
                                    scene.shapes[shape_light_id].vertices,
                                    scene.area_lights[light_id].polygon_silhouette,
                                    scene.area_lights[light_id].n_polygon_silhouette,
                                    polygon_radiance,
                                    envmap_radiance,
                                    bbox_edges_idxs);
                prob_sum += prob_roots[i];
            }
        }

        else {
            for (int i = 0; i < n_roots; i++) {
                prob_roots[i] = node_importance(nodes_buffer + i, p, m_inv,
                                                isotropic_frame,
                                                bbox_edges_idxs,
                                                bbox_edges_normals);
                prob_sum += prob_roots[i];
            }
        }

        if (prob_sum > 0) {
            for (int i = 0; i < n_roots; i++) {
                prob_roots[i] /= prob_sum;
            }
            int idx = sample_discrete_n(prob_roots, n_roots, sample);
            *stack_ptr++ = BVHStackItemH{
                BVHNodePtr{nodes_buffer + idx}, 1, prob_roots[idx]};
        }
        else {
            return -1;
        }

        while (stack_ptr != &buffer[0]) {
            assert(stack_ptr > &buffer[0] && stack_ptr < &buffer[buffer_size]);
            // pop from stack
            const auto &stack_item = *--stack_ptr;

            if (is_leaf(stack_item.node_ptr)) {

                Real probs[4];
                Real prob_sum = 0;
                for (int i = 0; i < stack_item.node_ptr.ptr->num_children; i++) {
                    const Edge &edge = edges[(stack_item.node_ptr.ptr)->edge_ids[i]];
                    probs[i] = edge_importance(edge, p, m, m_inv, isotropic_frame, scene.shapes);
                    prob_sum += probs[i];
                }

                if (prob_sum > 0) {
                    for (int i = 0; i < stack_item.node_ptr.ptr->num_children; i++){
                        probs[i] /= prob_sum;
                    }
                    for (int i = stack_item.node_ptr.ptr->num_children; i < 4; i++){
                        probs[i] = 0;
                    }
                    int idx = sample_discrete_4(probs, sample);
                    selected_edge = (stack_item.node_ptr.ptr)->edge_ids[idx];
                    edge_weight = 1.0 / (stack_item.pmf * probs[idx]);
                }
            } else {
                BVHNodePtr children[4];
                get_children(stack_item.node_ptr, children);

                Real probs[4];
                for (int i = 0; i < 4; i++) {
                    probs[i] = 0;
                }

                if (use_nee) {
                    int light_id = 0;
                    int shape_light_id = scene.area_lights[light_id].shape_id;
                    Real polygon_radiance = sum(scene.area_lights[light_id].intensity) / 3;
                    Real envmap_radiance = 0;
                    if (scene.envmap != nullptr) {
                        envmap_radiance = scene.envmap->mean_intensity;
                    }
                    for (int i = 0; i < stack_item.node_ptr.ptr->num_children; i++) {
                        probs[i] = node_importance_poly_light(children[i].ptr, p, m, m_inv, isotropic_frame,
                                            scene.shapes[shape_light_id].vertices,
                                            scene.area_lights[light_id].polygon_silhouette,
                                            scene.area_lights[light_id].n_polygon_silhouette,
                                            polygon_radiance,
                                            envmap_radiance,
                                            bbox_edges_idxs);
                    }
                }

                else {
                    for (int i = 0; i < stack_item.node_ptr.ptr->num_children; i++) {
                        probs[i] = node_importance(children[i].ptr, p, m_inv,
                                                   isotropic_frame,
                                                   bbox_edges_idxs,
                                                   bbox_edges_normals);
                    }
                }

                Real imp_total = probs[0] + probs[1] + probs[2] + probs[3];

                if (imp_total > 0) {
                    for (int i = 0; i < stack_item.node_ptr.ptr->num_children; i++) {
                        probs[i] /= imp_total;
                    }
                    int idx = sample_discrete_4(probs, sample);
                    auto current_pmf = stack_item.pmf;
                    *stack_ptr++ = BVHStackItemH{
                                children[idx], 1, current_pmf * probs[idx]};
                }
            }
        }
        if (edge_weight <= 0) {
            return -1;
        }
        sample_weight = edge_weight;
        return selected_edge;
    }

    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &edge_sample = edge_samples[idx];
        const auto &wi = -incoming_rays[pixel_id].dir;
        const auto &shading_isect = shading_isects[pixel_id];
        const auto &shading_point = shading_points[pixel_id];
        const auto &throughput = throughputs[pixel_id];
        const auto &min_rough = min_roughness[pixel_id];
        const auto &nee_isect = nee_isects[pixel_id];
        const auto &nee_point = nee_points[pixel_id];

        auto nee_ray = nee_rays[pixel_id];
        // nee_ray.tmax is used for marking occluded rays, so we need to recompute
        // it here
        // TODO: there is probably a more elegant solution
        if (nee_isect.valid()) {
            nee_ray.tmax = length(nee_point.position - nee_ray.org);
        } else {
            nee_ray.tmax = infinity<Real>();
        }

        // Initialize output
        edge_records[idx] = SecondaryEdgeRecord{};
        new_throughputs[2 * idx + 0] = Vector3{0, 0, 0};
        new_throughputs[2 * idx + 1] = Vector3{0, 0, 0};
        rays[2 * idx + 0] = Ray(Vector3{0, 0, 0}, Vector3{0, 0, 0});
        rays[2 * idx + 1] = Ray(Vector3{0, 0, 0}, Vector3{0, 0, 0});
        edge_min_roughness[2 * idx + 0] = min_rough;
        edge_min_roughness[2 * idx + 1] = min_rough;

        // XXX Hack: don't compute secondary edge derivatives if we already hit a diffuse vertex
        // before shading_point.
        // Such paths are extremely noisy and have very small contribution to the actual derivatives.
        if (min_rough > 1e-2f) {
            return;
        }

        // Setup the Linearly Transformed Cosine Distribution
        const Shape &shape = scene.shapes[shading_isect.shape_id];
        const Material &material = scene.materials[shape.material_id];
        // First decide which component of BRDF to sample
        auto diffuse_reflectance = get_diffuse_reflectance(material, shading_point);
        auto specular_reflectance = get_specular_reflectance(material, shading_point);
        auto diffuse_weight = luminance(diffuse_reflectance);
        auto specular_weight = luminance(specular_reflectance);

        auto weight_sum = diffuse_weight + specular_weight;
        if (weight_sum <= 0.f) {
            // black material
            return;
        }
        auto diffuse_pmf = diffuse_weight / weight_sum;
        auto specular_pmf = specular_weight / weight_sum;
        auto m_pmf = Real(0);
        auto n = shading_point.shading_frame.n;
        if (material.two_sided) {
            if (dot(wi, n) < 0.f) {
                n = -n;
            }
        }
        auto frame_x = normalize(wi - n * dot(wi, n));
        auto frame_y = cross(n, frame_x);
        if (dot(wi, n) > 1 - 1e-6f) {
            coordinate_system(n, frame_x, frame_y);
        }
        auto isotropic_frame = Frame{frame_x, frame_y, n};
        auto m = Matrix3x3{};
        auto m_inv = Matrix3x3{};
        auto roughness = max(get_roughness(material, shading_point), min_rough);
        if (edge_sample.bsdf_component <= diffuse_pmf) {
            // M is shading frame * identity
            m_inv = Matrix3x3(isotropic_frame);
            m = inverse(m_inv);
            m_pmf = diffuse_pmf;
        } else {
            m_inv = inverse(get_ltc_matrix(shading_point, wi, roughness, tabM)) *
                    Matrix3x3(isotropic_frame);
            m = inverse(m_inv);
            m_pmf = specular_pmf;
        }

        auto edge_id = -1;
        auto edge_weight = Real(0);
        auto sample_p = Vector3{};
        auto mwt = Vector3{};

        auto edge_sel = edge_sample.edge_sel;
        auto use_nee_ray = false;
        auto nee_ray_pmf = Real(1);
        auto is_diffuse_or_glossy =
            edge_sample.bsdf_component <= diffuse_pmf || roughness > Real(0.1);

        // sample using a tree traversal
        edge_id = sample_edge_h(edge_tree_roots,
            shading_point, m, m_inv, isotropic_frame, nee_ray,
            edge_sel, edge_sample.resample_sel, edge_weight,
            use_nee);

        if (edge_id == -1 || edge_weight <= 0) {
            return;
        }

        const auto &edge = edges[edge_id];
        if (!is_silhouette(scene.shapes, shading_point.position, edge)) {
            return;
        }

        auto v0 = Vector3{get_v0(scene.shapes, edge)};
        auto v1 = Vector3{get_v1(scene.shapes, edge)};

        // Transform the vertices to local coordinates
        auto v0tmp = Matrix3x3(isotropic_frame) * (v0 - shading_point.position);
        auto v1tmp = Matrix3x3(isotropic_frame) * (v1 - shading_point.position);
        if (v0tmp[2] <= 0.f && v1tmp[2] <= 0.f) {
            // Edge is below the shading point
            return;
        }

        // Clip to the horizon
        if (v0tmp[2] < 0.f) {
            v0tmp = (v0tmp*v1tmp[2] - v1tmp*v0tmp[2]) / (v1tmp[2] - v0tmp[2]);
        }
        if (v1tmp[2] < 0.f) {
            v1tmp = (v0tmp*v1tmp[2] - v1tmp*v0tmp[2]) / (v1tmp[2] - v0tmp[2]);
        }

        // Transform the vertices to the LTC space
        auto v0o = m_inv * transpose(Matrix3x3(isotropic_frame)) * v0tmp;
        auto v1o = m_inv * transpose(Matrix3x3(isotropic_frame)) * v1tmp;
        if (v0o[2] <= 0.f && v1o[2] <= 0.f) {
            // Edge is below the shading point
            return;
        }
        // Clip to the horizon
        if (v0o[2] < 0.f) {
            v0o = (v0o*v1o[2] - v1o*v0o[2]) / (v1o[2] - v0o[2]);
        }
        if (v1o[2] < 0.f) {
            v1o = (v0o*v1o[2] - v1o*v0o[2]) / (v1o[2] - v0o[2]);
        }
        auto vodir = v1o - v0o;
        auto wt = normalize(vodir);
        auto l0 = dot(v0o, wt);
        auto l1 = dot(v1o, wt);
        auto vo = v0o - l0 * wt;
        auto d = length(vo);
        auto I = [&](Real l) {
            return (l/(d*(d*d+l*l))+atan(l/d)/(d*d))*vo[2] +
                (l*l/(d*(d*d+l*l)))*wt[2];
        };
        auto Il0 = I(l0);
        auto Il1 = I(l1);
        auto normalization = Il1 - Il0;
        auto line_cdf = [&](Real l) {
            return (I(l)-Il0)/normalization;
        };
        auto line_pdf = [&](Real l) {
            auto dist_sq=d*d+l*l;
            return 2.f*d*(vo+l*wt)[2]/(normalization*dist_sq*dist_sq);
        };
        // Hybrid bisection & Newton iteration
        // Here we are trying to find a point l s.t. line_cdf(l) = edge_sample.t
        auto lb = l0;
        auto ub = l1;
        if (lb > ub) {
            swap_(lb, ub);
        }
        auto l = 0.5f * (lb + ub);
        for (int it = 0; it < 20; it++) {
            if (!(l >= lb && l <= ub)) {
                l = 0.5f * (lb + ub);
            }
            auto value = line_cdf(l) - edge_sample.t;
            if (fabs(value) < 1e-5f || it == 19) {
                break;
            }
            // The derivative may not be entirely accurate,
            // but the bisection is going to handle this
            if (value > 0.f) {
                ub = l;
            } else {
                lb = l;
            }
            auto derivative = line_pdf(l);
            l -= value / derivative;
        }
        if (line_pdf(l) <= 0.f) {
            // Numerical issue
            return;
        }
        // Convert from l to position
        sample_p = m * (vo + l * wt);
        auto edge_pdf = m_pmf * line_pdf(l);
        assert(edge_pdf > 0);
        edge_weight /= (m_pmf * line_pdf(l));
        mwt = m * wt;

        // shading_point.position, v0 and v1 forms a half-plane
        // that splits the spaces into upper half-space and lower half-space
        auto half_plane_normal =
            normalize(cross(v0 - shading_point.position,
                            v1 - shading_point.position));
        // Generate sample directions
        auto offset = 1e-5f / length(sample_p);
        auto sample_dir = normalize(sample_p);
        // Sample two rays on the two sides of the edge
        auto v_upper_dir = normalize(sample_dir + offset * half_plane_normal);
        auto v_lower_dir = normalize(sample_dir - offset * half_plane_normal);

        auto eval_bsdf = bsdf(material, shading_point, wi, sample_dir, min_rough);
        if (sum(eval_bsdf) < 1e-6f) {
            return;
        }

        // Setup output
        auto nd = channel_info.num_total_dimensions;
        auto rd = channel_info.radiance_dimension;
        auto d_color = Vector3{0, 0, 0};
        if (rd != -1) {
            d_color = Vector3{
                d_rendered_image[nd * pixel_id + rd + 0],
                d_rendered_image[nd * pixel_id + rd + 1],
                d_rendered_image[nd * pixel_id + rd + 2]
            };
        }
        edge_records[idx].edge = edge;
        edge_records[idx].edge_pt = sample_p; // for Jacobian computation 
        edge_records[idx].mwt = mwt; // for Jacobian computation
        edge_records[idx].use_nee_ray = use_nee_ray;
        edge_records[idx].is_diffuse_or_glossy = is_diffuse_or_glossy;
        // Set the two rays. The first one starts at the edge. The second ray points to the edge.
        rays[2 * idx + 0] = Ray(shading_point.position + sample_p, sample_dir);
        rays[2 * idx + 1] = Ray(shading_point.position, sample_dir);
        const auto &incoming_ray_differential = incoming_ray_differentials[pixel_id];
        // Propagate ray differentials
        auto bsdf_ray_differential = RayDifferential{};
        bsdf_ray_differential.org_dx = incoming_ray_differential.org_dx;
        bsdf_ray_differential.org_dy = incoming_ray_differential.org_dy;
        if (edge_sample.bsdf_component <= diffuse_pmf) {
            // HACK: Output direction has no dependencies w.r.t. input
            // However, since the diffuse BRDF serves as a low pass filter,
            // we want to assign a larger prefilter.
            bsdf_ray_differential.dir_dx = Vector3{0.03f, 0.03f, 0.03f};
            bsdf_ray_differential.dir_dy = Vector3{0.03f, 0.03f, 0.03f};
        } else {
            // HACK: we compute the half vector as the micronormal,
            // and use dndx/dndy to approximate the micronormal screen derivatives
            auto m = normalize(wi + sample_dir);
            auto m_local2 = dot(m, shading_point.shading_frame.n);
            auto dmdx = shading_point.dn_dx * m_local2;
            auto dmdy = shading_point.dn_dy * m_local2;
            auto dir_dx = incoming_ray_differential.dir_dx;
            auto dir_dy = incoming_ray_differential.dir_dy;
            // Igehy 1999, Equation 15
            auto ddotn_dx = dir_dx * m - wi * dmdx;
            auto ddotn_dy = dir_dy * m - wi * dmdy;
            // Igehy 1999, Equation 14
            bsdf_ray_differential.dir_dx =
                dir_dx - 2 * (-dot(wi, m) * shading_point.dn_dx + ddotn_dx * m);
            bsdf_ray_differential.dir_dy =
                dir_dy - 2 * (-dot(wi, m) * shading_point.dn_dy + ddotn_dy * m);
        }
        bsdf_differentials[2 * idx + 0] = bsdf_ray_differential;
        bsdf_differentials[2 * idx + 1] = bsdf_ray_differential;
        // edge_weight doesn't take the Jacobian between the shading point
        // and the ray intersection into account. We'll compute this later
        assert(nee_ray_pmf > 0);
        auto nt = throughput * eval_bsdf * d_color * edge_weight / nee_ray_pmf;
        // assert(isfinite(throughput));
        // assert(isfinite(eval_bsdf));
        // assert(isfinite(d_color));
        assert(isfinite(edge_weight));
        // Make sure that signs of the contributions are correct
        auto test_vertex = get_non_shared_v0(scene.shapes, edge);
        if (dot(test_vertex - shading_point.position, half_plane_normal) > 0) {
            nt = -nt;
        }
        new_throughputs[2 * idx + 0] = nt;
        new_throughputs[2 * idx + 1] = -nt;
        // The first intersection is unknown
        edge_shading_isects[2 * idx + 0] = -1;
        // The second intersection is on the edge
        int face_idx = dot(get_normal(scene.shapes[edge.shape_id], edge.f0), sample_dir) < 0 ? edge.f0 : edge.f1;
        edge_shading_isects[2 * idx + 1] = Intersection(edge.shape_id, face_idx);
        edge_shading_points[2 * idx + 1].position = shading_point.position + sample_p;
    }

    const FlattenScene scene;
    const Edge *edges;
    int num_edges;
    const Vector3 cam_org;
    const Real *edges_pmf;
    const Real *edges_cdf;
    const EdgeTreeRoots edge_tree_roots;
    const Real edge_bounds_expand;
    const int non_manifold_size;
    const int manifold_size;
    const int *active_pixels;
    const SecondaryEdgeSample *edge_samples;
    const Ray *incoming_rays;
    const RayDifferential *incoming_ray_differentials;
    const Intersection *shading_isects;
    const SurfacePoint *shading_points;
    const Ray *nee_rays;
    const Intersection *nee_isects;
    const SurfacePoint *nee_points;
    const Vector3 *throughputs;
    const Real *min_roughness;
    const float *d_rendered_image;
    const ChannelInfo channel_info;
    const bool use_nee;
    const float *tabM;
    const int *bbox_edges_idxs;
    const Vector3 *bbox_edges_normals;
    SecondaryEdgeRecord *edge_records;
    Ray *rays;
    RayDifferential *bsdf_differentials;
    Vector3 *new_throughputs;
    Real *edge_min_roughness;
    Intersection *edge_shading_isects;
    SurfacePoint *edge_shading_points;
};

void sample_secondary_edges(const Scene &scene,
                            const BufferView<int> &active_pixels,
                            const BufferView<SecondaryEdgeSample> &samples,
                            const BufferView<Ray> &incoming_rays,
                            const BufferView<RayDifferential> &incoming_ray_differentials,
                            const BufferView<Intersection> &shading_isects,
                            const BufferView<SurfacePoint> &shading_points,
                            const BufferView<Ray> &nee_rays,
                            const BufferView<Intersection> &nee_isects,
                            const BufferView<SurfacePoint> &nee_points,
                            const BufferView<Vector3> &throughputs,
                            const BufferView<Real> &min_roughness,
                            const float *d_rendered_image,
                            const ChannelInfo &channel_info,
                            const BufferView<int> &bbox_edges_idxs,
                            const BufferView<Vector3> &bbox_edges_normals,
                            BufferView<SecondaryEdgeRecord> edge_records,
                            BufferView<Ray> rays,
                            BufferView<RayDifferential> &bsdf_differentials,
                            BufferView<Vector3> new_throughputs,
                            BufferView<Real> edge_min_roughness,
                            BufferView<Intersection> &edge_shading_isects,
                            BufferView<SurfacePoint> &edge_shading_points) {
    auto cam_org = xfm_point(scene.camera.cam_to_world, Vector3{0, 0, 0});
    auto edge_tree = scene.edge_sampler.edge_tree.get();
    parallel_for(secondary_edge_sampler{
        get_flatten_scene(scene),
        scene.edge_sampler.edges.begin(),
        (int)scene.edge_sampler.edges.size(),
        cam_org,
        scene.edge_sampler.secondary_edges_pmf.begin(),
        scene.edge_sampler.secondary_edges_cdf.begin(),
        get_edge_tree_roots(edge_tree),
        edge_tree != nullptr ? edge_tree->edge_bounds_expand : Real(0),
        edge_tree->non_manifold_size,
        edge_tree->manifold_size,
        active_pixels.begin(),
        samples.begin(),
        incoming_rays.begin(),
        incoming_ray_differentials.begin(),
        shading_isects.begin(),
        shading_points.begin(),
        nee_rays.begin(),
        nee_isects.begin(),
        nee_points.begin(),
        throughputs.begin(),
        min_roughness.begin(),
        d_rendered_image,
        channel_info,
        scene.use_nee,
        ltc::tabM,
        bbox_edges_idxs.begin(),
        bbox_edges_normals.begin(),
        edge_records.begin(),
        rays.begin(),
        bsdf_differentials.begin(),
        new_throughputs.begin(),
        edge_min_roughness.begin(),
        edge_shading_isects.begin(),
        edge_shading_points.begin(),},
        active_pixels.size(), scene.use_gpu);
}

// The derivative of the intersection point w.r.t. a line parameter t
DEVICE
inline Vector3 intersect_jacobian(const Vector3 &org,
                                  const Vector3 &dir,
                                  const Vector3 &p,
                                  const Vector3 &n,
                                  const Vector3 &l) {
    // Jacobian of ray-plane intersection:
    // https://www.cs.princeton.edu/courses/archive/fall00/cs426/lectures/raycast/sld017.htm
    // d = -(p dot n)
    // t = -(org dot n + d) / (dir dot n)
    // p = org + t * dir
    // d p[i] / d dir[i] = t
    // d p[i] / d t = dir[i]
    // d t / d dir_dot_n = (org dot n - p dot n) / dir_dot_n^2
    // d dir_dot_n / d dir[j] = n[j]
    auto dir_dot_n = dot(dir, n);
    if (fabs(dir_dot_n) < 1e-10f) {
        return Vector3{0.f, 0.f, 0.f};
    }
    auto d = -dot(p, n);
    auto t = -(dot(org, n) + d) / dir_dot_n;
    if (t <= 0) {
        return Vector3{0.f, 0.f, 0.f};
    }
    return t * (l - dir * (dot(l, n) / dot(dir, n)));
}


struct secondary_edge_weights_updater {
    DEVICE void update_throughput(const Intersection &edge_isect,
                                  const SurfacePoint &edge_surface_point,
                                  const SurfacePoint &shading_point,
                                  const SecondaryEdgeRecord &edge_record,
                                  Vector3 &edge_throughput) {
        if (edge_isect.valid()) {
            // Hit a surface
            // Geometry term
            auto dir = edge_surface_point.position - shading_point.position;
            auto dist_sq = length_squared(dir);
            if (dist_sq < 1e-8f) {
                // Likely a self-intersection
                edge_throughput = Vector3{0, 0, 0};
                return;
            }

            auto v0 = Vector3{get_v0(scene.shapes, edge_record.edge)};
            auto v1 = Vector3{get_v1(scene.shapes, edge_record.edge)};
            auto p = shading_point.position;
            auto m = edge_surface_point.position;
            auto nm = edge_surface_point.geom_normal;
            auto wt = edge_record.edge_pt;
            auto norm = length(m - p);
            auto dirn = normalize(m - p);
            auto w_num = length(edge_record.mwt) * dot(nm, wt);
            auto w_den = length(v0 - v1) * length(wt) * length(wt) * length(wt) * dot(m - p, nm);
            auto w = w_num / w_den;
            edge_throughput *= w;
            assert(isfinite(w));
        } else if (scene.envmap != nullptr && edge_isect.infinity()) {
            // Hit an environment light
            auto v0 = Vector3{get_v0(scene.shapes, edge_record.edge)};
            auto v1 = Vector3{get_v1(scene.shapes, edge_record.edge)};
            Real w_num = length(edge_record.mwt) * dot(edge_record.edge_pt, shading_point.geom_normal);
            Real norm = length(edge_record.edge_pt);
            Real w_den = length(v0 - v1) * norm * norm * norm;
            Real w = w_num / w_den;
            edge_throughput *= w;
        } else if (scene.vmflight != nullptr && edge_isect.infinity()) {
            // Hit the vMF light.
            auto p = shading_point.position;
            auto v0 = Vector3{get_v0(scene.shapes, edge_record.edge)};
            auto v1 = Vector3{get_v1(scene.shapes, edge_record.edge)};
            auto d0 = v0 - p;
            auto d1 = v1 - p;
            auto dirac_jacobian = length(cross(d0, d1)); // Eq. 16 in the paper

            auto line_jacobian = 1 / length_squared(edge_record.edge_pt - p);
            auto w = line_jacobian / dirac_jacobian;

            edge_throughput *= w;
        }
    }

    DEVICE void operator()(int idx) {
        const auto &edge_record = edge_records[idx];
        if (edge_record.edge.shape_id < 0) {
            return;
        }

        auto pixel_id = active_pixels[idx];
        const auto &shading_point = shading_points[pixel_id];
        const auto &edge_isect0 = edge_isects[2 * idx + 0];
        const auto &edge_surface_point0 = edge_surface_points[2 * idx + 0];
        const auto &edge_isect1 = edge_isects[2 * idx + 1];
        const auto &edge_surface_point1 = edge_surface_points[2 * idx + 1];

        update_throughput(edge_isect0,
                          edge_surface_point0,
                          shading_point,
                          edge_record,
                          edge_throughputs[2 * idx + 0]);
        update_throughput(edge_isect1,
                          edge_surface_point1,
                          shading_point,
                          edge_record,
                          edge_throughputs[2 * idx + 1]);
    }

    const FlattenScene scene;
    const int *active_pixels;
    const SurfacePoint *shading_points;
    const Intersection *edge_isects;
    const SurfacePoint *edge_surface_points;
    const SecondaryEdgeRecord *edge_records;
    Vector3 *edge_throughputs;
};

void update_secondary_edge_weights(const Scene &scene,
                                   const BufferView<int> &active_pixels,
                                   const BufferView<SurfacePoint> &shading_points,
                                   const BufferView<Intersection> &edge_isects,
                                   const BufferView<SurfacePoint> &edge_surface_points,
                                   const BufferView<SecondaryEdgeRecord> &edge_records,
                                   BufferView<Vector3> edge_throughputs) {
    parallel_for(secondary_edge_weights_updater{
        get_flatten_scene(scene),
        active_pixels.begin(),
        shading_points.begin(),
        edge_isects.begin(),
        edge_surface_points.begin(),
        edge_records.begin(),
        edge_throughputs.begin()},
        active_pixels.size(), scene.use_gpu);
}

struct secondary_edge_derivatives_accumulator {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &shading_point = shading_points[pixel_id];
        const auto &edge_record = edge_records[idx];
        if (edge_record.edge.shape_id < 0) {
            return;
        }
        const auto &shading_isect = shading_isects[pixel_id];
        const int edge_isect_type0 = edge_intersection_type[2 * idx + 0];
        const int edge_isect_type1 = edge_intersection_type[2 * idx + 1];
        auto edge_contrib0 = edge_contribs[2 * idx + 0];
        auto edge_contrib1 = edge_contribs[2 * idx + 1];
        const auto &edge_surface_point0 = edge_surface_points[2 * idx + 0];
        const auto &edge_surface_point1 = edge_surface_points[2 * idx + 1];

        auto dcolor_dp = Vector3{0, 0, 0};
        auto dcolor_dv0 = Vector3{0, 0, 0};
        auto dcolor_dv1 = Vector3{0, 0, 0};
        auto v0 = Vector3{get_v0(shapes, edge_record.edge)};
        auto v1 = Vector3{get_v1(shapes, edge_record.edge)};
        auto grad = [&](const Vector3 &p, const Vector3 &x, Real edge_contrib) {
            if (edge_contrib == 0) {
                return;
            }
            auto d0 = v0 - p;
            auto d1 = v1 - p;
            // Eq. 16 in the paper (see the errata)
            auto dp = cross(d1, d0) + cross(x, d1) + cross(d0, x);
            auto dv0 = cross(d1, x);
            auto dv1 = cross(x, d0);
            dcolor_dp += dp * edge_contrib;
            dcolor_dv0 += dv0 * edge_contrib;
            dcolor_dv1 += dv1 * edge_contrib;
        };

        auto norm_dir_0 = normalize(edge_surface_point0 - shading_point.position);
        auto norm_dir_1 = normalize(edge_surface_point1 - shading_point.position);

        bool valid_geom0 = (edge_isect_type0 == 0) && (dot(shading_point.geom_normal, norm_dir_0) > 0);
        bool valid_geom1 = (edge_isect_type1 == 0) && (dot(shading_point.geom_normal, norm_dir_1) > 0);

        bool valid_envmap0 = (scene.envmap != nullptr && edge_isect_type0 == 1) && (dot(shading_point.geom_normal, normalize(edge_record.edge_pt)) > 0);
        bool valid_envmap1 = (scene.envmap != nullptr && edge_isect_type1 == 1) && (dot(shading_point.geom_normal, normalize(edge_record.edge_pt)) > 0);

        bool valid0 = valid_geom0 || valid_envmap0;
        bool valid1 = valid_geom1 || valid_envmap1;

        if (!valid0 || !valid1) {
            return;
        }

        if (edge_isect_type0 == 0) {
            grad(shading_point.position, edge_surface_point0 - shading_point.position, edge_contrib0);
        }
        else if (scene.envmap != nullptr && edge_isect_type0 == 1) {
            auto dir = edge_record.edge_pt / dot(edge_record.edge_pt, shading_point.geom_normal);
            grad(shading_point.position, dir, edge_contrib0);
        }
        else if (scene.vmflight != nullptr && edge_isect_type0 == 1) {
            assert (false);
        }

        if (edge_isect_type1 == 0) {
            grad(shading_point.position, edge_surface_point1 - shading_point.position, edge_contrib1);
        }
        else if (scene.envmap != nullptr && edge_isect_type1 == 1) {
            auto dir = edge_record.edge_pt / dot(edge_record.edge_pt, shading_point.geom_normal);
            grad(shading_point.position, dir, edge_contrib1);
        }
        else if (scene.vmflight != nullptr && edge_isect_type1 == 1) {
            assert (false);
        }

        assert(isfinite(edge_contrib0));
        assert(isfinite(edge_contrib1));
        assert(isfinite(dcolor_dp));

        d_points[pixel_id].position += dcolor_dp;
        if(edge_record.edge.shape_id == SHAPE_SELECT && debug_image != nullptr) {
            debug_image[pixel_id] += (dcolor_dv0[DIM_SELECT] + dcolor_dv1[DIM_SELECT]);
        }
        atomic_add(&(d_shapes[edge_record.edge.shape_id].vertices[3 * edge_record.edge.v0]), dcolor_dv0);
        atomic_add(&(d_shapes[edge_record.edge.shape_id].vertices[3 * edge_record.edge.v1]), dcolor_dv1);
        if (! TEASER) {
            if(edge_record.edge.shape_id == SHAPE_SELECT && screen_gradient_image != nullptr) {
                screen_gradient_image[pixel_id * 2 + 0] += (dcolor_dv0[DIM_SELECT] + dcolor_dv1[DIM_SELECT]);
                screen_gradient_image[pixel_id * 2 + 1] += (dcolor_dv0[DIM_SELECT] + dcolor_dv1[DIM_SELECT]);
            }
        }
        else {
            if(edge_record.edge.shape_id >= 1 && edge_record.edge.shape_id <= 4 && screen_gradient_image != nullptr) {
                screen_gradient_image[pixel_id * 2 + 0] += (dcolor_dv0[DIM_SELECT_TEASER] + dcolor_dv1[DIM_SELECT_TEASER]);
                screen_gradient_image[pixel_id * 2 + 1] += (dcolor_dv0[DIM_SELECT_TEASER] + dcolor_dv1[DIM_SELECT_TEASER]);
            }
        }
    }
    const FlattenScene scene;
    const Shape *shapes;
    const int *active_pixels;
    const SurfacePoint *shading_points;
    const Intersection *shading_isects;
    const SecondaryEdgeRecord *edge_records;
    const Vector3 *edge_surface_points;
    const int *edge_intersection_type;
    const Real *edge_contribs;
    SurfacePoint *d_points;
    DShape *d_shapes;
    float* debug_image;
    float* screen_gradient_image;
    const Matrix4x4 &m_transf;
};

void accumulate_secondary_edge_derivatives(const Scene &scene,
                                           const BufferView<int> &active_pixels,
                                           const BufferView<SurfacePoint> &shading_points,
                                           const BufferView<Intersection> &shading_isects,
                                           const BufferView<SecondaryEdgeRecord> &edge_records,
                                           const BufferView<Vector3> &edge_surface_points,
                                           const BufferView<int> &edge_intersection_type,
                                           const BufferView<Real> &edge_contribs,
                                           BufferView<SurfacePoint> d_points,
                                           BufferView<DShape> d_shapes,
                                           float* debug_image,
                                           float* screen_gradient_image,
                                           const Matrix4x4 &m_transf) {
    parallel_for(secondary_edge_derivatives_accumulator{
        get_flatten_scene(scene),
        scene.shapes.data,
        active_pixels.begin(),
        shading_points.begin(),
        shading_isects.begin(),
        edge_records.begin(),
        edge_surface_points.begin(),
        edge_intersection_type.begin(),
        edge_contribs.begin(),
        d_points.begin(),
        d_shapes.begin(),
        debug_image,
        screen_gradient_image,
        m_transf
    }, active_pixels.size(), scene.use_gpu);
}
