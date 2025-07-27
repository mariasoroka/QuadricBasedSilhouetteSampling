#pragma once

#include "redner.h"
#include "edge_tree.h"
#include "ptr.h"
#include "edge.h"
#include "ltc.h"
#include "intersection.h"


int get_n_nodes(const EdgeSampler &edge_sampler);
int get_n_edges(const EdgeSampler &edge_sampler);

bool intersect_one_ray(const Scene &scene, 
                       const Vector3 &ray_origin, 
                       const Vector3 &ray_dir, 
                       SurfacePoint &sp,
                       Matrix3x3 &m, 
                       Matrix3x3 &m_inv, 
                       Matrix3x3 &isotropic_frame);

// For a grid of (nx, ny, nz) points, run the rejection test
void compute_rejection_pattern(ptr<double> xs, 
                               ptr<double> ys, 
                               ptr<double> zs,
                               ptr<double> res, 
                               int nx, 
                               int ny, 
                               int nz,
                               int n_node, 
                               const EdgeSampler &edge_sampler);

// For each point on a grid of (nx, ny, nz) 3D points, 
// check all the edges in the edge_sampler and determine whether any of them is a silhouette
void compute_gt_rejection_pattern(ptr<double> xs, 
                                 ptr<double> ys, 
                                 ptr<double> zs,
                                 ptr<double> res, 
                                 int nx, 
                                 int ny, 
                                 int nz,
                                 int n_node, 
                                 const EdgeSampler &edge_sampler, 
                                 const Scene &scene);

// For a surface point sp, traverse the edge sampling hierarchy and compute the
// probabilities of sampling each edge.
Real compute_edge_probs(const SurfacePoint &sp, 
                        const Matrix3x3 &m,
                        const Matrix3x3 &m_inv, 
                        const Matrix3x3 &isotropic_frame,
                        ptr<double> probs, 
                        ptr<Vector3> edge_starts, 
                        ptr<Vector3> edge_ends,
                        const EdgeSampler &edge_sampler, 
                        const Scene &scene);

// For a surface point sp, compute ltc integral for each edge in the edge sampler.
Real compute_gt_edge_probs(const SurfacePoint &sp, 
                           const Matrix3x3 &m,
                           const Matrix3x3 &m_inv, 
                           const Matrix3x3 &isotropic_frame,
                           ptr<double> probs,
                           ptr<Vector3> edge_starts, 
                           ptr<Vector3> edge_ends,
                           const EdgeSampler &edge_sampler, 
                           const Scene &scene);