#include "redner.h"
#include "solid_angles.h"


DEVICE 
Real test_polygon_solid_angle(int num_vert, ptr<Vector3> vertices) {

    Buffer<Vector3> vertices_(false, num_vert);
    for(int i = 0; i < num_vert; i++){
        vertices_[i] = vertices.get_index(i);
    }
    Vector3 p(0.0, 0.0, 0.0);
    Vector3 n(0.0, 0.0, 1.0);
    Matrix3x3 frame(1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0);
    return polygon_solid_angle(num_vert, vertices_.view(0, num_vert), frame, n, p);
}

DEVICE
Real test_bbox_average_bsdf(const AABB3 &bounds,
                            const SurfacePoint &p,
                            const Matrix3x3 &m,
                            const Matrix3x3 &m_inv,
                            const Matrix3x3 &frame_inv
                            ) {
    int silhouette_edges[12];
    Buffer<int> bbox_edges_idxs(false, 24);
    Buffer<Vector3> bbox_edges_normals(false, 24);
    init_bbox_edges_and_normals(bbox_edges_idxs.view(0, 24), bbox_edges_normals.view(0, 24));
    return bbox_average_bsdf(bounds, p, m_inv, frame_inv, 
                             bbox_edges_idxs.begin(), 
                             bbox_edges_normals.begin());
}