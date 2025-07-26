#pragma once

#include "redner.h"
#include "vector.h"
#include "buffer.h"
#include "ptr.h"


// For a bbox silhouette, clip it against the horizon.
DEVICE
inline void clip_silhouette_horizon(BufferView<Vector3> bbox_silhouette, 
                                    int &n_bbox_silhouette, 
                                    const Vector3 n, 
                                    const Vector3 p) {

    Vector3 vertices_tmp[7];
    int n_vert_tmp = 0;
    Vector4 ineq(n.x, n.y, n.z, - dot(n, p));

    for (int j = 0; j < n_bbox_silhouette; j++){
        Vector4 vert_j(bbox_silhouette[j].x, 
                       bbox_silhouette[j].y, 
                       bbox_silhouette[j].z, 
                       1.0);
        Vector4 vert_j1(bbox_silhouette[(j - 1 + n_bbox_silhouette) % n_bbox_silhouette].x, 
                        bbox_silhouette[(j - 1 + n_bbox_silhouette) % n_bbox_silhouette].y, 
                        bbox_silhouette[(j - 1 + n_bbox_silhouette) % n_bbox_silhouette].z, 
                        1.0);
        if (dot(vert_j, ineq) < 0){
            if (dot(vert_j1, ineq) > 0){
                Real lam = - dot(vert_j, ineq) / dot(vert_j1 - vert_j, ineq);
                Vector4 vert_new = vert_j + lam * (vert_j1 - vert_j);
                vertices_tmp[n_vert_tmp] = Vector3(vert_new.x, vert_new.y, vert_new.z);
                n_vert_tmp++;
            }
        }
                
        else if (dot(vert_j, ineq) > 0){
            if (dot(vert_j1, ineq) < 0) {
                Real lam = - dot(vert_j, ineq) / dot(vert_j1 - vert_j, ineq);
                Vector4 vert_new = vert_j + lam * (vert_j1 - vert_j);
                vertices_tmp[n_vert_tmp] = Vector3(vert_new.x, vert_new.y, vert_new.z);
                n_vert_tmp++;
            }

            vertices_tmp[n_vert_tmp] = Vector3(vert_j.x, vert_j.y, vert_j.z);
            n_vert_tmp++;
        }
    }
    n_bbox_silhouette = n_vert_tmp;
    for (int j = 0; j < n_bbox_silhouette; j++){
        bbox_silhouette[j] = vertices_tmp[j];
    }
}

// For a bbox silhouette, clip it against the silhouette of a light source.
DEVICE
inline void clip_silhouette_polygon(BufferView<Vector3> bbox_silhouette, 
                                    int &n_bbox_silhouette, 
                                    const float *light_vertices, 
                                    const int *light_silhouette, 
                                    const int n_light_silhouette,
                                    const Vector3 &p) {
    Vector3 vertices_tmp[7 + n_light_silhouette];
    int idx1 = light_silhouette[0];
    Vector3 v1(light_vertices[3 * idx1 + 0], light_vertices[3 * idx1 + 1], light_vertices[3 * idx1 + 2]);
    int idx2 = light_silhouette[1];
    Vector3 v2(light_vertices[3 * idx2 + 0], light_vertices[3 * idx2 + 1], light_vertices[3 * idx2 + 2]);
    int idx3 = light_silhouette[2] == light_silhouette[1] ? light_silhouette[3] : light_silhouette[2];
    Vector3 v3(light_vertices[3 * idx3 + 0], light_vertices[3 * idx3 + 1], light_vertices[3 * idx3 + 2]);

    Real sign = dot(cross(v1 - p, v2 - p), v3 - p) > 0 ? 1 : -1;

    for (int i = 0; i < n_light_silhouette / 2; i++){
        int n_vertices_tmp = 0;
        int idx1 = light_silhouette[2 * i + 0];
        Vector3 v1(light_vertices[3 * idx1 + 0], light_vertices[3 * idx1 + 1], light_vertices[3 * idx1 + 2]);
        int idx2 = light_silhouette[2 * i + 1];
        Vector3 v2(light_vertices[3 * idx2 + 0], light_vertices[3 * idx2 + 1], light_vertices[3 * idx2 + 2]);
        Vector3 normal = normalize(cross(v1 - p, 
                                         v2 - p));

        Vector4 ineq = sign * Vector4(normal.x, normal.y, normal.z, - dot(normal, p));

        for (int j = 0; j < n_bbox_silhouette; j++){
            Vector4 vert_j(bbox_silhouette[j].x, bbox_silhouette[j].y, bbox_silhouette[j].z, 1.0);
            Vector4 vert_j1(bbox_silhouette[(j - 1 + n_bbox_silhouette) % n_bbox_silhouette].x, 
                            bbox_silhouette[(j - 1 + n_bbox_silhouette) % n_bbox_silhouette].y, 
                            bbox_silhouette[(j - 1 + n_bbox_silhouette) % n_bbox_silhouette].z, 
                            1.0);
            if (dot(vert_j, ineq) < 0){
                if (dot(vert_j1, ineq) > 0){
                    Real lam = - dot(vert_j, ineq) / dot(vert_j1 - vert_j, ineq);
                    Vector4 vert_new = vert_j + lam * (vert_j1 - vert_j);
                    vertices_tmp[n_vertices_tmp] = Vector3(vert_new.x, vert_new.y, vert_new.z);
                    n_vertices_tmp++;
                }
            }
                    
            else if (dot(vert_j, ineq) > 0){
                if (dot(vert_j1, ineq) < 0) {
                    Real lam = - dot(vert_j, ineq) / dot(vert_j1 - vert_j, ineq);
                    Vector4 vert_new = vert_j + lam * (vert_j1 - vert_j);
                    vertices_tmp[n_vertices_tmp] = Vector3(vert_new.x, vert_new.y, vert_new.z);
                    n_vertices_tmp++;
                }

                vertices_tmp[n_vertices_tmp] = Vector3(vert_j.x, vert_j.y, vert_j.z);
                n_vertices_tmp++;
            }
        }
        n_bbox_silhouette = n_vertices_tmp;
        for (int j = 0; j < n_vertices_tmp; j++){
            bbox_silhouette[j] = vertices_tmp[j];
        }
    }
}