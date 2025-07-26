import pyredner
import redner
import torch
import scipy
import numpy as np

def intersect_with_plane(p, v0, v1):
    if np.abs(np.dot(p, v0 - v1)) / (np.linalg.norm(p) * np.linalg.norm(v0 - v1)) < 1e-10:
        if np.abs(np.dot(p, v0)) / (np.linalg.norm(p) * np.linalg.norm(v0)) < 1e-10:
            return v0, v1
        else:
            return None, None
    else:
        lam = np.dot(p, v0) / np.dot(p, v0 - v1)
        if lam < 0 or lam > 1:
            return None, None
        else:
            return v0 + lam * (v1 - v0), None

def test_intersect_plane_1():
    np.random.seed(0)
    for k in range(10):
        np_p1 = np.random.randn(3)
        np_p2 = np.random.randn(3)
        np_p = np.random.randn(4)

        p1 = redner.Vector3(np_p1[0], np_p1[1], np_p1[2])
        p2 = redner.Vector3(np_p2[0], np_p2[1], np_p2[2])
        aabb = redner.AABB3(p1, p2)
        p = redner.Vector4(np_p[0], np_p[1], np_p[2], np_p[3])

        points = redner.Vector3_ptr(0)
        points.allocate(12)
        num_points = redner.int_ptr(0)
        num_points.allocate(1)

        redner.test_aabb(aabb, p, points, num_points)
            
        vertices_ = []
        bbox_vertices = np.zeros((8, 4))
        for i in range(8):
            bbox_vertices[i][0] = redner.corner(aabb, i).x
            bbox_vertices[i][1] = redner.corner(aabb, i).y
            bbox_vertices[i][2] = redner.corner(aabb, i).z
            bbox_vertices[i][3] = 1
        bbox_edges = np.array([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]])

        for i in range(12):
            v1, v2 = intersect_with_plane(np_p, bbox_vertices[bbox_edges[i, 0]], bbox_vertices[bbox_edges[i, 1]])
            if v1 is not None:
                vertices_.append(v1)
            if v2 is not None:
                vertices_.append(v2)
        assert num_points.get_index(0) == len(vertices_)

        if len(vertices_) != 0:

            vertices = np.array(vertices_)
            avg = np.mean(vertices, axis=0) 
            dirs = (vertices - avg)[:, 0:3]
            dirs /= np.linalg.norm(dirs, axis=1)[:, np.newaxis]
            matrix = np.zeros((3, 4))
            matrix[:, 0] = np_p[0:3] / np.linalg.norm(np_p[0:3])
            matrix[:, 1:4] = np.eye(3)
            Q, R = np.linalg.qr(matrix)
            u = Q[:, 1]
            v = Q[:, 2] 

            sin = np.dot(dirs, v)
            cos = np.dot(dirs, u)
            angles = np.arctan2(sin, cos)
            idxes = np.argsort(angles)
            vertices= vertices[idxes]

            vs = np.zeros((num_points.get_index(0), 4))
            for i in range(num_points.get_index(0)):
                vs[i, 0] = points.get_index(i).x
                vs[i, 1] = points.get_index(i).y
                vs[i, 2] = points.get_index(i).z
                vs[i, 3] = 1

            idx = np.argmin(np.linalg.norm(vs - vertices[0], axis=1))
            for i in range(num_points.get_index(0)):
                a = (np.linalg.norm(vs[(idx + i) % num_points.get_index(0)] - vertices[i]) < 1e-5)
                b = (np.linalg.norm(vs[(idx - i) % num_points.get_index(0)] - vertices[i]) < 1e-5)
                assert a or b

        points.destroy_array()
        num_points.destroy_array()

def test_intersect_plane_2():

    np_p1 = np.array([-0.0104849, -14.0698, -11.164])
    np_p2 = np.array([6.07932, 34.097, 1.85701])
    np_p_3d = np.array([-0.695839, 0.251019, -0.702374, 1])
    m = np.array([[0.999945, 2.18917e-19, -2.07412e-18, 0.0104843], 
                  [-0.00144066, 0.990514, -1.64401e-17, 0.137404], 
                  [0.00556754, 0.0736695, 0.84414, -0.531008], 
                  [-0.00876625, -0.115995, 0.536122, 0.836087]])
    np_p = m.T @ np_p_3d

    p1 = redner.Vector3(np_p1[0], np_p1[1], np_p1[2])
    p2 = redner.Vector3(np_p2[0], np_p2[1], np_p2[2])
    aabb = redner.AABB3(p1, p2)
    p = redner.Vector4(np_p[0], np_p[1], np_p[2], np_p[3])

    points = redner.Vector3_ptr(0)
    points.allocate(12)
    num_points = redner.int_ptr(0)
    num_points.allocate(1)

    redner.test_aabb(aabb, p, points, num_points)

    vertices_ = []
    bbox_vertices = np.zeros((8, 4))
    for i in range(8):
        bbox_vertices[i][0] = redner.corner(aabb, i).x
        bbox_vertices[i][1] = redner.corner(aabb, i).y
        bbox_vertices[i][2] = redner.corner(aabb, i).z
        bbox_vertices[i][3] = 1
    bbox_edges = np.array([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]])

    for i in range(12):
        v1, v2 = intersect_with_plane(np_p, bbox_vertices[bbox_edges[i, 0]], bbox_vertices[bbox_edges[i, 1]])
        if v1 is not None:
            vertices_.append(v1)
        if v2 is not None:
            vertices_.append(v2)
    assert num_points.get_index(0) == len(vertices_)


    if len(vertices_) != 0:

        vertices = np.array(vertices_)
        avg = np.mean(vertices, axis=0) 
        dirs = (vertices - avg)[:, 0:3]
        dirs /= np.linalg.norm(dirs, axis=1)[:, np.newaxis]
        matrix = np.zeros((3, 4))
        matrix[:, 0] = np_p[0:3] / np.linalg.norm(np_p[0:3])
        matrix[:, 1:4] = np.eye(3)
        Q, R = np.linalg.qr(matrix)
        u = Q[:, 1]
        v = Q[:, 2] 

        sin = np.dot(dirs, v)
        cos = np.dot(dirs, u)
        angles = np.arctan2(sin, cos)
        idxes = np.argsort(angles)
        vertices= vertices[idxes]

        vs = np.zeros((num_points.get_index(0), 4))
        for i in range(num_points.get_index(0)):
            vs[i, 0] = points.get_index(i).x
            vs[i, 1] = points.get_index(i).y
            vs[i, 2] = points.get_index(i).z
            vs[i, 3] = 1
        
        idx = np.argmin(np.linalg.norm(vs - vertices[0], axis=1))
        for i in range(num_points.get_index(0)):
            a = (np.linalg.norm(vs[(idx + i) % num_points.get_index(0)] - vertices[i]) < 1e-5)
            b = (np.linalg.norm(vs[(idx - i) % num_points.get_index(0)] - vertices[i]) < 1e-5)
            assert a or b

    points.destroy_array()
    num_points.destroy_array()


def test_get_bbox_silhouette_1():
    p_min = redner.Vector3(-0.39268199,  0.465646,   -0.45052901)
    p_max = redner.Vector3(-0.073913,    0.79917198, -0.12509)
    p = redner.Vector3(-1.10526316, -1.73684211, -0.47368421)
    aabb = redner.AABB3(p_min, p_max)

    silhouette = redner.Vector3_ptr(0)
    silhouette.allocate(6)

    n_points = 0
    n_points = redner.test_get_bbox_silhouette_py(aabb, p, silhouette)

    redner_silhouette = np.zeros((n_points, 3), dtype=np.float32)
    for i in range(n_points):
        redner_silhouette[i, 0] = silhouette.get_index(i).x
        redner_silhouette[i, 1] = silhouette.get_index(i).y
        redner_silhouette[i, 2] = silhouette.get_index(i).z

    correct_silhouette = np.array([[-0.073913  ,  0.465646  , -0.12509   ],
                                   [-0.39268199,  0.465646  , -0.12509   ],
                                   [-0.39268199,  0.79917198, -0.12509   ],
                                   [-0.39268199,  0.79917198, -0.45052901],
                                   [-0.073913  ,  0.79917198, -0.45052901],
                                   [-0.073913  ,  0.465646  , -0.45052901]])
    
    assert(n_points == 6)
    assert(np.allclose(redner_silhouette, correct_silhouette[::-1]))

def test_get_bbox_silhouette_2():
    p_min = redner.Vector3(-0.39268199,  0.465646,   -0.45052901)
    p_max = redner.Vector3(-0.073913,    0.79917198, -0.12509)
    p = redner.Vector3(-0.15789474, -1.73684211, -0.47368421)
    aabb = redner.AABB3(p_min, p_max)

    silhouette = redner.Vector3_ptr(0)
    silhouette.allocate(6)

    n_points = 0
    n_points = redner.test_get_bbox_silhouette_py(aabb, p, silhouette)

    redner_silhouette = np.zeros((n_points, 3), dtype=np.float32)
    for i in range(n_points):
        redner_silhouette[i, 0] = silhouette.get_index(i).x
        redner_silhouette[i, 1] = silhouette.get_index(i).y
        redner_silhouette[i, 2] = silhouette.get_index(i).z

    correct_silhouette = np.array([[-0.073913  ,  0.465646  , -0.45052901],
                                   [-0.073913  ,  0.79917198, -0.45052901],
                                   [-0.39268199,  0.79917198, -0.45052901],
                                   [-0.39268199,  0.465646  , -0.45052901],
                                   [-0.39268199,  0.465646  , -0.12509   ],
                                   [-0.073913  ,  0.465646  , -0.12509   ]])
    correct_silhouette = np.roll(correct_silhouette, 2, axis=0)
    
    assert(n_points == 6)
    assert(np.allclose(redner_silhouette, correct_silhouette[::-1]))

def test_get_bbox_silhouette_3():
    p_min = redner.Vector3(-0.39268199,  0.465646,   -0.45052901)
    p_max = redner.Vector3(-0.073913,    0.79917198, -0.12509)
    p = redner.Vector3(-0.15789474, -1.73684211, -0.15789474)
    aabb = redner.AABB3(p_min, p_max)

    silhouette = redner.Vector3_ptr(0)
    silhouette.allocate(6)

    n_points = 0
    n_points = redner.test_get_bbox_silhouette_py(aabb, p, silhouette)

    redner_silhouette = np.zeros((n_points, 3), dtype=np.float32)
    for i in range(n_points):
        redner_silhouette[i, 0] = silhouette.get_index(i).x
        redner_silhouette[i, 1] = silhouette.get_index(i).y
        redner_silhouette[i, 2] = silhouette.get_index(i).z

    correct_silhouette = np.array([[-0.39268199,  0.465646,   -0.12509   ],
                                   [-0.39268199,  0.465646,   -0.45052901],
                                   [-0.073913  ,  0.465646,   -0.45052901],
                                   [-0.073913  ,  0.465646,   -0.12509   ]])
    
    assert(n_points == 4)
    assert(np.allclose(redner_silhouette, correct_silhouette[::-1]))

