import pyredner
import redner
import torch
import scipy
import numpy as np
import igl

def node_compute(mesh_filename):
    geometry_obj = pyredner.load_obj(mesh_filename, return_objects=True)[0]

    vertices = geometry_obj.vertices.numpy()
    indexes = geometry_obj.indices.numpy()
    edges = igl.edges(indexes)
    _, _, edge_flaps, _ = igl.edge_flaps(indexes)
    good_edges = np.nonzero(np.logical_and((edge_flaps[:, 0] != -1), edge_flaps[:, 1] != -1))[0]
    edges = edges[good_edges]
    edge_flaps = edge_flaps[good_edges]

    ptr_edges = redner.int_ptr(0)
    ptr_edges.allocate(len(edges) * 4)

    ptr_vertices = redner.float_ptr(0)
    ptr_vertices.allocate(len(vertices) * 3)

    ptr_indexes = redner.int_ptr(0)
    ptr_indexes.allocate(len(indexes) * 3)

    for i in range(len(vertices)):
        ptr_vertices.set_index(i * 3 + 0, vertices[i][0])
        ptr_vertices.set_index(i * 3 + 1, vertices[i][1])
        ptr_vertices.set_index(i * 3 + 2, vertices[i][2])

    for i in range(len(indexes)):
        ptr_indexes.set_index(i * 3 + 0, indexes[i][0])
        ptr_indexes.set_index(i * 3 + 1, indexes[i][1])
        ptr_indexes.set_index(i * 3 + 2, indexes[i][2])

    for i in range(len(edges)):
        ptr_edges.set_index(i * 4 + 0, edges[i][0])
        ptr_edges.set_index(i * 4 + 1, edges[i][1])
        ptr_edges.set_index(i * 4 + 2, edge_flaps[i][0])
        ptr_edges.set_index(i * 4 + 3, edge_flaps[i][1])

    shape = redner.Shape(ptr_vertices,
                         ptr_indexes,
                         redner.float_ptr(0),
                         redner.float_ptr(0),
                         redner.int_ptr(0),
                         redner.int_ptr(0),
                         redner.float_ptr(0),
                         len(vertices),
                         0,
                         0,
                         len(indexes),
                         0,
                         -1)
    
    ptr_output = redner.double_ptr(0)
    ptr_output.allocate(50)

    redner.test_compute_stack_item(len(edges), ptr_edges, ptr_output, shape, False)

    bounds_p_min = np.array([ptr_output.get_index(0), ptr_output.get_index(1), ptr_output.get_index(2)])
    bounds_p_max = np.array([ptr_output.get_index(3), ptr_output.get_index(4), ptr_output.get_index(5)])

    one_hemisphere = ptr_output.get_index(6)

    quadric_offset_vector = np.array([ptr_output.get_index(7), ptr_output.get_index(8), ptr_output.get_index(9), ptr_output.get_index(10)])

    dual_bounds_p_min = np.array([ptr_output.get_index(11), ptr_output.get_index(12), ptr_output.get_index(13)])
    dual_bounds_p_max = np.array([ptr_output.get_index(14), ptr_output.get_index(15), ptr_output.get_index(16)])

    q0 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            q0[i, j] = ptr_output.get_index(17 + i * 4 + j)
    
    q1 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            q1[i, j] = ptr_output.get_index(33 + i * 4 + j)

    ptr_edges.destroy_array()
    ptr_vertices.destroy_array()
    ptr_indexes.destroy_array()
    ptr_output.destroy_array()

    edge_vertices = np.concatenate([vertices[edges[:, 0]], vertices[edges[:, 1]]], axis=0)

    gt_bounds_p_min = np.array([np.min(edge_vertices[:, 0]), np.min(edge_vertices[:, 1]), np.min(edge_vertices[:, 2])])
    gt_bounds_p_max = np.array([np.max(edge_vertices[:, 0]), np.max(edge_vertices[:, 1]), np.max(edge_vertices[:, 2])])

    assert np.allclose(bounds_p_min, gt_bounds_p_min)
    assert np.allclose(bounds_p_max, gt_bounds_p_max)                

    return one_hemisphere, quadric_offset_vector, q0, q1

def test_mesh_1():
    one_hemisphere, quadric_offset_vector, q0, q1 = node_compute("./test_mesh_1.obj")
    assert one_hemisphere == 1
    
    gt_quadric_offset_vector = np.array([0.68891535, -0.02461455, -0.72442374,  0])
    quadric_offset_vector /= np.linalg.norm(quadric_offset_vector)
    assert np.allclose(quadric_offset_vector, gt_quadric_offset_vector, atol=1e-6)
    
    gt_q0 = np.array([[ 0.01053136,  0.11180628, -0.03241376,  0.13837845],
                      [ 0.11180628,  0.32621267, -0.06681653,  0.46801327],
                      [-0.03241376, -0.06681653, -0.02110372, -0.0811466 ],
                      [ 0.13837845,  0.46801327, -0.0811466 ,  0.60618311]])
    gt_q1 = np.array([[ 0.01168124,  0.11175934, -0.03362179,  0.13837123],
                      [ 0.11175934,  0.32619713, -0.06676982,  0.46798886],
                      [-0.03362179, -0.06676982, -0.01983055, -0.08114237],
                      [ 0.13837123,  0.46798886, -0.08114237,  0.6061515 ]])

    gt_q0 /= np.linalg.norm(gt_q0)
    gt_q0 *= np.sign(gt_q0[0, 0])
    gt_q1 /= np.linalg.norm(gt_q1)
    gt_q1 *= np.sign(gt_q1[0, 0])

    q0 /= np.linalg.norm(q0)
    q0 *= np.sign(q0[0, 0])
    q1 /= np.linalg.norm(q1)
    q1 *= np.sign(q1[0, 0])

    assert np.allclose(q0, gt_q0, atol=1e-6) or np.allclose(q1, gt_q0, atol=1e-6)
    assert np.allclose(q0, gt_q1, atol=1e-6) or np.allclose(q1, gt_q1, atol=1e-6)
    
def test_mesh_2():
    one_hemisphere, quadric_offset_vector, q0, q1 = node_compute("./test_mesh_2.obj")
    assert one_hemisphere == 1
    
    gt_quadric_offset_vector = np.array([0.50357774, -0.63601444, -0.58471796,  0])
    quadric_offset_vector /= np.linalg.norm(quadric_offset_vector)
    assert np.allclose(quadric_offset_vector, gt_quadric_offset_vector, atol=1e-6)

    gt_q0 = np.array([[ 0.26126197,  0.15095213, -0.03675481,  0.24742226],
                      [ 0.15095213, -0.1782377 , -0.07693629, -0.40618916],
                      [-0.03675481, -0.07693629,  0.15668469, -0.10037079],
                      [ 0.24742226, -0.40618916, -0.10037079, -0.58544573]])
    gt_q1 = np.array([[ 0.2761695 ,  0.13364523, -0.05322107,  0.24820486],
                      [ 0.13364523, -0.15633996, -0.05652972, -0.40747394],
                      [-0.05322107, -0.05652972,  0.17616473, -0.10068826],
                      [ 0.24820486, -0.40747394, -0.10068826, -0.5872975 ]])

    gt_q0 /= np.linalg.norm(gt_q0)
    gt_q0 *= np.sign(gt_q0[0, 0])
    gt_q1 /= np.linalg.norm(gt_q1)
    gt_q1 *= np.sign(gt_q1[0, 0])

    q0 /= np.linalg.norm(q0)
    q0 *= np.sign(q0[0, 0])
    q1 /= np.linalg.norm(q1)
    q1 *= np.sign(q1[0, 0])

    assert np.allclose(q0, gt_q0, atol=1e-6) or np.allclose(q1, gt_q0, atol=1e-6)
    assert np.allclose(q0, gt_q1, atol=1e-6) or np.allclose(q1, gt_q1, atol=1e-6)