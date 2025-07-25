import pyredner
import redner
import torch
import scipy
import numpy as np

def test_compute_bounding_sphere_1():
    n_points = 1
    np_points = np.random.randn(n_points, 4)

    dual_planes = redner.Vector4_ptr(0)
    dual_planes.allocate(n_points)

    for i in range(n_points):
        dual_planes.set_index(i, redner.Vector4(np_points[i,0], np_points[i,1], np_points[i,2], np_points[i,3]))

    optsphere = redner.test_find_approx_bounding_sphere(dual_planes, n_points)

    np_optsphere_center = np.array([optsphere.center.x, optsphere.center.y, optsphere.center.z])
    np_optsphere_radius = optsphere.radius

    gt_center = np_points[0, 0:3]

    dual_planes.destroy_array()
    assert np.allclose(np_optsphere_center, gt_center)
    assert np.allclose(np_optsphere_radius, 0)

def test_compute_bounding_sphere_2():
    n_points = 2
    np_points = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

    dual_planes = redner.Vector4_ptr(0)
    dual_planes.allocate(n_points)

    for i in range(n_points):
        dual_planes.set_index(i, redner.Vector4(np_points[i,0], np_points[i,1], np_points[i,2], np_points[i,3]))

    optsphere = redner.test_find_approx_bounding_sphere(dual_planes, n_points)

    np_optsphere_center = np.array([optsphere.center.x, optsphere.center.y, optsphere.center.z])
    np_optsphere_radius = optsphere.radius

    gt_center = ((np_points[0] + np_points[1]) / 2)[0:3]
    gt_radius = np.linalg.norm(np_points[0, 0:3] - gt_center)

    dual_planes.destroy_array()
    assert np.allclose(np_optsphere_center, gt_center)
    assert np.allclose(np_optsphere_radius, gt_radius, atol=1e-4)

def test_compute_bounding_sphere_3():
    n_points = 3
    np_points = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1]])

    dual_planes = redner.Vector4_ptr(0)
    dual_planes.allocate(n_points)

    for i in range(n_points):
        dual_planes.set_index(i, redner.Vector4(np_points[i,0], np_points[i,1], np_points[i,2], np_points[i,3]))

    optsphere = redner.test_find_approx_bounding_sphere(dual_planes, n_points)

    np_optsphere_center = np.array([optsphere.center.x, optsphere.center.y, optsphere.center.z])
    np_optsphere_radius = optsphere.radius

    gt_center = ((np_points[0] + np_points[1] + np_points[2]) / 3)[0:3]
    gt_radius = np.linalg.norm(np_points[0, 0:3] - gt_center)

    dual_planes.destroy_array()
    assert np.allclose(np_optsphere_center, gt_center)
    assert np.allclose(np_optsphere_radius, gt_radius, atol=1e-4)

def test_compute_bounding_sphere_4():
    n_points = 4
    np_points = np.array([[1, 1, 1, 1], [1, 2, 2, 1], [2, 2, 1, 1], [2, 1, 2, 1]])

    dual_planes = redner.Vector4_ptr(0)
    dual_planes.allocate(n_points)

    for i in range(n_points):
        dual_planes.set_index(i, redner.Vector4(np_points[i,0], np_points[i,1], np_points[i,2], np_points[i,3]))

    optsphere = redner.test_find_approx_bounding_sphere(dual_planes, n_points)

    np_optsphere_center = np.array([optsphere.center.x, optsphere.center.y, optsphere.center.z])
    np_optsphere_radius = optsphere.radius

    gt_center = ((np_points[0] + np_points[1] + np_points[2] + np_points[3]) / 4)[0:3]
    gt_radius = np.linalg.norm(np_points[0, 0:3] - gt_center)

    dual_planes.destroy_array()
    assert np.allclose(np_optsphere_center, gt_center)
    assert np.allclose(np_optsphere_radius, gt_radius, atol=1e-4)

def test_compute_bounding_sphere_n():
    n_points = 30
    np_points = np.random.rand(n_points, 4) + np.array([1, 1, 1, 1])
    np_points[:, 3] = 1
    np_points_cube = np.array([[1, 1, 1, 1], [1, 2, 2, 1], [2, 2, 1, 1], [2, 1, 2, 1]])
    np_points[0:4] = np_points_cube

    dual_planes = redner.Vector4_ptr(0)
    dual_planes.allocate(n_points)

    for i in range(n_points):
        dual_planes.set_index(i, redner.Vector4(np_points[i,0], np_points[i,1], np_points[i,2], np_points[i,3]))

    optsphere = redner.test_find_approx_bounding_sphere(dual_planes, n_points)

    np_optsphere_center = np.array([optsphere.center.x, optsphere.center.y, optsphere.center.z])
    np_optsphere_radius = optsphere.radius

    gt_center = ((np_points[0] + np_points[1] + np_points[2] + np_points[3]) / 4)[0:3]
    gt_radius = np.linalg.norm(np_points[0, 0:3] - gt_center)

    dual_planes.destroy_array()
    assert np.allclose(np_optsphere_center, gt_center)
    assert np.allclose(np_optsphere_radius, gt_radius, atol=1e-4)

def test_compute_offset_quadric_vector():
    n_points = 60
    np_points = np.random.rand(n_points, 4) + np.array([1, 1, 1, 1])
    np_points[:, 3] = 1
    np_points_cube = np.array([[1, 1, 1, 1], [1, 2, 2, 1], [2, 2, 1, 1], [2, 1, 2, 1]])
    np_points[0:4] = np_points_cube

    dual_planes = redner.Vector4_ptr(0)
    dual_planes.allocate(n_points)

    for i in range(n_points):
        dual_planes.set_index(i, redner.Vector4(np_points[i,0], np_points[i,1], np_points[i,2], np_points[i,3]))

    dir = redner.test_find_offset_quadric_vector(dual_planes, n_points)
    dual_planes.destroy_array()

    np_points_normalized = np_points[:, 0:3] / np.linalg.norm(np_points[:, 0:3], axis=1)[:, None]
    gt_center = np.mean(np_points_normalized[0:4], axis=0)

    gt_dir = gt_center / np.linalg.norm(gt_center)

    assert np.allclose(np.array([dir.x, dir.y, dir.z]), gt_dir, atol=1e-4)
