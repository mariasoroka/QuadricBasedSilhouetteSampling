import pyredner
import redner
import torch
import scipy
import numpy as np

def solve_lp_scipy(np_points):
    c = np.array([-1, 0, 0, 0])
    b = np_points[:, 3]
    A = np.concatenate([np.ones((len(np_points), 1)), -np_points[:, 0:3]], axis=1)
    res = scipy.optimize.linprog(c, A, b)
    return res

def test_lp_solve_0():
    # there is a unique solution
    np_points = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 2], [0, -1, 0, 2], [-1, 0, 0, 2]])

    points = redner.Vector4_ptr(0)
    points.allocate(6)
    for i in range(6):
        points.set_index(i, redner.Vector4(np_points[i, 0], np_points[i, 1], np_points[i, 2], np_points[i, 3]))
    result = redner.test_solve_lp(points, 6)
    np_result = np.array([result.x, result.y, result.z, result.w])

    res = solve_lp_scipy(np_points)
    vector = np.array([res.x[1], res.x[2], res.x[3], 1])    
    if not res.success or res.x[0] < 0:
        assert np.linalg.norm(np_result) == 0
    else:
        assert np.allclose(np_result, vector)

    points.destroy_array()

def test_lp_solve_1():
    # there is a unique solution
    np_points = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1], [0, 0, -1, 1], [0, -1, 0, 1], [-1, 0, 0, 1]])

    points = redner.Vector4_ptr(0)
    points.allocate(6)
    for i in range(6):
        points.set_index(i, redner.Vector4(np_points[i, 0], np_points[i, 1], np_points[i, 2], np_points[i, 3]))
    result = redner.test_solve_lp(points, 6)
    np_result = np.array([result.x, result.y, result.z, result.w])

    res = solve_lp_scipy(np_points)
    vector = np.array([res.x[1], res.x[2], res.x[3], 1])
    if not res.success or res.x[0] < 0:
        assert np.linalg.norm(np_result) == 0
    else:
        assert np.allclose(np_result, vector)

    points.destroy_array()

def test_lp_solve_2():
    # there is no solution
    np_points = np.array([[0, 0, -1, -1], [0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, -1], [0, -1, 0, 1], [-1, 0, 0, 1]])

    points = redner.Vector4_ptr(0)
    points.allocate(6)
    for i in range(6):
        points.set_index(i, redner.Vector4(np_points[i, 0], np_points[i, 1], np_points[i, 2], np_points[i, 3]))
    result = redner.test_solve_lp(points, 6)
    np_result = np.array([result.x, result.y, result.z, result.w])

    res = solve_lp_scipy(np_points)
    
    if not res.success or res.x[0] < 0:
        assert np.allclose(np.linalg.norm(np_result[0:3]), 0)
    else:
        vector = np.array([res.x[1], res.x[2], res.x[3], 1])
        assert np.allclose(np_result, vector)

    points.destroy_array()

def test_lp_solve_3():
    np.random.seed(0)
    for j in range(15):

        n = np.random.randint(5, 10)
        np_points = np.random.rand(n, 4)

        points = redner.Vector4_ptr(0)
        points.allocate(n)
        for i in range(n):
            points.set_index(i, redner.Vector4(np_points[i, 0], np_points[i, 1], np_points[i, 2], np_points[i, 3]))
        result = redner.test_solve_lp(points, n)
        np_result = np.array([result.x, result.y, result.z, result.w])

        res = solve_lp_scipy(np_points)
        if not res.success or res.x[0] < 0:
            assert np.allclose(np.linalg.norm(np_result[0:3]), 0)
        else:
            vector = np.array([res.x[1], res.x[2], res.x[3], 1])
            dot_prods = np.einsum('ij, j -> i', np_points, np_result)
            obj = np.min(dot_prods)
            assert np.allclose(np_result, vector) or np.allclose(obj, res.x[0])

        points.destroy_array()