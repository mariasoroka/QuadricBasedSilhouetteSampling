import redner
import numpy as np

import scipy

def test_atan2_approx():
    angles = np.linspace(-np.pi, np.pi, 100)
    xs = np.cos(angles)
    ys = np.sin(angles)
    for i in range(100):
        atan2_redner = redner.atan2_approx(ys[i], xs[i])
        assert np.isclose(atan2_redner, angles[i], atol=1e-4)


def intersect_bbox(p_min, p_max, ray_origin, ray_dir):
    tmin_0 = (p_min[0] - ray_origin[:, 0]) / ray_dir[:, 0]
    tmax_0 = (p_max[0] - ray_origin[:, 0]) / ray_dir[:, 0]

    tmin_1 = (p_min[1] - ray_origin[:, 1]) / ray_dir[:, 1]
    tmax_1 = (p_max[1] - ray_origin[:, 1]) / ray_dir[:, 1]

    tmin_2 = (p_min[2] - ray_origin[:, 2]) / ray_dir[:, 2]
    tmax_2 = (p_max[2] - ray_origin[:, 2]) / ray_dir[:, 2]

    tmin = np.maximum(np.minimum(tmin_0, tmax_0), np.minimum(tmin_1, tmax_1))
    tmax = np.minimum(np.maximum(tmin_0, tmax_0), np.maximum(tmin_1, tmax_1))

    tmin = np.maximum(tmin, np.minimum(tmin_2, tmax_2))
    tmax = np.minimum(tmax, np.maximum(tmin_2, tmax_2))

    return tmin < tmax

def test_bbox_solid_angle_1():
    p_min = redner.Vector3(-8, -1, 1)
    p_max = redner.Vector3(1, 1, 3)
    p_min_np = np.array([-8, -1, 1])
    p_max_np = np.array([1, 1, 3])
    bbox = redner.AABB3(p_min, p_max)
    surface_point = redner.SurfacePoint()
    surface_point.position = redner.Vector3(0, 0, 0)
    surface_point.geom_normal = redner.Vector3(0, 0, 1)
    frame_inv = redner.Matrix3x3(1, 0, 0,
                                    0, 1, 0,
                                    0, 0, 1)
    
    solid_angle = redner.bbox_solid_angle(bbox, frame_inv, surface_point)

    nsamples = 1000000
    np.random.seed(0)
    samples = np.random.rand(nsamples, 2)
    phis = 2 * np.pi * samples[:, 0]
    z = samples[:, 1]
    thetas = np.arccos(z)
    dirs = np.zeros((nsamples, 3))
    dirs[:, 0] = np.sin(thetas) * np.cos(phis)
    dirs[:, 1] = np.sin(thetas) * np.sin(phis)
    dirs[:, 2] = np.cos(thetas)

    origins = np.zeros((nsamples, 3))

    res = intersect_bbox(p_min_np, p_max_np, origins, dirs)

    assert np.isclose(2 * np.pi * np.count_nonzero(res) / nsamples, solid_angle, atol=0.01)


def test_bbox_solid_angle_2():
    p_min = redner.Vector3(-8, -8, 1)
    p_max = redner.Vector3(1, 1, 3)
    p_min_np = np.array([-8, -8, 1])
    p_max_np = np.array([1, 1, 3])
    bbox = redner.AABB3(p_min, p_max)
    surface_point = redner.SurfacePoint()
    surface_point.position = redner.Vector3(0, 0, 0)
    surface_point.geom_normal = redner.Vector3(0, 0, 1)
    frame_inv = redner.Matrix3x3(1, 0, 0,
                                    0, 1, 0,
                                    0, 0, 1)
    
    solid_angle = redner.bbox_solid_angle(bbox, frame_inv, surface_point)

    nsamples = 1000000
    np.random.seed(0)
    samples = np.random.rand(nsamples, 2)
    phis = 2 * np.pi * samples[:, 0]
    z = samples[:, 1]
    thetas = np.arccos(z)
    dirs = np.zeros((nsamples, 3))
    dirs[:, 0] = np.sin(thetas) * np.cos(phis)
    dirs[:, 1] = np.sin(thetas) * np.sin(phis)
    dirs[:, 2] = np.cos(thetas)

    origins = np.zeros((nsamples, 3))

    res = intersect_bbox(p_min_np, p_max_np, origins, dirs)

    assert np.isclose(2 * np.pi * np.count_nonzero(res) / nsamples, solid_angle, atol=0.01)


def triangle_solid_angle(v0, v1, v2):
    len_a = np.linalg.norm(v0)
    len_b = np.linalg.norm(v1)
    len_c = np.linalg.norm(v2)

    num = np.dot(np.cross(v0, v1), v2)
    den = len_a * len_b * len_c + np.dot(v0, v1) * len_c + np.dot(v1, v2) * len_a + np.dot(v2, v0) * len_b

    if (num > 0 and den < 0):
        return 2 * np.arctan(num / den) + np.pi
    
    return 2 * np.arctan(np.abs(num / den))


def test_polygon_solid_angle_1():

    vertices = redner.Vector3_ptr(0)
    n_vert = 3
    vertices.allocate(n_vert)
    np_v_1 = np.array([0.0, 0.0, 1.0])
    np_v_2 = np.array([1.0, 0.0, 0.0])
    np_v_3 = np.array([0.0, 1.0, 0.0])
    vertices.set_index(0, redner.Vector3(np_v_1[0], np_v_1[1], np_v_1[2]))
    vertices.set_index(1, redner.Vector3(np_v_2[0], np_v_2[1], np_v_2[2]))
    vertices.set_index(2, redner.Vector3(np_v_3[0], np_v_3[1], np_v_3[2]))

    angle = redner.test_polygon_solid_angle(n_vert, vertices)

    vertices.destroy_array()

    assert np.isclose(angle, np.pi / 2, atol=1e-4)

def test_polygon_solid_angle_2():

    vertices = redner.Vector3_ptr(0)
    n_vert = 3
    vertices.allocate(n_vert)
    np_v_1 = np.array([0.0, 0.0, -1.0])
    np_v_2 = np.array([-1.0, 0.0, 0.0])
    np_v_3 = np.array([0.0, -1.0, 0.0])
    vertices.set_index(0, redner.Vector3(np_v_1[0], np_v_1[1], np_v_1[2]))
    vertices.set_index(1, redner.Vector3(np_v_2[0], np_v_2[1], np_v_2[2]))
    vertices.set_index(2, redner.Vector3(np_v_3[0], np_v_3[1], np_v_3[2]))

    angle = redner.test_polygon_solid_angle(n_vert, vertices)

    vertices.destroy_array()

    assert np.isclose(angle, 0)

def test_polygon_solid_angle_3():

    vertices = redner.Vector3_ptr(0)
    n_vert = 4
    vertices.allocate(n_vert)
    np_v_1 = np.array([1.0, 1.0, 1.0])
    np_v_2 = np.array([1.0, 1.0, -1.0])
    np_v_3 = np.array([1.0, -1.0, -1.0])
    np_v_4 = np.array([1.0, -1.0, 1.0])
    vertices.set_index(0, redner.Vector3(np_v_1[0], np_v_1[1], np_v_1[2]))
    vertices.set_index(1, redner.Vector3(np_v_2[0], np_v_2[1], np_v_2[2]))
    vertices.set_index(2, redner.Vector3(np_v_3[0], np_v_3[1], np_v_3[2]))
    vertices.set_index(3, redner.Vector3(np_v_4[0], np_v_4[1], np_v_4[2]))

    angle = redner.test_polygon_solid_angle(n_vert, vertices)

    vertices.destroy_array()
    gt_angle = triangle_solid_angle(np_v_1, np_v_2, np_v_3) + \
               triangle_solid_angle(np_v_1, np_v_3, np_v_4)
    
    assert np.isclose(angle, gt_angle)

def ltc_value(ray_dirs, m, m_inv):
    ray_dirs_transformed = np.einsum('ij, nj -> ni', m_inv, ray_dirs)
    norms = np.linalg.norm(ray_dirs_transformed, axis=1)
    determ = np.linalg.det(m_inv)
    return ray_dirs_transformed[:, 2] * determ / (norms**4 * np.pi)

def test_bbox_ltc_1():
    p_min = redner.Vector3(-1, -1, 1)
    p_max = redner.Vector3(1, 1, 3)
    p_min_np = np.array([-1, -1, 1])
    p_max_np = np.array([1, 1, 3])
    bbox = redner.AABB3(p_min, p_max)
    surface_point = redner.SurfacePoint()
    surface_point.position = redner.Vector3(0, 0, 0)
    surface_point.geom_normal = redner.Vector3(0, 0, 1)
    frame_inv = redner.Matrix3x3(1, 0, 0,
                                    0, 1, 0,
                                    0, 0, 1)
    m = redner.Matrix3x3(0.8, 0, 0,
                         0, 0.2, 0,
                            0, 0, 1)
    m_inv = redner.Matrix3x3(1 / 0.8, 0, 0,
                                0, 1 / 0.2, 0,
                                0, 0, 1)
    
    m_inv_np = np.array([[1 / 0.8, 0, 0],
                        [0, 1 / 0.2, 0],
                        [0, 0, 1]])
    
    integral_ltc = redner.bbox_ltc(bbox, surface_point, m, m_inv, frame_inv)

    nsamples = 100000
    np.random.seed(0)
    samples = np.random.rand(nsamples, 2)
    phis = 2 * np.pi * samples[:, 0]
    z = samples[:, 1]
    thetas = np.arccos(z)
    dirs = np.zeros((nsamples, 3))
    dirs[:, 0] = np.sin(thetas) * np.cos(phis)
    dirs[:, 1] = np.sin(thetas) * np.sin(phis)
    dirs[:, 2] = np.cos(thetas)

    origins = np.zeros((nsamples, 3))

    res = intersect_bbox(p_min_np, p_max_np, origins, dirs)
    values = ltc_value(dirs, m, m_inv_np)
    integral = 2 * np.pi * np.sum(values[res]) / nsamples

    assert np.isclose(integral, integral_ltc, atol=0.01)


def test_bbox_ltc_2():
    p_min = redner.Vector3(-1, -1, 0.5)
    p_max = redner.Vector3(1, 1, 3)
    p_min_np = np.array([-1, -1, 0.5])
    p_max_np = np.array([1, 1, 3])
    bbox = redner.AABB3(p_min, p_max)
    surface_point = redner.SurfacePoint()
    surface_point.position = redner.Vector3(0, 0, 0)
    surface_point.geom_normal = redner.Vector3(0, 0, 1)
    frame_inv = redner.Matrix3x3(1, 0, 0,
                                    0, 1, 0,
                                    0, 0, 1)
    m = redner.Matrix3x3(0.8, 0, 0,
                         0, 0.2, 0,
                            0, 0, 1)
    m_inv = redner.Matrix3x3(1 / 0.8, 0, 0,
                                0, 1 / 0.2, 0,
                                0, 0, 1)
    
    m_inv_np = np.array([[1 / 0.8, 0, 0],
                        [0, 1 / 0.2, 0],
                        [0, 0, 1]])
    
    integral_ltc = redner.bbox_ltc(bbox, surface_point, m, m_inv, frame_inv)

    nsamples = 100000
    np.random.seed(0)
    samples = np.random.rand(nsamples, 2)
    phis = 2 * np.pi * samples[:, 0]
    z = samples[:, 1]
    thetas = np.arccos(z)
    dirs = np.zeros((nsamples, 3))
    dirs[:, 0] = np.sin(thetas) * np.cos(phis)
    dirs[:, 1] = np.sin(thetas) * np.sin(phis)
    dirs[:, 2] = np.cos(thetas)

    origins = np.zeros((nsamples, 3))

    res = intersect_bbox(p_min_np, p_max_np, origins, dirs)
    values = ltc_value(dirs, m, m_inv_np)
    integral = 2 * np.pi * np.sum(values[res]) / nsamples

    assert np.isclose(integral, integral_ltc, atol=0.01)

def test_bbox_ltc_3():
    p_min = redner.Vector3(-1, -2, 1)
    p_max = redner.Vector3(1, -1, 3)
    p_min_np = np.array([-1, -2, 1])
    p_max_np = np.array([1, -1, 3])
    bbox = redner.AABB3(p_min, p_max)
    surface_point = redner.SurfacePoint()
    surface_point.position = redner.Vector3(0, 0, 0)
    surface_point.geom_normal = redner.Vector3(0, 0, 1)
    frame_inv = redner.Matrix3x3(1, 0, 0,
                                    0, 1, 0,
                                    0, 0, 1)
    m = redner.Matrix3x3(0.8, 0, 0,
                         0, 0.2, 0,
                            0, 0, 1)
    m_inv = redner.Matrix3x3(1 / 0.8, 0, 0,
                                0, 1 / 0.2, 0,
                                0, 0, 1)
    
    m_inv_np = np.array([[1 / 0.8, 0, 0],
                        [0, 1 / 0.2, 0],
                        [0, 0, 1]])
    
    integral_ltc = redner.bbox_ltc(bbox, surface_point, m, m_inv, frame_inv)

    nsamples = 100000
    np.random.seed(0)
    samples = np.random.rand(nsamples, 2)
    phis = 2 * np.pi * samples[:, 0]
    z = samples[:, 1]
    thetas = np.arccos(z)
    dirs = np.zeros((nsamples, 3))
    dirs[:, 0] = np.sin(thetas) * np.cos(phis)
    dirs[:, 1] = np.sin(thetas) * np.sin(phis)
    dirs[:, 2] = np.cos(thetas)

    origins = np.zeros((nsamples, 3))

    res = intersect_bbox(p_min_np, p_max_np, origins, dirs)
    values = ltc_value(dirs, m, m_inv_np)
    integral = 2 * np.pi * np.sum(values[res]) / nsamples

    assert np.isclose(integral, integral_ltc, atol=0.01)


def test_bbox_average_bsdf_1():

    p_min = redner.Vector3(-1, -1, 1)
    p_max = redner.Vector3(1, 1, 3)
    bbox = redner.AABB3(p_min, p_max)
    surface_point = redner.SurfacePoint()
    surface_point.position = redner.Vector3(0, 0, 0)
    surface_point.geom_normal = redner.Vector3(0, 0, 1)
    frame_inv = redner.Matrix3x3(1, 0, 0,
                                 0, 1, 0,
                                 0, 0, 1)
    m = redner.Matrix3x3(0.8, 0, 0,
                         0, 0.2, 0,
                         0, 0, 1)
    m_inv = redner.Matrix3x3(1 / 0.8, 0, 0,
                             0, 1 / 0.2, 0,
                             0, 0, 1)
    
    solid_angle = redner.bbox_solid_angle(bbox, frame_inv, surface_point)
    bsdf = redner.bbox_ltc(bbox, surface_point, m, m_inv, frame_inv)

    bsdf_average = redner.test_bbox_average_bsdf(bbox, surface_point, m, m_inv, frame_inv)

    assert np.isclose(bsdf_average, bsdf / solid_angle, atol=0.01)

def test_bbox_average_bsdf_2():
    
    p_min = redner.Vector3(-8, -1, 1)
    p_max = redner.Vector3(1, 1, 3)
    bbox = redner.AABB3(p_min, p_max)
    surface_point = redner.SurfacePoint()
    surface_point.position = redner.Vector3(0, 0, 0)
    surface_point.geom_normal = redner.Vector3(0, 0, 1)
    frame_inv = redner.Matrix3x3(1, 0, 0,
                                 0, 1, 0,
                                 0, 0, 1)
    m = redner.Matrix3x3(0.8, 0, 0,
                         0, 0.2, 0,
                         0, 0, 1)
    m_inv = redner.Matrix3x3(1 / 0.8, 0, 0,
                             0, 1 / 0.2, 0,
                             0, 0, 1)
    
    solid_angle = redner.bbox_solid_angle(bbox, frame_inv, surface_point)
    bsdf = redner.bbox_ltc(bbox, surface_point, m, m_inv, frame_inv)

    bsdf_average = redner.test_bbox_average_bsdf(bbox, surface_point, m, m_inv, frame_inv)

    assert np.isclose(bsdf_average, bsdf / solid_angle, atol=0.01)
