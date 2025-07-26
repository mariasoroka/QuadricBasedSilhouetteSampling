import pyredner
import redner
import torch
import scipy
import numpy as np

def test_find_basis_3D_1():
    np.random.seed(0)
    for k in range(10):
        N_np = np.random.randn(3)
        N = redner.Vector3(N_np[0], N_np[1], N_np[2])
        m = redner.find_basis(N)

        m_np = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                m_np[i, j] = m(i, j)

        assert np.allclose(m_np @ m_np.T, np.eye(3))
        assert np.allclose(m_np.T @ m_np, np.eye(3))
        assert np.allclose(m_np[:, 2], N_np / np.linalg.norm(N_np))

def test_find_basis_3D_2():
    N_np = np.array([0, 0, 1])
    N = redner.Vector3(N_np[0], N_np[1], N_np[2])
    m = redner.find_basis(N)

    m_np = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            m_np[i, j] = m(i, j)

    assert np.allclose(m_np @ m_np.T, np.eye(3))
    assert np.allclose(m_np.T @ m_np, np.eye(3))
    assert np.allclose(m_np[:, 2], N_np / np.linalg.norm(N_np))

def test_find_basis_3D_3():
    N_np = np.array([0, 1, 0])
    N = redner.Vector3(N_np[0], N_np[1], N_np[2])
    m = redner.find_basis(N)

    m_np = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            m_np[i, j] = m(i, j)

    assert np.allclose(m_np @ m_np.T, np.eye(3))
    assert np.allclose(m_np.T @ m_np, np.eye(3))
    assert np.allclose(m_np[:, 2], N_np / np.linalg.norm(N_np))

def test_find_basis_3D_4():
    N_np = np.array([1, 0, 0])
    N = redner.Vector3(N_np[0], N_np[1], N_np[2])
    m = redner.find_basis(N)

    m_np = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            m_np[i, j] = m(i, j)

    assert np.allclose(m_np @ m_np.T, np.eye(3))
    assert np.allclose(m_np.T @ m_np, np.eye(3))
    assert np.allclose(m_np[:, 2], N_np / np.linalg.norm(N_np))


def test_find_basis_4D_1():
    np.random.seed(0)
    for k in range(10):
        N_np = np.random.randn(4)
        N = redner.Vector4(N_np[0], N_np[1], N_np[2], N_np[3])
        m = redner.find_basis(N)

        m_np = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                m_np[i, j] = m(i, j)

        assert np.allclose(m_np @ m_np.T, np.eye(4))
        assert np.allclose(m_np.T @ m_np, np.eye(4))
        assert np.allclose(m_np[:, 3], N_np / np.linalg.norm(N_np))

def test_find_basis_4D_2():
    N_np = np.array([0, 0, 0, 1])
    N = redner.Vector4(N_np[0], N_np[1], N_np[2], N_np[3])
    m = redner.find_basis(N)

    m_np = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            m_np[i, j] = m(i, j)

    assert np.allclose(m_np @ m_np.T, np.eye(4))
    assert np.allclose(m_np.T @ m_np, np.eye(4))
    assert np.allclose(m_np[:, 3], N_np / np.linalg.norm(N_np))

def test_find_basis_4D_3():
    N_np = np.array([0, 0, 1, 0])
    N = redner.Vector4(N_np[0], N_np[1], N_np[2], N_np[3])
    m = redner.find_basis(N)

    m_np = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            m_np[i, j] = m(i, j)

    assert np.allclose(m_np @ m_np.T, np.eye(4))
    assert np.allclose(m_np.T @ m_np, np.eye(4))
    assert np.allclose(m_np[:, 3], N_np / np.linalg.norm(N_np))

def test_find_basis_4D_4():
    N_np = np.array([0, 1, 0, 0])
    N = redner.Vector4(N_np[0], N_np[1], N_np[2], N_np[3])
    m = redner.find_basis(N)

    m_np = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            m_np[i, j] = m(i, j)

    assert np.allclose(m_np @ m_np.T, np.eye(4))
    assert np.allclose(m_np.T @ m_np, np.eye(4))
    assert np.allclose(m_np[:, 3], N_np / np.linalg.norm(N_np))

def test_find_basis_4D_5():
    N_np = np.array([1, 0, 0, 0])
    N = redner.Vector4(N_np[0], N_np[1], N_np[2], N_np[3])
    m = redner.find_basis(N)

    m_np = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            m_np[i, j] = m(i, j)

    assert np.allclose(m_np @ m_np.T, np.eye(4))
    assert np.allclose(m_np.T @ m_np, np.eye(4))
    assert np.allclose(m_np[:, 3], N_np / np.linalg.norm(N_np))

def test_find_basis_4D_6():
    N_np = np.array([-0.509334, -0.843424, -0.170923, 0])
    N = redner.Vector4(N_np[0], N_np[1], N_np[2], N_np[3])
    m = redner.find_basis(N)

    m_np = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            m_np[i, j] = m(i, j)

    assert np.allclose(m_np @ m_np.T, np.eye(4))
    assert np.allclose(m_np.T @ m_np, np.eye(4))
    assert np.allclose(m_np[:, 3], N_np / np.linalg.norm(N_np))