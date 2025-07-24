import pyredner
import redner
import torch
import scipy
import numpy as np


def test_intersect_with_line():
    np.random.seed(0)
    for i in range(100):
        np_matrix = np.random.randn(3, 3)
        np_matrix = np_matrix + np_matrix.T
        np_matrix /= np.linalg.norm(np_matrix)
        np_line = np.random.randn(3)

        matrix = redner.Matrix3x3(np_matrix[0, 0], np_matrix[0, 1], np_matrix[0, 2], \
                                np_matrix[1, 0], np_matrix[1, 1], np_matrix[1, 2], \
                                np_matrix[2, 0], np_matrix[2, 1], np_matrix[2, 2])
        conic = redner.Conic(matrix)
        line = redner.Vector3(np_line[0], np_line[1], np_line[2])

        intersection = redner.intersect_with_line(conic, line)
        np_p1 = np.array([intersection.p1.x, intersection.p1.y, intersection.p1.z])
        np_p2 = np.array([intersection.p2.x, intersection.p2.y, intersection.p2.z])

        mu_h = np.array([[0, np_line[2], -np_line[1]], [-np_line[2], 0, np_line[0]], [np_line[1], -np_line[0], 0]])

        B = mu_h.T @ np_matrix @ mu_h
        best_idx = np.argmax(np.abs(np_line))
        minor_idxs_0 = np.array(list(range(best_idx))+list(range(best_idx+1,3)))[:,np.newaxis]
        minor_idxs_1 = np.array(list(range(best_idx))+list(range(best_idx+1,3)))
        scale = np_line[best_idx]
        det_tmp = np.linalg.det(B[minor_idxs_0, minor_idxs_1])
        if det_tmp > 0:
            assert np.all(np_p1 == 0)
            assert np.all(np_p2 == 0)
        else:
            alpha = np.sqrt(-det_tmp) / scale
            C = B + alpha * mu_h
            idxs = np.nonzero(C)
            best_idx = 0
            p = np.copy(C[idxs[0][best_idx], :])
            q = np.copy(C[:, idxs[1][best_idx]])
            p /= p[2]
            q /= q[2]

            assert np.allclose(p, np_p1, rtol=1e-5) or np.allclose(p, np_p2, rtol=1e-5)
            assert np.allclose(q, np_p2, rtol=1e-5) or np.allclose(q, np_p1, rtol=1e-5)

            assert np.abs(np.dot(np_p1, np_line)) / (np.linalg.norm(np_p1) * np.linalg.norm(np_line)) <= 1e-5
            assert np.abs(np.dot(np_p2, np_line)) / (np.linalg.norm(np_p2) * np.linalg.norm(np_line)) <= 1e-5
            assert np.abs(np.einsum('i, ij, j', np_p1, np_matrix, np_p1)) / (np.linalg.norm(np_p1) * np.linalg.norm(np_p2)) <= 1e-5
            assert np.abs(np.einsum('i, ij, j', np_p2, np_matrix, np_p2)) / (np.linalg.norm(np_p1) * np.linalg.norm(np_p2)) <= 1e-5

def test_get_point_on_conic():
    np.random.seed(0)
    j = 0
    for i in range(100):
        np_matrix = np.random.randn(3, 3)
        np_matrix = np_matrix + np_matrix.T
        np_matrix /= np.linalg.norm(np_matrix)
        matrix = redner.Matrix3x3(np_matrix[0, 0], np_matrix[0, 1], np_matrix[0, 2], \
                                np_matrix[1, 0], np_matrix[1, 1], np_matrix[1, 2], \
                                np_matrix[2, 0], np_matrix[2, 1], np_matrix[2, 2])
        conic = redner.Conic(matrix)
        p = redner.get_point_on_conic(conic)
        np_p = np.array([p.x, p.y, p.z])
        
        if (np.linalg.norm(np_p) > 0):
            j += 1
            assert np.einsum('i, ij, j', np_p, np_matrix, np_p) / (np.linalg.norm(np_p) ** 2) <= 1e-5
    assert j > 0

def test_intersect_with_segment():
    np.random.seed(0)
    for i in range(100):
        np_matrix = np.random.randn(4, 4)
        np_matrix = np_matrix + np_matrix.T
        np_matrix /= np.linalg.norm(np_matrix)

        np_v0 = np.random.randn(3)
        np_v1 = np.random.randn(3)

        matrix = redner.Matrix4x4(np_matrix[0, 0], np_matrix[0, 1], np_matrix[0, 2], np_matrix[0, 3], \
                                    np_matrix[1, 0], np_matrix[1, 1], np_matrix[1, 2], np_matrix[1, 3], \
                                    np_matrix[2, 0], np_matrix[2, 1], np_matrix[2, 2], np_matrix[2, 3], \
                                    np_matrix[3, 0], np_matrix[3, 1], np_matrix[3, 2], np_matrix[3, 3])
        
        quadric = redner.Quadric(matrix)
        
        v1 = redner.Vector3(np_v0[0], np_v0[1], np_v0[2])
        v2 = redner.Vector3(np_v1[0], np_v1[1], np_v1[2])

        intersection = redner.intersect_with_segment(quadric, v1, v2)

        np_p1 = np.array([intersection.p1.x, intersection.p1.y, intersection.p1.z, intersection.p1.w])
        np_p2 = np.array([intersection.p2.x, intersection.p2.y, intersection.p2.z, intersection.p2.w])

        np_v0_hom = np.concatenate([np_v0, [1]])
        np_v1_hom = np.concatenate([np_v1, [1]])

        a0 = np.einsum('i, j, ij', np_v0_hom, np_v0_hom, np_matrix)
        a1 = 2 * np.einsum('i, j, ij', np_v0_hom, np_v1_hom - np_v0_hom, np_matrix)
        a2 = np.einsum('i, j, ij', np_v1_hom - np_v0_hom, np_v1_hom - np_v0_hom, np_matrix)

        roots = np.roots([a2, a1, a0])

        real_roots = roots[np.isreal(roots)]

        if len(real_roots) == 0:
            root_0 = np.zeros(4)
            root_1 = np.zeros(4)
        elif len(real_roots) == 1:
            root_0 = np_v0_hom + real_roots[0] * (np_v1_hom - np_v0_hom) if real_roots[0] >= 0 and real_roots[0] <= 1 else np.zeros(4)
            root_1 = np.zeros(4)
        else:
            root_0 = np_v0_hom + real_roots[0] * (np_v1_hom - np_v0_hom) if real_roots[0] >= 0 and real_roots[0] <= 1 else np.zeros(4)
            root_1 = np_v0_hom + real_roots[1] * (np_v1_hom - np_v0_hom) if real_roots[1] >= 0 and real_roots[1] <= 1 else np.zeros(4)

        
        assert np.allclose(np_p1, root_0, rtol=1e-5) or np.allclose(np_p1, root_1, rtol=1e-5)
        assert np.allclose(np_p2, root_1, rtol=1e-5) or np.allclose(np_p2, root_0, rtol=1e-5)

        if np.linalg.norm(np_p1) > 0:
            assert np.abs(np.einsum('i, ij, j', np_p1, np_matrix, np_p1)) / (np.linalg.norm(np_p1) ** 2) <= 1e-5
        if np.linalg.norm(np_p2) > 0:
            assert np.abs(np.einsum('i, ij, j', np_p2, np_matrix, np_p2)) / (np.linalg.norm(np_p2) ** 2) <= 1e-5


def test_solve_gen_eig():
    np.random.seed(0)
    for k in range(100):
        tmp = np.random.randn(4, 4)
        Q_A = np.linalg.qr(tmp)[0]
        np_A_small = Q_A @ np.diag(np.random.rand(4)) @ Q_A.T
        np_A = np.zeros((10, 10))
        idxs = np.array([3, 6, 8, 9], dtype=np.int32)
        idxs1_blk, idxs2_blk = np.meshgrid(idxs, idxs)
        np_A[idxs1_blk, idxs2_blk] = np_A_small

        tmp = np.random.randn(10, 10)
        Q_B = np.linalg.qr(tmp)[0]
        np_B = Q_B @ np.diag(np.random.rand(10)) @ Q_B.T

        A = redner.double_ptr(0)
        B = redner.double_ptr(0)
        eigvect = redner.double_ptr(0)
        A.allocate(100)
        B.allocate(100)
        eigvect.allocate(10)

        for i in range(10):
            for j in range(10):
                A.set_index(i * 10 + j, np_A[i, j])
                B.set_index(i * 10 + j, np_B[i, j])

        redner.solve_gen_eig_py(A, B, eigvect)
        res_eigvect = np.zeros(10)
        for i in range(10):
            res_eigvect[i] = eigvect.get_index(i)

        eigvals, eigvects = scipy.linalg.eig(np_A, np_B)
        idxes = np.argsort(eigvals)
        np_res = eigvects[:, idxes[-1]]
        np_res_2 = eigvects[:, idxes[-2]]

        A.destroy_array()
        B.destroy_array()
        eigvect.destroy_array()

        res_eigvect *= np.sign(res_eigvect[0])
        np_res *= np.sign(np_res[0])
        np_res_2 *= np.sign(np_res_2[0])
        
        assert np.allclose(res_eigvect, np_res, atol=1e-5) or np.allclose(res_eigvect, np_res_2, atol=1e-5)



def test_compute_LU():
    np.random.seed(0)
    for k in range(100):
        np_A = np.random.randn(10, 10)
        np_A = np_A + np_A.T

        A = redner.double_ptr(0)
        A.allocate(100)
        L = redner.double_ptr(0)
        L.allocate(100)
        U = redner.double_ptr(0)
        U.allocate(100)
        P = redner.int_ptr(0)
        P.allocate(10)

        for i in range(10):
            for j in range(10):
                A.set_index(i * 10 + j, np_A[i, j])
        
        redner.compute_LU_py(A, L, U, P)

        np_L = np.zeros((10, 10))
        np_U = np.zeros((10, 10))
        np_P = np.zeros(10, dtype=np.int32)

        for i in range(10):
            for j in range(10):
                np_L[i, j] = L.get_index(i * 10 + j)
                np_U[i, j] = U.get_index(i * 10 + j)
            np_P[i] = P.get_index(i)

        np_P_matrix = np.eye(10)[np_P]


        test = np_P_matrix.T @ np_L @ np_U

        assert np.allclose(test, np_A, atol=1e-5)

def test_solve_LU():
    np.random.seed(0)
    for k in range(100):
        np_A = np.random.randn(10, 10)
        np_A = np_A + np_A.T

        np_rhs = np.random.randn(10)

        A = redner.double_ptr(0)
        A.allocate(100)
        L = redner.double_ptr(0)
        L.allocate(100)
        U = redner.double_ptr(0)
        U.allocate(100)
        P = redner.int_ptr(0)
        P.allocate(10)
        rhs = redner.double_ptr(0)
        rhs.allocate(10)
        res = redner.double_ptr(0)
        res.allocate(10)

        for i in range(10):
            for j in range(10):
                A.set_index(i * 10 + j, np_A[i, j])
            rhs.set_index(i, np_rhs[i])

        redner.compute_LU_py(A, L, U, P)
        
        redner.solve_LU_py(L, U, P, rhs, res)

        np_res = np.zeros(10)
        for i in range(10):
            np_res[i] = res.get_index(i)

        gt_res = scipy.linalg.solve(np_A, np_rhs)

        assert np.allclose(gt_res, np_res, atol=1e-5)


def test_quadric_fit():
    np.random.seed(0)
    for k in range(10):
        np_points = np.random.randn(100, 4)
        np_points = np_points / np.linalg.norm(np_points[:, 0:3], axis=1)[:,np.newaxis]

        with open('./points.txt', 'w') as f:
            for i in range(100):
                f.write(f'{np_points[i, 0]} {np_points[i, 1]} {np_points[i, 2]} {np_points[i, 3]}\n')

        quadric = redner.fit_quadric_file_input('./points.txt')

        res_matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                res_matrix[i, j] = quadric.matrix(i, j)

        idxs_0 = [0] * 4 + [1] * 3 + [2] * 2 + [3]
        idxs_1 = [0, 1, 2, 3] + [1, 2, 3] + [2, 3] + [3]

        tmp = np.einsum('ki, kj -> kij', np_points, np_points)

        l = np.zeros((len(np_points), 10))
        l[:, 0] = tmp[:, 0, 0]          # = p_x^2
        l[:, 1:4] = 2 * tmp[:, 0, 1:4]  # = p_x * p_y, p_x * p_z, p_x * p_w
        l[:, 4] = tmp[:, 1, 1]
        l[:, 5:7] = 2 * tmp[:, 1, 2:4]
        l[:, 7] = tmp[:, 2, 2]
        l[:, 8] = 2 * tmp[:, 2, 3]
        l[:, 9] = tmp[:, 3, 3]

        lx = np.zeros((len(np_points), 10))
        lx[:, 3] = np_points[:, 0]
        lx[:, 6] = np_points[:, 1]
        lx[:, 8] = np_points[:, 2]
        lx[:, 9] = np_points[:, 3]

        M = np.sum(np.einsum('ki, kj -> kij', l, l), axis=0)
        N = np.sum(np.einsum('ki, kj -> kij', lx, lx), axis=0)

        eigvals, eigvects = scipy.linalg.eig(M, N)
        
        idxes = np.argsort(np.real(eigvals))
        Q_coef = eigvects[:, idxes[0]]
        Q = np.zeros((4, 4))
        Q[idxs_0, idxs_1] = Q_coef[0:10]
        Q[idxs_1, idxs_0] = Q_coef[0:10]

        Q /= np.sign(Q[0, 0])
        res_matrix /= np.sign(res_matrix[0, 0])

        assert np.allclose(Q, res_matrix, atol=1e-5)


