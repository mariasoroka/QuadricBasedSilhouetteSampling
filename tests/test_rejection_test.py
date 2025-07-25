import pyredner
import redner
import torch
import scipy
import numpy as np

# Sphere mesh
def test_rejection_test_1():
    #l 3 n 2
    np_Q0 = np.array([[-0.3053997452036104, -0.176468043367048, 0.0433218945385722, -0.2898525432252712], 
                      [-0.176468043367048, 0.20910632413636981, 0.0899814928903975, 0.47555066105742627], 
                      [0.0433218945385722, 0.0899814928903975, -0.18351239649062462, 0.11826377375899261], 
                      [-0.2898525432252712, 0.47555066105742627, 0.11826377375899261, 0.6837749834678876]])
    np_Q1 = np.array([[-0.32133882296676924, -0.15553831094232215, 0.06174248533470212, -0.28853825527138366], 
                      [-0.15553831094232215, 0.18162332255499064, 0.06579326555396471, 0.47382485887409753], 
                      [0.06174248533470212, 0.06579326555396471, -0.20480084054745437, 0.11674486777543544], 
                      [-0.28853825527138366, 0.47382485887409753, 0.11674486777543544, 0.6836666112740003]])

    np_m = np.array([[6.4266721778802745e-18, -0.8683883246268973, 1.6607657278708303e-16, -0.4958847826377524], 
                     [-8.868463886175335e-18, -0.3718331221615655, 0.6616217736662758, 0.6511503343117268], 
                     [-0.07116793328438986, -0.3272562519573304, -0.7479364072043868, 0.573087778272284], 
                     [0.9974643478701525, -0.023349356952886927, -0.05336440188819389, 0.04088915343918686]])

    np_p_min = np.array([-0.35649348, -0.41473308, -0.76998413])
    np_p_max = np.array([0.30753324, 0.76037743, 0.75563012])
    p_min = redner.Vector3(np_p_min[0], np_p_min[1], np_p_min[2])
    p_max = redner.Vector3(np_p_max[0], np_p_max[1], np_p_max[2])
    dual_bounds = redner.AABB3(p_min, p_max)

    m0 = redner.Matrix4x4(np_Q0[0, 0], np_Q0[0, 1], np_Q0[0, 2], np_Q0[0, 3], \
                            np_Q0[1, 0], np_Q0[1, 1], np_Q0[1, 2], np_Q0[1, 3], \
                            np_Q0[2, 0], np_Q0[2, 1], np_Q0[2, 2], np_Q0[2, 3], \
                            np_Q0[3, 0], np_Q0[3, 1], np_Q0[3, 2], np_Q0[3, 3])
    
    m1 = redner.Matrix4x4(np_Q1[0, 0], np_Q1[0, 1], np_Q1[0, 2], np_Q1[0, 3], \
                            np_Q1[1, 0], np_Q1[1, 1], np_Q1[1, 2], np_Q1[1, 3], \
                            np_Q1[2, 0], np_Q1[2, 1], np_Q1[2, 2], np_Q1[2, 3], \
                            np_Q1[3, 0], np_Q1[3, 1], np_Q1[3, 2], np_Q1[3, 3])

    Q0 = redner.Quadric(m0)
    Q1 = redner.Quadric(m1)
    quad_pair = redner.QuadricPair(Q0, Q1)

    m = redner.Matrix4x4(np_m[0, 0], np_m[0, 1], np_m[0, 2], np_m[0, 3], \
                            np_m[1, 0], np_m[1, 1], np_m[1, 2], np_m[1, 3], \
                            np_m[2, 0], np_m[2, 1], np_m[2, 2], np_m[2, 3], \
                            np_m[3, 0], np_m[3, 1], np_m[3, 2], np_m[3, 3])

    np_p_1 = np.array([25.82069756090641, 1.831420206709911, -23.103496772678277])
    rej_1 = False

    np_p_2 = np.array([[25.82069756090641, 23.393193406494035, -23.103496772678277]])
    rej_2 = True

    #10, 7, 1
    np_p_3 = np.array([1.563702711149265, -6.254244743209135, -23.103496772678277])
    rej_3 = False

    np_p_4 = np.array([-14.60762718868883, -6.254244743209135, -23.103496772678277])
    rej_4 = True

    #gt can reject, quadrics can't
    #18, 11, 6
    np_p_5 = np.array([23.12547591093339, 4.526641856682929, -9.627388522813195, 1.0])
    rej_5 = True

    # reject because no intersection with bbox
    # 14, 10, 6
    np_p_6 = np.array([12.344589311041332, 1.831420206709911, -9.627388522813195, 1.0])
    rej_6 = False

    # reject because no intersection with quadric
    # 14, 10, 6
    np_p_7 = np.array([9.649367661068311, 1.831420206709911, -9.627388522813195, 1.0])
    rej_7 = False

    # reject because no intersection with quadric
    # 9, 2, 8
    np_p_8 = np.array([-1.1315189388237528, -19.730352993074217, -4.236945222867163, 1.0])
    rej_8 = False

    # reject because no intersection with quadric
    # 8, 2, 8
    np_p_9 = np.array([-3.826740588796767, -19.730352993074217, -4.236945222867163, 1.0])
    rej_9 = False

    p_1 = redner.Vector3(np_p_1[0], np_p_1[1], np_p_1[2])
    assert redner.rejection_test_py(quad_pair, m, dual_bounds, p_1) == rej_1

    p_2 = redner.Vector3(np_p_2[0, 0], np_p_2[0, 1], np_p_2[0, 2])
    assert redner.rejection_test_py(quad_pair, m, dual_bounds, p_2) == rej_2

    p_3 = redner.Vector3(np_p_3[0], np_p_3[1], np_p_3[2])
    assert redner.rejection_test_py(quad_pair, m, dual_bounds, p_3) == rej_3

    p_4 = redner.Vector3(np_p_4[0], np_p_4[1], np_p_4[2])
    assert redner.rejection_test_py(quad_pair, m, dual_bounds, p_4) == rej_4

    p_5 = redner.Vector3(np_p_5[0], np_p_5[1], np_p_5[2])
    assert redner.rejection_test_py(quad_pair, m, dual_bounds, p_5) == rej_5

    p_6 = redner.Vector3(np_p_6[0], np_p_6[1], np_p_6[2])
    assert redner.rejection_test_py(quad_pair, m, dual_bounds, p_6) == rej_6

    p_7 = redner.Vector3(np_p_7[0], np_p_7[1], np_p_7[2])
    assert redner.rejection_test_py(quad_pair, m, dual_bounds, p_7) == rej_7

    p_8 = redner.Vector3(np_p_8[0], np_p_8[1], np_p_8[2])
    assert redner.rejection_test_py(quad_pair, m, dual_bounds, p_8) == rej_8

    p_9 = redner.Vector3(np_p_9[0], np_p_9[1], np_p_9[2])
    assert redner.rejection_test_py(quad_pair, m, dual_bounds, p_9) == rej_9


# Bob mesh
def test_rejection_test_2():
    #l 5 n 1
    np_Q0 = np.array([[-0.041082021243850464, -0.05756578745157025, 0.047490367340799144, -0.10437594208847893], 
                      [-0.05756578745157025, 0.29768494490757214, -0.11518773857806391, 0.49630717621444886], 
                      [0.047490367340799144, -0.11518773857806391, -0.009334648398587225, -0.16895238444992675], 
                      [-0.10437594208847893, 0.49630717621444886, -0.16895238444992675, 0.7780446308222037]])
    np_Q1 = np.array([[-0.04156854048974695, -0.057182337180241335, 0.04714806659700321, -0.10500811998131847], 
                      [-0.057182337180241335, 0.29738272846201064, -0.11491795412421507, 0.4968054274472331], 
                      [0.04714806659700321, -0.11491795412421507, -0.00957548121318475, -0.1693971664309861], 
                      [-0.10500811998131847, 0.4968054274472331, -0.1693971664309861, 0.7772231855394222]])

    np_m = np.array([[-4.1985264814122333e-17, -0.8585806134184404, -1.6794105925648933e-16, -0.5126785837754635], 
                     [1.1215133912614608e-17, -0.24127833268387824, -0.8823343995866975, 0.4040677832780909], 
                     [-0.8793670831000964, 0.2153858209665453, -0.2240845270469317, -0.3607057055616787], 
                     [0.47614444568852027, 0.39778517388895995, -0.41385037398090535, -0.6661691153378145]])

    np_p_min = np.array([-0.47116936, -1.02764677, -0.7489897])
    np_p_max = np.array([0.79112551, 1.0276467,  0.74898957])
    p_min = redner.Vector3(np_p_min[0], np_p_min[1], np_p_min[2])
    p_max = redner.Vector3(np_p_max[0], np_p_max[1], np_p_max[2])
    dual_bounds = redner.AABB3(p_min, p_max)


    m0 = redner.Matrix4x4(np_Q0[0, 0], np_Q0[0, 1], np_Q0[0, 2], np_Q0[0, 3], \
                            np_Q0[1, 0], np_Q0[1, 1], np_Q0[1, 2], np_Q0[1, 3], \
                            np_Q0[2, 0], np_Q0[2, 1], np_Q0[2, 2], np_Q0[2, 3], \
                            np_Q0[3, 0], np_Q0[3, 1], np_Q0[3, 2], np_Q0[3, 3])
    
    m1 = redner.Matrix4x4(np_Q1[0, 0], np_Q1[0, 1], np_Q1[0, 2], np_Q1[0, 3], \
                            np_Q1[1, 0], np_Q1[1, 1], np_Q1[1, 2], np_Q1[1, 3], \
                            np_Q1[2, 0], np_Q1[2, 1], np_Q1[2, 2], np_Q1[2, 3], \
                            np_Q1[3, 0], np_Q1[3, 1], np_Q1[3, 2], np_Q1[3, 3])

    Q0 = redner.Quadric(m0)
    Q1 = redner.Quadric(m1)
    quad_pair = redner.QuadricPair(Q0, Q1)

    m = redner.Matrix4x4(np_m[0, 0], np_m[0, 1], np_m[0, 2], np_m[0, 3], \
                            np_m[1, 0], np_m[1, 1], np_m[1, 2], np_m[1, 3], \
                            np_m[2, 0], np_m[2, 1], np_m[2, 2], np_m[2, 3], \
                            np_m[3, 0], np_m[3, 1], np_m[3, 2], np_m[3, 3])
    
    
    np_p_1 = np.array([-19.858499090138235, 13.405419390452536, 3.1712813298953186])
    rej_1 = True

    np_p_2 = np.array([-19.858499090138235, 1.8628811114712747, 3.1712813298953186])
    rej_2 = True

    np_p_3 = np.array([5.535085123620533, -9.679657167509983, 3.1712813298953186])
    rej_3 = False

    np_p_4 = np.array([12.460608091009291, -9.679657167509983, 3.1712813298953186])
    rej_4 = False

    np_p_5 = np.array([12.460608091009291, 13.405419390452536, 3.1712813298953186])
    rej_5 = True

    p_1 = redner.Vector3(np_p_1[0], np_p_1[1], np_p_1[2])
    assert redner.rejection_test_py(quad_pair, m, dual_bounds, p_1) == rej_1

    p_2 = redner.Vector3(np_p_2[0], np_p_2[1], np_p_2[2])
    assert redner.rejection_test_py(quad_pair, m, dual_bounds, p_2) == rej_2

    p_3 = redner.Vector3(np_p_3[0], np_p_3[1], np_p_3[2])
    assert redner.rejection_test_py(quad_pair, m, dual_bounds, p_3) == rej_3

    p_4 = redner.Vector3(np_p_4[0], np_p_4[1], np_p_4[2])
    assert redner.rejection_test_py(quad_pair, m, dual_bounds, p_4) == rej_4

    p_5 = redner.Vector3(np_p_5[0], np_p_5[1], np_p_5[2])
    assert redner.rejection_test_py(quad_pair, m, dual_bounds, p_5) == rej_5
