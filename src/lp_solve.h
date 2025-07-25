#pragma once

#include "redner.h"
#include "matrix.h"
#include "vector.h"
#include "ptr.h"
#include "thrust_utils.h"
#include "aabb.h"

#include <ClpSimplex.hpp>
#include <ClpDualRowSteepest.hpp>

// Use Clp to solve the constrained optimization problem from Eq. (6) in the 
// paper "Quadric-Based Silhouette Sampling for Differentiable Rendering"
// Vector of optimized variables is x = (\rho, D_x, D_y, D_z). D_w is assumed to be 1.
template <typename T>
DEVICE 
TVector4<T> solve_constrained_lp(const BufferView<TVector4<T>> &dual_planes) {

    ClpSimplex  model;
    model.setLogLevel(0);
    int N_constr = dual_planes.size();
    model.resize(N_constr, 4);

    double *obj = model.objective();
    obj[0] = -1; // The objective is to maximize \rho
    for (int i = 1; i < 4; i++) {
        obj[i] = 0; // D_x, D_y and D_z do not contribute to the objective function
    }

    int numberElements = N_constr * 4;
    double * elements = new double[numberElements];
               
    CoinBigIndex * starts = new CoinBigIndex [4 + 1];
    int * rows = new int[numberElements];
    int * lengths = new int[4];

    double * columnUpper = model.columnUpper();
    double * columnLower = model.columnLower();
    double * rowUpper = model.rowUpper();
    double * rowLower = model.rowLower();

    // Inequality constraints. Only lower bounds are set.
    for (int k = 0; k < N_constr; k++) {
        rowUpper[k] = COIN_DBL_MAX; 
        rowLower[k] = -dual_planes[k][3];
    }
    for(int k = 0; k < 4; k++) {
        columnLower[k] = -COIN_DBL_MAX;
        columnUpper[k] = COIN_DBL_MAX;
    }

    CoinBigIndex put = 0;
    for (int k = 0; k < 4; k++) {
        starts[k] = N_constr * k;
        lengths[k] = N_constr;
        for (int i = 0; i < N_constr; i++) {
            rows[put] = i;
            if (k == 0) {
                elements[put] = -1;
            } else {
                elements[put] = dual_planes[i][k - 1];
            }
            put++;
        }
    }
    starts[4] = put;

    CoinPackedMatrix * matrix = new CoinPackedMatrix(true, 0.0, 0.0);
    matrix->assignMatrix(true, N_constr, 4, numberElements,
                        elements, rows, starts, lengths);
    
    ClpPackedMatrix * clpMatrix = new ClpPackedMatrix(matrix);
    model.replaceMatrix(clpMatrix, true);

    model.dual();

    int status = model.status();
    if (status != 0) {
        return TVector4<T>(0, 0, 0, 0);
    }
    double objval = model.objectiveValue();
    if (objval > 0) {
        return TVector4<T>(0, 0, 0, 0);
    }

    const double * columnPrimal = model.getColSolution();
    TVector4<T> result;
    for (int i = 1; i < 4; i++) {
        result[i - 1] = columnPrimal[i];
    }
    result[3] = 1;

    return result;
}

template <typename T>
DEVICE
TVector4<T> test_solve_lp(ptr<TVector4<T>> dual_planes, int size) {
    TVector4<T> result = solve_constrained_lp(BufferView<TVector4<T>>(dual_planes.get(), size));
    return result;
}