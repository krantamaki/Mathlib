#include "declare_lalib.h"

#define BASE_TOL 0.000001
#define MAX_ITER 1000


/*

TODO: Description

*/

template<class Matrix> Matrix jacobiSolve(Matrix A, Matrix x_0, Matrix b, int max_iter=MAX_ITER, double tol = BASE_TOL) {
    if (A.nrows() != x_0.nrows() || A.ncols() != b.ncols()) {
        throw std::invalid_argument("Improper dimensions!");
    }

    if (A.nrows() != A.ncols()) {
        throw std::invalid_argument("Coefficient matrix must be symmetric!");
    }

    Matrix x_k = Matrix(x_0);

    for (int iter = 0; iter < max_iter; iter++) {
        Matrix x_temp = Matrix(1, A.ncols());

        #pragma omp parallel for schedule(dynamic, 1)
        for (int row = 0; row < A.nrows(); row++) {
            double x_i = 0.0;
            for (int col = 0; col < A.ncols(); col++) {
                x_i += A(row, col) * x_k(1, col);
            }

            double a_ii = A(row, row);
            if (a_ii != 0.0) {
                x_i = (b(row, 1) - x_i) / a_ii;
            }
            else {
                throw std::invalid_argument("Coefficient matrix must have a non-zero diagonal!");
            }

            x_temp.place(row, 1, x_i);
        }

        x_k = x_temp;

        if ((A.matmul(x_k) - b).norm() < tol) {
            return x_k;
        }
    }

    std::cout << "\nWARNING: Jacobi method did not converge to wanted tolerance!" << "\n\n";
    return x_k;
}





