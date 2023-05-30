#ifndef STATIONARY_SOLVERS_HPP
#define STATIONARY_SOLVERS_HPP

/*
TODO: DESCRIPTION HERE
*/

#define BASE_TOL 0.000001
#define MAX_ITER 1000


template<class Matrix> Matrix jacobiSolve(const Matrix& A, const Matrix& x_0, const Matrix& b, int max_iter=MAX_ITER, double tol=BASE_TOL) {
    if (A.nrows() != x_0.nrows() || A.nrows() != b.nrows()) {
        throw std::invalid_argument("Improper dimensions!");
    }

    if (A.nrows() != A.ncols()) {
        throw std::invalid_argument("Coefficient matrix must be symmetric!");
    }

    Matrix x_k = Matrix(x_0);

    for (int iter = 0; iter < max_iter; iter++) {
        Matrix x_temp = Matrix(A.nrows(), 1);

        for (int row = 0; row < A.nrows(); row++) {
            double s = 0.0;
            for (int col = 0; col < A.ncols(); col++) {
	        if (col == row) continue;
                s += A(row, col) * x_k(col, 0);
            }

            double a_ii = A(row, row);
            if (a_ii != 0.0) {
                s = (b(row, 0) - s) / a_ii;
            }
            else {
                throw std::invalid_argument("Coefficient matrix must have a non-zero diagonal!");
            }

            x_temp.place(row, 0, s);
        }

        x_k = x_temp;

        if ((A.matmul(x_k) - b).norm() < tol) {
            return x_k;
        }
    }

    std::cout << "\nWARNING: Jacobi method did not converge to wanted tolerance!" << "\n\n";
    return x_k;
}

#endif
