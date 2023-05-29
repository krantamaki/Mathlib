#ifndef LALIB_HPP
#define LALIB_HPP


/*
This is a general linear algebra library for C++. Main functionality consist of different
matrix classes (currently only DenseMatrix is implemented) which override the basic math operators
(+, -, ...) to function as element-wise operations equivalent to Matlab's .* etc and contain methods
like matmul for computing the common matrix operations. Additionally, there are some general template 
functions that work independent of the matrix type (currently only DenseMatrix).

The methods and functions are made as optimal as feasible with parallelization utilizing multi-threading
and SIMD commands as well as making sure memory calls are linear.
*/


#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <float.h>
#include <time.h>
#include <math.h>
#include <omp.h>


// Define a vector for SIMD commands. Currently set up to hold 4 doubles (for 256 bit vector registers)

#define VECT_ELEMS 4
typedef double vect_t __attribute__ ((__vector_size__ (VECT_ELEMS * sizeof(double))));

#define BASE_TOL 0.000001
#define MAX_ITER 1000

// Generally useful functions

// Function for "dividing up" two integers
inline int _ceil(int a, int b) {
    return (a + b - 1) / b;
}


// STATIONARY SOLVERS

template<class Matrix> Matrix jacobiSolve(Matrix& A, Matrix& x_0, Matrix& b, int max_iter=MAX_ITER, double tol=BASE_TOL) {
    if (A.nrows() != x_0.nrows() || A.nrows() != b.nrows()) {
        throw std::invalid_argument("Improper dimensions!");
    }

    if (A.nrows() != A.ncols()) {
        throw std::invalid_argument("Coefficient matrix must be symmetric!");
    }

    Matrix x_k = Matrix(x_0);

    for (int iter = 0; iter < max_iter; iter++) {
        Matrix x_temp = Matrix(A.nrows(), 1);

        #pragma omp parallel for schedule(dynamic, 1)
        for (int row = 0; row < A.nrows(); row++) {
            double x_i = 0.0;
            for (int col = 0; col < A.ncols(); col++) {
                x_i += A(row, col) * x_k(row, 1);
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


#endif
