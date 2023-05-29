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


// Generally useful functions

// Function for "dividing up" two integers
inline int _ceil(int a, int b) {
    return (a + b - 1) / b;
}


// STATIONARY SOLVERS
template<class Matrix> Matrix jacobiSolve(Matrix A, Matrix x_0, Matrix b, int max_iter=MAX_ITER, double tol = BASE_TOL);

#endif
