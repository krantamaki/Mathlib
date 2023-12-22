#ifndef NONSTATIONARY_SOLVERS_HPP
#define NONSTATIONARY_SOLVERS_HPP


#include "declare_lalib.hpp"
#include "vector/Vector.hpp"
#include "matrix/Matrix.hpp"


#ifndef BASE_TOL
#define BASE_TOL 0.0000001
#endif
#ifndef MAX_ITER
#define MAX_ITER 1000
#endif
#ifndef CHECK_SYMMETRIC
#define CHECK_SYMMETRIC 0
#endif
#ifndef PRINT_INTERVAL
#define PRINT_INTERVAL 20
#endif


/*
  Methods that use values that change in each iteration, e.g. the residuals, are 
  nonstationary. Some such methods are implemented here.


  References:

  [1] Barrett, Richard, et al. Templates for the solution of linear systems: building blocks for iterative methods. Society for Industrial and Applied Mathematics, 1994.
  [2] Hannukainen, Antti. Numerical matrix computations. Course lecture notes, 2022.
*/


namespace lalib {


  /*
    Conjugate gradient method doesn't solve the system of linear equations Ax = b per se, but finds the minimizer
    for an auxiliary problem x^T Ax - x^T b. The equality between the solutions of these problems holds 
    only in the case that A is symmetric and positive definite. If that is the case we can note that the
    auxiliary problem is quadratic in nature and thus can be solved as an unconstrained nonlinear optimization 
    problem with gradient descent. Furthermore, if the search directions are chosen to be A-orthogonal we get 
    the traditional formulation for conjugate gradient method [2].
  */

  template<class type, bool vectorize, bool sparse> 
  Vector<type, vectorize> cgSolve(const Matrix<type, vectorize, sparse>& A, const Vector<type, vectorize>& x_0, const Vector<type, vectorize>& b, 
                                  int max_iter=MAX_ITER, type tol=BASE_TOL, bool check_symmetric=CHECK_SYMMETRIC) {
    
    if (A.nrows() != x_0.len() || A.nrows() != b.len()) {
      ERROR("Improper dimensions!");
    }

    if (A.nrows() != A.ncols()) {
      ERROR("Coefficient matrix must be square!");
    }

    if (check_symmetric) {
      for (int row = 0; row < A.nrows(); row++) {
	      for (int col = 0; col < A.ncols(); col++) {
	        if (A(row, col) != A(col, row)) {
	          ERROR("Coefficient matrix must be symmetric!");
	        }
	      } 
      }
    }

    Vector x_k = Vector<type, vectorize>(x_0);
    Vector r = b - A.matmul(x_k);
    Vector p = Vector<type, vectorize>(r);

    type rsold = r.dot(r);

    for (int iter = 1; iter <= max_iter; iter++) {

      Vector Ap = A.matmul(p);
      
      type alpha = rsold / (p.dot(Ap));

      x_k += alpha * p;
      r -= alpha * Ap;

      type norm = r.norm();
      type rsnew = norm * norm;
      
      if (norm < tol) {
	      ITER(iter, norm);
	      return x_k;
      }

      type beta = rsnew / rsold;

      p *= beta;
      p += r;
      
      rsold = rsnew;

      if (iter % PRINT_INTERVAL == 0) {
        ITER(iter, norm);
      }
    }

    WARNING("Solver did not converge to the wanted tolerance!");

    
    return x_k;
  }


  /*
    Conjugate gradient on the normal equations

    Conjugate gradient (CG) method requires that the coefficient matrix A is symmetric and positive definite.
    While there is many variants of the CG method to overcome this challenge the simplest would be 
    conjugate gradient on the normal equations (CGNR). CGNR doesn't solve the system Ax = b, but an
    equivalent one A^T Ax = A^T b. It can be shown that for any matrix A the corresponding matrix 
    A^T A is symmetric and positive definite and thus solvable via CG method.

    The issue with CGNR is that the number of iterations taken by the CG method is bounded by the condition
    number of the matrix and the condition number of the normal equations would be the square of the original
    ones. Thus, the algorithm can be very slow to converge. 
  */
  template<class type, bool vectorize, bool sparse> 
  Vector<type, vectorize> cgnrSolve(const Matrix<type, vectorize, sparse>& A, const Vector<type, vectorize>& x_0, const Vector<type, vectorize>& b, 
                                    int max_iter=MAX_ITER, type tol=BASE_TOL) {

    if (A.nrows() != x_0.len() || A.nrows() != b.len()) {
      ERROR("Improper dimensions!");
    }

    Matrix A_T = A.T();

    Vector x_k = Vector<type, vectorize>(x_0);
    Vector r = A_T.matmul(b) - A_T.matmul(A.matmul(x_k));
    Vector p = Vector<type, vectorize>(r);

    type rsold = r.dot(r);

    for (int iter = 1; iter <= max_iter; iter++) {

      Vector Ap = A.matmul(p);
      
      type alpha = rsold / (Ap.dot(Ap));

      x_k += alpha * p;
      r -= alpha * A_T.matmul(Ap);

      type norm = r.norm();
      type rsnew = norm * norm;
      
      if (norm < tol) {
	      ITER(iter, norm);
	      return x_k;
      }

      type beta = rsnew / rsold;

      p *= beta;
      p += r;
      
      rsold = rsnew;

      if (iter % (PRINT_INTERVAL * 5) == 0) {
        ITER(iter, norm);
      }
    }

    WARNING("Solver did not converge to the wanted tolerance!");

    
    return x_k;
  }

}


#endif
