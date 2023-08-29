#ifndef NONSTATIONARY_SOLVERS_HPP
#define NONSTATIONARY_SOLVERS_HPP


#include "declare_lalib.hpp"


#ifndef BASE_TOL
#define BASE_TOL 0.000001
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
    Conjugate gradient method doesn't solve the system of linear equations Ax = b  explicitly, but finds the minimizer
    for an auxiliary problem x^T Ax - x^T b. The equality between the solutions of these problems holds 
    only in the case that A is symmetric and positive definite. If that is the case we can note that the
    auxiliary problem is quadratic in nature and thus can be solved as an unconstrained nonlinear optimization 
    problem with gradient descent. Furthermore, if the search directions are chosen to be A-orthogonal we get 
    the traditional formulation for conjugate gradient method [2].
  */

  template<class Matrix, class Vector> Vector cgSolve(const Matrix& A, const Vector& x_0, const Vector& b, int max_iter=MAX_ITER, double tol=BASE_TOL, bool check_symmetric=CHECK_SYMMETRIC) {
    
    if (A.nrows() != x_0.len() || A.nrows() != b.len()) {
      throw std::invalid_argument(_formErrorMsg("Improper dimensions!", __FILE__, __func__, __LINE__));
    }

    if (A.nrows() != A.ncols()) {
      throw std::invalid_argument(_formErrorMsg("Coefficient matrix must be square!", __FILE__, __func__, __LINE__));
    }

    if (check_symmetric) {
      for (int row = 0; row < A.nrows(); row++) {
	for (int col = 0; col < A.ncols(); col++) {
	  if (A(row, col) != A(col, row)) {
	    throw std::invalid_argument(_formErrorMsg("Coefficient matrix must be symmetric!", __FILE__, __func__, __LINE__));
	  }
	}
      }
    }

    Vector x_k = Vector(x_0);
    Vector r = b - A.matmul(x_k);
    Vector p = Vector(r);

    double rsold = r.dot(r);

    for (int iter = 0; iter < max_iter; iter++) {

      Vector Ap = A.matmul(p);
      
      double alpha = rsold / (p.dot(Ap));

      x_k += alpha * p;
      r -= alpha * Ap;

      double norm = r.norm();
      double rsnew = norm * norm;
      
      if (norm < tol) {
	return x_k;
      }

      p *= (rsnew / rsold);
      p += r;
      
      rsold = rsnew;

      if (iter % PRINT_INTERVAL == 0 && verbosity() >= 3) {
	std::cout << _formIterMsg(__func__, iter, norm);
      }
    }

    if (verbosity() >= 2) {
      std::cout << _formWarningMsg("Solver did not converge to the wanted tolerance!", __func__);
    }
    
    return x_k;
  }
  
}

#endif
