#ifndef NONSTATIONARY_SOLVERS_HPP
#define NONSTATIONARY_SOLVERS_HPP


/*
  Methods that use values that change in each iteration, e.g. the residuals, are 
  nonstationary. Some such methods are implemented here.


  References:

  [1] Barrett, Richard, et al. Templates for the solution of linear systems: building blocks for iterative methods. Society for Industrial and Applied Mathematics, 1994.
  [2] Hannukainen, Antti. Numerical matrix computations. Course lecture notes, 2022.
*/


#define BASE_TOL 0.0001
#define MAX_ITER 1000
#define CHECK_SYMMETRIC 0


namespace lalib {


  /*
    Conjugate gradient method doesn't solve the system of linear equations Ax = b  explicitly, but solves
    an auxiliary problem x^T Ax - x^T b = 0. The equality between the solutions of these problems holds 
    only in the case that A is symmetric and positive definite. If that is the case we can note that the
    auxiliary problem is quadratic in nature and thus can be solved as an unconstrained nonlinear optimization 
    problem with gradient descent. Furthermore, if the search directions are chosen to be A-orthogonal we get 
    the traditional formulation for conjugate gradient method [2].
  */

  template<class Matrix> Matrix cgSolve(const Matrix& A, const Matrix& x_0, const Matrix& b, int max_iter=MAX_ITER, double tol=BASE_TOL, bool check_symmetric=CHECK_SYMMETRIC) {
    
    if (A.nrows() != x_0.nrows() || A.nrows() != b.nrows()) {
      throw std::invalid_argument("Improper dimensions!");
    }

    if (A.nrows() != A.ncols()) {
      throw std::invalid_argument("Coefficient matrix must be square!");
    }

    if (check_symmetric) {
      for (int row = 0; row < A.nrows(); row++) {
	for (int col = 0; col < A.ncols(); col++) {
	  if (A(row, col) != A(col, row)) {
	    throw std::invalid_argument("Coefficient matrix must be symmetric!");
	  }
	}
      }
    }

    Matrix x_k = Matrix(x_0);
    Matrix r = b - A.matmul(x_k);
    Matrix p = Matrix(r);

    for (int iter = 0; iter < max_iter; iter++) {

      Matrix p_T = p.T();
      
      double alpha = (p_T.matmul(r)).asDouble() / (p_T.matmul(A.matmul(p))).asDouble();
      
      x_k += alpha * p;

      Matrix r_tmp = Matrix(r);

      r -= alpha * A.matmul(p);
      p = r + (r.T().matmul(r)).asDouble() / (r_tmp.T().matmul(r_tmp)).asDouble() * p;

      if (r.norm() < tol) {
	return x_k;
      }
    }

    std::cout << "\nWARNING: Conjugate gradient method did not converge to the wanted tolerance!" << "\n\n";
    
    return x_k;
  }
  
}

#endif
