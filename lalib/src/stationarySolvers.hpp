#ifndef STATIONARY_SOLVERS_HPP
#define STATIONARY_SOLVERS_HPP


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
#ifndef OMEGA
#define OMEGA 0.75
#endif
#ifndef PRINT_INTERVAL
#define PRINT_INTERVAL 20
#endif


/*
  Stationary methods are iterative methods that can be expressed as [1]:
    
    x^{(k)} = Ax^{(k-1)} + c

  where A and c remain constant throughout the iterations. Some of these methods are
  implemented here. Note that as the implementations are of template form, none of them are
  parallelized as while e.g. the Jacobi method could trivially be made to run in parallel 
  there might end up being memory races for sparse matrices.


  References:

  [1] Barrett, Richard, et al. Templates for the solution of linear systems: building blocks for iterative methods. Society for Industrial and Applied Mathematics, 1994.
*/


namespace lalib {

  
  /*
    Jacobi method solves a system of linear equations by looking at the equations in isolation. That is when
    solving for x_i all other values in x are held constant giving [1]:
   
      x_i = (b_i - sum_{j != i} a_{i, j} * x_j) / a_{i, i}

    This forms an iterative process:
    
      x_i^{(k)} = (b_i - sum_{j != i} a_{i, j} * x_j^{(k-1)}) / a_{i, i}

    Which is implemented below
  */

  template<class Matrix, class Vector> Vector jacobiSolve(const Matrix& A, const Vector& x_0, const Vector& b, int max_iter=MAX_ITER, double tol=BASE_TOL) {
    
    if (A.nrows() != x_0.len() || A.nrows() != b.len()) {
      throw std::invalid_argument(_formErrorMsg("Improper dimensions!", __FILE__, __func__, __LINE__));
    }

    if (A.nrows() != A.ncols()) {
      throw std::invalid_argument(_formErrorMsg("Coefficient matrix must be square!", __FILE__, __func__, __LINE__));
    }

    Vector x_k = Vector(x_0);

    for (int iter = 0; iter < max_iter; iter++) {
      Vector x_temp = Vector(A.nrows());

      // Go over all i in {0, ..., nrows}
      for (int row = 0; row < A.nrows(); row++) {

	// Compute the sum s = sum_{j != i} a_{i, j} * x_j
	double s = 0.0;
	for (int col = 0; col < A.ncols(); col++) {
	  if (col == row) continue;
	  s += A(row, col) * x_k(col);
	}

	// Compute the new value of x_i = (b_i - s) / a_{i, i}
	double a_ii = A(row, row);
	if (a_ii != 0.0) {
	  s = (b(row) - s) / a_ii;
	}
	else {
	  throw std::invalid_argument(_formErrorMsg("Coefficient matrix must have a non-zero diagonal!", __FILE__, __func__, __LINE__));
	}

	// Update the vector x
	x_temp.place(row, s);
      }

      x_k = x_temp;

      double norm = (A.matmul(x_k) - b).norm();

      if (norm < tol) {
	return x_k;
      }

      if (iter % PRINT_INTERVAL == 0 && verbosity() >= 3) {
	std::cout << _formIterMsg(__func__, iter, norm);
      }
    }

    if (verbosity() >= 2) {
      std::cout << _formWarningMsg("Solver did not converge to the wanted tolerance!", __func__);
    }
    
    return x_k;
  }


  /*
    Gauss-Seidel method is a natural extension for the Jacobi method where the linear equations are
    not considered in isolation, but in sequence. Then the values already computed in the same 
    iteration can be used instantly. This gives an iterative process of form [1]:

      x_i^{(k)} = (b_i - sum_{j < i} a_{i, j} * x_j^{(k)} - sum_{j > i} a_{i, j} * x_j^{(k-1)}) / a_{i, i}

    Note that unlike the Jacobi method, the Gauss-Seidel method is not trivially parallelizable due to its 
    fundamentally serial nature.
  */

  template<class Matrix, class Vector> Vector gsSolve(const Matrix& A, const Vector& x_0, const Vector& b, int max_iter=MAX_ITER, double tol=BASE_TOL) {
    
    if (A.nrows() != x_0.len() || A.nrows() != b.len()) {
      throw std::invalid_argument(_formErrorMsg("Improper dimensions!", __FILE__, __func__, __LINE__));
    }

    if (A.nrows() != A.ncols()) {
      throw std::invalid_argument(_formErrorMsg("Coefficient matrix must be square!", __FILE__, __func__, __LINE__));
    }

    Vector x_k = Vector(x_0);

    for (int iter = 0; iter < max_iter; iter++) {

      // Go over all i in {0, ..., nrows}
      for (int row = 0; row < A.nrows(); row++) {

	// Compute the sum s_l = sum_{j < i} a_{i, j} * x_j^{(k)}
	double s_l = 0.0;
	for (int col = 0; col < row; col++) {
	  s_l += A(row, col) * x_k(col);
	}

	// Compute the sum s_g = sum_{j > i} a_{i, j} * x_j^{(k-1)}
	double s_g = 0.0;
	for (int col = row + 1; col < A.ncols(); col++) {
	  s_g += A(row, col) * x_k(col);
	}

	double s = s_l + s_g;

	// Compute the new value of x_i = (b_i - s) / a_{i, i}
	double a_ii = A(row, row);
	if (a_ii != 0.0) {
	  s = (b(row) - s) / a_ii;
	}
	else {
	  throw std::invalid_argument(_formErrorMsg("Coefficient matrix must have a non-zero diagonal!", __FILE__, __func__, __LINE__));
	}

	// Update the vector x
	x_k.place(row, s);
      }

      double norm = (A.matmul(x_k) - b).norm();

      if (norm < tol) {
	return x_k;
      }

      if (iter % PRINT_INTERVAL == 0 && verbosity() >= 3) {
	std::cout << _formIterMsg(__func__, iter, norm);
      }
    }

    if (verbosity() >= 2) {
      std::cout << _formWarningMsg("Solver did not converge to the wanted tolerance!", __func__);
    }
    
    return x_k;
  }


  /*
    Successive overrelaxation method builds on the Gauss-Seidel method by taking a weighted average 
    between the current and previous iterate [1]: 

      s = (b_i - sum_{j < i} a_{i, j} * x_j^{(k)} - sum_{j > i} a_{i, j} * x_j^{(k-1)}) / a_{i, i}
      x_i^{(k)} = x_i^{(k-1)} + w(s - x_i^{(k-1)})

    where w is the extrapolation factor. There is no closed form way for choosing the optimal w, but
    for the method to converge it should be chosen from interval (0, 2). It is good to note that if w = 1
    SOR methods is equivalent to Gauss-Seidel method. Generally, if w < 1 convergence is improved, but with
    the drawback of greater number of required iterations. Logically then, if w > 1 convergence might be 
    less certain, but required number of iterations is decreased.
  */

  template<class Matrix, class Vector> Vector sorSolve(const Matrix& A, const Vector& x_0, const Vector& b, int max_iter=MAX_ITER, double tol=BASE_TOL, double w=OMEGA) {
    
    if (A.nrows() != x_0.len() || A.nrows() != b.len()) {
      throw std::invalid_argument(_formErrorMsg("Improper dimensions!", __FILE__, __func__, __LINE__));
    }

    if (A.nrows() != A.ncols()) {
      throw std::invalid_argument(_formErrorMsg("Coefficient matrix must be square!", __FILE__, __func__, __LINE__));
    }

    Vector x_k = Vector(x_0);

    for (int iter = 0; iter < max_iter; iter++) {

      // Go over all i in {0, ..., nrows}
      for (int row = 0; row < A.nrows(); row++) {

	// Compute the sum s_l = sum_{j < i} a_{i, j} * x_j^{(k)}
	double s_l = 0.0;
	for (int col = 0; col < row; col++) {
	  s_l += A(row, col) * x_k(col);
	}

	// Compute the sum s_g = sum_{j > i} a_{i, j} * x_j^{(k-1)}
	double s_g = 0.0;
	for (int col = row + 1; col < A.ncols(); col++) {
	  s_g += A(row, col) * x_k(col);
	}

	// Compute the new value of x_i = x_i^{(k-1)} + w(s - x_i^{(k-1)})
	double a_ii = A(row, row);
	double x_i = x_k(row);
	double s = (b(row) - s_l - s_g) / a_ii;
	if (a_ii != 0.0) {
	  x_i = x_i + w * (s - x_i);
	}
	else {
	  throw std::invalid_argument(_formErrorMsg("Coefficient matrix must have a non-zero diagonal!", __FILE__, __func__, __LINE__));
	}

	// Update the vector x
	x_k.place(row, x_i);
      }

      double norm = (A.matmul(x_k) - b).norm();
      
      if (norm < tol) {
	return x_k;
      }

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
