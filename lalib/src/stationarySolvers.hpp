#ifndef STATIONARY_SOLVERS_HPP
#define STATIONARY_SOLVERS_HPP


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


#define BASE_TOL 0.0001
#define MAX_ITER 1000


namespace lalib {

  
  /*
    Jacobi method solves a system of linear equations by looking at the equations in isolation. That is when
    solving for x_i all other values in x are held constant giving [1]:
   
      x_i = (b_i - sum_{j != i} a_{i, j} * x_j) / a_{i, i}

    This forms an iterative process:
    
      x_i^{(k)} = (b_i - sum_{j != i} a_{i, j} * x_j^{(k-1)}) / a_{i, i}

    Which is implemented below
  */

  template<class Matrix> Matrix jacobiSolve(const Matrix& A, const Matrix& x_0, const Matrix& b, int max_iter=MAX_ITER, double tol=BASE_TOL) {
    if (A.nrows() != x_0.nrows() || A.nrows() != b.nrows()) {
      throw std::invalid_argument("Improper dimensions!");
    }

    if (A.nrows() != A.ncols()) {
      throw std::invalid_argument("Coefficient matrix must be square!");
    }

    Matrix x_k = Matrix(x_0);

    for (int iter = 0; iter < max_iter; iter++) {
      Matrix x_temp = Matrix(A.nrows(), 1);

      // Go over all i in {0, ..., nrows}
      for (int row = 0; row < A.nrows(); row++) {

	// Compute the sum s = sum_{j != i} a_{i, j} * x_j
	double s = 0.0;
	for (int col = 0; col < A.ncols(); col++) {
	  if (col == row) continue;
	  s += A(row, col) * x_k(col, 0);
	}

	// Compute the new value of x_i = (b_i - s) / a_{i, i}
	double a_ii = A(row, row);
	if (a_ii != 0.0) {
	  s = (b(row, 0) - s) / a_ii;
	}
	else {
	  throw std::invalid_argument("Coefficient matrix must have a non-zero diagonal!");
	}

	// Update the vector x
	x_temp.place(row, 0, s);
      }

      x_k = x_temp;

      if ((A.matmul(x_k) - b).norm() < tol) {
	return x_k;
      }
    }

    std::cout << "\nWARNING: Jacobi method did not converge to the wanted tolerance!" << "\n\n";
    
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

  template<class Matrix> Matrix gsSolve(const Matrix& A, const Matrix& x_0, const Matrix& b, int max_iter=MAX_ITER, double tol=BASE_TOL) {
    if (A.nrows() != x_0.nrows() || A.nrows() != b.nrows()) {
      throw std::invalid_argument("Improper dimensions!");
    }

    if (A.nrows() != A.ncols()) {
      throw std::invalid_argument("Coefficient matrix must be square!");
    }

    Matrix x_k = Matrix(x_0);

    for (int iter = 0; iter < max_iter; iter++) {

      // Go over all i in {0, ..., nrows}
      for (int row = 0; row < A.nrows(); row++) {

	// Compute the sum s_l = sum_{j < i} a_{i, j} * x_j^{(k)}
	double s_l = 0.0;
	for (int col = 0; col < row; col++) {
	  s_l += A(row, col) * x_k(col, 0);
	}

	// Compute the sum s_g = sum_{j > i} a_{i, j} * x_j^{(k-1)}
	double s_g = 0.0;
	for (int col = row + 1; col < A.ncols(); col++) {
	  s_g += A(row, col) * x_k(col, 0);
	}

	double s = s_l + s_g;

	// Compute the new value of x_i = (b_i - s) / a_{i, i}
	double a_ii = A(row, row);
	if (a_ii != 0.0) {
	  s = (b(row, 0) - s) / a_ii;
	}
	else {
	  throw std::invalid_argument("Coefficient matrix must have a non-zero diagonal!");
	}

	// Update the vector x
	x_k.place(row, 0, s);
      }

      if ((A.matmul(x_k) - b).norm() < tol) {
	return x_k;
      }
    }

    std::cout << "\nWARNING: Gauss-Seidel method did not converge to the wanted tolerance!" << "\n\n";
    
    return x_k;
  }

}

#endif
