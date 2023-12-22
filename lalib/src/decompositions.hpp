#ifndef DECOMPOSITIONS_HPP
#define DECOMPOSITIONS_HPP


#include "declare_lalib.hpp"
#include "vector/Vector.hpp"
#include "matrix/Matrix.hpp"


#ifndef CHECK_SYMMETRIC
#define CHECK_SYMMETRIC 0
#endif


/*
  TODO: Proper comments

  [1] Press, William H., and Saul A. Teukolsky. Numerical recipes 3rd edition: The art of scientific computing. Cambridge university press, 2007.
  [2] Hannukainen, Antti. Numerical matrix computations. Course lecture notes, 2022.
*/


namespace lalib {

  /*
    Solve the LU decomposition of a given square matrix using Doolittle algorithm
  */

  template<class type, bool vectorize, bool sparse> 
  std::tuple<Matrix<type, vectorize, sparse>, Matrix<type, vectorize, sparse>> lu(const Matrix<type, vectorize, sparse>& A) {

    if (A.nrows() != A.ncols()) {
      ERROR("Given matrix must be square!");
    }

    Matrix L = Matrix<type, vectorize, sparse>(A.nrows(), A.ncols());
    Matrix U = Matrix<type, vectorize, sparse>(A.nrows(), A.ncols());

    for (int row = 0; row < A.nrows(); row++) {

      // Upper triangular
      for (int k = row; k < A.nrows(); k++) {
        type sum = { };

        for (int j = 0; j < i; j++) {
          // sum += ...

          // TODO: FINISH IMPLEMENTATION
        }
      }

    }
    
  }


  /*
    Cholesky
  */

  template<class Matrix> std::tuple<Matrix, Matrix> chol(const Matrix& A, bool check_symmetric=CHECK_SYMMETRIC) {

    if (A.nrows() != A.ncols()) {
      ERROR("Given matrix must be square!");
    }

    if (check_symmetric) {
      for (int row = 0; row < A.nrows(); row++) {
        for (int col = 0; col < A.ncols(); col++) {
          if (A(row, col) != A(col, row)) {
            ERROR("Given matrix must be symmetric!");
          }
        }
      }
    }

    // TODO: FINISH IMPLEMENTATION
    
  }

  
  /*
    QR
  */

  template<class Matrix> std::tuple<Matrix, Matrix> qr(const Matrix& A) {

    if (A.nrows() != A.ncols()) {
      ERROR("Given matrix must be square!");
    }

    // TODO: FINISH IMPLEMENTATION
    
  }


  /*
    SVD
  */

  template<class Matrix> std::tuple<Matrix, Matrix> svd(const Matrix& A) {

    if (A.nrows() != A.ncols()) {
      ERROR("Given matrix must be square!");
    }
    
    // TODO: FINISH IMPLEMENTATION

  }
  
}

#endif
