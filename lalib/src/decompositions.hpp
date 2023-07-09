#ifndef DECOMPOSITIONS_HPP
#define DECOMPOSITIONS_HPP


/*
  [1] Press, William H., and Saul A. Teukolsky. Numerical recipes 3rd edition: The art of scientific computing. Cambridge university press, 2007.
  [2] Hannukainen, Antti. Numerical matrix computations. Course lecture notes, 2022.
*/


#define CHECK_SYMMETRIC 0


namespace lalib {


  /*
    PLU
  */

  template<class Matrix> std::tuple<Matrix, Matrix, Matrix> plu(const Matrix& A) {

    if (A.nrows() != A.ncols()) {
      throw std::invalid_argument("Given matrix must be square!");
    }
    
  }


  /*
    Cholesky
  */

  template<class Matrix> std::tuple<Matrix, Matrix> chol(const Matrix& A, bool check_symmetric=CHECK_SYMMETRIC) {

    if (A.nrows() != A.ncols()) {
      throw std::invalid_argument("Given matrix must be square!");
    }

    if (check_symmetric) {
      for (int row = 0; row < A.nrows(); row++) {
	for (int col = 0; col < A.ncols(); col++) {
	  if (A(row, col) != A(col, row)) {
	    throw std::invalid_argument("Given matrix must be symmetric!");
	  }
	}
      }
    }
    
  }

  
  /*
    QR
  */

  template<class Matrix> std::tuple<Matrix, Matrix> qr(const Matrix& A) {

    if (A.nrows() != A.ncols()) {
      throw std::invalid_argument("Given matrix must be square!");
    }
    
  }


  /*
    SVD
  */

  template<class Matrix> std::tuple<Matrix, Matrix> svd(const Matrix& A) {

    if (A.nrows() != A.ncols()) {
      throw std::invalid_argument("Given matrix must be square!");
    }
    
  }
  
}

#endif
