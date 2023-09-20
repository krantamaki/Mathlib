#ifndef MATRIX_DECL_HPP
#define MATRIX_DECL_HPP


#include "../declare_lalib.hpp"
#include "../vector/Vector.hpp"


#ifndef STRASSEN_THRESHOLD
#define STRASSEN_THRESHOLD 10000
#endif


using namespace utils;


namespace lalib {
  

  template <class type = double, bool vectorize = false, bool sparse = true>
  class Matrix;


  /**
   * @brief Default insertion operator
   *
   * Method that adds a representation of the dense Matrix object into a
   * std::ostream.
   *
   * @param os A reference of a std::ostream into which representation of 
   * the dense Matrix object is to be added
   * @param A A reference to the dense Matrix object to be inserted
   *
   * @return A reference of the updated std::ostream
   */
  template <class type, bool vectorize, bool sparse>
  std::ostream& operator<<(std::ostream& os, Matrix<type, vectorize, sparse>& A) {
    if (A.ncols() == 0 || A.nrows() == 0) {
      os << "[]" << std::endl;  // Signifies uninitialized matrix
          
      return os;
    }
      
    os << "[";
    for (int row = 0; row < A.nrows(); row++) {
      if (row > 0) os << ' ';

      os << "[";
      for (int col = 0; col < A.ncols() - 1; col++) {
        os << A(row, col) << ' ';
      }
      os << A(row, A.ncols() - 1) << "]";

      if (row < A.nrows() - 1) os << std::endl; 
    }
    os << "]" << std::endl;

    return os;
  }


  /**
   * @brief Scalar (left) multiplication
   *
   * Method that performs the standard scalar-matrix multiplication
   *
   * @param scalar The scalar used in the multiplication
   * @param matrix A reference to the dense Matrix object used in multiplication
   *
   * @return A dense Matrix object
   */
  template <class type, bool vectorize, bool sparse>
  const Matrix<type, vectorize, sparse> operator* (type scalar, const Matrix<type, vectorize, sparse>& matrix) {
    return Matrix(matrix) *= scalar;
  }

}


#endif