#ifndef DIRECT_SOLVERS_HPP
#define DIRECT_SOLVERS_HPP


#include "declare_lalib.hpp"
#include "decompositions.hpp"
#include "vector/Vector.hpp"
#include "matrix/Matrix.hpp"


#ifndef CHECK_TRIANGULAR
#define CHECK_TRIANGULAR 0
#endif


/**
 * TODO: Proper comments
 * 
 * [1] Hannukainen, Antti. Numerical matrix computations. Course lecture notes, 2022.
 */


using namespace utils;


namespace lalib {


  /**
   * Solve system Lx = b, where L is a lower triangular matrix
   */
  template<class type, bool vectorize, bool sparse>
  Vector<type, vectorize> trilSolve(Matrix<type, vectorize, sparse> L, Vector<type, vectorize> b) {

    if (L.nrows() != b.len()) {
      _errorMsg("Improper dimensions!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
    }

    if (L.nrows() != L.ncols()) {
      _errorMsg("Coefficient matrix must be square!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
    }

    /*
    if (CHECK_TRIANGULAR) {

    }
    */

    Vector ret = Vector<type, vectorize>(b.len());

    for (int i = 0; i < L.nrows(); i++) {
      type tmp = b(i);

      tmp -= L.rowDot(i, ret);

      ret.place(i, tmp / L(i, i));
    }

    return ret;
  }


  /**
   * Solve system Ux = b, where U is an upper triangular matrix
   */
  template<class type, bool vectorize, bool sparse>
  Vector<type, vectorize> triuSolve(Matrix<type, vectorize, sparse> U, Vector<type, vectorize> b) {
    
    if (U.nrows() != b.len()) {
      _errorMsg("Improper dimensions!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
    }

    if (U.nrows() != U.ncols()) {
      _errorMsg("Coefficient matrix must be square!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
    }

    /*
    if (CHECK_TRIANGULAR) {

    }
    */

    Vector ret = Vector<type, vectorize>(b.len());

    for (int i = U.nrows() - 1; i >= 0; i++) {
      type tmp = b(i);

      tmp -= U.rowDot(i, ret);

      ret.place(i, tmp / U(i, i));
    }

    return ret;
  }


  /**
   * Solve system Ax = b by using the PLU decomposition of A -> PLUx = b by
   *   Solving Ly = b  for y using trilSolve
   *   Solving Ux = y  for x using triuSolve
   *   Returning Px
   */
  template<class type, bool vectorize, bool sparse>
  Vector<type, vectorize> luSolve(Matrix<type, vectorize, sparse> A, Vector<type, vectorize> b) {

    if (A.nrows() != b.len()) {
      _errorMsg("Improper dimensions!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
    }

    if (A.nrows() != A.ncols()) {
      _errorMsg("Coefficient matrix must be square!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
    }

    std::tuple<Matrix<type, vectorize, sparse>, Matrix<type, vectorize, sparse>, Matrix<type, vectorize, sparse>> plu_tup = plu(A);

    Vector y = trilSolve<type, vectorize, sparse>(plu_tup[1], b);
    Vector x = trilSolve<type, vectorize, sparse>(plu_tup[2], y);
    
    Vector ret = plu_tup[0].matmul(x);

    return ret;
  }

}


#endif