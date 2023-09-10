#ifndef VECTOR_MATMUL_HPP
#define VECTOR_MATMUL_HPP


#include "vector_impl.hpp"
#include "../dense/DenseMatrix.hpp"
#include "../declare_lalib.hpp"


using namespace lalib;
using namespace utils;


// Dense vector-matrix multiplication
template <class type, bool vectorize> 
const Vector<type, vectorize> Vector<type, vectorize>::matmul(const DenseMatrix<type, vectorize>& that, bool is_symmetric) const {
  if (_len != that.nrows()) {
    _errorMsg("Improper dimensions given!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (!is_symmetric) {
    DenseMatrix that_T = that.T();
    return that_T.matmul(*this);
  }

  return that.matmul(*this);
}


// Dot (inner) product
template <class type, bool vectorize>
type Vector<type, vectorize>::dot(const Vector<type, vectorize>& that) const {
  if (_len != that._len) {
    _errorMsg("Vector dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  type ret = t_zero;

  if constexpr (vectorize) {
    var_t sum = v_zero;

    for (int vect = 0; vect < _total_vects - 1; vect++) {
      sum += data[vect] * that.data[vect];
    }
    for (int elem = 0; elem < _len % var_size; elem++) {
      ret += data[_total_vects - 1][elem] * that.data[_total_vects - 1][elem];
    }

    ret += _reduce(sum);
  }
  else {
    for (int elem = 0; elem < _len; elem++) {
      ret += data[elem] * that.data[elem];
    }
  }

  return ret;
}


#endif