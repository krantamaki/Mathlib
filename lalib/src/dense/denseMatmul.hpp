#ifndef DENSEMATRIX_MATMUL_HPP
#define DENSEMATRIX_MATMUL_HPP


#include "denseMatrix_impl.hpp"
#include "../vector/Vector.hpp"
#include "../declare_lalib.hpp"


#ifndef STRASSEN_THRESHOLD
#define STRASSEN_THRESHOLD 1000
#endif


using namespace lalib;
using namespace utils;


// Matrix-matrix multiplication in the (naive) textbook way
template <class type, bool vectorize>
const DenseMatrix<type, vectorize> DenseMatrix<type, vectorize>::matmulNaive(const DenseMatrix<type, vectorize>& that) const {
  if (_ncols != that._nrows) {
    _errorMsg("Improper dimensions given!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  DenseMatrix<type, vectorize> ret = DenseMatrix<type, vectorize>(_nrows, that._ncols);

  // Transpose that for linear memory reads
  DenseMatrix<type, vectorize> that_T = that.T();

  if constexpr (vectorize) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int row = 0; row < _nrows; row++) {
      for (int col = 0; col < that._ncols; col++) {    

        var_t sum = v_zero;

        for (int vect = 0; vect < _vects_per_row - 1; vect++) {
          sum += data[_vects_per_row * row + vect] * that_T.data[_vects_per_row * col + vect];
        }

        type val = t_zero;
        for (int elem = 0; elem < (that._ncols % var_size); elem++) {
          val += data[_vects_per_row * (row + 1) - 1][elem] * that_T.data[_vects_per_row * (col + 1) - 1][elem];
        }

        val += _reduce(sum);
        
        ret.place(row, col, val);
      }
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int row = 0; row < _nrows; row++) {
      for (int col = 0; col < that._ncols; col++) {    

        type val = t_zero;

        for (int i = 0; i < _nrows; i++) {
          val += this->operator() (row, i) * that(i, col);
        }
        
        ret.place(row, col, val);
      }
    }
  }

  return ret;
}


// TODO: Implement Strassen algorithm
// const DenseMatrix DenseMatrix::matmulStrassen(const DenseMatrix& that) const {}


// Wrapper for matrix-matrix multiplication
template <class type, bool vectorize>
const DenseMatrix<type, vectorize> DenseMatrix<type, vectorize>::matmul(const DenseMatrix<type, vectorize>& that) const {
  if (_ncols != that._nrows) {
    _errorMsg("Improper dimensions given!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  // 100 chosen as arbitrary threshold
  if (_ncols > STRASSEN_THRESHOLD && _nrows > STRASSEN_THRESHOLD && that._ncols > STRASSEN_THRESHOLD) {
    return this->matmulNaive(that);  // Should call Strassen algorithm, but that is not implemented yet
  }

  return this->matmulNaive(that);
}


// Matrix-vector multiplication
template <class type, bool vectorize>
const Vector<type, vectorize> DenseMatrix<type, vectorize>::matmul(const Vector<type, vectorize>& that) const {
  if (_ncols != that.len()) {
    _errorMsg("Improper dimensions given!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  Vector<type, vectorize> ret = Vector<type, vectorize>(_nrows);

  if constexpr (vectorize) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int row = 0; row < _nrows; row++) {

      var_t sum = v_zero;

      for (int vect = 0; vect < _vects_per_row - 1; vect++) {
        sum += data[_vects_per_row * row + vect] * *(var_t*)that.getSIMD(vect);
      }

      type val = t_zero;
      for (int elem = 0; elem < (that.len() % var_size); elem++) {
        val += data[_vects_per_row * (row + 1) - 1][elem] * (*(var_t*)that.getSIMD(_vects_per_row - 1))[elem];
      }

      val += _reduce(sum);
      
      ret.place(row, val);
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int row = 0; row < _nrows; row++) {

      type val = t_zero;

      for (int col = 0; col < _ncols; col++) {
        val += data[_ncols * row + col] * that(col);
      }

      ret.place(row, val);
    }
  }

  return ret;
}


#endif