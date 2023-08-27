#include "denseMatrix.hpp"
#include "denseVector.hpp"
#include "../declare_lalib.hpp"


#ifndef STRASSEN_THRESHOLD
#define STRASSEN_THRESHOLD 1000
#endif


using namespace lalib;


// Matrix-matrix multiplication in the (naive) textbook way
const DenseMatrix DenseMatrix::matmulNaive(const DenseMatrix& that) const {
  if (_ncols != that._nrows) {
    throw std::invalid_argument(_formErrorMsg("Improper dimensions given!", __FILE__, __func__, __LINE__));
  }

  // Allocate memory for the resulting matrix
  DenseMatrix ret = DenseMatrix(_nrows, that._ncols);

  // Transpose that for linear memory reads
  DenseMatrix that_T = that.T();

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < that._ncols; col++) {    

      vect_t sum = zeros;

      for (int vect = 0; vect < vects_per_row - 1; vect++) {
	      sum += data[vects_per_row * row + vect] * that_T.data[vects_per_row * col + vect];
      }

      double val = 0.0;
      for (int elem = 0; elem < (that._ncols % VECT_ELEMS); elem++) {
        val += data[vects_per_row * (row + 1) - 1][elem] * that_T.data[vects_per_row * (col + 1) - 1][elem];
      }

      val += _reduce(sum);
			
      ret.place(row, col, val);
    }
  }

  return ret;
}


// TODO: Implement Strassen algorithm
// const DenseMatrix DenseMatrix::matmulStrassen(const DenseMatrix& that) const {}


// Wrapper for matrix-matrix multiplication
const DenseMatrix DenseMatrix::matmul(const DenseMatrix& that) const {
  if (_ncols != that._nrows) {
    throw std::invalid_argument(_formErrorMsg("Improper dimensions given!", __FILE__, __func__, __LINE__));
  }

  // 100 chosen as arbitrary threshold
  if (_ncols > STRASSEN_THRESHOLD && _nrows > STRASSEN_THRESHOLD && that._ncols > STRASSEN_THRESHOLD) {
    return this->matmulNaive(that);  // Should call Strassen algorithm, but that is not implemented yet
  }

  return this->matmulNaive(that);
}


const DenseVector DenseMatrix::matmul(const DenseVector& that) const {
  if (_ncols != that.len()) {
    throw std::invalid_argument(_formErrorMsg("Improper dimensions given!", __FILE__, __func__, __LINE__));
  }

  // Allocate memory for the resulting vector
  DenseVector ret = DenseVector(_nrows);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < _nrows; row++) {

    vect_t sum = zeros;

    for (int vect = 0; vect < vects_per_row - 1; vect++) {
      sum += data[vects_per_row * row + vect] * that.getSIMD(vect);
    }

    double val = 0.0;
    for (int elem = 0; elem < (that.len() % VECT_ELEMS); elem++) {
      val += data[vects_per_row * (row + 1) - 1][elem] * that.getSIMD(vects_per_row - 1)[elem];
    }

    val += _reduce(sum);
    
    ret.place(row, val);
  }

  return ret;
}


const DenseVector DenseVector::matmul(const DenseMatrix& that, bool is_symmetric) const {
  if (_len != that.nrows()) {
    throw std::invalid_argument(_formErrorMsg("Improper dimensions given!", __FILE__, __func__, __LINE__));
  }

  if (!is_symmetric) {
    DenseMatrix that_T = that.T();
    return that_T.matmul(*this);
  }

  return that.matmul(*this);
}

// Dot product behaves similarly to matmul, but always returns a scalar value
// (even when multiplying column vector with row vector or column vector with column vector etc.)

double DenseVector::dot(const DenseVector& that) const {
  if (_len != that._len) {
    throw std::invalid_argument(_formErrorMsg("Improper dimensions given!", __FILE__, __func__, __LINE__));
  }

  vect_t sum = zeros;

  for (int vect = 0; vect < total_vects - 1; vect++) {
    sum += data[vect] * that.data[vect];
  }

  double ret = 0.0;
  for (int elem = 0; elem < (_len % VECT_ELEMS); elem++) {
    ret += data[total_vects - 1][elem] * that.data[total_vects - 1][elem];
  }

  ret += _reduce(sum);

  return ret;
}