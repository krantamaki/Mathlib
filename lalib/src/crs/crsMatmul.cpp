#include "crsMatrix.hpp"
#include "crsVector.hpp"
#include "../declare_lalib.hpp"


#ifndef STRASSEN_THRESHOLD
#define STRASSEN_THRESHOLD 1000
#endif


using namespace lalib;
using namespace utils;


// Matrix-matrix multiplication in the (naive) textbook way
const CRSMatrix CRSMatrix::matmulNaive(const CRSMatrix& that) const {

  if (_ncols != that._nrows) {
    _errorMsg("Improper dimensions!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  // Create the matrix that will be filled
  CRSMatrix ret = CRSMatrix(_nrows, that._ncols);

  // Transpose that to have constant time access to columns
  CRSMatrix that_T = that.T();

  for (int row = 0; row < _nrows; row++) {
    
    int n_this_row_elems = rowPtrs[row + 1] - rowPtrs[row];
    if (n_this_row_elems == 0) continue;
    
    else {
      
      for (int col = 0; col < that._ncols; col++) {
	
	int n_that_col_elems = that_T.rowPtrs[col + 1] - that_T.rowPtrs[col];
	if (n_that_col_elems == 0) continue;

	else {

	  double sum = 0.0;
	  for (int col_i = rowPtrs[row]; col_i < rowPtrs[row + 1]; col_i++) {
	    for (int row_i = that_T.rowPtrs[col]; row_i < that_T.rowPtrs[col + 1]; row_i++) {
	      if (colInds[col_i] == that_T.colInds[row_i]) {
		sum += vals[col_i] * that_T.vals[row_i];
	      }
	    }
	  }

	  if (sum != 0.0) {
	    ret.vals.push_back(sum);
	    ret.colInds.push_back(col);

	    for (int row_i = row + 1; row_i <= _nrows; row_i++) {
	      ret.rowPtrs[row_i] += 1;
	    }
	  }
	}
      }
    }
  }
  
  return ret;
}


// TODO: Implements Strassen algorithm
// const CRSMatrix CRSMatrix::matmulStrassen(const CRSMatrix& that) const {}


// Wrapper for matrix-matrix multiplication
const CRSMatrix CRSMatrix::matmul(const CRSMatrix& that) const {
  if (_ncols != that._nrows) {
    _errorMsg("Improper dimensions!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (_ncols > STRASSEN_THRESHOLD && _nrows > STRASSEN_THRESHOLD && that._ncols > STRASSEN_THRESHOLD) {
    return this->matmulNaive(that);  // Should call Strassen once implemented
  }

  return this->matmulNaive(that);
}


// Efficient matrix-vector multiplication
const CRSVector CRSMatrix::matmul(const CRSVector& that) const {
  if (_ncols != that.len()) {
    _errorMsg("Improper dimensions!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  CRSVector ret = CRSVector(_nrows);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < _nrows; row++) {
    int n_row_elems = rowPtrs[row + 1] - rowPtrs[row];
    if (n_row_elems == 0) continue;
    else {
      double sum = 0.0;
      for (int col_i = rowPtrs[row]; col_i < rowPtrs[row + 1]; col_i++) {
	int col = colInds[col_i];
	double val = vals[col_i];

	sum += val * that(col);
      }
      ret.place(row, sum);
    }
  }

  return ret;
}


// Vector-matrix multiplication
const CRSVector CRSVector::matmul(const CRSMatrix& that, bool is_symmetric) const {
  if (_len != that.nrows()) {
    _errorMsg("Improper dimensions!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (is_symmetric) {
    return that.matmul(*this);
  }
  else {
    CRSMatrix that_T = that.T();
    return that_T.matmul(*this);
  }
}


// Dot (inner) product
double CRSVector::dot(const CRSVector& that) const {
  if (_len != that._len) {
    _errorMsg("Improper dimensions!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  double ret = 0.0;

  for (int i = 0; i < _len; i++) {
    ret += data[i] * that.data[i];
  }

  return ret;
}
