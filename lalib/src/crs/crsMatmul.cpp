#include "crsMatrix.hpp"
#include "../declare_lalib.hpp"


#define STRASSEN_THRESHOLD 1000


using namespace lalib;


/*
  TODO: PROPER DESCRIPTION
*/


// Generic matrix multiplication (Not really optimized)

const CRSMatrix CRSMatrix::matmulNaive(const CRSMatrix& that) const {

  if (_ncols != that._nrows) {
    throw std::invalid_argument("Improper dimensions!");
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
	
	int n_that_col_elems = that_T.rowPtrs[col + 1] - rowPtrs[col];
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

	  ret.vals.push_back(sum);
	  ret.colInds.push_back(col);

	  for (int row_i = row + 1; row_i <= _nrows; row_i++) {
	    ret.rowPtrs[row_i] += 1;
	  }

	}
      }
    }
  }
  
  return ret;
}


// TODO: Implements Strassen algorithm
// const CRSMatrix CRSMatrix::matmulStrassen(const CRSMatrix& that) const {}


// Wrapper function for calling wanted matrix multiplication algorithm

const CRSMatrix CRSMatrix::matmul(const CRSMatrix& that) const {
  if (_ncols != that._nrows) {
    throw std::invalid_argument("Improper dimensions!");
  }

  if (_ncols > STRASSEN_THRESHOLD && _nrows > STRASSEN_THRESHOLD && that._ncols > STRASSEN_THRESHOLD) {
    return this->matmulNaive(that);  // Should call Strassen once implemented
  }

  return this->matmulNaive(that);
}
