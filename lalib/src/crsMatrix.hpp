#ifndef CRSMATRIX_HPP
#define CRSMATRIX_HPP

/*
  CRSMatrix is a sparse matrix storage format that allows for constant time 
  access to the rows and linear time access to columns. The elements are stored
  in three std::vectors - one for values themselves, one for row indeces and one 
  for column indeces. Unlike DenseMatrix this implementation doesn't utilize SIMD
  commands.
*/

#include "declare_lalib.hpp"

class CRSVector;  // To avoid circular dependencies

class CRSMatrix {

protected:
  // Initialize values to signify 'empty' matrix
  int _ncols = 0;
  int _nrows = 0;
  std::vector<double> vals;
  std::vector<int> colInds;
  std::vector<int> rowPtrs;
  
public:
  // Constructors

  CRSMatrix(void);
  CRSMatrix(const CRSMatrix& that);
  CRSMatrix(int rows, int cols);
  CRSMatrix(int rows, int cols, double init_val);
  CRSMatrix(int rows, int cols, double* elems);
  CRSMatrix(int rows, int cols, std::vector<double> elems);
  CRSMatrix(int rows, int cols, std::vector<double> new_vals, std::vector<int> new_colInds, std::vector<int> new_rowPtrs);

  // ~CRSMatrix();  // Destructor not needed


  // Overload indexing operators

  // NOTE! To access a single element there are two alternatives:
  // 1. double elem = A[num]
  // 2. double elem = A(row, col)
  // For these to return the same value must hold that:
  //    num = row * num_rows + col
  // Thus it is recommended to use () for indexing

  // Additionally for slicing there exists a overloaded operator:
  // CRSMatrix B = A(rowStart, rowEnd, colStart, colEnd)
  // Requires that *Start < *End and *Start >= 0, but does allow having the *End 
  // going out of bounds, but only returns the values that exist. The values *Start
  // understandably must be in bounds

  double operator[] (int num) const;
  double operator() (int row, int col) const;
  double get(int row, int col) const;  // Alias for operator()
  const CRSMatrix operator() (int rowStart, int rowEnd, int colStart, int colEnd) const;
  const CRSMatrix get(int rowStart, int rowEnd, int colStart, int colEnd) const;  // Alias for operator()
  const CRSVector getCol(int col) const;
  const CRSVector getRow(int row) const;

  // Functions for placing values into existing matrices
  
  void place(int row, int col, double val);
  void place(int rowStart, int rowEnd, int colStart, int colEnd, CRSMatrix matrix);

  
  // Overload basic math operators

  // NOTE! The operators will function as elementwise operators

  const CRSMatrix operator+ (const CRSMatrix& that) const;
  CRSMatrix& operator+= (const CRSMatrix& that);
  const CRSMatrix operator- (const CRSMatrix& that) const;
  CRSMatrix& operator-= (const CRSMatrix& that);
  const CRSMatrix operator* (const CRSMatrix& that) const;
  CRSMatrix& operator*= (const CRSMatrix& that);
  const CRSMatrix operator* (const double that) const;
  const CRSMatrix operator/ (const CRSMatrix& that) const;
  CRSMatrix& operator/= (const CRSMatrix& that);
  const CRSMatrix operator/ (const double that) const;
  // ... ?

  // Friend methods
  
  friend std::ostream& operator<<(std::ostream& os, CRSMatrix& A);

  // Other methods

  int ncols() { return _ncols; }
  int nrows() { return _nrows; }
  std::tuple<int, int> shape() { return std::make_tuple(_nrows, _ncols); }

  const int ncols() const { return _ncols; }
  const int nrows() const { return _nrows; }
  const std::tuple<int, int> shape() const { return std::make_tuple(_nrows, _ncols); }

  void _printArrays();
  
};




#endif
