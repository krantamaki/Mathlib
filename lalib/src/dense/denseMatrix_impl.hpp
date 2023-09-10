#ifndef DENSEMATRIX_IMPL_HPP
#define DENSEMATRIX_IMPL_HPP


#include "denseMatrix_decl.hpp"
#include "../vector/Vector.hpp"
#include "../declare_lalib.hpp"


using namespace lalib;
using namespace utils;


// -------------------CONSTRUCTORS AND DESTRUCTORS--------------------------


// Default constuctor
template <class type, bool vectorize> 
DenseMatrix<type, vectorize>::DenseMatrix(void) { }


// Copying constructor
template <class type, bool vectorize>
DenseMatrix<type, vectorize>::DenseMatrix(const DenseMatrix& that) {

  if (that._ncols < 1 || that._nrows < 1) {
    _errorMsg("Matrix dimensions must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _ncols = that._ncols;
  _nrows = that._nrows;

  if constexpr (vectorize) {
    data = that.data;
    _total_vects = that._total_vects;
    _vects_per_row = that._vects_per_row;
  }
  else {
    data = that.data;
  }
}


// Zeros constructor
template <class type, bool vectorize> 
DenseMatrix<type, vectorize>::DenseMatrix(int rows, int cols) {

  if (cols < 1 || rows < 1) {
    _errorMsg("Matrix dimensions must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _ncols = cols;
  _nrows = rows;

  if constexpr (vectorize) {
    _vects_per_row = _ceil(cols, var_size);
    _total_vects = rows * _vects_per_row;

    data = std::vector<var_t>(_total_vects, v_zero);
  }
  else {
    data = std::vector<var_t>(_ncols * _nrows, v_zero);
  }
}


// Default value constructor
template <class type, bool vectorize> 
DenseMatrix<type, vectorize>::DenseMatrix(int rows, int cols, type init_val) {

  if (cols < 1 || rows < 1) {
    _errorMsg("Matrix dimensions must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _ncols = cols;
  _nrows = rows;

  if constexpr (vectorize) {
    var_t init_vect = _fill(init_val);

    _vects_per_row = _ceil(cols, var_size);
    _total_vects = rows * _vects_per_row;
    data = std::vector<var_t>(_total_vects, init_vect);
  }
  else {
    data = std::vector<var_t>(_ncols * _nrows, init_val);
  }
}


// Array copying constructor
template <class type, bool vectorize>
DenseMatrix<type, vectorize>::DenseMatrix(int rows, int cols, type* elems) {

  _warningMsg("Initializing a matrix with double array might lead to undefined behaviour!", __func__);

  if (cols < 1 || rows < 1) {
    _errorMsg("Matrix dimensions must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _ncols = cols;
  _nrows = rows;

  if constexpr (vectorize) {
    _vects_per_row = _ceil(cols, var_size);
    _total_vects = rows * _vects_per_row;

    for (int vect = 0; vect < _total_vects; vect++) {
      var_t tmp_vect;
      for (int elem = 0; elem < var_size; elem++) {
        int i = vect * var_size + elem;
        tmp_vect[elem] = i < _ncols * _nrows ? elems[i] : t_zero;
      }
      data.push_back(tmp_vect);
    }
  }
  else {
    data.assign(elems, elems + _ncols * _nrows);
  }
}


// Vector copying constructor
template <class type, bool vectorize> 
DenseMatrix<type, vectorize>::DenseMatrix(int rows, int cols, std::vector<type>& elems) {

  if (cols < 1 || rows < 1) {
    _errorMsg("Matrix dimensions must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (rows * cols != (int)elems.size()) {
    _warningMsg("Given dimensions don't match with the size of the std::vector!", __func__);
  } 

  _ncols = cols;
  _nrows = rows;

  if constexpr (vectorize) {
    _vects_per_row = _ceil(cols, var_size);
    _total_vects = rows * _vects_per_row;
    for (int vect = 0; vect < _total_vects; vect++) {
      var_t tmp_vect;
      for (int elem = 0; elem < var_size; elem++) {
        int i = vect * var_size + elem;
        tmp_vect[elem] = (i < (int)elems.size()) && (i < _ncols * _nrows) ? elems[i] : t_zero;
      }
      data.push_back(tmp_vect);
    }
  }
  else {
    std::copy(elems.begin(), elems.begin() + _ncols * _nrows, std::back_inserter(data));
  }
}


// ---------------------OVERLOADED BASIC MATH OPERATORS------------------------


// Element-wise addition assignment
template <class type, bool vectorize> 
DenseMatrix<type, vectorize>& DenseMatrix<type, vectorize>::operator+= (const DenseMatrix<type, vectorize>& that) {
  if (_ncols != that._ncols || _nrows != that._nrows) {
    _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  } 

  if constexpr (vectorize) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects - 1; vect++) {
      data[vect] = data[vect] + that.data[vect];
    }
    for (int elem = 0; elem < (_nrows * _ncols) % var_size; elem++) {
      data[_total_vects - 1][elem] = data[_total_vects - 1][elem] + that.data[_total_vects - 1][elem];
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int row = 0; row < _nrows; row++) {
      for (int col = 0; col < _ncols; col++) {
        data[row * _nrows + col] = data[row * _nrows + col] + that.data[row * _nrows + col];
      }
    }
  }
  
  return *this;
}


// Element-wise addition
template <class type, bool vectorize>
const DenseMatrix<type, vectorize> DenseMatrix<type, vectorize>::operator+ (const DenseMatrix<type, vectorize>& that) const {
  return DenseMatrix(*this) += that;
}


// Element-wise subtraction assignment
template <class type, bool vectorize> 
DenseMatrix<type, vectorize>& DenseMatrix<type, vectorize>::operator-= (const DenseMatrix<type, vectorize>& that) {
  if (_ncols != that._ncols || _nrows != that._nrows) {
    _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  } 

  if constexpr (vectorize) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects - 1; vect++) {
      data[vect] = data[vect] - that.data[vect];
    }
    for (int elem = 0; elem < (_nrows * _ncols) % var_size; elem++) {
      data[_total_vects - 1][elem] = data[_total_vects - 1][elem] - that.data[_total_vects - 1][elem];
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int row = 0; row < _nrows; row++) {
      for (int col = 0; col < _ncols; col++) {
        data[row * _nrows + col] = data[row * _nrows + col] - that.data[row * _nrows + col];
      }
    }
  }

  return *this;
}


// Element-wise subtraction
template <class type, bool vectorize> 
const DenseMatrix<type, vectorize> DenseMatrix<type, vectorize>::operator- (const DenseMatrix<type, vectorize>& that) const {
  return DenseMatrix(*this) -= that;
}


// Element-wise multiplication assignment
template <class type, bool vectorize> 
DenseMatrix<type, vectorize>& DenseMatrix<type, vectorize>::operator*= (const DenseMatrix<type, vectorize>& that) {
  if (_ncols != that._ncols || _nrows != that._nrows) {
    _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  } 

  if constexpr (vectorize) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects - 1; vect++) {
      data[vect] = data[vect] * that.data[vect];
    }
    for (int elem = 0; elem < (_nrows * _ncols) % var_size; elem++) {
      data[_total_vects - 1][elem] = data[_total_vects - 1][elem] * that.data[_total_vects - 1][elem];
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int row = 0; row < _nrows; row++) {
      for (int col = 0; col < _ncols; col++) {
        data[row * _nrows + col] = data[row * _nrows + col] * that.data[row * _nrows + col];
      }
    }
  }

  return *this;
}


// Element-wise multiplication
template <class type, bool vectorize> 
const DenseMatrix<type, vectorize> DenseMatrix<type, vectorize>::operator* (const DenseMatrix<type, vectorize>& that) const {
  return DenseMatrix(*this) *= that;
}


// Scalar (right) multiplication assignment
template <class type, bool vectorize> 
DenseMatrix<type, vectorize>& DenseMatrix<type, vectorize>::operator*= (type that) {
  if (_nrows < 1 || _ncols < 1) {
    return *this;
  }

  if constexpr (vectorize) {
    var_t mult = _fill(that);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects - 1; vect++) {
      data[vect] = data[vect] * mult;
    }
    for (int elem = 0; elem < (_nrows * _ncols) % var_size; elem++) {
      data[_total_vects - 1][elem] = data[_total_vects - 1][elem] * that;
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int row = 0; row < _nrows; row++) {
      for (int col = 0; col < _ncols; col++) {
        data[row * _nrows + col] = data[row * _nrows + col] * that;
      }
    }
  }

  return *this;    
}


// Scalar (right) multiplication
template <class type, bool vectorize> 
const DenseMatrix<type, vectorize> DenseMatrix<type, vectorize>::operator* (const type that) const {
  return DenseMatrix(*this) *= that;  
}


// Scalar (left) multiplication
template <class type, bool vectorize> 
const DenseMatrix<type, vectorize> operator* (type scalar, const DenseMatrix<type, vectorize>& matrix) {
  return DenseMatrix(matrix) *= scalar;
}


// Element-wise division assignment
template <class type, bool vectorize> 
DenseMatrix<type, vectorize>& DenseMatrix<type, vectorize>::operator/= (const DenseMatrix<type, vectorize>& that) {
  if (ncols() != that._ncols || nrows() != that._ncols) {
    _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  } 

  if constexpr (vectorize) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects - 1; vect++) {
      data[vect] = data[vect] / that.data[vect];
    }
    for (int elem = 0; elem < (_nrows * _ncols) % var_size; elem++) {
      data[_total_vects - 1][elem] = data[_total_vects - 1][elem] / that.data[_total_vects - 1][elem];
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int row = 0; row < _nrows; row++) {
      for (int col = 0; col < _ncols; col++) {
        data[row * _nrows + col] = data[row * _nrows + col] / that.data[row * _nrows + col];
      }
    }
  }

  return *this;
}


// Element-wise division
template <class type, bool vectorize> 
const DenseMatrix<type, vectorize> DenseMatrix<type, vectorize>::operator/ (const DenseMatrix<type, vectorize>& that) const {
  return DenseMatrix(*this) /= that;
}


// Scalar division assignment
template <class type, bool vectorize> 
DenseMatrix<type, vectorize>& DenseMatrix<type, vectorize>::operator/= (type that) {
  if (that == t_zero) {
    _errorMsg("Division by zero undefined!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
  if (_nrows < 1 || _ncols < 1) {
    return *this;
  }

  if constexpr (vectorize) {
    var_t div = _fill(that);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects - 1; vect++) {
      data[vect] = data[vect] / div;
    }
    for (int elem = 0; elem < (_nrows * _ncols) % var_size; elem++) {
      data[_total_vects - 1][elem] = data[_total_vects - 1][elem] / that;
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int row = 0; row < _nrows; row++) {
      for (int col = 0; col < _ncols; col++) {
        data[row * _nrows + col] = data[row * _nrows + col] / that;
      }
    }
  }
  
  return *this;
}


// Scalar division
template <class type, bool vectorize> 
const DenseMatrix<type, vectorize> DenseMatrix<type, vectorize>::operator/ (const type that) const {
  return DenseMatrix(*this) /= that;
}


// ----------------------OVERLOADED INDEXING OPERATORS--------------------------


// Standard single value placement
template <class type, bool vectorize>
void DenseMatrix<type, vectorize>::place(int row, int col, type val) {
  if (row < 0 || col < 0 || row >= _nrows || col >= _ncols) {
    _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if constexpr (vectorize) {
    const int vect = col / var_size;
    const int elem = col % var_size;

    data[_vects_per_row * row + vect][elem] = val;
  }
  else {
    data[_ncols * row + col] = val;
  }
}


// Standard matrix placement
template <class type, bool vectorize>
void DenseMatrix<type, vectorize>::place(int rowStart, int rowEnd, int colStart, int colEnd, const DenseMatrix<type, vectorize>& mat) {
  if (_nrows < rowEnd - rowStart || _ncols < colEnd - colStart || mat._nrows < rowEnd - rowStart || mat._ncols < colEnd - colStart) {
    _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row0 = 0; row0 < rowEnd - rowStart; row0++) {
    int row = row0 + rowStart;
    for (int col0 = 0; col0 < colEnd - colStart; col0++) {
      int col = col0 + colStart;
      this->place(row, col, mat(row0, col0));
    }
  }
}


// Standard indexing method
template <class type, bool vectorize>
type DenseMatrix<type, vectorize>::operator() (int row, int col) const {
  if (row < 0 || col < 0 || row >= _nrows || col >= _ncols) {
    _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if constexpr (vectorize) {
    const int vect = col / var_size;
    const int elem = col % var_size;

    return data[_vects_per_row * row + vect][elem];
  }
  else {
    return data[_ncols * row + col];
  }
}


// Named indexing method
template <class type, bool vectorize>
type DenseMatrix<type, vectorize>::get(int row, int col) const {
  return this->operator() (row, col);
}


// Squared bracket indexing method
template <class type, bool vectorize>
type DenseMatrix<type, vectorize>::operator[] (int num) const {
  if (num >= _ncols * _nrows) {
    _errorMsg("Given index out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  const int row = num / _ncols;
  const int col = num % _ncols;

  return this->operator() (row, col);
}


// Standard slicing method
template <class type, bool vectorize>
const DenseMatrix<type, vectorize> DenseMatrix<type, vectorize>::operator() (int rowStart, int rowEnd, int colStart, int colEnd) const {
  if (rowStart >= rowEnd || rowStart < 0 || colStart >= colEnd || colStart < 0) {
    _errorMsg("Improper dimensions given!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (rowEnd > _nrows || colEnd > _ncols) {
    _warningMsg("End index out of bounds", __func__);
  }

  int _rowEnd = rowEnd > _nrows ? _nrows : rowEnd;
  int _colEnd = colEnd > _ncols ? _ncols : colEnd;

  // Allocate memory for a new matrix
  DenseMatrix ret = DenseMatrix(_rowEnd - rowStart, _colEnd - colStart);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row0 = 0; row0 < _rowEnd - rowStart; row0++) {
    int row = row0 + rowStart;
    for (int col0 = 0; col0 < _colEnd - colStart; col0++) {
      int col = col0 + colStart;
      ret.place(row0, col0, this->operator()(row, col));
    }
  }

  return ret;
}


// Named slicing method
template <class type, bool vectorize>
const DenseMatrix<type, vectorize> DenseMatrix<type, vectorize>::get(int rowStart, int rowEnd, int colStart, int colEnd) const {
  return this->operator() (rowStart, rowEnd, colStart, colEnd);
}


// ------------------------OTHER OVERLOADED OPERATORS---------------------------


// Default assignment operator
template <class type, bool vectorize>
DenseMatrix<type, vectorize>& DenseMatrix<type, vectorize>::operator= (const DenseMatrix& that) {
  // Check for self-assignment ie. case where a = a is called by comparing the pointers of the objects
  if (this == &that) return *this; 

  data = that.data;
  _total_vects = that._total_vects;
  _vects_per_row = that._vects_per_row;
  _ncols = that._ncols;
  _nrows = that._nrows;

  return *this;
}


// Default (equality) comparison operator
template <class type, bool vectorize>
bool DenseMatrix<type, vectorize>::operator== (const DenseMatrix<type, vectorize>& that) {
  if (_nrows != that._nrows || _ncols != that._ncols) {
    return false;
  }

  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      if (this->operator() (row, col) != that(row, col)) {
	      return false;
      }
    }
  }

  return true;
}


// Default (inequality) comparison operator
template <class type, bool vectorize>
bool DenseMatrix<type, vectorize>::operator!= (const DenseMatrix<type, vectorize>& that) {
  return !(*this == that);
}


// Default insertion operator
template <class type, bool vectorize>
std::ostream& lalib::operator<<(std::ostream& os, DenseMatrix<type, vectorize>& A) {
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


// ----------------------------------MISC----------------------------------------


// Approximative equality comparison
template <class type, bool vectorize>
bool DenseMatrix<type, vectorize>::isclose(const DenseMatrix<type, vectorize>& that, type tol, type (*abs_func)(type)) {
  if (_nrows != that._nrows || _ncols != that._ncols) {
    return false;
  }

  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      if (abs_func(this->operator() (row, col) - that(row, col)) > tol) {
        return false;
      }
    }
  }

  return true;
}


// Standard transpose
template <class type, bool vectorize>
const DenseMatrix<type, vectorize> DenseMatrix<type, vectorize>::transpose() const{
  // Check that matrix initialized
  if (_ncols <= 0 || _nrows <= 0) {
    return *this;  // Maybe change to fatal error
  }

  // Allocate memory for needed sized matrix
  DenseMatrix ret = DenseMatrix(_ncols, _nrows);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      ret.place(col, row, this->operator() (row, col));
    }
  }

  return ret;
}


// Standard transpose
template <class type, bool vectorize>
const DenseMatrix<type, vectorize> DenseMatrix<type, vectorize>::T() const{
  return this->transpose();
}


// Convert DenseMatrix into std::vector
template <class type, bool vectorize>
std::vector<type> DenseMatrix<type, vectorize>::tovector() const {
  if (_ncols <= 0 || _nrows <= 0) {
    _errorMsg("Matrix must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  std::vector<type> ret;
  ret.reserve(_ncols * _nrows);

  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      ret.push_back(this->operator() (row, col));
    }
  }

  return ret;
}


// Convert DenseMatrix into a scalar
template <class type, bool vectorize>
type DenseMatrix<type, vectorize>::asScalar() const {
  if (_ncols != 1 || _nrows != 1) {
    _errorMsg("Matrix must be a 1 x 1 matrix!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  return this->operator() (0, 0);
}


// Convert DenseMatrix into a Vector
template <class type, bool vectorize>
const Vector<type, vectorize> DenseMatrix<type, vectorize>::asVector() const {
  if (_ncols != 1 && _nrows != 1) {
    _errorMsg("Matrix must be a 1 x n or n x 1 matrix!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
    
  Vector ret = Vector(_nrows * _ncols);

  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      ret.place(row * _ncols + col, this->operator() (row, col));
    }
  }

  return ret;
}


// The Frobenius norm
template <class type, bool vectorize>
type DenseMatrix<type, vectorize>::norm(type (*pow_func)(type, type)) const {
  if (_ncols <= 0 || _nrows <= 0) {
    _errorMsg("Matrix must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  type ret = t_zero;

  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      ret += pow_func(this->operator() (row, col), (type)2.0);
    }
  }

  return pow_func(ret, (type)(1.0 / 2.0));
}

// const DenseVector DenseMatrix::mean(int dim = 0) {}

// const DenseVector DenseMatrix::sd(int dim = 0) {}


#endif