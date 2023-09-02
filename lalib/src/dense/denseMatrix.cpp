#include "denseMatrix.hpp"
#include "denseVector.hpp"
#include "../declare_lalib.hpp"


using namespace lalib;

// -------------------CONSTRUCTORS AND DESTRUCTORS--------------------------


// Default constuctor
DenseMatrix::DenseMatrix(void) {}


// Copying constructor
DenseMatrix::DenseMatrix(const DenseMatrix& that) {
  if (that._ncols > 0 && that._nrows > 0) {
    _ncols = that._ncols;
    _nrows = that._nrows;

    vects_per_row = that.vects_per_row;
    total_vects = that.total_vects;

    // Allocate aligned memory
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(vect_t), total_vects * sizeof(vect_t))) {
      throw std::bad_alloc();
    }

    data = (vect_t*)tmp;

    // Copy the data from that
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < _nrows; i++) {
      for (int vect = 0; vect < vects_per_row; vect++) {
	data[vects_per_row * i + vect] = that.data[vects_per_row * i + vect];
      }
    } 
  }
}


// Zeros constructor
DenseMatrix::DenseMatrix(int rows, int cols) {
  if (cols < 1 || rows < 1) {
    _errorMsg("Matrix dimensions must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _ncols = cols;
  _nrows = rows;

  vects_per_row = _ceil(cols, VECT_ELEMS);
  total_vects = rows * vects_per_row;

  // Allocate aligned memory
  void* tmp = 0;
  if (posix_memalign(&tmp, sizeof(vect_t), total_vects * sizeof(vect_t))) {
    throw std::bad_alloc();
  }

  // Initialize the data values as zeros
  data = (vect_t*)tmp;
  vect_t init = zeros;

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < rows; i++) {
    for (int vect = 0; vect < vects_per_row; vect++) {
      data[vects_per_row * i + vect] = init;
    }
  }
}


// Default value constructor
DenseMatrix::DenseMatrix(int rows, int cols, double init_val) {
  if (cols < 1 || rows < 1) {
    _errorMsg("Matrix dimensions must be non-negative!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _ncols = cols;
  _nrows = rows;

  vects_per_row = _ceil(cols, VECT_ELEMS);
  total_vects = rows * vects_per_row;

  // Allocate aligned memory
  void* tmp = 0;
  if (posix_memalign(&tmp, sizeof(vect_t), total_vects * sizeof(vect_t))) {
    throw std::bad_alloc();
  }

  // Initialize the data values as init_val
  data = (vect_t*)tmp;

  vect_t init_vals;
  for (int i = 0; i < VECT_ELEMS; i++) {
    init_vals[i] = init_val;
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < rows; i++) {
    for (int vect = 0; vect < vects_per_row; vect++) {
      data[vects_per_row * i + vect] = init_vals;
    }
  }
}


// Vector copying constructor
DenseMatrix::DenseMatrix(int rows, int cols, std::vector<double>& elems) {
  if (cols < 1 || rows < 1) {
    _errorMsg("Matrix dimensions must be non-negative!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
  if (rows * cols != (int)elems.size()) {
    _warningMsg("Given dimensions don't match with the size of the std::vector!", __func__);
  } 

  _ncols = cols;
  _nrows = rows;

  vects_per_row = _ceil(cols, VECT_ELEMS);
  total_vects = rows * vects_per_row;

  // Allocate aligned memory
  void* tmp = 0;
  if (posix_memalign(&tmp, sizeof(vect_t), total_vects * sizeof(vect_t))) {
    throw std::bad_alloc();
  }

  data = (vect_t*)tmp;

  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      int vect = col / VECT_ELEMS;
      int elem = col % VECT_ELEMS;

      data[row * vects_per_row + vect][elem] = (int)elems.size() > row * _ncols + col ? elems[row * _ncols + col] : 0.0;
    }
  }
}


// Array copying constructor
DenseMatrix::DenseMatrix(int rows, int cols, double* elems) {
  _warningMsg("Initializing a matrix with double array might lead to undefined behaviour!", __func__);

  if (cols < 1 || rows < 1) {
    _errorMsg("Matrix dimensions must be non-negative!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _ncols = cols;
  _nrows = rows;

  vects_per_row = _ceil(cols, VECT_ELEMS);
  total_vects = rows * vects_per_row;

  // Allocate aligned memory
  void* tmp = 0;
  if (posix_memalign(&tmp, sizeof(vect_t), total_vects * sizeof(vect_t))) {
    throw std::bad_alloc();
  }

  data = (vect_t*)tmp;

  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      int vect = col / VECT_ELEMS;
      int elem = col % VECT_ELEMS;

      data[row * vects_per_row + vect][elem] = elems[row * _ncols + col];
    }
  }
}


// Default destructor
DenseMatrix::~DenseMatrix() {
  free(data);
}


// ---------------------OVERLOADED BASIC MATH OPERATORS------------------------


// Element-wise addition assignment
DenseMatrix& DenseMatrix::operator+= (const DenseMatrix& that) {
  if (_ncols != that._ncols || _nrows != that._nrows) {
    _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < _nrows; row++) {
    for (int vect = 0; vect < vects_per_row; vect++) {
      data[vects_per_row * row + vect] = data[vects_per_row * row + vect] + that.data[that.vects_per_row * row + vect];
    }
  }
  
  return *this;
}


// Element-wise addition
const DenseMatrix DenseMatrix::operator+ (const DenseMatrix& that) const {
  return DenseMatrix(*this) += that;
}


// Element-wise subtraction assignment
DenseMatrix& DenseMatrix::operator-= (const DenseMatrix& that) {
  if (_ncols != that._ncols || _nrows != that._nrows) {
    _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < _nrows; row++) {
    for (int vect = 0; vect < vects_per_row; vect++) {
      data[vects_per_row * row + vect] = data[vects_per_row * row + vect] - that.data[that.vects_per_row * row + vect];
    }
  }

  return *this;
}


// Element-wise subtraction
const DenseMatrix DenseMatrix::operator- (const DenseMatrix& that) const {
  return DenseMatrix(*this) -= that;
}


// Element-wise multiplication assignment
DenseMatrix& DenseMatrix::operator*= (const DenseMatrix& that) {
  if (_ncols != that._ncols || _nrows != that._nrows) {
    _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < nrows(); row++) {
    for (int vect = 0; vect < vects_per_row; vect++) {
      data[vects_per_row * row + vect] = data[vects_per_row * row + vect] * that.data[that.vects_per_row * row + vect];		}
  }

  return *this;
}


// Element-wise multiplication
const DenseMatrix DenseMatrix::operator* (const DenseMatrix& that) const {
  return DenseMatrix(*this) *= that;
}


// Scalar (right) multiplication
const DenseMatrix DenseMatrix::operator* (const double that) const {
  if (_ncols < 1 || _nrows < 1) {
    return *this;
  }

  DenseMatrix ret = DenseMatrix(*this);

  vect_t mult;
  for (int i = 0; i < VECT_ELEMS; i++) {
    mult[i] = that;
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < _nrows; row++) {
    for (int vect = 0; vect < vects_per_row; vect++) {
      ret.data[vects_per_row * row + vect] = ret.data[vects_per_row * row + vect] * mult;
    }
  }

  return ret;
}


// Scalar (left) multiplication
const DenseMatrix lalib::operator* (double scalar, const DenseMatrix& matrix) {
  return matrix * scalar;
}


// Element-wise division assignment
DenseMatrix& DenseMatrix::operator/= (const DenseMatrix& that) {
  if (ncols() != that._ncols || nrows() != that._ncols) {
    _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < nrows(); row++) {
    for (int vect = 0; vect < vects_per_row; vect++) {
      data[vects_per_row * row + vect] = data[vects_per_row * row + vect] / that.data[that.vects_per_row * row + vect];
    }
  }

  return *this;
}


// Element-wise division
const DenseMatrix DenseMatrix::operator/ (const DenseMatrix& that) const {
  return DenseMatrix(*this) /= that;
}


// Scalar (right) division
const DenseMatrix DenseMatrix::operator/ (const double that) const {
  if (that == 0) {
    _errorMsg("Division by zero undefined!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (_ncols < 1 || _nrows < 1) {
    return *this;
  }

  DenseMatrix ret = DenseMatrix(*this);

  vect_t div;
  for (int i = 0; i < VECT_ELEMS; i++) {
    div[i] = that;
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < _nrows; row++) {
    for (int vect = 0; vect < vects_per_row; vect++) {
      ret.data[vects_per_row * row + vect] = ret.data[vects_per_row * row + vect] / div;
    }
  }

  return ret;
}


// ----------------------OVERLOADED INDEXING OPERATORS--------------------------


// Standard single value placement
void DenseMatrix::place(int row, int col, double val) {
  if (row < 0 || col < 0 || row >= _nrows || col >= _ncols) {
    _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  // Find the proper vector and element in said vector for column col
  const int vect = col / VECT_ELEMS;
  const int elem = col % VECT_ELEMS;

  data[vects_per_row * row + vect][elem] = val;
}


// Standard matrix placement
void DenseMatrix::place(int rowStart, int rowEnd, int colStart, int colEnd, const DenseMatrix& mat) {
  // Check that the matrix to be placed fits
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
double DenseMatrix::operator() (int row, int col) const {
  if (row < 0 || col < 0 || row >= _nrows || col >= _ncols) {
    _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  // Find the proper vector and element in said vector for column col
  const int vect = col / VECT_ELEMS;
  const int elem = col % VECT_ELEMS;

  return data[vects_per_row * row + vect][elem];
}


// Named indexing method
double DenseMatrix::get(int row, int col) const {
  return this->operator() (row, col);
}


// Squared bracket indexing method
double DenseMatrix::operator[] (int num) const {
  if (num >= _ncols * _nrows) {
    _errorMsg("Given index out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  const int row = num / _ncols;
  const int col = num % _ncols;

  const int vect = col / VECT_ELEMS;
  const int elem = col % VECT_ELEMS;

  return data[vects_per_row * row + vect][elem];
}


// Standard slicing method
const DenseMatrix DenseMatrix::operator() (int rowStart, int rowEnd, int colStart, int colEnd) const {
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
const DenseMatrix DenseMatrix::get(int rowStart, int rowEnd, int colStart, int colEnd) const {
  return this->operator() (rowStart, rowEnd, colStart, colEnd);
}


// ------------------------OTHER OVERLOADED OPERATORS---------------------------


// Default assignment operator
DenseMatrix& DenseMatrix::operator= (const DenseMatrix& that) {
  // Check for self-assignment ie. case where a = a is called by comparing the pointers of the objects
  if (this == &that) return *this; 

  // Free the existing memory and allocate new one that matches the dimensions of that
  free(data);

  _ncols = that._ncols;
  _nrows = that._nrows;

  vects_per_row = that.vects_per_row;
  total_vects = that.total_vects;

  // Allocate aligned memory
  void* tmp = 0;
  if (posix_memalign(&tmp, sizeof(vect_t), total_vects * sizeof(vect_t))) {
    throw std::bad_alloc();
  }

  data = (vect_t*)tmp;

  // Copy the data from that
  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < _nrows; i++) {
    for (int vect = 0; vect < vects_per_row; vect++) {
      data[vects_per_row * i + vect] = that.data[vects_per_row * i + vect];
    }
  }

  return *this;
}


// Default (equality) comparison operator
bool DenseMatrix::operator== (const DenseMatrix& that) {
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
  
  for (int vect = 0; vect < total_vects; vect++) {
    for (int elem = 0; elem < VECT_ELEMS; elem++) {
      if (data[vect][elem] != that.data[vect][elem]) {
        return false;
      }
    }
  }

  return true;
}


// Default (inequality) comparison operator
bool DenseMatrix::operator!= (const DenseMatrix& that) {
  return !(*this == that);
}


// Default insertion operator
std::ostream& lalib::operator<<(std::ostream& os, DenseMatrix& A) {
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
bool DenseMatrix::isclose(const DenseMatrix& that, double tol) {
  if (_nrows != that._nrows || _ncols != that._ncols) {
    return false;
  }

  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      if (fabs(this->operator() (row, col) - that(row, col)) > tol) {
	return false;
      }
    }
  }

  return true;
}


// Standard transpose
const DenseMatrix DenseMatrix::transpose() const{
  // Check that matrix initialized
  if (_ncols <= 0 || _nrows <= 0) {
    return *this;  // Maybe change to fatal error
  }

  // Allocate memory for needed sized matrix
  DenseMatrix ret = DenseMatrix(_ncols, _nrows);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {

      const int vect = col / VECT_ELEMS;
      const int elem = col % VECT_ELEMS;

      ret.place(col, row, this->data[vects_per_row * row + vect][elem]);
    }
  }

  return ret;
}


// Standard transpose
const DenseMatrix DenseMatrix::T() const{
  return this->transpose();
}


// Convert DenseMatrix into std::vector
std::vector<double> DenseMatrix::toVector() const {
  if (_ncols <= 0 || _nrows <= 0) {
    _errorMsg("Matrix must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  std::vector<double> ret;
  ret.reserve(_ncols * _nrows);

  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      ret.push_back(this->operator() (row, col));
    }
  }

  return ret;
}


// Convert DenseMatrix into a double
double DenseMatrix::asDouble() const {
  if (_ncols != 1 || _nrows != 1) {
    _errorMsg("Matrix must be a 1 x 1 matrix!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  return this->operator() (0, 0);
}


// Convert DenseMatrix into a DenseVector
const DenseVector DenseMatrix::asDenseVector() const {
  if (_ncols != 1 && _nrows != 1) {
    _errorMsg("Matrix must be a 1 x n or n x 1 matrix!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
    
  DenseVector ret = DenseVector(_nrows, _ncols);

  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      ret.place(row * _ncols + col, this->operator() (row, col));
    }
  }

  return ret;
}


// The Frobenius norm
double DenseMatrix::norm() const {
  if (_ncols <= 0 || _nrows <= 0) {
    _errorMsg("Matrix must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  double ret = 0;

  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      ret += pow(this->operator() (row, col), 2.0);
    }
  }

  return pow(ret, 1.0 / 2.0);
}

// const DenseVector DenseMatrix::mean(int dim = 0) {}

// const DenseVector DenseMatrix::sd(int dim = 0) {}


