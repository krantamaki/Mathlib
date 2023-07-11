#include "denseMatrix.hpp"
#include "denseVector.hpp"
#include "../declare_lalib.hpp"


using namespace lalib;

// -------------------CONSTRUCTORS AND DESTRUCTORS--------------------------

// Constructor that doesn't allocate memory
DenseMatrix::DenseMatrix(void) {}

// Constructor that copies the contents of a given matrix
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

// Constructor that allocates memory for wanted sized matrix and initializes
// the values as zeros
DenseMatrix::DenseMatrix(int rows, int cols) {
  if (cols < 1 || rows < 1) {
    throw std::invalid_argument("Matrix dimensions must be positive!");
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
  vect_t zeros;
  for (int i = 0; i < VECT_ELEMS; i++) {
    zeros[i] = 0.0;
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < rows; i++) {
    for (int vect = 0; vect < vects_per_row; vect++) {
      data[vects_per_row * i + vect] = zeros;
    }
  }
}

// Constructor that allocates memory for wanted sized matrix and initializes
// the values as a given double
DenseMatrix::DenseMatrix(int rows, int cols, double init_val) {
  if (cols < 1 || rows < 1) {
    throw std::invalid_argument("Matrix dimensions must be non-negative!");
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

// Constructor that copies the contents of a std::vector into a matrix.
// NOTE! If the number of elements in the std::vector doesn't match the 
// dimensions of the matrix either the extra elements are ignored or 
// the matrix is padded with zeros at the last rows. In either case a 
// warning is printed.
DenseMatrix::DenseMatrix(int rows, int cols, std::vector<double> elems) {
  if (cols < 1 || rows < 1) {
    throw std::invalid_argument("Matrix dimensions must be non-negative!");
  }
  if (rows * cols != (int)elems.size()) {
    std::cout << "\nWARNING: Given dimensions don't match with the size of the std::vector!" << "\n\n";
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

// Constructor that copies the contents of a double array into a matrix.
// NOTE! SHOULD NOT BE USED UNLESS ABSOLUTELY NECESSARY! This function will
// read the needed amount of elements from the array independent of the size
// of the array (which can not be verified) and thus might read unwanted memory.
DenseMatrix::DenseMatrix(int rows, int cols, double* elems) {
  std::cout << "\nWARNING: Initializing a matrix with double array might lead to undefined behaviour!" << "\n\n";

  if (cols < 1 || rows < 1) {
    throw std::invalid_argument("Matrix dimensions must be non-negative!");
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

DenseMatrix::~DenseMatrix() {
  free(data);
}


// ---------------------OVERLOADED BASIC MATH OPERATORS------------------------

DenseMatrix& DenseMatrix::operator+= (const DenseMatrix& that) {
  if (_ncols != that._ncols || _nrows != that._nrows) {
    throw std::invalid_argument("Matrix dimensions must match!");
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < _nrows; row++) {
    for (int vect = 0; vect < vects_per_row; vect++) {
      data[vects_per_row * row + vect] = data[vects_per_row * row + vect] + that.data[that.vects_per_row * row + vect];
    }
  }
  
  return *this;
}

const DenseMatrix DenseMatrix::operator+ (const DenseMatrix& that) const {
  return DenseMatrix(*this) += that;
}

DenseMatrix& DenseMatrix::operator-= (const DenseMatrix& that) {
  if (_ncols != that._ncols || _nrows != that._nrows) {
    throw std::invalid_argument("Matrix dimensions must match!");
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < _nrows; row++) {
    for (int vect = 0; vect < vects_per_row; vect++) {
      data[vects_per_row * row + vect] = data[vects_per_row * row + vect] - that.data[that.vects_per_row * row + vect];
    }
  }

  return *this;
}

const DenseMatrix DenseMatrix::operator- (const DenseMatrix& that) const {
  return DenseMatrix(*this) -= that;
}

// NOTE! The overloaded operator for multiplication will do an element-wise multiplication
// For standard matrix multiplication use DenseMatrix::matmul
DenseMatrix& DenseMatrix::operator*= (const DenseMatrix& that) {
  if (_ncols != that._ncols || _nrows != that._nrows) {
    throw std::invalid_argument("Matrix dimensions must match!");
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < nrows(); row++) {
    for (int vect = 0; vect < vects_per_row; vect++) {
      data[vects_per_row * row + vect] = data[vects_per_row * row + vect] * that.data[that.vects_per_row * row + vect];		}
  }

  return *this;
}

const DenseMatrix DenseMatrix::operator* (const DenseMatrix& that) const {
  return DenseMatrix(*this) *= that;
}

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

const DenseMatrix lalib::operator* (double scalar, const DenseMatrix& matrix) {
  return matrix * scalar;
}

// NOTE! The overloaded operator for division will do an element-wise division and is not equivalent to
// Matlabs backslash operator
DenseMatrix& DenseMatrix::operator/= (const DenseMatrix& that) {
  if (ncols() != that._ncols || nrows() != that._ncols) {
    throw std::invalid_argument("Matrix dimensions must match!");
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < nrows(); row++) {
    for (int vect = 0; vect < vects_per_row; vect++) {
      data[vects_per_row * row + vect] = data[vects_per_row * row + vect] / that.data[that.vects_per_row * row + vect];
    }
  }

  return *this;
}

const DenseMatrix DenseMatrix::operator/ (const DenseMatrix& that) const {
  return DenseMatrix(*this) /= that;
}

const DenseMatrix DenseMatrix::operator/ (const double that) const {
  if (that == 0) {
    throw std::invalid_argument("Division by zero undefined!");
  }

  if (_ncols < 1 || _nrows < 1) {
    return *this;
  }

  vect_t div;
  for (int i = 0; i < VECT_ELEMS; i++) {
    div[i] = that;
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int row = 0; row < _nrows; row++) {
    for (int vect = 0; vect < vects_per_row; vect++) {
      data[vects_per_row * row + vect] = data[vects_per_row * row + vect] / div;
    }
  }

  return *this;
}


// ----------------------OVERLOADED INDEXING OPERATORS--------------------------

void DenseMatrix::place(int row, int col, double val) {
  if (row < 0 || col < 0 || row >= _nrows || col >= _ncols) {
    throw std::invalid_argument("Given dimensions out of bounds!");
  }

  // Find the proper vector and element in said vector for column col
  const int vect = col / VECT_ELEMS;
  const int elem = col % VECT_ELEMS;

  data[vects_per_row * row + vect][elem] = val;
}

void DenseMatrix::place(int rowStart, int rowEnd, int colStart, int colEnd, DenseMatrix mat) {
  // Check that the matrix to be placed fits
  if (_nrows < rowEnd - rowStart || _ncols < colEnd - colStart || mat._nrows < rowEnd - rowStart || mat._ncols < colEnd - colStart) {
    throw std::invalid_argument("Given dimensions out of bounds!");
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

double DenseMatrix::operator() (int row, int col) const {
  if (row < 0 || col < 0 || row >= _nrows || col >= _ncols) {
    throw std::invalid_argument("Given dimensions out of bounds!");
  }

  // Find the proper vector and element in said vector for column col
  const int vect = col / VECT_ELEMS;
  const int elem = col % VECT_ELEMS;

  return data[vects_per_row * row + vect][elem];
}

double DenseMatrix::get(int row, int col) const {
  return this->operator() (row, col);
}

double DenseMatrix::operator[] (int num) const {
  if (num >= _ncols * _nrows) {
    throw std::invalid_argument("Given index out of bounds!");
  }

  const int row = num / _ncols;
  const int col = num % _ncols;

  const int vect = col / VECT_ELEMS;
  const int elem = col % VECT_ELEMS;

  return data[vects_per_row * row + vect][elem];
}

const DenseMatrix DenseMatrix::operator() (int rowStart, int rowEnd, int colStart, int colEnd) const {
  if (rowStart >= rowEnd || rowStart < 0 || colStart >= colEnd || colStart < 0) {
    throw std::invalid_argument("Improper dimensions given!");
  }

  if (rowEnd > _nrows || colEnd > _ncols) {
    std::cout << "\nWARNING: End index out of bounds" << "\n\n";
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

const DenseMatrix DenseMatrix::get(int rowStart, int rowEnd, int colStart, int colEnd) const {
  return this->operator() (rowStart, rowEnd, colStart, colEnd);
}


// ------------------------OTHER OVERLOADED OPERATORS---------------------------

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

bool DenseMatrix::operator!= (const DenseMatrix& that) {
  return !(*this == that);
}

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

const DenseMatrix DenseMatrix::T() const{
  return this->transpose();
}

std::vector<double> DenseMatrix::toVector() const {
  if (_ncols <= 0 || _nrows <= 0) {
    throw std::invalid_argument("Matrix must be initialized!");
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

double DenseMatrix::asDouble() const {
  if (_ncols != 1 || _nrows != 1) {
    throw std::invalid_argument("Matrix must be a 1 x 1 matrix!");
  }

  return this->operator() (0, 0);
}

const DenseVector DenseMatrix::asDenseVector() const {
  if (_ncols != 1 && _nrows != 1) {
    throw std::invalid_argument("Matrix must be a 1 x n or n x 1 matrix!");
  }
    
  DenseVector ret = DenseVector(_nrows, _ncols);

  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      ret.place(row * _ncols + col, this->operator() (row, col));
    }
  }

  return ret;
}

double DenseMatrix::norm() const {
  if (_ncols <= 0 || _nrows <= 0) {
    throw std::invalid_argument("Matrix must be initialized!");
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


