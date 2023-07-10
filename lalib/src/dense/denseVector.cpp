#include "denseVector.hpp"
#include "denseMatrix.hpp"
#include "../declare_lalib.hpp"


using namespace lalib;

// -------------------CONSTRUCTORS AND DESTRUCTORS--------------------------

// Constructor that doesn't allocate memory
DenseVector::DenseVector(void) {}

// Constructor that copies the contents of a given vector
DenseVector::DenseVector(const DenseVector& that) {
  if (that._ncols > 0 || that._nrows > 0) {
    _ncols = that._ncols;
    _nrows = that._nrows;

    total_vects = that.total_vects;

    // Allocate aligned memory
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(vect_t), total_vects * sizeof(vect_t))) {
      throw std::bad_alloc();
    }

    data = (vect_t*)tmp;

    // Copy the data from that
    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < total_vects; vect++) {
      data[vect] = that.data[vect];
    }
  }
}

// Constructor that allocates memory for wanted sized matrix and initializes
// the values as zeros
DenseVector::DenseVector(int rows, int cols) {
  if (!((rows > 0 && cols == 1) || (cols > 0 && rows == 1))) {
    throw std::invalid_argument("Improper dimensions! Either rows or cols has to be positive and the other 0!");
  }

  _ncols = cols;
  _nrows = rows;

  total_vects = _ceil(cols > 1 ? cols : rows, VECT_ELEMS);

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
  for (int vect = 0; vect < total_vects; vect++) {
    data[vect] = zeros;
  }
}

// Constructor that allocates memory for wanted sized matrix and initializes
// the values as wanted double
DenseVector::DenseVector(int rows, int cols, double init_val) {
  if (!((rows > 0 && cols == 1) || (cols > 0 && rows == 1))) {
    throw std::invalid_argument("Improper dimensions! Either rows or cols has to be positive and the other 0!");
  }

  _ncols = cols;
  _nrows = rows;

  total_vects = _ceil(cols > 1 ? cols : rows, VECT_ELEMS);

  // Allocate aligned memory
  void* tmp = 0;
  if (posix_memalign(&tmp, sizeof(vect_t), total_vects * sizeof(vect_t))) {
    throw std::bad_alloc();
  }

  // Initialize the data values as zeros
  data = (vect_t*)tmp;
  vect_t vals;
  for (int i = 0; i < VECT_ELEMS; i++) {
    vals[i] = init_val;
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int vect = 0; vect < total_vects; vect++) {
    data[vect] = vals;
  }
}

// Constructor that copies the contents of a std::vector into a matrix.
// NOTE! If the number of elements in the std::vector doesn't match the 
// dimensions of the matrix either the extra elements are ignored or 
// the matrix is padded with zeros at the last rows. In either case a 
// warning is printed.
DenseVector::DenseVector(int rows, int cols, std::vector<double> elems) {
  if (!((rows > 0 && cols == 1) || (cols > 0 && rows == 1))) {
    throw std::invalid_argument("Improper dimensions! Either rows or cols has to be positive and the other 0!");
  }
  if (rows * cols != (int)elems.size()) {
    std::cout << "\nWARNING: Given dimensions don't match with the size of the std::vector!" << "\n\n";
  } 

  _ncols = cols;
  _nrows = rows;

  total_vects = _ceil(cols > 1 ? cols : rows, VECT_ELEMS);

  // Allocate aligned memory
  void* tmp = 0;
  if (posix_memalign(&tmp, sizeof(vect_t), total_vects * sizeof(vect_t))) {
    throw std::bad_alloc();
  }

  data = (vect_t*)tmp;

  for (int vect = 0; vect < total_vects; vect++) {
    for (int elem = 0; elem < VECT_ELEMS; elem++) {
      int i = vect * VECT_ELEMS + elem;
      data[vect][elem] = (int)elems.size() > i ? elems[i] : 0.0;
    }
  }
}

// Constructor that copies the contents of double array into a matrix.
// NOTE! SHOULD NOT BE USED UNLESS ABSOLUTELY NECESSARY! This function will
// read the needed amount of elements from the array independent of the size
// of the array (which can not be verified) and thus might read unwanted memory.
DenseVector::DenseVector(int rows, int cols, double* elems) {
  std::cout << "\nWARNING: Initializing a vector with double array might lead to undefined behaviour!" << "\n\n";
 
  if (!((rows > 0 && cols == 1) || (cols > 0 && rows == 1))) {
    throw std::invalid_argument("Improper dimensions! Either rows or cols has to be positive and the other 0!");
  }

  _ncols = cols;
  _nrows = rows;

  total_vects = _ceil(cols > 1 ? cols : rows, VECT_ELEMS);

  // Allocate aligned memory
  void* tmp = 0;
  if (posix_memalign(&tmp, sizeof(vect_t), total_vects * sizeof(vect_t))) {
    throw std::bad_alloc();
  }

  data = (vect_t*)tmp;

  for (int vect = 0; vect < total_vects; vect++) {
    for (int elem = 0; elem < VECT_ELEMS; elem++) {
      int i = vect * VECT_ELEMS + elem;
      data[vect][elem] = elems[i];
    }
  }
}

DenseVector::~DenseVector() {
  free(data);
}


// ---------------------OVERLOADED BASIC MATH OPERATORS------------------------

DenseVector& DenseVector::operator+= (const DenseVector& that) {
  if ((_ncols > 1 ? _ncols : _nrows) != (that._ncols > 1 ? that._ncols : that._nrows)) {
    throw std::invalid_argument("Vectors must have equal amount of elements!");
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int vect = 0; vect < total_vects; vect++) {
    data[vect] = data[vect] + that.data[vect];
  }

  return *this;
}

const DenseVector DenseVector::operator+ (const DenseVector& that) const {
  return DenseVector(*this) += that;
}

DenseVector& DenseVector::operator-= (const DenseVector& that) {
  if ((_ncols > 1 ? _ncols : _nrows) != (that._ncols > 1 ? that._ncols : that._nrows)) {
    throw std::invalid_argument("Vectors must have equal amount of elements!");
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int vect = 0; vect < total_vects; vect++) {
    data[vect] = data[vect] - that.data[vect];
  }

  return *this;
}

const DenseVector DenseVector::operator- (const DenseVector& that) const {
  return DenseVector(*this) -= that;
}

DenseVector& DenseVector::operator*= (const DenseVector& that) {
  if ((_ncols > 1 ? _ncols : _nrows) != (that._ncols > 1 ? that._ncols : that._nrows)) {
    throw std::invalid_argument("Vectors must have equal amount of elements!");
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int vect = 0; vect < total_vects; vect++) {
    data[vect] = data[vect] * that.data[vect];
  }

  return *this;
}

const DenseVector DenseVector::operator* (const DenseVector& that) const {
  return DenseVector(*this) *= that;
}

const DenseVector DenseVector::operator* (const double that) const {
  if (_ncols < 1 || _nrows < 1) {
    return *this;
  }

  DenseVector ret = DenseVector(*this);

  vect_t mult;
  for (int i = 0; i < VECT_ELEMS; i++) {
    mult[i] = that;
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int vect = 0; vect < total_vects; vect++) {
    ret.data[vect] = ret.data[vect] * mult;
  }

  return ret;    
}

const DenseVector operator* (double scalar, const DenseVector& vector) {
  return vector * scalar;
}

DenseVector& DenseVector::operator/= (const DenseVector& that) {
  if ((_ncols > 1 ? _ncols : _nrows) != (that._ncols > 1 ? that._ncols : that._nrows)) {
    throw std::invalid_argument("Vectors must have equal amount of elements!");
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int vect = 0; vect < total_vects; vect++) {
    data[vect] = data[vect] / that.data[vect];
  }

  return *this;
}

const DenseVector DenseVector::operator/ (const DenseVector& that) const {
  return DenseVector(*this) /= that;
}

const DenseVector DenseVector::operator/ (const double that) const {
  if (that == 0) {
    throw std::invalid_argument("Division by zero undefined!");
  }
  if (_ncols < 1 || _nrows < 1) {
    return *this;
  }

  DenseVector ret = DenseVector(*this);

  vect_t div;
  for (int i = 0; i < VECT_ELEMS; i++) {
    div[i] = that;
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int vect = 0; vect < total_vects; vect++) {
    ret.data[vect] = ret.data[vect] / div;
  }

  return ret;
}


// ---------------------OVERLOADED INDEXING OPERATORS---------------------------

void DenseVector::place(int num, double val) {
  if ((_ncols > 1 ? _ncols : _nrows) < num) {
    throw std::invalid_argument("Index out of bounds!");
  }

  const int vect = num / VECT_ELEMS;
  const int elem = num % VECT_ELEMS;

  data[vect][elem] = val;
}

void DenseVector::place(int start, int end, DenseVector vector) {
  if ((_ncols > 1 ? _ncols : _nrows) < end - start) {
    throw std::invalid_argument("Given dimensions out of bounds!");
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < end - start; i++) {
    this->place(i + start, vector(i));
  }
}

double DenseVector::operator() (int num) const {
  if ((_ncols > 1 ? _ncols : _nrows) < num) {
    throw std::invalid_argument("Index out of bounds!");
  }

  const int vect = num / VECT_ELEMS;
  const int elem = num % VECT_ELEMS;

  return data[vect][elem];
}

double DenseVector::operator[] (int num) const {
  return this->operator() (num);
}

double DenseVector::get(int num) const {
  return this->operator() (num);
}

const DenseVector DenseVector::operator() (int start, int end) const {
  if (start >= end || start < 0) {
    throw std::invalid_argument("Improper dimensions given!");
  }

  int max = (_ncols > 1 ? _ncols : _nrows);

  if (end >= max) {
    std::cout << "\nWARNING: End index out of bounds" << "\n\n";
  }

  end = end > max ? max : end;
    
  DenseVector ret = DenseVector(_nrows > 0 ? end - start : 0, _ncols > 0 ? end - start : 0);

  for (int i = 0; i < end - start; i++) {
    ret.place(i, this->operator() (i + start));
  }

  return ret;
}

const DenseVector DenseVector::get(int start, int end) const {
  return this->operator() (start, end);
}

vect_t DenseVector::getSIMD(int num) const {
  if (num > total_vects) {
    throw std::invalid_argument("Index out of bounds!");
  }

  return data[num / VECT_ELEMS];
}



// ----------------------OTHER OVERLOADED OPERATORS-----------------------------

DenseVector& DenseVector::operator= (const DenseVector& that) {
  // Check for self-assignment ie. case where a = a is called by comparing the pointers of the objects
  if (this == &that) return *this; 

  // Free the existing memory and allocate new one that matches the dimensions of that
  free(data);

  _ncols = that._ncols;
  _nrows = that._nrows;

  total_vects = that.total_vects;

  // Allocate aligned memory
  void* tmp = 0;
  if (posix_memalign(&tmp, sizeof(vect_t), total_vects * sizeof(vect_t))) {
    throw std::bad_alloc();
  }

  data = (vect_t*)tmp;

  // Copy the data from that
  #pragma omp parallel for schedule(dynamic, 1)
  for (int vect = 0; vect < total_vects; vect++) {
    data[vect] = that.data[vect];
  }
  
  return *this;
}

bool DenseVector::operator== (const DenseVector& that) {
  if (_nrows != that._nrows || _ncols != that._ncols) {
    return false;
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

bool DenseVector::operator!= (const DenseVector& that) {
  return !(*this == that);
}

std::ostream& lalib::operator<<(std::ostream& os, DenseVector& v) {
  if (v.ncols() == 0 && v.nrows() == 0) {
    os << "[]" << std::endl;  // Signifies uninitialized vector
        
    return os;
  }
    
  if (v.ncols() > v.nrows()) {
    os << "[";
    for (int i = 0; i < v.ncols(); i++) {
      if (i > 0) {
	os << " ";
      }
      os << v(i);
    }
    os << "]" << std::endl;
  }
  else {
    os << "[";
    for (int i = 0; i < v.nrows(); i++) {
      if (i > 0) {
	os << "\n ";
      }
      os << v(i);
    }
    os << "]" << std::endl;
  }
  
  return os;
}


// ----------------------------------MISC----------------------------------------


bool DenseVector::isclose(const DenseVector& that, double tol) {
  if (_nrows != that._nrows || _ncols != that._ncols) {
    return false;
  }

  for (int vect = 0; vect < total_vects; vect++) {
    for (int elem = 0; elem < VECT_ELEMS; elem++) {
      if (fabs(data[vect][elem] - that.data[vect][elem]) > tol) {
	return false;
      }
    }
  }

  return true;
}


const DenseVector DenseVector::transpose() const {
  DenseVector ret = DenseVector(*this);

  ret._nrows = _ncols;
  ret._ncols = _nrows;

  return ret;
}

const DenseVector DenseVector::T() const {
  return this->transpose();
}

std::vector<double> DenseVector::toVector() const {
  if (_ncols <= 0 || _nrows <= 0) {
    throw std::invalid_argument("Vector must be initialized!");
  }

  std::vector<double> ret;
  ret.reserve(_ncols * _nrows);
  
  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      ret.push_back(this->operator() (row * _nrows + col));
    }
  }

  return ret;
}


const DenseMatrix DenseVector::asDenseMatrix() const {
  if (_ncols < 1 && _nrows < 1) {
    throw std::invalid_argument("Vector must be initialized!");
  }
    
  DenseMatrix ret = DenseMatrix(_nrows, _ncols);

  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      ret.place(row, col, this->operator() (row * _ncols + col));
    }
  }
  
  return ret;
}

double DenseVector::asDouble() const {
  if (_ncols != 1 || _nrows != 1) {
    throw std::invalid_argument("Vector must be a 1 x 1 Vector!");
  }

  return this->operator() (0);
}

double DenseVector::norm(double p) const {
  if (_ncols <= 0 || _nrows <= 0) {
    throw std::invalid_argument("Vector must be initialized!");
  }
  
  double ret = 0;

  for (int row = 0; row < _nrows; row++) {
    for (int col = 0; col < _ncols; col++) {
      ret += pow(this->operator() (row * _nrows + col), p);
    }
  }

  return pow(ret, 1.0 / p);
}

// double DenseVector::mean() {}
// double DenseVector::sd() {}


