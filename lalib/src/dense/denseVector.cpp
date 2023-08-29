#include "denseVector.hpp"
#include "denseMatrix.hpp"
#include "../declare_lalib.hpp"


using namespace lalib;

// -------------------CONSTRUCTORS AND DESTRUCTORS--------------------------


// Default constructor
DenseVector::DenseVector(void) {}


// Copying constructor
DenseVector::DenseVector(const DenseVector& that) {
  if (that._len > 0) {
    _len = that._len;

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

// Zeros constructor
DenseVector::DenseVector(int len) {
  if (len > 0) {
    _len = len;

    total_vects = _ceil(len, VECT_ELEMS);

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
}


// Default value constructor
DenseVector::DenseVector(int len, double init_val) {
  if (len > 0) {
    _len = len;
    total_vects = _ceil(len, VECT_ELEMS);

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
}


// Vector copying constructor
DenseVector::DenseVector(int len, std::vector<double>& elems) {
  if (len > 0) {
    
    if (len != (int)elems.size()) {
      _warningMsg("Given dimensions don't match with the size of the std::vector!", __func__);
    } 

    _len = len;
  
    total_vects = _ceil(len, VECT_ELEMS);

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
}


// Array copying constructor
DenseVector::DenseVector(int len, double* elems) {
  _warningMsg("Initializing a vector with double array might lead to undefined behaviour!", __func__);
 
  if (len > 0) {
    _len = len;

    total_vects = _ceil(len, VECT_ELEMS);

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
}

DenseVector::~DenseVector() {
  free(data);
}


// ---------------------OVERLOADED BASIC MATH OPERATORS------------------------


// Element-wise addition assignment
DenseVector& DenseVector::operator+= (const DenseVector& that) {
  if (_len != that._len) {
    _errorMsg("Vectors must have equal amount of elements!", __FILE__, __func__, __LINE__);
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int vect = 0; vect < total_vects; vect++) {
    data[vect] = data[vect] + that.data[vect];
  }

  return *this;
}


// Element-wise addition
const DenseVector DenseVector::operator+ (const DenseVector& that) const {
  return DenseVector(*this) += that;
}


// Element-wise subtraction assignment
DenseVector& DenseVector::operator-= (const DenseVector& that) {
  if (_len != that._len) {
    _errorMsg("Vectors must have equal amount of elements!", __FILE__, __func__, __LINE__);
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int vect = 0; vect < total_vects; vect++) {
    data[vect] = data[vect] - that.data[vect];
  }

  return *this;
}


// Element-wise subtraction
const DenseVector DenseVector::operator- (const DenseVector& that) const {
  return DenseVector(*this) -= that;
}


// Element-wise multiplication assignment
DenseVector& DenseVector::operator*= (const DenseVector& that) {
  if (_len != that._len) {
    _errorMsg("Vectors must have equal amount of elements!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int vect = 0; vect < total_vects; vect++) {
    data[vect] = data[vect] * that.data[vect];
  }

  return *this;
}


// Element-wise multiplication
const DenseVector DenseVector::operator* (const DenseVector& that) const {
  return DenseVector(*this) *= that;
}


// Scalar (right) multiplication
const DenseVector DenseVector::operator* (const double that) const {
  if (_len < 1) {
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


// Scalar (left) multiplication
const DenseVector operator* (double scalar, const DenseVector& vector) {
  return vector * scalar;
}


// Element-wise division assignment
DenseVector& DenseVector::operator/= (const DenseVector& that) {
  if (_len != that._len) {
    _errorMsg("Vectors must have equal amount of elements!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  } 

  #pragma omp parallel for schedule(dynamic, 1)
  for (int vect = 0; vect < total_vects; vect++) {
    data[vect] = data[vect] / that.data[vect];
  }

  return *this;
}


// Element-wise division
const DenseVector DenseVector::operator/ (const DenseVector& that) const {
  return DenseVector(*this) /= that;
}


// Scalar division
const DenseVector DenseVector::operator/ (const double that) const {
  if (that == 0) {
    _errorMsg("Division by zero undefined!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
  if (_len < 1) {
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


// Standard single value placement
void DenseVector::place(int num, double val) {
  if (_len < num) {
    _errorMsg("Index out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  const int vect = num / VECT_ELEMS;
  const int elem = num % VECT_ELEMS;

  data[vect][elem] = val;
}


// Standard vector placement
void DenseVector::place(int start, int end, DenseVector& vector) {
  if (_len < end - start) {
    _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < end - start; i++) {
    this->place(i + start, vector(i));
  }
}


// Standard indexing method
double DenseVector::operator() (int num) const {
  if (_len < num || num < 0) {
    _errorMsg("Index out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  const int vect = num / VECT_ELEMS;
  const int elem = num % VECT_ELEMS;

  return data[vect][elem];
}


// Squared bracket indexing method
double DenseVector::operator[] (int num) const {
  return this->operator() (num);
}


// Named indexing method
double DenseVector::get(int num) const {
  return this->operator() (num);
}


// Standard slicing method
const DenseVector DenseVector::operator() (int start, int end) const {
  if (start >= end || start < 0) {
    _errorMsg("Improper dimensions given!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (end >= _len) {
    _warningMsg("End index out of bounds", __func__);
  }

  end = end > _len ? _len : end;
    
  DenseVector ret = DenseVector(end - start);

  for (int i = 0; i < end - start; i++) {
    ret.place(i, this->operator() (i + start));
  }

  return ret;
}


// Named slicing method
const DenseVector DenseVector::get(int start, int end) const {
  return this->operator() (start, end);
}


// SIMD accessing method
vect_t DenseVector::getSIMD(int num) const {
  if (num >= total_vects) {
    _errorMsg("Index out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__));
  }

  return data[num];
}


// ----------------------OTHER OVERLOADED OPERATORS-----------------------------


// Default assignment operator
DenseVector& DenseVector::operator= (const DenseVector& that) {
  // Check for self-assignment ie. case where a = a is called by comparing the pointers of the objects
  if (this == &that) return *this; 

  // Free the existing memory and allocate new one that matches the dimensions of that
  free(data);

  _len = that._len;

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


// Default (equality) comparison operator
bool DenseVector::operator== (const DenseVector& that) {
  if (_len != that._len) {
    return false;
  }

  for (int i = 0; i < _len; i++) {
    if (this->operator() (i) != that(i)) {
      return false;
    }
  }

  return true;
}


// Default (inequality) comparison operator
bool DenseVector::operator!= (const DenseVector& that) {
  return !(*this == that);
}


// Default insertion operator
std::ostream& lalib::operator<<(std::ostream& os, DenseVector& v) {
  if (v.len() == 0) {
    os << "[]" << std::endl;  // Signifies uninitialized vector
        
    return os;
  }
    
  os << "[";
  for (int i = 0; i < v.len(); i++) {
    if (i > 0) {
      os << "\n ";
    }
    os << v(i);
  }
  os << "]" << std::endl;
  
  return os;
}


// ----------------------------------MISC----------------------------------------


// Approximative equality comparison
bool DenseVector::isclose(const DenseVector& that, double tol) {
  if (_len != that._len) {
    return false;
  }

  for (int i = 0; i < _len; i++) {
    if (fabs(this->operator() (i) - that(i)) > tol) {
      return false;
    }
  }

  return true;
}


// Convert DenseVector into std::vector
std::vector<double> DenseVector::toVector() const {
  if (_len < 1) {
    _errorMsg("Vector must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__));
  }

  std::vector<double> ret;
  ret.reserve(_len);
  
  for (int i = 0; i < _len; i++) {
    ret.push_back(this->operator() (i));
  }

  return ret;
}


// Convert DenseVector object into a DenseMatrix object
const DenseMatrix DenseVector::asDenseMatrix() const {
  if (_len < 1) {
    _errorMsg("Vector must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
    
  DenseMatrix ret = DenseMatrix(_len, 1);

  for (int i = 0; i < _len; i++) {
    ret.place(i, 0, this->operator() (i));
  }
  
  return ret;
}


// Convert DenseVector into a double
double DenseVector::asDouble() const {
  if (_len != 1) {
    _errorMsg("Vector must be a 1 x 1 Vector!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  return this->operator() (0);
}


// The l_p norm
double DenseVector::norm(double p) const {
  if (_len < 1) {
    _errorMsg("Vector must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
  
  double ret = 0;

  for (int i = 0; i < _len; i++) {
    ret += pow(this->operator() (i), p);
  }

  return pow(ret, 1.0 / p);
}

// double DenseVector::mean() {}
// double DenseVector::sd() {}


