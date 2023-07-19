#include "crsVector.hpp"
#include "crsMatrix.hpp"
#include "../declare_lalib.hpp"


using namespace lalib;


// -------------------CONSTRUCTORS AND DESTRUCTORS--------------------------

// Constructor that doesn't allocate memory
CRSVector::CRSVector(void) {}

// Constructor that copies the contents of a given vector
CRSVector::CRSVector(const CRSVector& that) {
  if (that._len > 0) {
    _len = that._len;

    data = that.data;
  }
}

// Constructor that allocates memory for wanted sized vector and initializes
// the values as zeros
CRSVector::CRSVector(int len) {
  if (len < 1) {
    throw std::invalid_argument(_formErrorMsg("Vector length must be positive!", __FILE__, __func__, __LINE__));
  }

  _len = len;

  data = std::vector<double>(len, 0.0); 
}

// Constructor that allocates memory for wanted sized vector and initializes
// the values as wanted double
CRSVector::CRSVector(int len, double init_val) {
  if (len < 1) {
    throw std::invalid_argument(_formErrorMsg("Vector length must be positive!", __FILE__, __func__, __LINE__));
  }

  _len = len;

  data = std::vector<double>(len, init_val); 
}

// Constructor that copies the contents of a std::vector into a matrix.
// NOTE! If the number of elements in the std::vector doesn't match the 
// dimensions of the matrix either the extra elements are ignored or 
// the matrix is padded with zeros at the last rows. In either case a 
// warning is printed.
CRSVector::CRSVector(int len, std::vector<double> elems) {
  if (len < 1) {
    throw std::invalid_argument(_formErrorMsg("Vector length must be positive!", __FILE__, __func__, __LINE__));
  }
  
  if (len != (int)elems.size()) {
    std::cout << "\nWARNING: Given length doesn't match with the size of the std::vector!" << "\n\n";
  }

  _len = len;

  if (len > (int)elems.size()) {
    data.reserve(len);
    
    for (int i = 0; i < len; i++) {
      if (i < (int)elems.size()) data.push_back(elems[i]);
      else data.push_back(0.0);
    }

  }
  else {
    std::copy(elems.begin(), elems.begin() + len, std::back_inserter(data));
  }
}

// Constructor that copies the contents of double array into a matrix.
// NOTE! SHOULD NOT BE USED UNLESS ABSOLUTELY NECESSARY! This function will
// read the needed amount of elements from the array independent of the size
// of the array (which can not be verified) and thus might read unwanted memory
CRSVector::CRSVector(int len, double* elems) {
  std::cout << "\nWARNING: Initializing a vector with double array might lead to undefined behaviour!" << "\n\n";

  if (len < 1) {
    throw std::invalid_argument(_formErrorMsg("Vector length must be positive!", __FILE__, __func__, __LINE__));
  }

  _len = len;

  data.assign(elems, elems + len);
}

// Destructor not needed


// ---------------------OVERLOADED BASIC MATH OPERATORS------------------------

CRSVector& CRSVector::operator+= (const CRSVector& that) {
  if (_len != that._len) {
    throw std::invalid_argument(_formErrorMsg("Vector lengths must match!", __FILE__, __func__, __LINE__));
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < _len; i++) {
    data[i] = data[i] + that.data[i];
  }

  return *this;
}

const CSRVector CRSVector::operator+ (const CRSVector& that) const {
  return CRSVector(*this) += that;
}

CRSVector& CRSVector::operator-= (const CRSVector& that) {
  if (_len != that._len) {
    throw std::invalid_argument(_formErrorMsg("Vector lengths must match!", __FILE__, __func__, __LINE__));
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < _len; i++) {
    data[i] = data[i] - that.data[i];
  }

  return *this;
}

const CSRVector CRSVector::operator- (const CRSVector& that) const {
  return CRSVector(*this) -= that;
}

CRSVector& CRSVector::operator*= (const CRSVector& that) {
  if (_len != that._len) {
    throw std::invalid_argument(_formErrorMsg("Vector lengths must match!", __FILE__, __func__, __LINE__));
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < _len; i++) {
    data[i] = data[i] * that.data[i];
  }

  return *this;
}

const CSRVector CRSVector::operator* (const CRSVector& that) const {
  return CRSVector(*this) *= that;
}

const CRSVector CRSVector::operator* (const double that) const {
  if (_len < 1) return *this;

  CRSMatrix ret = CRSMatrix(*this);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < _len; i++) {
    ret.data[i] = ret.data[i] * that;
  }

  return ret;
}

const CRSVector lalib::operator* (double scalar, const CRSVector& vector) {
  return vector * scalar;
}

CRSVector& CRSVector::operator/= (const CRSVector& that) {
  if (_len != that._len) {
    throw std::invalid_argument(_formErrorMsg("Vector lengths must match!", __FILE__, __func__, __LINE__));
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < _len; i++) {
    data[i] = data[i] / that.data[i];
  }

  return *this;
}

const CSRVector CRSVector::operator/ (const CRSVector& that) const {
  return CRSVector(*this) /= that;
}

const CRSVector CRSVector::operator/ (const double that) const {
  if (_len < 1) return *this;

  CRSMatrix ret = CRSMatrix(*this);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < _len; i++) {
    ret.data[i] = ret.data[i] / that;
  }

  return ret;
}


// ---------------------OVERLOADED INDEXING OPERATORS---------------------------

void CRSVector::place(int num, double val) {
  if (num < 0 || num >= _len) {
    throw std::invalid_argument(_formErrorMsg("Index out of bounds!", __FILE__, __func__, __LINE__));
  }

  data[num] = val;
}

void CRSVector::place(int start, int end, CRSVector vector) {
  if (_len < end - start || start < 0 || start >= end) {
    throw std::invalid_argument(_formErrorMsg("Given dimensions out of bounds!", __FILE__, __func__, __LINE__));
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < end - start; i++) {
    this->place(i + start, vector(i));
  }
}

double CRSMatrix::operator() (int num) const {
  if (num < 0 || num >= _len) {
    throw std::invalid_argument(_formErrorMsg("Index out of bounds!", __FILE__, __func__, __LINE__));
  }

  return data[num];
}

double CRSMatrix::operator[] (int num) const {
  return this->operator() (num);
}

double CRSMatrix::get(int num) const {
  return this->operator() (num);
}

const CRSVector CRSVector::operator() (int start, int end) const {
  if (_len < end - start || start < 0 || start >= end) {
    throw std::invalid_argument(_formErrorMsg("Given dimensions out of bounds!", __FILE__, __func__, __LINE__));
  }

  if (end >= _len) {
    std::cout << "\nWARNING: End index out of bounds" << "\n\n";
    end = _len;
  }

  CRSVector ret = CRSVector(end - start);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < end - start; i++) {
    ret.place(i, this->operator() (i + start));
  }

  return ret;
}

const CRSVector CRSVector::get(int start, int end) {
  return this->operator() (start, end);
}


// ----------------------OTHER OVERLOADED OPERATORS-----------------------------

CRSVector& CRSVector::operator= (const CRSVector& that) {
  // Check for self-assignment ie. case where a = a is called by comparing the pointers of the objects
  if (this == &that) return *this;

  data = that.data;
  _len = that._len;
}

bool CRSVector::operator== (const CRSVector& that) {
  if (_len != that._len) return false;

  for (int i = 0; i < _len; i++) {
    if (this->operator() (i) != that(i)) {
      return false;
    }
  }

  return true;
}

bool CRSVector::operator!= (const CRSVector& that) {
  return !(*this == that);
}


// ----------------------------------MISC----------------------------------------

bool CRSVector::isclose(const CRSVector& that, double tol) {
  if (_len != that._len) return false;

  for (int i = 0; i < _len; i++) {
    if (fabs(this->operator() (i) - that(i)) > tol) {
      return false;
    }
  }

  return true;
}

std::vector<double> CRSVector::toVector() const {
  if (_len < 1) {
    throw std::invalid_argument(_formErrorMsg("Vector must be initialized!", __FILE__, __func__, __LINE__));
  }

  std::vector<double> ret = data;

  return ret;
}

const CRSMatrix CRSVector::asCRSMatrix() const {
  if (_len < 1) {
    throw std::invalid_argument(_formErrorMsg("Vector must be initialized!", __FILE__, __func__, __LINE__));
  }

  CRSMatrix ret = CRSMatrix(_len, 1);  // By default a vector is considered as a column vector

  for (int i = 0; i < _len; i++) {
    ret.place(i, 1, this->operator() (i));
  }

  return ret;
}

double CRSVector::asDouble() const {
  if (_len != 1) {
    throw std::invalid_argument(_formErrorMsg("Vector must have just a single element!", __FILE__, __func__, __LINE__));
  }

  return this->operator() (0);
}

double CRSVector::norm(double p) const {
  if (_len < 1) {
    throw std::invalid_argument(_formErrorMsg("Vector must be initialized!", __FILE__, __func__, __LINE__));
  }

  double ret = 0.0;

  for (int i = 0; i < _len; i++) {
    ret += pow(this->operator() (i), p);
  }

  return pow(ret, 1.0 / p);
}
