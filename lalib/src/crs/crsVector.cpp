#include "crsVector.hpp"
#include "crsMatrix.hpp"
#include "../declare_lalib.hpp"


using namespace lalib;


// -------------------CONSTRUCTORS AND DESTRUCTORS--------------------------


// Default constructor
CRSVector::CRSVector(void) {}


// Copying constructor
CRSVector::CRSVector(const CRSVector& that) {
  if (that._len > 0) {
    _len = that._len;

    data = that.data;
  }
}


// Zeros constructor
CRSVector::CRSVector(int len) {
  if (len < 1) {
    _errorMsg("Vector length must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _len = len;

  data = std::vector<double>(len, 0.0); 
}


// Default value constructor
CRSVector::CRSVector(int len, double init_val) {
  if (len < 1) {
    _errorMsg("Vector length must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _len = len;

  data = std::vector<double>(len, init_val); 
}


// Vector copying constructor
CRSVector::CRSVector(int len, std::vector<double>& elems) {
  if (len < 1) {
    _errorMsg("Vector length must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
  
  if (len != (int)elems.size()) {
    _warningMsg("Given length doesn't match with the size of the std::vector!", __func__);
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

// Array copying constructor
CRSVector::CRSVector(int len, double* elems) {
  _warningMsg("Initializing a vector with double array might lead to undefined behaviour!", __func__);

  if (len < 1) {
    _errorMsg("Vector length must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _len = len;

  data.assign(elems, elems + len);
}


// Load from file constructor
CRSVector::CRSVector(const std::string& path, int offset, std::string format) {
  // Variables to read the line contents to
  int row, col;
  double val;
  
  // Read the last line of the file to get the dimensions of the matrix
  std::stringstream lastLine = _lastLine(path);

  int nTokens = _numTokens(lastLine.str());

  if (format != ".dat") {
    _errorMsg("Support for other formats than .dat not implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
  
  if (nTokens == 3) {
    lastLine >> row >> col >> val;

    if (col > 1 && row > 1) {
      _errorMsg("Improper data file!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
    }

    _len = row * col + 1 - offset;

    data = std::vector<double>(_len, 0.0);

    // Start reading the lines from the beginning of the file
    std::ifstream file(path);

    while (file >> row >> col >> val) {
      this->place(row > col ? row - offset : col - offset, val);
    }
    
    file.close();
  }

  else if (nTokens == 2) {
    lastLine >> row  >> val;
    
    _len = row + 1 - offset;

    data = std::vector<double>(_len, 0.0);

    // Start reading the lines from the beginning of the file
    std::ifstream file(path);

    while (file >> row >> val) {
      this->place(row - offset, val);
    }

    file.close();
  }
  else {
    _errorMsg("Improper data file!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
}

// Destructor not needed


// ---------------------OVERLOADED BASIC MATH OPERATORS------------------------


// Element-wise addition assignment
CRSVector& CRSVector::operator+= (const CRSVector& that) {
  if (_len != that._len) {
    _errorMsg("Vector lengths must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < _len; i++) {
    data[i] = data[i] + that.data[i];
  }

  return *this;
}


// Element-wise addition
const CRSVector CRSVector::operator+ (const CRSVector& that) const {
  return CRSVector(*this) += that;
}


// Element-wise subtraction assignment
CRSVector& CRSVector::operator-= (const CRSVector& that) {
  if (_len != that._len) {
    _errorMsg("Vector lengths must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
  
  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < _len; i++) {
    data[i] = data[i] - that.data[i];
  }

  return *this;
}


// Element-wise subtraction
const CRSVector CRSVector::operator- (const CRSVector& that) const {
  return CRSVector(*this) -= that;
}

CRSVector& CRSVector::operator*= (const CRSVector& that) {
  if (_len != that._len) {
    _errorMsg("Vector lengths must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < _len; i++) {
    data[i] = data[i] * that.data[i];
  }

  return *this;
}


// Element-wise multiplication
const CRSVector CRSVector::operator* (const CRSVector& that) const {
  return CRSVector(*this) *= that;
}


// Scalar (right) multiplication assignment
CRSVector& CRSVector::operator*= (double that) {
  if (_len < 1) return *this;
  
  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < _len; i++) {
    data[i] *= that;
  }

  return *this;
}


// Scalar (right) multiplication
const CRSVector CRSVector::operator* (const double that) const {
  return CRSVector(*this) *= that;
}


// Scalar (left) multiplication
const CRSVector lalib::operator* (double scalar, const CRSVector& vector) {
  return vector * scalar;
}

CRSVector& CRSVector::operator/= (const CRSVector& that) {
  if (_len != that._len) {
    _errorMsg("Vector lengths must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < _len; i++) {
    data[i] = data[i] / that.data[i];
  }

  return *this;
}


// Element-wise division
const CRSVector CRSVector::operator/ (const CRSVector& that) const {
  return CRSVector(*this) /= that;
}


// Scalar division
const CRSVector CRSVector::operator/ (const double that) const {
  if (_len < 1) return *this;

  CRSVector ret = CRSVector(*this);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < _len; i++) {
    ret.data[i] = ret.data[i] / that;
  }

  return ret;
}


// ---------------------OVERLOADED INDEXING OPERATORS---------------------------


// Standard single value placement
void CRSVector::place(int num, double val) {
  if (num < 0 || num >= _len) {
    _errorMsg("Index out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  data[num] = val;
}


// Standard vector placement
void CRSVector::place(int start, int end, CRSVector& vector) {
  if (_len < end - start || start < 0 || start >= end) {
    _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < end - start; i++) {
    this->place(i + start, vector(i));
  }
}


// Standard indexing method
double CRSVector::operator() (int num) const {
  if (num < 0 || num >= _len) {
    _errorMsg("Index out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  return data[num];
}


// Squared bracket indexing method
double CRSVector::operator[] (int num) const {
  return this->operator() (num);
}


// Named indexing method
double CRSVector::get(int num) const {
  return this->operator() (num);
}


// Standard slicing method
const CRSVector CRSVector::operator() (int start, int end) const {
  if (_len < end - start || start < 0 || start >= end) {
    _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (end >= _len) {
    _warningMsg("End index out of bounds", __func__);
    end = _len;
  }

  CRSVector ret = CRSVector(end - start);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < end - start; i++) {
    ret.place(i, this->operator() (i + start));
  }

  return ret;
}

const CRSVector CRSVector::get(int start, int end) const {
  return this->operator() (start, end);
}


// ----------------------OTHER OVERLOADED OPERATORS-----------------------------


// Default assignment operator
CRSVector& CRSVector::operator= (const CRSVector& that) {
  // Check for self-assignment ie. case where a = a is called by comparing the pointers of the objects
  if (this == &that) return *this;

  data = that.data;
  _len = that._len;

  return *this;
}


// Default (equality) comparison operator
bool CRSVector::operator== (const CRSVector& that) {
  if (_len != that._len) return false;

  for (int i = 0; i < _len; i++) {
    if (this->operator() (i) != that(i)) {
      return false;
    }
  }

  return true;
}


// Default (inequality) comparison operator
bool CRSVector::operator!= (const CRSVector& that) {
  return !(*this == that);
}


// ----------------------------------MISC----------------------------------------


// Approximative equality comparison
bool CRSVector::isclose(const CRSVector& that, double tol) {
  if (_len != that._len) return false;

  for (int i = 0; i < _len; i++) {
    if (fabs(this->operator() (i) - that(i)) > tol) {
      return false;
    }
  }

  return true;
}


// Vector saving
bool CRSVector::save(std::string& path, int offset, std::string format) {
  if (_len <= 0) {
    _errorMsg("Cannot save an unitialized matrix!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (format != ".dat") {
    _errorMsg("Support for other formats than .dat not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  std::ofstream file(path);
  bool success = true;

  for (int i = 0; i < _len; i++) {
    double val = data[i];
    if (val != 0.0) {
      if (!(file << i + offset << " " << data[i] << "\n")) {
        success = false;
      }
    }
  }
  
  file.close();
  
  return success;
}


// Convert CRSVector into std::vector
std::vector<double> CRSVector::toVector() const {
  if (_len < 1) {
    _errorMsg("Vector must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  std::vector<double> ret = data;

  return ret;
}


// Convert CRSVector object into a CRSMatrix object
const CRSMatrix CRSVector::asCRSMatrix() const {
  if (_len < 1) {
    _errorMsg("Vector must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  CRSMatrix ret = CRSMatrix(_len, 1);  // By default a vector is considered as a column vector

  for (int i = 0; i < _len; i++) {
    ret.place(i, 1, this->operator() (i));
  }

  return ret;
}


// Convert CRSVector into a double
double CRSVector::asDouble() const {
  if (_len != 1) {
    _errorMsg("Vector must have just a single element!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  return this->operator() (0);
}

double CRSVector::norm(double p) const {
  if (_len < 1) {
    _errorMsg("Vector must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  double ret = 0.0;

  for (int i = 0; i < _len; i++) {
    ret += pow(this->operator() (i), p);
  }

  return pow(ret, 1.0 / p);
}
