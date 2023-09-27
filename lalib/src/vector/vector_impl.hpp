#ifndef VECTOR_IMPL_HPP
#define VECTOR_IMPL_HPP


#include "vector_decl.hpp"
#include "../declare_lalib.hpp"


using namespace lalib;
using namespace utils;


// -------------------CONSTRUCTORS AND DESTRUCTORS--------------------------


// Default constructor
template <class type, bool vectorize> 
Vector<type, vectorize>::Vector(void) { }


// Copying constructor
template <class type, bool vectorize> 
Vector<type, vectorize>::Vector(const Vector& that) {

  if (that._len > 0) {

    _len = that._len;

    if constexpr (vectorize) {
      data = that.data;
      _total_vects = that._total_vects;
    }
    else {
      data = that.data;
    }
  }
}


// Zeros constructor
template <class type, bool vectorize> 
Vector<type, vectorize>::Vector(int len) {

  if (len < 1) {
    _errorMsg("Vector length must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _len = len;

  if constexpr (vectorize) {
    _total_vects = _ceil(_len, var_size);
    data = std::vector<var_t>(_total_vects, v_zero);
  }
  else {
    data = std::vector<var_t>(_len, v_zero);
  }
}


// Default value constructor
template <class type, bool vectorize> 
Vector<type, vectorize>::Vector(int len, type init_val) {

  if (len < 1) {
    _errorMsg("Vector length must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _len = len;

  if constexpr (vectorize) {
    var_t init = _fill(init_val);

    _total_vects = _ceil(_len, var_size);
    data = std::vector<var_t>(_total_vects, init);
  }
  else {
    data = std::vector<var_t>(_len, init_val);
  }
}


// Array copying constructor
template <class type, bool vectorize>
Vector<type, vectorize>::Vector(int len, type* elems) {

  _warningMsg("Initializing a vector with double array might lead to undefined behaviour!", __func__);

  if (len < 1) {
    _errorMsg("Vector length must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _len = len;

  if constexpr (vectorize) {
    _total_vects = _ceil(_len, var_size);
    for (int vect = 0; vect < _total_vects; vect++) {
      var_t tmp_vect;
      for (int elem = 0; elem < var_size; elem++) {
        int i = vect * var_size + elem;
        tmp_vect[elem] = i < len ? elems[i] : t_zero;
      }
      data.push_back(tmp_vect);
    }
  }
  else {
    data.assign(elems, elems + len);
  }
}


// Vector copying constructor
template <class type, bool vectorize>
Vector<type, vectorize>::Vector(int len, std::vector<type>& elems) {

  if (len < 1) {
    _errorMsg("Vector length must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
    
  if (len != (int)elems.size()) {
    _warningMsg("Given dimensions don't match with the size of the std::vector!", __func__);
  } 

  _len = len;

  if constexpr (vectorize) {
    _total_vects = _ceil(len, var_size);
    for (int vect = 0; vect < _total_vects; vect++) {
      var_t tmp_vect;
      for (int elem = 0; elem < var_size; elem++) {
        int i = vect * var_size + elem;
        tmp_vect[elem] = (i < (int)elems.size()) && (i < len) ? elems[i] : t_zero;
      }
      data.push_back(tmp_vect);
    }
  }
  else {
    std::copy(elems.begin(), elems.begin() + len, std::back_inserter(data));
  }
}


// Load from file constructor
template <class type, bool vectorize>
Vector<type, vectorize>::Vector(const std::string& path, int offset, std::string format) {
  // Variables to read the line contents to
  int row, col;
  type val;
  
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

    data = std::vector<var_t>(_len, v_zero);

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

    data = std::vector<var_t>(_len, v_zero);

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


// ---------------------OVERLOADED BASIC MATH OPERATORS------------------------


// Element-wise addition assignment
template <class type, bool vectorize> 
Vector<type, vectorize>& Vector<type, vectorize>::operator+= (const Vector<type, vectorize>& that) {
  if (_len != that._len) {
    _errorMsg("Vectors must have equal amount of elements!", __FILE__, __func__, __LINE__);
  } 

  if constexpr (vectorize) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects - 1; vect++) {
      data[vect] = data[vect] + that.data[vect];
    }
    for (int elem = 0; elem < _len % var_size; elem++) {
      data[_total_vects - 1][elem] = data[_total_vects - 1][elem] + that.data[_total_vects - 1][elem];
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int elem = 0; elem < _len; elem++) {
      data[elem] = data[elem] + that.data[elem];
    }
  }
  
  return *this;
}


// Element-wise addition
template <class type, bool vectorize> 
const Vector<type, vectorize> Vector<type, vectorize>::operator+ (const Vector<type, vectorize>& that) const {
  return Vector<type, vectorize>(*this) += that;
}


// Element-wise subtraction assignment
template <class type, bool vectorize> 
Vector<type, vectorize>& Vector<type, vectorize>::operator-= (const Vector<type, vectorize>& that) {
  if (_len != that._len) {
    _errorMsg("Vectors must have equal amount of elements!", __FILE__, __func__, __LINE__);
  } 

  if constexpr (vectorize) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects - 1; vect++) {
      data[vect] = data[vect] - that.data[vect];
    }
    for (int elem = 0; elem < _len % var_size; elem++) {
      data[_total_vects - 1][elem] = data[_total_vects - 1][elem] - that.data[_total_vects - 1][elem];
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int elem = 0; elem < _len; elem++) {
      data[elem] = data[elem] - that.data[elem];
    }
  }
  
  return *this;
}


// Element-wise subtraction
template <class type, bool vectorize> 
const Vector<type, vectorize> Vector<type, vectorize>::operator- (const Vector<type, vectorize>& that) const {
  return Vector<type, vectorize>(*this) -= that;
}


// Element-wise multiplication assignment
template <class type, bool vectorize> 
Vector<type, vectorize>& Vector<type, vectorize>::operator*= (const Vector<type, vectorize>& that) {
  if (_len != that._len) {
    _errorMsg("Vectors must have equal amount of elements!", __FILE__, __func__, __LINE__);
  } 

  if constexpr (vectorize) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects; vect++) {
      data[vect] = data[vect] * that.data[vect];
    }
    for (int elem = 0; elem < _len % var_size; elem++) {
      data[_total_vects - 1][elem] = data[_total_vects - 1][elem] * that.data[_total_vects - 1][elem];
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int elem = 0; elem < _len; elem++) {
      data[elem] = data[elem] * that.data[elem];
    }
  }
  
  return *this;
}


// Element-wise multiplication
template <class type, bool vectorize> 
const Vector<type, vectorize> Vector<type, vectorize>::operator* (const Vector<type, vectorize>& that) const {
  return Vector<type, vectorize>(*this) *= that;
}


// Scalar (right) multiplication assignment
template <class type, bool vectorize> 
Vector<type, vectorize>& Vector<type, vectorize>::operator*= (type that) {
  if (_len < 1) {
    return *this;
  }

  if constexpr (vectorize) {
    var_t mult = _fill(that);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects - 1; vect++) {
      data[vect] = data[vect] * mult;
    }
    for (int elem = 0; elem < _len % var_size; elem++) {
      data[_total_vects - 1][elem] = data[_total_vects - 1][elem] * that;
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int elem = 0; elem < _len; elem++) {
      data[elem] = data[elem] * that;
    }
  }

  return *this;    
}


// Scalar (right) multiplication
template <class type, bool vectorize> 
const Vector<type, vectorize> Vector<type, vectorize>::operator* (const type that) const {
  return Vector<type, vectorize>(*this) *= that;  
}


// Scalar (left) multiplication
template <class type, bool vectorize> 
const Vector<type, vectorize> lalib::operator* (type scalar, const Vector<type, vectorize>& vector) {
  return Vector<type, vectorize>(vector) *= scalar;
}


// Element-wise division assignment
template <class type, bool vectorize> 
Vector<type, vectorize>& Vector<type, vectorize>::operator/= (const Vector<type, vectorize>& that) {
  if (_len != that._len) {
    _errorMsg("Vectors must have equal amount of elements!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  } 

  if constexpr (vectorize) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects - 1; vect++) {
      data[vect] = data[vect] / that.data[vect];
    }
    for (int elem = 0; elem < _len % var_size; elem++) {
      data[_total_vects - 1][elem] = data[_total_vects - 1][elem] / that.data[_total_vects - 1][elem];
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int elem = 0; elem < _len; elem++) {
      data[elem] = data[elem] / that.data[elem];
    }
  }

  return *this;
}


// Element-wise division
template <class type, bool vectorize> 
const Vector<type, vectorize> Vector<type, vectorize>::operator/ (const Vector<type, vectorize>& that) const {
  return Vector<type, vectorize>(*this) /= that;
}


// Scalar division assignment
template <class type, bool vectorize> 
Vector<type, vectorize>& Vector<type, vectorize>::operator/= (type that) {
  if (that == t_zero) {
    _errorMsg("Division by zero undefined!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
  if (_len < 1) {
    return *this;
  }

  if constexpr (vectorize) {
    var_t div = _fill(that);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects - 1; vect++) {
      data[vect] = data[vect] / div;
    }
    for (int elem = 0; elem < _len % var_size; elem++) {
      data[_total_vects - 1][elem] = data[_total_vects - 1][elem] / that;
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int elem = 0; elem < _len; elem++) {
      data[elem] = data[elem] / that;
    }
  }
  
  return *this;
}


// Scalar division
template <class type, bool vectorize>  
const Vector<type, vectorize> Vector<type, vectorize>::operator/ (const type that) const {
  return Vector<type, vectorize>(*this) /= that;
}


// ---------------------OVERLOADED INDEXING OPERATORS---------------------------


// Standard single value placement
template <class type, bool vectorize>
void Vector<type, vectorize>::place(int num, type val) {
  if (num < 0 || num >= _len) {
    _errorMsg("Index out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if constexpr (vectorize) {
    const int vect = num / var_size;
    const int elem = num % var_size;

    data[vect][elem] = val;
  }
  else {
    data[num] = val;
  }
}


// Standard vector placement
template <class type, bool vectorize>
void Vector<type, vectorize>::place(int start, int end, Vector<type, vectorize>& vector) {
  if (_len < end - start || start < 0 || start >= end) {
    _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < end - start; i++) {
    this->place(i + start, vector(i));
  }
}


// Standard indexing method
template <class type, bool vectorize>
type Vector<type, vectorize>::operator() (int num) const {
  if (num < 0 || num >= _len) {
    _errorMsg("Index out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if constexpr (vectorize) {
    const int vect = num / var_size;
    const int elem = num % var_size;

    return data[vect][elem];
  }
  else {
    return data[num];
  }
}
  

// Squared bracket indexing method
template <class type, bool vectorize>
type Vector<type, vectorize>::operator[] (int num) const {
  return this->operator() (num);
}


// Named indexing method
template <class type, bool vectorize>
type Vector<type, vectorize>::get(int num) const {
  return this->operator() (num);
}


// Standard slicing method
template <class type, bool vectorize>
const Vector<type, vectorize> Vector<type, vectorize>::operator() (int start, int end) const {
  if (_len < end - start || start < 0 || start >= end) {
    _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (end >= _len) {
    _warningMsg("End index out of bounds", __func__);
    end = _len;
  }

  Vector ret = Vector<type, vectorize>(end - start);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < end - start; i++) {
    ret.place(i, this->operator() (i + start));
  }

  return ret;
}


// Named slicing method
template <class type, bool vectorize>
const Vector<type, vectorize> Vector<type, vectorize>::get(int start, int end) const {
  return this->operator() (start, end);
}


// SIMD accessing method
template <class type, bool vectorize>
type* Vector<type, vectorize>::getSIMD(int num) const {
  if (!vectorize) {
    _errorMsg("To access SIMD vectors implementation must be vectorized", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (num >= _total_vects) {
    _errorMsg("Index out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  return (type*)&data.data()[num];
}


// ----------------------OTHER OVERLOADED OPERATORS-----------------------------


// Default assignment operator
template <class type, bool vectorize>
Vector<type, vectorize>& Vector<type, vectorize>::operator= (const Vector<type, vectorize>& that) {
  // Check for self-assignment ie. case where a = a is called by comparing the pointers of the objects
  if (this == &that) return *this;

  data = that.data;
  _total_vects = that._total_vects;
  _len = that._len;

  return *this;
}


// Default (equality) comparison operator
template <class type, bool vectorize>
bool Vector<type, vectorize>::operator== (const Vector<type, vectorize>& that) {
  if (_len != that._len) return false;

  for (int i = 0; i < _len; i++) {
    if (this->operator() (i) != that(i)) {
      return false;
    }
  }

  return true;
}


// Default (inequality) comparison operator
template <class type, bool vectorize>
bool Vector<type, vectorize>::operator!= (const Vector<type, vectorize>& that) {
  return !(*this == that);
}


// ----------------------------------MISC----------------------------------------


// Approximative equality comparison
template <class type, bool vectorize>
bool Vector<type, vectorize>::isclose(const Vector<type, vectorize>& that, type tol, type (*abs_func)(type)) {
  if (_len != that._len) return false;

  for (int i = 0; i < _len; i++) {
    if (abs_func(this->operator() (i) - that(i)) > tol) {
      return false;
    }
  }

  return true;
}


// Vector saving
template <class type, bool vectorize>
bool Vector<type, vectorize>::save(const std::string& path, int offset, std::string format) {
  if (_len <= 0) {
    _errorMsg("Cannot save an unitialized vector!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (format != ".dat") {
    _errorMsg("Support for other formats than .dat not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  std::ofstream file(path);
  bool success = true;

  for (int i = 0; i < _len; i++) {
    type val = this->operator() (i);
    if (!(file << i + offset << " " << val << std::endl)) {
      success = false;
    }
  }
  
  file.close();
  
  return success;
}


// Convert Vector into std::vector
template <class type, bool vectorize>
std::vector<type> Vector<type, vectorize>::tovector() const {
  if (_len < 1) {
    _errorMsg("Vector must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  std::vector<type> ret;

  for (int i = 0; i < _len; i++) {
    ret.push_back(this->operator() (i));
  }

  return ret;
}


// Convert Vector into a double
template <class type, bool vectorize>
type Vector<type, vectorize>::asScalar() const {
  if (_len != 1) {
    _errorMsg("Vector must have just a single element!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  return this->operator() (0);
}


// The l_p norm
template <class type, bool vectorize>
type Vector<type, vectorize>::norm(type p, type (*pow_func)(type, type)) const {
  if (_len < 1) {
    _errorMsg("Vector must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  type ret = t_zero;

  for (int i = 0; i < _len; i++) {
    ret += pow_func(this->operator() (i), p);
  }

  return pow_func(ret, (type)1.0 / p);
}



// Default insertion operator
template <class type, bool vectorize>
std::ostream& lalib::operator<<(std::ostream& os, Vector<type, vectorize>& v) {
  if (v.len() == 0) {
    os << "[]" << std::endl;  // Signifies uninitialized vector
        
    return os;
  }
    
  os << "[";
  for (int i = 0; i < v.len(); i++) {
    if (i > 0) {
      os << std::endl << ' ';
    }
    os << v(i);
  }
  os << "]" << std::endl;
  
  return os;
}


#endif