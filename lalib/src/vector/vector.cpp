#include "vector.hpp"
#include "../declare_lalib.hpp"


using namespace lalib;
using namespace utils;


// -------------------CONSTRUCTORS AND DESTRUCTORS--------------------------


// Default constructor
template <class vect_type, int vect_size> 
Vector<vect_type, vect_size>::Vector(void) { _check(); }


// Copying constructor
template <class vect_type, int vect_size> 
Vector<vect_type, vect_size>::Vector(const Vector& that) {

  _check();

  if (that._len > 0) {
    _len = that._len;

    if (vectorized) {
      dataV = that.dataV;
      _total_vects = that._total_vects;
    }
    else {
      dataS = that.dataS;
    }
  }
}


// Zeros constructor
template <class vect_type, int vect_size> 
Vector<vect_type, vect_size>::Vector(int len) {

  _check()

  if (len < 1) {
    _errorMsg("Vector length must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _len = len;

  if (vectorized) {
    _total_vects = _ceil(_len, vect_size);
    dataV = std::vector<vect_t>(_total_vects, zeros);
  }
  else {
    dataS = std::vector<vect_type>(_len, (vect_type)0.0);
  }
}


// Default value constructor
template <class vect_type, int vect_size> 
Vector<vect_type, vect_size>::Vector(int len, vect_type init_val) {

  _check()

  if (len < 1) {
    _errorMsg("Vector length must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _len = len;

  if (vectorized) {
    vect_t init_vect = _fill(init_val);

    _total_vects = _ceil(_len, vect_size);
    dataV = std::vector<vect_t>(_total_vects, init_vect);
  }
  else {
    dataS = std::vector<vect_type>(_len, init_val);
  }
}


// Array copying constructor
template <class vect_type, int vect_size> 
Vector<vect_type, vect_size>::Vector(int len, vect_type* elems) {

  _check();

  _warningMsg("Initializing a vector with double array might lead to undefined behaviour!", __func__);

  if (len < 1) {
    _errorMsg("Vector length must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  _len = len;

  if (vectorized) {
    _total_vects = _ceil(_len, vect_size);
    for (int vect = 0; vect < _total_vects; vect++) {
      vect_t tmp_vect;
      for (int elem = 0; elem < vect_size; elem++) {
        int i = vect * vect_size + elem;
        tmp_vect[elem] = elems[i];
      }
      dataV.push_back(tmp_vect);
    }
  }
  else {
    dataS.assign(elems, elems + len);
  }
}


// Vector copying constructor
template <class vect_type, int vect_size> 
Vector<vect_type, vect_size>::Vector(int len, std::vector<vect_type>& elems) {

  _check();

  if (len < 1) {
    _errorMsg("Vector length must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
    
  if (len != (int)elems.size()) {
    _warningMsg("Given dimensions don't match with the size of the std::vector!", __func__);
  } 

  _len = len;

  if (vectorized) {
    _total_vects = _ceil(len, vect_size);
    for (int vect = 0; vect < _total_vects; vect++) {
      vect_t tmp_vect;
      for (int elem = 0; elem < vect_size; elem++) {
        int i = vect * vect_size + elem;
        tmp_vect[elem] = (int)elems.size() > i ? elems[i] : (vect_type)0.0;
      }
      dataV.push_back(tmp_vect);
    }
  }
  else {
    dataS.assign(elems, elems + len);
  }
}


// ---------------------OVERLOADED BASIC MATH OPERATORS------------------------


// Element-wise addition assignment
template <class vect_type, int vect_size> 
Vector<vect_type, vect_size>& Vector<vect_type, vect_size>::operator+= (const Vector<vect_type, vect_size>& that) {
  if (_len != that._len) {
    _errorMsg("Vectors must have equal amount of elements!", __FILE__, __func__, __LINE__);
  } 

  if (vectorized) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects; vect++) {
      dataV[vect] = dataV[vect] + that.dataV[vect];
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int elem = 0; elem < _len; elem++) {
      dataS[elem] = dataS[elem] + that.dataS[elem];
    }
  }
  
  return *this;
}


// Element-wise addition
template <class vect_type, int vect_size>
const Vector<vect_type, vect_size> Vector<vect_type, vect_size>::operator+ (const Vector<vect_type, vect_size>& that) const {
  return Vector(*this) += that;
}


// Element-wise subtraction assignment
template <class vect_type, int vect_size> 
Vector<vect_type, vect_size>& Vector<vect_type, vect_size>::operator-= (const Vector<vect_type, vect_size>& that) {
  if (_len != that._len) {
    _errorMsg("Vectors must have equal amount of elements!", __FILE__, __func__, __LINE__);
  } 

  if (vectorized) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects; vect++) {
      dataV[vect] = dataV[vect] - that.dataV[vect];
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int elem = 0; elem < _len; elem++) {
      dataS[elem] = dataS[elem] - that.dataS[elem];
    }
  }
  
  return *this;
}


// Element-wise subtraction
template <class vect_type, int vect_size>
const Vector<vect_type, vect_size> Vector<vect_type, vect_size>::operator- (const Vector<vect_type, vect_size>& that) const {
  return Vector(*this) -= that;
}


// Element-wise multiplication assignment
template <class vect_type, int vect_size> 
Vector<vect_type, vect_size>& Vector<vect_type, vect_size>::operator*= (const Vector<vect_type, vect_size>& that) {
  if (_len != that._len) {
    _errorMsg("Vectors must have equal amount of elements!", __FILE__, __func__, __LINE__);
  } 

  if (vectorized) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects; vect++) {
      dataV[vect] = dataV[vect] * that.dataV[vect];
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int elem = 0; elem < _len; elem++) {
      dataS[elem] = dataS[elem] * that.dataS[elem];
    }
  }
  
  return *this;
}


// Element-wise multiplication
template <class vect_type, int vect_size>
const Vector<vect_type, vect_size> Vector<vect_type, vect_size>::operator* (const Vector<vect_type, vect_size>& that) const {
  return Vector(*this) *= that;
}


// Scalar (right) multiplication assignment
template <class vect_type, int vect_size> 
Vector<vect_type, vect_size>& Vector<vect_type, vect_size>::operator*= (vect_type that) {
  if (_len < 1) {
    return *this;
  }

  if (vectorized) {
    vect_t mult = _fill(that);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects; vect++) {
      dataV[vect] = dataV[vect] * mult;
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int elem = 0; elem < _len; elem++) {
      dataS[elem] = dataS[elem] * that;
    }
  }

  return *this;    
}


// Scalar (right) multiplication
template <class vect_type, int vect_size> 
const Vector<vect_type, vect_size> Vector<vect_type, vect_size>::operator* (const vect_type that) const {
  return Vector(*this) *= that;  
}


// Scalar (left) multiplication
template <class vect_type, int vect_size> 
const Vector<vect_type, vect_size> operator* (double scalar, const Vector<vect_type, vect_size>& vector) {
  return Vector(vector) *= scalar;
}


// Element-wise division assignment
template <class vect_type, int vect_size> 
Vector<vect_type, vect_size>& Vector<vect_type, vect_size>::operator/= (const Vector<vect_type, vect_size>& that) {
  if (_len != that._len) {
    _errorMsg("Vectors must have equal amount of elements!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  } 

  if (vectorized) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects; vect++) {
      dataV[vect] = dataV[vect] / that.dataV[vect];
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int elem = 0; elem < _len; elem++) {
      dataS[elem] = dataS[elem] / that.dataS[elem];
    }
  }

  return *this;
}


// Element-wise division
template <class vect_type, int vect_size> 
const Vector<vect_type, vect_size> Vector<vect_type, vect_size>::operator/ (const Vector<vect_type, vect_size>& that) const {
  return Vector(*this) /= that;
}


// Scalar division assignment
template <class vect_type, int vect_size> 
Vector<vect_type, vect_size>& Vector<vect_type, vect_size>::operator/= (vect_type that) {
  if (that == (vect_type)0) {
    _errorMsg("Division by zero undefined!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
  if (_len < 1) {
    return *this;
  }

  if (vectorized) {
    vect_t div = _fill(that);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < _total_vects; vect++) {
      dataV[vect] = dataV[vect] / div;
    }
  }
  else {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int elem = 0; elem < _len; elem++) {
      dataS[elem] = dataS[elem] / that;
    }
  }
  
  return *this;
}


// Scalar division
template <class vect_type, int vect_size> 
const Vector<vect_type, vect_size> Vector<vect_type, vect_size>::operator/ (const vect_type that) const {
  return Vector(*this) /= that;
}


// ---------------------OVERLOADED INDEXING OPERATORS---------------------------


// Standard single value placement
template <class vect_type, int vect_size>
void Vector<vect_type, vect_size>::place(int num, vect_type val) {
  if (num < 0 || num >= _len) {
    _errorMsg("Index out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (vectorized) {
    const int vect = num / vect_size;
    const int elem = num % vect_size;

    dataV[vect][elem] = val;
  }
  else {
    dataS[num] = val;
  }
}


// Standard vector placement
template <class vect_type, int vect_size>
void Vector<vect_type, vect_size>::place(int start, int end, Vector<vect_type, vect_size>& vector) {
  if (_len < end - start || start < 0 || start >= end) {
    _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < end - start; i++) {
    this->place(i + start, vector(i));
  }
}


// Standard indexing method
template <class vect_type, int vect_size>
vect_type Vector<vect_type, vect_size>::operator() (int num) const {
  if (num < 0 || num >= _len) {
    _errorMsg("Index out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (vectorized) {
    const int vect = num / vect_size;
    const int elem = num % vect_size;

    return dataV[vect][elem];
  }

  // Else
  return dataS[num];
}
  

// Squared bracket indexing method
template <class vect_type, int vect_size>
vect_type Vector<vect_type, vect_size>::operator[] (int num) const {
  return this->operator() (num);
}


// Named indexing method
template <class vect_type, int vect_size>
vect_type Vector<vect_type, vect_size>::get(int num) const {
  return this->operator() (num);
}


// Standard slicing method
template <class vect_type, int vect_size>
const Vector<vect_type, vect_size> Vector<vect_type, vect_size>::operator() (int start, int end) const {
  if (_len < end - start || start < 0 || start >= end) {
    _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (end >= _len) {
    _warningMsg("End index out of bounds", __func__);
    end = _len;
  }

  Vector ret = Vector(end - start);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < end - start; i++) {
    ret.place(i, this->operator() (i + start));
  }

  return ret;
}


// Named slicing method
template <class vect_type, int vect_size>
const Vector<vect_type, vect_size> Vector<vect_type, vect_size>::get(int start, int end) const {
  return this->operator() (start, end);
}


// ----------------------OTHER OVERLOADED OPERATORS-----------------------------


// Default assignment operator
template <class vect_type, int vect_size>
Vector<vect_type, vect_size>& Vector<vect_type, vect_size>::operator= (const Vector<vect_type, vect_size>& that) {
  // Check for self-assignment ie. case where a = a is called by comparing the pointers of the objects
  if (this == &that) return *this;

  dataS = that.dataS;
  dataV = that.dataV;
  _total_vects = that._total_vects
  _len = that._len;

  return *this;
}


// Default (equality) comparison operator
template <class vect_type, int vect_size>
bool Vector<vect_type, vect_size>::operator== (const Vector<vect_type, vect_size>& that) {
  if (_len != that._len) return false;

  for (int i = 0; i < _len; i++) {
    if (this->operator() (i) != that(i)) {
      return false;
    }
  }

  return true;
}


// Default (inequality) comparison operator
template <class vect_type, int vect_size>
bool Vector<vect_type, vect_size>::operator!= (const Vector<vect_type, vect_size>& that) {
  return !(*this == that);
}


// ----------------------------------MISC----------------------------------------


// Approximative equality comparison
template <class vect_type, int vect_size>
bool Vector<vect_type, vect_size>::isclose(const Vector<vect_type, vect_size>& that, vect_type tol, vect_type (*abs_func)(vect_type)) {
  if (_len != that._len) return false;

  for (int i = 0; i < _len; i++) {
    if (abs_func(this->operator() (i) - that(i)) > tol) {
      return false;
    }
  }

  return true;
}


// Vector saving
template <class vect_type, int vect_size>
bool Vector<vect_type, vect_size>::save(std::string& path, int offset, std::string format) {
  if (_len <= 0) {
    _errorMsg("Cannot save an unitialized vector!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  if (format != ".dat") {
    _errorMsg("Support for other formats than .dat not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  std::ofstream file(path);
  bool success = true;

  for (int i = 0; i < _len; i++) {
    vect_type val = this->operator() (i);
    if (!(file << i + offset << " " << val << std::endl)) {
      success = false;
    }
  }
  
  file.close();
  
  return success;
}


// Convert Vector into std::vector
template <class vect_type, int vect_size>
std::vector<vect_type> Vector<vect_type, vect_size>::tovector() const {
  if (_len < 1) {
    _errorMsg("Vector must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  std::vector<vect_type> ret;

  for (int i = 0; i < _len; i++) {
    ret.push_back(this->operator() (i));
  }

  return ret;
}


// Convert Vector into a double
template <class vect_type, int vect_size>
vect_type Vector<vect_type, vect_size>::asScalar() const {
  if (_len != 1) {
    _errorMsg("Vector must have just a single element!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  return this->operator() (0);
}


// The l_p norm
template <class vect_type, int vect_size>
vect_type Vector<vect_type, vect_size>::norm(vect_type p, vect_type (*pow_func)(vect_type, vect_type) = &std::pow) const {
  if (_len < 1) {
    _errorMsg("Vector must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  vect_type ret = (vect_type)0.0;

  for (int i = 0; i < _len; i++) {
    ret += pow_func(this->operator() (i), p);
  }

  return pow_func(ret, (vect_type)1.0 / p);
}



// Default insertion operator
template <class vect_type, int vect_size>
std::ostream& lalib::operator<<(std::ostream& os, Vector<vect_type, vect_size>& v) {
  if (v.len() == 0) {
    os << "[]" << std::endl;  // Signifies uninitialized vector
        
    return os;
  }
    
  os << "[";
  for (int i = 0; i < v.len(); i++) {
    if (i > 0) {
      os << std::endl;
    }
    os << v(i);
  }
  os << "]" << std::endl;
  
  return os;
}