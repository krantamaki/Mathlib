#ifndef VECTOR_HPP
#define VECTOR_HPP


#include "../declare_lalib.hpp"


using namespace utils;


namespace lalib {

  class CRSMatrix;  // To avoid circular dependencies
  class DenseMatrix;  // To avoid circular dependencies

  /**
   * @brief General vector class 
   *
   * Vector class compatible with all matrix classes. Templated to allow arbitrary
   * data type and vectorization if wanted. Defaults to unvectorized version using
   * doubles. 
   * 
   * Stores the values in a dense fashion, which allows for trivial parallelization
   * in e.g. matrix-vector multiplication.
   * 
   * Note! The checks used for the SIMD vector size and type are not perfect and might
   * cause runtime errors if the choice of data type is not some standard double, float,
   * int etc.
   * 
   */
  template <class vect_type = double, int vect_size = 1>
  class Vector {

    protected:

      // Initialize a boolean flag telling if the implementation should be vectorized
      bool vectorized;

      // Check if the implementation should be vectorized. Should be called by all constructors
      inline void _check() {
        if (vect_size > 1) {
          vectorized = true;
        }
        else if (vect_size == 1) {
          vectorized = false;
        }
        else {
          _errorMsg("Improper SIMD vector size passed!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        // Check that the vector size is sensible
        if ((vect_size * sizeof(vect_type) * __CHAR_BIT__ != SIMD_SIZE) && vectorized) {
          _errorMsg("SIMD vector size doesn't match with the size of the vector register!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
      }


      // Define the SIMD vector that would be used if the implementation is vectorized
      typedef vect_type vect_t __attribute__ ((__vector_size__ (vect_size * sizeof(vect_type))));

      // Constant expression that should define a SIMD vector of zeros
      constexpr vect_t zeros = { };


      // Function that sums together the elements in a SIMD vector
      inline vect_type _reduce(vect_t vector) {
        vect_type ret = 0.0;
        for (int i = 0; i < vect_size; i++) {
          ret += vector[i];
        }

        return ret;
      }

      // Function that fills the SIMD vector with wanted value
      inline vect_t _fill(vect_type init_val) {
        vect_t ret;
        for (int i = 0; i < vect_size; i++) {
          ret[i] = init_val;
        }

        return ret
      }


      // Define the data arrays
      // Both the vectorized and non-vectorized versions are initialized, but only one will get used in practice

      // Array for vectorized data
      std::vector<vect_t> dataV;

      // Array for non-vectorized data
      std::vector<vect_type> dataS;


      // Define the size of the vector
      int _len = 0;

      // Define the number of SIMD vectors in the vector
      int _total_vects = 0;

    public:

      // ---------- Constructors ------------

      /**
       * @brief Default constructor
       *
       * Constructor that creates an uninitialized Vector object.
       */
      Vector(void);

      /**
       * @brief Copying constructor
       *
       * Constructor that copies the values from a given Vector object.
       *
       * @param that The object to be copied
       */
      Vector(const Vector& that);

      /**
       * @brief Zeros constructor
       *
       * Constructor that initializes a Vector object of wanted size
       * and fills it with zeros. 
       * 
       * NOTE! Might not work with arbitrary data types
       *
       * @param len The size of the vector
       * data type
       */
      Vector(int len);

      /**
       * @brief Default value constructor
       *
       * Constructor that initializes a Vector object of wanted size
       * and fills it with the wanted value. 
       *
       * @param len The size of the vector
       * @param init_val The value with which the vector is to be filled
       */
      Vector(int len, vect_type init_val);

      /**
       * @brief Array copying constructor
       *
       * Constructor that initializes a Vector object of wanted size
       * and copies values from a C style array into it.
       *
       * NOTE! As there is no way to verify the length of the C array
       * this constructor might end up reading unwanted memory.
       *
       * @param len The size of the vector
       * @param elems A pointer to the start of the C style array
       */
      Vector(int len, vect_type* elems);

      /**
       * @brief Vector copying constructor
       *
       * Constructor that initializes a Vector object of wanted size
       * and copies values from a std::vector into it.
       *
       * NOTE! If the length of the passed std::vector doesn't match with
       * the vector size either the extra elements are ignored or the
       * vector is padded with zeros. In either case a warning is printed.
       *
       * @param len The size of the vector
       * @param elems A reference to a std::vector
       */
      Vector(int len, std::vector<vect_type>& elems);

      /**
       * @brief Load from file constructor
       *
       * Constructor that initializes a Vector with values read from a
       * file. Supported formats for the files are standard whitespace
       * separated <row col val> (or <row val>) tuples or the MTX format. In
       *  the whitespace separated format the last row is assumed to contain 
       * the bottom left element of the matrix even if it were to be 0.
       *
       * NOTE! The support for MTX format is not yet implemented!
       *
       * @param path A reference to a std::string that defines the path to the 
       * file with the matrix values
       * @param offset OPTIONAL, DEFAULTS TO < int offset = 0 >. The offset between the 
       * indexing conventions. That is if the values in the file are indexed 
       * e.g. starting from 1 that should be passed as the offset.
       * @param format OPTIONAL, DEFAULTS TO < std::string format = ".dat" >. the extension
       * of the used format. Choices are ".dat" and ".mtx"..
       */
      Vector(const std::string& path, int offset = 0, std::string format = ".dat");


      // ------------ Overloaded basic math operators ------------

      // NOTE! The operators will function as elementwise operators

      /**
       * @brief Element-wise addition assignment
       *
       * Method that performs an element-wise addition assignment between this Vector
       * object and the one passed as argument.
       *
       * @param that A reference to a Vector object used in the sum assignment
       *
       * @return A reference to (this) Vector object
       */
      Vector& operator+= (const Vector& that);

      /**
       * @brief Element-wise addition
       *
       * Method that performs an element-wise addition between this Vector
       * object and the one passed as argument.
       *
       * @param that A reference to a Vector object used in the sum
       *
       * @return A Vector object
       */
      const Vector operator+ (const Vector& that) const;

      /**
       * @brief Element-wise subtraction assignment
       *
       * Method that performs an element-wise subtraction assignment between this Vector
       * object and the one passed as argument.
       *
       * @param that A reference to a Vector object used in the difference assignment
       *
       * @return A reference to (this) Vector object
       */
      Vector& operator-= (const Vector& that);

      /**
       * @brief Element-wise subtraction
       *
       * Method that performs an element-wise subtraction between this Vector
       * object and the one passed as argument.
       *
       * @param that A reference to a Vector object used in the difference
       *
       * @return A Vector object
       */
      const Vector operator- (const Vector& that) const;

      /**
       * @brief Element-wise multiplication assignment
       *
       * Method that performs an element-wise multiplication assignment between this
       * Vector object and the one passed as argument.
       *
       * @param that A reference to a Vector object used in the multiplication assignment
       *
       * @return A reference to (this) Vector object
       */
      Vector& operator*= (const Vector& that);

      /**
       * @brief Element-wise multiplication
       *
       * Method that performs an element-wise multiplication between this Vector
       * object and the one passed as argument.
       *
       * @param that A reference to a Vector object used in the multiplication
       *
       * @return A Vector object
       */
      const Vector operator* (const Vector& that) const;

      /**
       * @brief Scalar (right) multiplication assignment
       *
       * Method that performs the standard matrix-scalar multiplication assignment
       *
       * @param that The scalar used in the multiplication
       *
       * @return A reference to (this) Vector object
       */
      Vector& operator*= (vect_type that);

      /**
       * @brief Scalar (right) multiplication
       *
       * Method that performs the standard matrix-scalar multiplication
       *
       * @param that The scalar used in the multiplication
       *
       * @return A Vector object
       */
      const Vector operator* (const vect_type that) const;

      /**
       * @brief Element-wise division assignment
       *
       * Method that performs an element-wise division assignment between this
       * Vector object and the one passed as argument.
       *
       * @param that A reference to a Vector object used in the division assignment
       *
       * @return A reference to (this) Vector object
       */
      Vector& operator/= (const Vector& that);

      /**
       * @brief Element-wise division
       *
       * Method that performs an element-wise division between this Vector
       * object and the one passed as argument.
       *
       * @param that A reference to a Vector object used in the division
       *
       * @return A Vector object
       */
      const Vector operator/ (const Vector& that) const;

      /**
       * @brief Scalar division assignment
       *
       * Method that performs the standard matrix-scalar division assignment
       *
       * @param that The scalar used in the division
       *
       * @return A reference to (this) Vector object
       */
      Vector& operator/= (vect_type that);

      /**
       * @brief Scalar division
       *
       * Method that performs the standard matrix-scalar division
       *
       * @param that The scalar used in the division
       *
       * @return A Vector object
       */
      const Vector operator/ (const vect_type that) const;


      // --------- Overloaded indexing operators -----------

      /**
       * @brief Squared bracket indexing method
       *
       * Method that accesses the wanted element in the vector by a single value
       *
       * Alias for Vector::operator()
       * @see Vector::operator()
       *
       * @param num The index of the element
       *
       * @return The value at specified index
       */
      vect_type operator[] (int num) const;

      /**
       * @brief Standard indexing method
       *
       * Method that accesses the wanted element in the vector
       *
       * @param num The index of the element
       *
       * @return The value at specified index
       */
      vect_type operator() (int num) const;

      /**
       * @brief Named indexing method
       *
       * Method that accesses the wanted element in the vector
       *
       * Alias for Vector::operator()
       * @see Vector::operator()
       *
       * @param num The index of the element
       *
       * @return The value at specified index
       */
      vect_type get(int num) const;

      /**
       * @brief Standard slicing method
       *
       * Method that slices a wanted sized vector from a Vector object
       *
       * NOTE! If the end index is out of bounds only the elements in bounds
       * are returned. In this case a warning is printed.
       *
       * @param start The starting index for the slice
       * @param end The ending index for the slice
       *
       * @return A Vector object
       */
      const Vector operator() (int start, int end) const;

      /**
       * @brief Named slicing method
       *
       * Method that slices a wanted sized vector from a CRSVector object
       *
       * Alias for Vector::operator()
       * @see Vector::operator()
       *
       * @param start The starting row index for the slice
       * @param end The ending row index for the slice
       *
       * @return A Vector object
       */
      const Vector get(int start, int end) const;


      // -------- Placement methods ---------

      /**
       * @brief Standard single value placement
       *
       * Method that places a given value at wanted location in a Vector object
       *
       * @param num The index of interest
       * @param val The value to be placed
       */
      void place(int num, vect_type val);

      /**
       * @brief Standard vector placement
       * 
       * Method that places the values in a Vector object into the wanted location
       * in another Vector object.
       *
       * @param start The starting row index for the placement
       * @param end The ending row index for the placement
       * @param vector A reference to the Vector object of which values are to be placed
       */
      void place(int start, int end, Vector& vector);


      // --------- Other overloaded operators ----------

      /**
       * @brief Default assignment operator
       * 
       * Method that assigns the values in a given Vector object into
       * this Vector object
       *
       * @param that A reference to the Vector object of which values are assigned
       *
       * @return A reference to (this) Vector object
       */
      Vector& operator= (const Vector& that);

      /**
       * @brief Default (equality) comparison operator
       *
       * Method that compares the elements of two Vector objects element-wise
       *
       * NOTE! As the elements are stored as double precision floating pointsthere 
       * might be some floating point errors. Thus in some cases it might be better 
       * to use Vector::isclose() method.
       * @see Vector::isclose()
       *
       * @param that A reference to the Vector object of comparison
       *
       * @return A boolean signifying true if equal and false if unequal
       */
      bool operator== (const Vector& that);

      /**
       * @brief Default (inequality) comparison operator
       *
       * Method that compares the elements of two Vector objects element-wise
       *
       * @param that A reference to the Vector object of comparison
       *
       * @return A boolean signifying false if equal and true if unequal
       */
      bool operator!= (const Vector& that);


      // --------- Other methods ----------

      /**
       * @brief Approximative equality comparison
       * 
       * Method that compares two Vector object element-wise up to a tolerance
       *
       * @param that A reference to the Vector object of comparison
       * @param tol OPTIONAL, DEFAULTS TO < vect_type tol = 1e-7 > The tolerance
       * used in the comparison
       * @param abs_func OPTIONAL, DEFAULTS TO < std::abs > Function returning the absolute value
       *
       * @return A boolean signifying false if equal and true if unequal 
       */
      bool isclose(const Vector& that, vect_type tol = (vect_type)1e-7, vect_type (*abs_func)(vect_type) = &std::abs);

      /**
       * @brief Vector saving
       *
       * Method that saves the Vector object in a wanted format
       *
       * NOTE! The support for MTX format is not yet implemented!
       *
       * @param path A reference to a std::string that defines the path to the 
       * file where the matrix is to be stored
       * @param offset OPTIONAL, DEFAULTS TO < int offset = 0 >. The offset between the 
       * indexing conventions. That is if the values should be indexed 
       * e.g. starting from 1 that should be passed as the offset.
       * @param format OPTIONAL, DEFAULTS TO < std::string format ".dat" >. the extension
       * of the used format. Choices are ".dat" and ".mtx".
       */
      bool save(std::string& path, int offset = 0, std::string format = ".dat");

      /**
       * @brief Size of the vector
       *
       * Method that returns the number of elements in the CRSVector object
       *
       * @return The number of elements
       */
      int len() { return _len; }

      /**
       * @brief Size of the vector
       *
       * Method that returns the number of elements in the CRSVector object
       *
       * @return The number of elements as const
       */
      const int len() const { return _len; }

      /**
       * @brief Sparse vector-matrix multiplication
       *
       * Method that computes the vector-matrix multiplication in a (relatively) efficient way.
       *
       * @param that A reference to the CRSMatrix object used in multiplication
       * @param is_symmetric OPTIONAL, DEFAULTS TO < bool is_symmetric = ".dat" > Boolean flag
       * telling if the CRSMatrix is symmetric
       *
       * @return The resulting Vector object
       */
      const Vector matmul(const CRSMatrix<vect_type, vect_size>& that, bool is_symmetric=false) const;

      /**
       * @brief Dense vector-matrix multiplication
       *
       * Method that computes the vector-matrix multiplication in a (relatively) efficient way.
       *
       * @param that A reference to the DenseMatrix object used in multiplication
       * @param is_symmetric OPTIONAL, DEFAULTS TO < bool is_symmetric = ".dat" > Boolean flag
       * telling if the DenseMatrix is symmetric
       *
       * @return The resulting Vector object
       */
      const Vector matmul(const DenseMatrix<vect_type, vect_size>& that, bool is_symmetric=false) const;

      /**
       * @brief Dot (inner) product
       *
       * Method that computes the inner product between two Vector objects in an efficient way.
       * This method is crucial in most iterative methods.
       *
       * @param that A reference to the Vector object used in the product
       *
       * @return The resulting Vector object
       */
      vect_type dot(const Vector& that) const;

      /**
       * @brief Convert Vector into std::vector
       *
       * Method that returns the vector elements in a std::vector.
       *
       * @return The vector elements in a std::vector
       */
      std::vector<vect_type> tovector() const;

      /**
       * @brief Convert Vector into a scalar
       *
       * Method that returns the vector element as a scalar.
       *
       * NOTE! The vector should be contain only a single element.
       *
       * @return The vector element as a scalar
       */
      vect_type asScalar() const;

      /**
     * @brief The l_p norm
     *
     * Method that computes the l_p norm of the Vector object
     *
     * @param p OPTIONAL, DEFAULTS TO < p = 2 > The degree of the norm
     * @param pow_func OPTIONAL, DEFAULTS TO < std::pow > Pointer to the function used for calculating the powers
     * should be passed if the data type is not some standard library type
     *  
     * @return The computed norm
     */
    vect_type norm(vect_type p = 2.0, vect_type (*pow_func)(vect_type, vect_type) = &std::pow) const;

    // Statistics

    // TODO: double mean();
    // TODO: double sd();
  };

  /**
   * @brief Default insertion operator
   *
   * Method that adds a representation of the Vector object into a
   * std::ostream.
   *
   * @param os A reference of a std::ostream into which representation of 
   * Vector object is to be added
   * @param v A reference to the Vector object to be inserted
   *
   * @return A reference of the updated std::ostream
   */
  template <class vect_type, int vect_size>
  std::ostream& operator<<(std::ostream& os, Vector<vect_type, vect_size>& v);

  /**
   * @brief Scalar (left) multiplication
   *
   * Method that performs the standard scalar-vector multiplication
   *
   * @param scalar The scalar used in the multiplication
   * @param vector A reference to the Vector object used in multiplication
   *
   * @return A Vector object
   */
  template <class vect_type, int vect_size>
  const Vector<vect_type, vect_size> operator* (vect_type scalar, const Vector<vect_type, vect_size>& vector);

}


#endif