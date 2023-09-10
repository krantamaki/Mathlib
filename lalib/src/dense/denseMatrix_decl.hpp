#ifndef DENSEMATRIX_DECL_HPP
#define DENSEMATRIX_DECL_HPP


#include "../declare_lalib.hpp"


using namespace utils;


namespace lalib {

  template <class type, bool vectorize> 
  class Vector;  // To avoid circular dependencies
  class CRSMatrix;  // To avoid circular dependencies

  /**
   * @brief Dense representation matrix class
   *
   * Dense representation of a matrix is the simplest way of storing matrix information.
   * In a dense representation every element of the matrix is held in memory and can be
   * accessed in constant time. This allows for very efficient matrix operations and
   * trivial parallelization. 
   *
   * The current implementation uses SIMD commands and OpenMP threading for speeding up
   * computations.
   *
   * As each element of the matrix are held in memory the required amount of memory can
   * be vast for very large matrices. As such if the matrices are sparse (that is consists
   * of mainly zeros) it might be worthwhile to use CRSMatrix objects, which utilize a
   * a sparse matrix storage format.
   * @see CRSMatrix
   */
  template <class type = double, bool vectorize = false>
  class DenseMatrix {

    protected:

      // Alias the used variable type in computations
      using var_t = decltype(_choose_simd<type, vectorize>());


      // Should define a SIMD vector of zeros or scalar zero depending on vectorization
      var_t v_zero = { };

      // Defines a zero of the underlying type
      type t_zero = { };

      // Defines the number of elements in the SIMD vector if vectorized
      int var_size = SIMD_SIZE / (int)sizeof(type);


      // Const function that sums together the elements in a SIMD vector
      inline const type _reduce(const var_t val) const {

        // Check if the implementation is vectorized at compile time
        if constexpr (!vectorize) return val;
        // Otherwise reduce to scalar
        else {
          type ret = t_zero;
          for (int i = 0; i < var_size; i++) {
            ret += val[i];
          }

          return ret;
        }        
      }

      // Const function that fills the SIMD vector with wanted value
      inline const var_t _fill(const type val) const {

        // Check if the implementation is vectorized at compile time
        if constexpr (!vectorize) return val;
        // Otherwise fill a SIMD vector
        else {
          var_t ret;
          for (int i = 0; i < var_size; i++) {
            ret[i] = val;
          }

          return ret;
        }
      }


      // Define the data array
      std::vector<var_t> data;


      // Define the size of the matrix
      int _ncols = 0;
      int _nrows = 0;

      // Define the number of SIMD vectors in the matrix and per row of the matrix
      int _total_vects = 0;
      int _vects_per_row = 0;

    public:
    
      // ---------- Constructors ------------
      
      /**
       * @brief Default constuctor
       *
       * Constructor that initializes an empty DenseMatrix object
       */
      DenseMatrix(void);

      /**
       * @brief Copying constructor
       *
       * Constructor that initializes a DenseMatrix object and copies contents
       * of an another DenseMatrix object into it
       *
       * @param that A reference to a DenseMatrix object wanted to be copied
       */
      DenseMatrix(const DenseMatrix& that);

      /**
       * @brief Zeros constructor
       *
       * Constructor that initializes a DenseMatrix object of wanted shape and
       * fills it with zeros
       *
       * @param rows The number of rows
       * @param cols The number of columns
       */
      DenseMatrix(int rows, int cols);

      /**
       * @brief Default value constructor
       *
       * Constructor that initializes a DenseMatrix object of wanted shape and
       * fills it with given value
       *
       * @param rows The number of rows
       * @param cols The number of columns
       * @param init_val The value with which the matrix is to be filled
       */
      DenseMatrix(int rows, int cols, type init_val);

      /**
       * @brief Array copying constructor
       *
       * Constructor that initializes a DenseMatrix object of wanted shape and
       * copies the contents of a C style array into it
       *
       * NOTE! As there is no proper way to verify the length of the C style
       * array this constructor might end up reading unwanted memory.
       *
       * @param rows The number of rows
       * @param cols The number of columns
       * @param elems A pointer to the start of the C style array
       */
      DenseMatrix(int rows, int cols, type* elems);

      /**
       * @brief Vector copying constructor
       *
       * Constructor that initializes a DenseMatrix object of wanted shape and
       * copies the contents of a std::vector into it
       *
       * NOTE! If the length of the std::vector doesn't match with the dimensions
       * of the matrix either the extra elements are ignored or the last rows are
       * filled with zeros. In either case a warning is printed.
       *
       * @param rows The number of rows
       * @param cols The number of columns
       * @param elems A reference to a std::vector
       */
      DenseMatrix(int rows, int cols, std::vector<type>& elems);
    
    
      // ---------  Overloaded basic math operators ----------
    
      // NOTE! The operators will function as elementwise operators

      /**
       * @brief Element-wise addition assignment
       *
       * Method that performs an element-wise addition assignment between this DenseMatrix
       * object and the one passed as argument.
       *
       * @param that A reference to a DenseMatrix object used in the sum assignment
       *
       * @return A reference to (this) DenseMatrix object
       */
      DenseMatrix& operator+= (const DenseMatrix& that);

      /**
       * @brief Element-wise addition
       *
       * Method that performs an element-wise addition between this DenseMatrix
       * object and the one passed as argument.
       *
       * @param that A reference to a DenseMatrix object used in the sum
       *
       * @return A DenseMatrix object
       */
      const DenseMatrix operator+ (const DenseMatrix& that) const;

      /**
       * @brief Element-wise subtraction assignment
       *
       * Method that performs an element-wise subtraction assignment between this DenseMatrix
       * object and the one passed as argument.
       *
       * @param that A reference to a DenseMatrix object used in the difference assignment
       *
       * @return A reference to (this) DenseMatrix object
       */
      DenseMatrix& operator-= (const DenseMatrix& that);

      /**
       * @brief Element-wise subtraction
       *
       * Method that performs an element-wise subtraction between this DenseMatrix
       * object and the one passed as argument.
       *
       * @param that A reference to a DenseMatrix object used in the difference
       *
       * @return A DenseMatrix object
       */
      const DenseMatrix operator- (const DenseMatrix& that) const;

      /**
       * @brief Element-wise multiplication assignment
       *
       * Method that performs an element-wise multiplication assignment between this
       * DenseMatrix object and the one passed as argument.
       *
       * @param that A reference to a DenseMatrix object used in the multiplication assignment
       *
       * @return A reference to (this) DenseMatrix object
       */
      DenseMatrix& operator*= (const DenseMatrix& that);

      /**
       * @brief Element-wise multiplication
       *
       * Method that performs an element-wise multiplication between this DenseMatrix
       * object and the one passed as argument.
       *
       * @param that A reference to a DenseMatrix object used in the multiplication
       *
       * @return A DenseMatrix object
       */
      const DenseMatrix operator* (const DenseMatrix& that) const;

      /**
       * @brief Scalar (right) multiplication assignment
       *
       * Method that performs the standard matrix-scalar multiplication assignment
       *
       * @param that The scalar used in the multiplication
       *
       * @return A reference to (this) DenseMatrix object
       */
      DenseMatrix& operator*= (type that);

      /**
       * @brief Scalar (right) multiplication
       *
       * Method that performs the standard matrix-scalar multiplication
       *
       * @param that The scalar used in the multiplication
       *
       * @return A DenseMatrix object
       */
      const DenseMatrix operator* (const type that) const;

      /**
       * @brief Element-wise division assignment
       *
       * Method that performs an element-wise division assignment between this
       * DenseMatrix object and the one passed as argument.
       *
       * @param that A reference to a DenseMatrix object used in the division assignment
       *
       * @return A reference to (this) DenseMatrix object
       */
      DenseMatrix& operator/= (const DenseMatrix& that);

      /**
       * @brief Element-wise division
       *
       * Method that performs an element-wise division between this DenseMatrix
       * object and the one passed as argument.
       *
       * @param that A reference to a DenseMatrix object used in the division
       *
       * @return A DenseMatrix object
       */
      const DenseMatrix operator/ (const DenseMatrix& that) const;

      /**
       * @brief Scalar division assignment
       *
       * Method that performs the standard matrix-scalar division assignment
       *
       * @param that The scalar used in the division
       *
       * @return A reference to (this) DenseMatrix object
       */
      DenseMatrix& operator/= (type that);

      /**
       * @brief Scalar (right) division
       *
       * Method that performs a matrix-scalar division
       *
       * @param that The scalar used in the division
       *
       * @return A DenseMatrix object
       */
      const DenseMatrix operator/ (const type that) const;


      // ---------- Overloaded indexing operators -----------

      /**
       * @brief Standard indexing method
       *
       * Method that accesses the wanted element in the matrix
       *
       * @param row The row of the wanted element
       * @param col The column of the wanted element
       *
       * @return The value on row 'row' and column 'col' in the matrix
       */
      type operator() (int row, int col) const;

      /**
       * @brief Named indexing method
       *
       * Method that accesses the wanted element in the matrix
       *
       * Alias for DenseMatrix::operator()
       * @see DenseMatrix::operator()
       *
       * @param row The row of the wanted element
       * @param col The column of the wanted element
       *
       * @return The value on row 'row' and column 'col' in the matrix
       */
      type get(int row, int col) const;

      /**
       * @brief Squared bracket indexing method
       *
       * Method that accesses the wanted element in the matrix by a single value
       *
       * Returns the element on row $num / _ncols$ and column $num % _ncols$. 
       *
       * @param num The index of the element
       *
       * @return The value at specified index
       */
      type operator[] (int num) const;

      /**
       * @brief Standard slicing method
       *
       * Method that slices a wanted sized matrix from a DenseMatrix object
       *
       * NOTE! If the end indeces are out of bounds only the elements in bounds
       * are returned. In this case a warning is printed.
       *
       * @param rowStart The starting row index for the slice
       * @param rowEnd The ending row index for the slice
       * @param colStart The starting column index for the slice
       * @param colEnd The ending column index for the slice
       *
       * @return A DenseMatrix object
       */
      const DenseMatrix operator() (int rowStart, int rowEnd, int colStart, int colEnd) const;

      /**
       * @brief Named slicing method
       *
       * Method that slices a wanted sized matrix from a DenseMatrix object
       *
       * Alias for DenseMatrix::operator()
       * @see DenseMatrix::operator()
       *
       * @param rowStart The starting row index for the slice
       * @param rowEnd The ending row index for the slice
       * @param colStart The starting column index for the slice
       * @param colEnd The ending column index for the slice
       *
       * @return A DenseMatrix object
       */
      const DenseMatrix get(int rowStart, int rowEnd, int colStart, int colEnd) const;

      
      // TODO: const DenseVector getCol(int col) const;
      // TODO: const DenseVector getRow(int row) const;
      // TODO: vect_t getSIMD(int num);  // Allows user to access the SIMD vectors for further parallelization


      // -------- PLACEMENT METHODS ---------

      /**
       * @brief Standard single value placement
       *
       * Method that places a given value at wanted location in a DenseMatrix object
       *
       * @param row The row of interest
       * @param col The column of interest
       * @param val The value to be placed
       */
      void place(int row, int col, type val);

      /**
       * @brief Standard matrix placement
       * 
       * Method that places the values in a DenseMatrix object into the wanted location
       * in another DenseMatrix object.
       *
       * @param rowStart The starting row index for the placement
       * @param rowEnd The ending row index for the placement
       * @param colStart The starting column index for the placement
       * @param colEnd The ending column index for the placement
       * @param matrix A reference to the DenseMatrix object of which values are to be placed
       */
      void place(int rowStart, int rowEnd, int colStart, int colEnd, const DenseMatrix& matrix);

      
      // TODO: void placeCol(int col, DenseVector vector);
      // TODO: void placeRow(int row, DenseVector vector);

      
      // -------- Other overloaded operators ----------

      
      /**
       * @brief Default assignment operator
       * 
       * Method that assigns the values in a given DenseMatrix object into
       * this DenseMatrix object
       *
       * @param that A reference to the DenseMatrix object of which values are assigned
       *
       * @return A reference to (this) DenseMatrix object
       */
      DenseMatrix& operator= (const DenseMatrix& that);

      /**
       * @brief Default (equality) comparison operator
       *
       * Method that compares the elements of two DenseMatrix objects element-wise
       *
       * NOTE! As the elements are stored as double precision floating pointsthere 
       * might be some floating point errors. Thus in some cases it might be better 
       * to use DenseMatrix::isclose() method.
       * @see DenseMatrix::isclose()
       *
       * @param that A reference to the DenseMatrix object of comparison
       *
       * @return A boolean signifying true if equal and false if unequal
       */
      bool operator== (const DenseMatrix& that);

      /**
       * @brief Default (inequality) comparison operator
       *
       * Method that compares the elements of two DenseMatrix objects element-wise
       *
       * @param that A reference to the DenseMatrix object of comparison
       *
       * @return A boolean signifying false if equal and true if unequal
       */
      bool operator!= (const DenseMatrix& that);

    
      // -------- Other methods ---------


      /**
       * @brief Number of columns
       *
       * Method that returns the number of columns in the DenseMatrix object
       *
       * @return The number of columns
       */
      int ncols() { return _ncols; }

      /**
       * @brief Number of rows
       *
       * Method that returns the number of rows in the DenseMatrix object
       *
       * @return The number of rows
       */
      int nrows() { return _nrows; }

      /**
       * @brief The shape of the matrix
       *
       * Method that returns a tuple containing the number of rows and columns
       * in the DenseMatrix object
       *
       * @return Tuple of form < nrows, ncols >
       */
      std::tuple<int, int> shape() { return std::make_tuple(_nrows, _ncols); }

      /**
       * @brief Number of columns
       *
       * Method that returns the number of columns in the DenseMatrix object
       *
       * @return The number of columns as const
       */
      const int ncols() const { return _ncols; }

      /**
       * @brief Number of rows
       *
       * Method that returns the number of rows in the DenseMatrix object
       *
       * @return The number of rows as const
       */
      const int nrows() const { return _nrows; }

      /**
       * @brief The shape of the matrix
       *
       * Method that returns a tuple containing the number of rows and columns
       * in the DenseMatrix object
       *
       * @return Tuple of form < nrows, ncols > as const
       */
      const std::tuple<int, int> shape() const { return std::make_tuple(_nrows, _ncols); }

      /**
       * @brief Approximative equality comparison
       * 
       * Method that compares two DenseMatrix object element-wise up to a tolerance
       *
       * @param that A reference to the DenseMatrix object of comparison
       * @param tol OPTIONAL, DEFAULTS TO < type tol = 1e-7 > The tolerance
       * used in the comparison
       * @param abs_func OPTIONAL, DEFAULTS TO < std::abs > Function returning the absolute value
       *
       * @return A boolean signifying false if equal and true if unequal 
       */
      bool isclose(const DenseMatrix& that, type tol = (type)1e-7, type (*abs_func)(type) = &std::abs);

      /**
       * @brief Standard transpose
       *
       * Method that finds the transpose of the DenseMatrix object
       *
       * @return The transpose as a DenseMatrix object
       */
      const DenseMatrix transpose() const;

      /**
       * @brief Standard transpose
       *
       * Method that finds the transpose of the DenseMatrix object
       *
       * Alias for DenseMatrix::transpose()
       * @see DenseMatrix::transpose()
       *
       * @return The transpose as a DenseMatrix object
       */
      const DenseMatrix T() const;
      
      // TODO: DenseMatrix inv();

      /**
       * @brief Wrapper for matrix-matrix multiplication
       *
       * Method that efficiently computes the matrix-matrix product between
       * two DenseMatrix objects by calling appropriate methods based on matrix
       * properties.
       *
       * NOTE! Currently only calls the DenseMatrix::matmulNaive method as others
       * are not implemented.
       * @see DenseMatrix::matmulNaive
       *
       * @param that A reference to the DenseMatrix object used in multiplication
       *
       * @return The resulting DenseMatrix object
       */
      const DenseMatrix matmul(const DenseMatrix& that) const;

      // TODO: const DenseMatrix matmulStrassen(const DenseMatrix& that) const;

      /**
       * @brief Matrix-matrix multiplication in the (naive) textbook way
       *
       * Method that computes the matrix-matrix multiplication in the most naive
       * way. That is as a dot product between the rows of left matrix and columns
       * of right matrix.
       *
       * @param that A reference to the DenseMatrix object used in multiplication
       *
       * @return The resulting DenseMatrix object
       */
      const DenseMatrix matmulNaive(const DenseMatrix& that) const;

      /**
       * @brief Matrix-vector multiplication
       *
       * Method that computes the matrix-vector multiplication in an efficient way.
       * This method is crucial in most iterative methods.
       *
       * @param that A reference to the Vector object used in multiplication
       *
       * @return The resulting Vector object
       */
      const Vector<type, vectorize> matmul(const Vector<type, vectorize>& that) const;

      /**
       * @brief Convert DenseMatrix into std::vector
       *
       * Method that returns the matrix elements in a std::vector.
       *
       * @return The matrix elements in a std::vector
       */
      std::vector<type> tovector() const;
      
      // TODO: CRSMatrix asCRSMatrix() const;

      /**
       * @brief Convert DenseMatrix into a scalar
       *
       * Method that returns the matrix element as a scalar.
       *
       * NOTE! The matrix should be a 1 x 1 matrix.
       *
       * @return The matrix element as a scalar
       */
      type asScalar() const;

      /**
       * @brief Convert DenseMatrix into a DenseVector
       *
       * Method that returns the matrix elements in a DenseVector object
       *
       * NOTE! The matrix should be a 1 x n or a n x 1 matrix.
       *
       * @return The matrix elements in a DenseVector
       */
      const Vector<type, vectorize> asVector() const;

      /**
       * @brief The Frobenius norm
       *
       * Method that computes the Frobenius norm of the DenseMatrix object
       * 
       * @param pow_func OPTIONAL, DEFAULTS TO < std::pow > Pointer to the function used for calculating the powers
       * should be passed if the data type is not some standard library type
       *
       * @return The computed norm
       */
      type norm(type (*pow_func)(type, type) = &std::pow) const;

      
      // --------- Statistics ----------

      // TODO: const DenseVector mean(int dim = 0);
      // TODO: const DenseVector sd(int dim = 0);

  };

  /**
   * @brief Default insertion operator
   *
   * Method that adds a representation of the DenseMatrix object into a
   * std::ostream.
   *
   * @param os A reference of a std::ostream into which representation of 
   * DenseMatrix object is to be added
   * @param A A reference to the DenseMatrix object to be inserted
   *
   * @return A reference of the updated std::ostream
   */
  template <class type, bool vectorize>
  std::ostream& operator<<(std::ostream& os, DenseMatrix<type, vectorize>& A);

  /**
   * @brief Scalar (left) multiplication
   *
   * Method that performs the standard scalar-matrix multiplication
   *
   * @param scalar The scalar used in the multiplication
   * @param matrix A reference to the DenseMatrix object used in multiplication
   *
   * @return A DenseMatrix object
   */
  template <class type, bool vectorize>
  const DenseMatrix<type, vectorize> operator* (type scalar, const DenseMatrix<type, vectorize>& matrix);

}
  

#endif
