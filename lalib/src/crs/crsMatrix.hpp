#ifndef CRSMATRIX_HPP
#define CRSMATRIX_HPP


#include "../declare_lalib.hpp"


namespace lalib {

  class CRSVector;  // To avoid circular dependencies
  class DenseMatrix;  // To avoid circular dependencies

  /**
   * @brief Compressed row storage (CRS) matrix class
   *
   * CRS is a sparse matrix storage format, where only the non-zero values 
   * in the matrix are held in memory. To access these values additional
   * arrays for the column indeces and row pointers are used. This structure
   * allows for a constant time access to rows, which is very useful when
   * defining fast matrix-vector multiplication. However, the access to columns
   * is a linear time operation.
   */
  class CRSMatrix {

  protected:
    // Initialize values to signify 'empty' matrix
    int _ncols = 0;
    int _nrows = 0;
    std::vector<double> vals;
    std::vector<int> colInds;
    std::vector<int> rowPtrs;
  
  public:
    // ------------- CONSTRUCTORS ---------------

    /**
     * @brief Default constructor
     *
     * Constructor that creates an uninitialized CRSMatrix object.
     */
    CRSMatrix(void);

    /**
     * @brief Copying constructor
     *
     * Constructor that copies the values from a given CRSMatrix object.
     *
     * @param that The object to be copied
     */
    CRSMatrix(const CRSMatrix& that);

    /**
     * @brief Zeros constructor
     *
     * Constructor that initializes a CRSMatrix object of wanted shape
     * and fills it with zeros.
     *
     * @param rows The number of rows in the matrix
     * @param cols The number of columns in the matrix
     */
    CRSMatrix(int rows, int cols);

    /**
     * @brief Default value constructor
     *
     * Constructor that initializes a CRSMatrix object of wanted shape
     * and fills it with the wanted value. 
     *
     * NOTE! As the point of CRS format is to store sparse matrices in 
     * a memory efficient fashion filling the matrix isn't recommended.
     * Instead, user could use the DenseMatrix class.
     * @see DenseMatrix
     *
     * @param rows The number of rows in the matrix
     * @param cols The number of columns in the matrix
     * @param init_val The value with which the matrix is to be filled
     */
    CRSMatrix(int rows, int cols, double init_val);

    /**
     * @brief Array copying constructor
     *
     * Constructor that initializes a CRSMatrix object of wanted shape
     * and copies values from a C style array into it.
     *
     * NOTE! As there is no way to verify the length of the C array
     * this constructor might end up reading unwanted memory.
     *
     * @param rows The number of rows in the matrix
     * @param cols The number of columns in the matrix
     * @param elems A pointer to the start of the C style array
     */
    CRSMatrix(int rows, int cols, double* elems);

    /**
     * @brief Vector copying constructor
     *
     * Constructor that initializes a CRSMatrix object of wanted shape
     * and copies values from a std::vector into it.
     *
     * NOTE! If the length of the passed std::vector doesn't match with
     * the matrix dimensions either the extra elements are ignored or the
     * last rows of the matrix are filled with zeros. In either case a
     * warning is printed.
     *
     * @param rows The number of rows in the matrix
     * @param cols The number of columns in the matrix
     * @param elems A reference to a std::vector
     */
    CRSMatrix(int rows, int cols, const std::vector<double>& elems);

    /**
     * @brief CRS array constructor
     *
     * Constructor that initializes a CRSMatrix object of wanted shape
     * and copies the contents of CRS compatible arrays into it.
     *
     * @param rows The number of rows in the matrix
     * @param cols The number of columns in the matrix
     * @param new_vals A reference to a std::vector containing the non-zero
     * elements of the matrix
     * @param new_colInds A reference to a std::vector containing the column
     * indices of the non-zero elements of the matrix
     * @param new_rowPtrs A reference to a std::vector containing the row pointers
     */
    CRSMatrix(int rows, int cols, const std::vector<double>& new_vals, const std::vector<int>& new_colInds, const std::vector<int> new_rowPtrs);

    /**
     * @brief Load from file constructor
     *
     * Constructor that initializes a CRSMatrix with values read from a
     * file. Supported formats for the files are standard whitespace
     * separated <row col val> tuples or the MTX format. In the whitespace
     * separated format the last row is assumed to contain the bottom left
     * element of the matrix even if it were to be 0.
     *
     * NOTE! The support for MTX format is not yet implemented!
     *
     * @param path A reference to a std::string that defines the path to the 
     * file with the matrix values
     * @param offset OPTIONAL, DEFAULTS TO < int offset = 0 >. The offset between the 
     * indexing conventions. That is if the values in the file are indexed 
     * e.g. starting from 1 that should be passed as the offset.
     * @param format OPTIONAL, DEFAULTS TO < std::string& format ".dat" >. the extension
     * of the used format. Choices are ".dat" and ".mtx".
     * @param safe_indexing OPTIONAL, DEFAULTS TO < bool safe_indexing = false >. 
     * Boolean flag telling if the elements are sorted by rows and columns in the file.
     */
    CRSMatrix(const std::string& path, int offset = 0, const std::string& format = ".dat", bool safe_indexing = false);

    // ~CRSMatrix();  // Destructor not needed

    
    // ---------  Overloaded basic math operators ----------
  
    // NOTE! The operators will function as elementwise operators

    /**
     * @brief Element-wise addition
     *
     * Method that performs an element-wise addition between this CRSMatrix
     * object and the one passed as argument.
     *
     * @param that A reference to a CRSMatrix object used in the sum
     *
     * @return A CRSMatrix object
     */
    const CRSMatrix operator+ (const CRSMatrix& that) const;

    /**
     * @brief Element-wise addition assignment
     *
     * Method that performs an element-wise addition assignment between this CRSMatrix
     * object and the one passed as argument.
     *
     * @param that A reference to a CRSMatrix object used in the sum assignment
     *
     * @return A reference to (this) CRSMatrix object
     */
    CRSMatrix& operator+= (const CRSMatrix& that);

    /**
     * @brief Element-wise subtraction
     *
     * Method that performs an element-wise subtraction between this CRSMatrix
     * object and the one passed as argument.
     *
     * @param that A reference to a CRSMatrix object used in the difference
     *
     * @return A CRSMatrix object
     */
    const CRSMatrix operator- (const CRSMatrix& that) const;

    /**
     * @brief Element-wise subtraction assignment
     *
     * Method that performs an element-wise subtraction assignment between this CRSMatrix
     * object and the one passed as argument.
     *
     * @param that A reference to a CRSMatrix object used in the difference assignment
     *
     * @return A reference to (this) CRSMatrix object
     */
    CRSMatrix& operator-= (const CRSMatrix& that);

    /**
     * @brief Element-wise multiplication
     *
     * Method that performs an element-wise multiplication between this CRSMatrix
     * object and the one passed as argument.
     *
     * @param that A reference to a CRSMatrix object used in the multiplication
     *
     * @return A CRSMatrix object
     */
    const CRSMatrix operator* (const CRSMatrix& that) const;

    /**
     * @brief Element-wise multiplication assignment
     *
     * Method that performs an element-wise multiplication assignment between this
     * CRSMatrix object and the one passed as argument.
     *
     * @param that A reference to a CRSMatrix object used in the multiplication assignment
     *
     * @return A reference to (this) CRSMatrix object
     */
    CRSMatrix& operator*= (const CRSMatrix& that);

    /**
     * @brief Scalar (right) multiplication
     *
     * Method that performs the standard matrix-scalar multiplication
     *
     * @param that The scalar used in the multiplication
     *
     * @return A CRSMatrix object
     */
    const CRSMatrix operator* (const double that) const;

    /**
     * @brief Scalar (right) multiplication assignment
     *
     * Method that performs the standard matrix-scalar multiplication
     *
     * @param that The scalar used in the multiplication
     *
     * @return A reference to (this) CRSMatrix object
     */
    CRSMatrix& operator*= (double that);

    /**
     * @brief Element-wise division
     *
     * Method that performs an element-wise division between this CRSMatrix
     * object and the one passed as argument.
     *
     * @param that A reference to a CRSMatrix object used in the division
     *
     * @return A CRSMatrix object
     */
    const CRSMatrix operator/ (const CRSMatrix& that) const;

    /**
     * @brief Element-wise division assignment
     *
     * Method that performs an element-wise division assignment between this
     * CRSMatrix object and the one passed as argument.
     *
     * @param that A reference to a CRSMatrix object used in the division assignment
     *
     * @return A reference to (this) CRSMatrix object
     */
    CRSMatrix& operator/= (const CRSMatrix& that);

    
    // TODO: const CRSMatrix operator/ (const double that) const;
    
    // ... ?


    // ----------- Overloaded indexing operators ------------

    
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
    double operator[] (int num) const;

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
    double operator() (int row, int col) const;

    /**
     * @brief Named indexing method
     *
     * Method that accesses the wanted element in the matrix
     *
     * Alias for CRSMatrix::operator()
     * @see CRSMatrix::operator()
     *
     * @param row The row of the wanted element
     * @param col The column of the wanted element
     *
     * @return The value on row 'row' and column 'col' in the matrix
     */
    double get(int row, int col) const;

    /**
     * @brief Standard slicing method
     *
     * Method that slices a wanted sized matrix from a CRSMatrix object
     *
     * NOTE! If the end indeces are out of bounds only the elements in bounds
     * are returned. In this case a warning is printed.
     *
     * @param rowStart The starting row index for the slice
     * @param rowEnd The ending row index for the slice
     * @param colStart The starting column index for the slice
     * @param colEnd The ending column index for the slice
     *
     * @return A CRSMatrix object
     */
    const CRSMatrix operator() (int rowStart, int rowEnd, int colStart, int colEnd) const;

    /**
     * @brief Named slicing method
     *
     * Method that slices a wanted sized matrix from a CRSMatrix object
     *
     * Alias for CRSMatrix::operator()
     * @see CRSMatrix::operator()
     *
     * @param rowStart The starting row index for the slice
     * @param rowEnd The ending row index for the slice
     * @param colStart The starting column index for the slice
     * @param colEnd The ending column index for the slice
     *
     * @return A CRSMatrix object
     */
    const CRSMatrix get(int rowStart, int rowEnd, int colStart, int colEnd) const;

    /**
     * @brief Access a column
     *
     * Method that retrieves a wanted column of a CRSMatrix object
     *
     * @param col The index of the wanted column
     *
     * @return The wanted column as a CRSVector object
     */
    const CRSVector getCol(int col) const;

    /**
     * @brief Access a row
     *
     * Method that retrieves a wanted row of a CRSMatrix object
     *
     * @param row The index of the wanted row
     *
     * @return The wanted row as a CRSVector object
     */
    const CRSVector getRow(int row) const;

    // -------- PLACEMENT METHODS ---------

    /**
     * @brief Standard single value placement
     *
     * Method that places a given value at wanted location in a CRSMatrix object
     *
     * @param row The row of interest
     * @param col The column of interest
     * @param val The value to be placed
     */
    void place(int row, int col, double val);

    /**
     * @brief Standard matrix placement
     * 
     * Method that places the values in a CRSMatrix object into the wanted location
     * in another CRSMatrix object.
     *
     * @param rowStart The starting row index for the placement
     * @param rowEnd The ending row index for the placement
     * @param colStart The starting column index for the placement
     * @param colEnd The ending column index for the placement
     * @param matrix A reference to the CRSMatrix object of which values are to be placed
     */
    void place(int rowStart, int rowEnd, int colStart, int colEnd, CRSMatrix matrix);

    /**
     * @brief Standard column placement
     *
     * Method that places the values in a CRSVector object into the wanted column of a
     * CRSMatrix object.
     *
     * @param col The column on to which the elements of the CRSVector are to be placed
     * @param vector The CRSVector object with wanted values
     */
    void placeCol(int col, CRSVector vector);

    /**
     * @brief Standard row placement
     *
     * Method that places the values in a CRSVector object into the wanted row of a
     * CRSMatrix object.
     *
     * @param row The row on to which the elements of the CRSVector are to be placed
     * @param vector The CRSVector object with wanted values
     */
    void placeRow(int row, CRSVector vector);


    // -------- Other overloaded operators ----------

    
    /**
     * @brief Default assignment operator
     * 
     * Method that assigns the values in a given CRSMatrix object into
     * this CRSMatrix object
     *
     * @param that A reference to the CRSMatrix object of which values are assigned
     *
     * @return A reference to (this) CRSMatrix object
     */
    CRSMatrix& operator= (const CRSMatrix& that);

    /**
     * @brief Default (equality) comparison operator
     *
     * Method that compares the elements of two CRSMatrix objects element-wise
     *
     * NOTE! As the elements are stored as double precision floating pointsthere 
     * might be some floating point errors. Thus in some cases it might be better 
     * to use CRSMatrix::isclose() method.
     * @see CRSMatrix::isclose()
     *
     * @param that A reference to the DenseMatrix object of comparison
     *
     * @return A boolean signifying true if equal and false if unequal
     */
    bool operator== (const CRSMatrix& that);

    /**
     * @brief Default (inequality) comparison operator
     *
     * Method that compares the elements of two CRSMatrix objects element-wise
     *
     * @param that A reference to the CRSMatrix object of comparison
     *
     * @return A boolean signifying false if equal and true if unequal
     */
    bool operator!= (const CRSMatrix& that);


    // -------- Other methods ---------

    /**
     * @brief Approximative equality comparison
     * 
     * Method that compares two CRSMatrix object element-wise up to a tolerance
     *
     * @param that A reference to the CRSMatrix object of comparison
     * @param tol OPTIONAL, DEFAULTS TO < double tol = 0.000001 > The tolerance
     * used in the comparison
     *
     * @return A boolean signifying false if equal and true if unequal 
     */
    bool isclose(const CRSMatrix& that, double tol = 0.000001);

    /**
     * @brief Matrix saving
     *
     * Method that saves the CRSMatrix object in a wanted format
     *
     * NOTE! The support for MTX format is not yet implemented!
     *
     * @param path A reference to a std::string that defines the path to the 
     * file where the matrix is to be stored
     * @param offset OPTIONAL, DEFAULTS TO < int offset = 0 >. The offset between the 
     * indexing conventions. That is if the values should be indexed 
     * e.g. starting from 1 that should be passed as the offset.
     * @param format OPTIONAL, DEFAULTS TO < std::string& format ".dat" >. the extension
     * of the used format. Choices are ".dat" and ".mtx".
     */
    bool save(std::string& path, int offset = 0, std::string format = ".dat");

    /**
     * @brief Number of columns
     *
     * Method that returns the number of columns in the CRSMatrix object
     *
     * @return The number of columns
     */
    int ncols() { return _ncols; }

    /**
     * @brief Number of rows
     *
     * Method that returns the number of rows in the CRSMatrix object
     *
     * @return The number of rows
     */
    int nrows() { return _nrows; }

    /**
     * @brief The shape of the matrix
     *
     * Method that returns a tuple containing the number of rows and columns
     * in the CRSMatrix object
     *
     * @return Tuple of form < nrows, ncols >
     */
    std::tuple<int, int> shape() { return std::make_tuple(_nrows, _ncols); }

    /**
     * @brief Number of columns
     *
     * Method that returns the number of columns in the CRSMatrix object
     *
     * @return The number of columns as const
     */
    const int ncols() const { return _ncols; }

    /**
     * @brief Number of rows
     *
     * Method that returns the number of rows in the CRSMatrix object
     *
     * @return The number of rows as const
     */
    const int nrows() const { return _nrows; }

    /**
     * @brief The shape of the matrix
     *
     * Method that returns a tuple containing the number of rows and columns
     * in the CRSMatrix object
     *
     * @return Tuple of form < nrows, ncols > as const
     */
    const std::tuple<int, int> shape() const { return std::make_tuple(_nrows, _ncols); }

    /**
     * @brief CRS format array printing
     *
     * Method that prints the arrays defining the CRSMatrix object into ostream
     *
     * FOR DEBUGGING PURPOSES ONLY!
     */
    void _printArrays();

    /**
     * @brief Naive transpose
     *
     * Very inefficient, but certainly functional method for finding
     * the transpose of a CRSMatrix object
     *
     * @return The transpose as a CRSMatrix object
     */
    const CRSMatrix naiveTranspose() const;

    /**
     * @brief Standard transpose
     *
     * Method that finds the transpose of the CRSMatrix object
     *
     * @return The transpose as a CRSMatrix object
     */
    const CRSMatrix transpose() const;

    /**
     * @brief Standard transpose
     *
     * Method that finds the transpose of the CRSMatrix object
     *
     * Alias for CRSMatrix::transpose()
     * @see CRSMatrix::transpose()
     *
     * @return The transpose as a CRSMatrix object
     */
    const CRSMatrix T() const;
    
    // TODO: CRSMatrix inv();

    /**
     * @brief Wrapper for matrix-matrix multiplication
     *
     * Method that efficiently computes the matrix-matrix product between
     * two CRSMatrix objects by calling appropriate methods based on matrix
     * properties.
     *
     * NOTE! Currently only calls the CRSMatrix::matmulNaive method as others
     * are not implemented.
     * @see CRSMatrix::matmulNaive
     *
     * @param that A reference to the CRSMatrix object used in multiplication
     *
     * @return The resulting CRSMatrix object
     */
    const CRSMatrix matmul(const CRSMatrix& that) const;
    
    // TODO: const CRSMatrix matmulStrassen(const CRSMatrix& that) const;

    /**
     * @brief Matrix-matrix multiplication in the (naive) textbook way
     *
     * Method that computes the matrix-matrix multiplication in the most naive
     * way. That is as a dot product between the rows of left matrix and columns
     * of right matrix.
     *
     * @param that A reference to the CRSMatrix object used in multiplication
     *
     * @return The resulting CRSMatrix object
     */
    const CRSMatrix matmulNaive(const CRSMatrix& that) const;

    /**
     * @brief Matrix-vector multiplication
     *
     * Method that computes the matrix-vector multiplication in an efficient way.
     * This method is crucial in most iterative methods.
     *
     * @param that A reference to the CRSVector object used in multiplication
     *
     * @return The resulting CRSVector object
     */
    const CRSVector matmul(const CRSVector& that) const;

    /**
     * @brief Convert CRSMatrix into std::vector
     *
     * Method that returns the matrix elements in a std::vector.
     *
     * @return The matrix elements in a std::vector
     */
    std::vector<double> toVector() const;
    
    // TODO: DenseMatrix asDenseMatrix() const;

    /**
     * @brief Convert CRSMatrix into a double
     *
     * Method that returns the matrix element as a double.
     *
     * NOTE! The matrix should be a 1 x 1 matrix.
     *
     * @return The matrix element as a double
     */
    double asDouble() const;
    
    // TODO: const CRSVector asCRSVector();

    /**
     * @brief The Frobenius norm
     *
     * Method that computes the Frobenius norm of the CRSMatrix object
     *
     * @return The computed norm
     */
    double norm() const;

    // Statistics

    // TODO: const CRSVector mean(int dim = 0);
    // TODO: const CRSVector sd(int dim = 0);
  };

  /**
   * @brief Default insertion operator
   *
   * Method that adds a representation of the CRSMatrix object into a
   * std::ostream.
   *
   * @param os A reference of a std::ostream into which representation of 
   * CRSMatrix object is to be added
   * @param A A reference to the CRSMatrix object to be inserted
   *
   * @return A reference of the updated std::ostream
   */
  std::ostream& operator<<(std::ostream& os, CRSMatrix& A);

  /**
   * @brief Scalar (left) multiplication
   *
   * Method that performs the standard scalar-matrix multiplication
   *
   * @param scalar The scalar used in the multiplication
   * @param matrix A reference to the CRSMatrix object used in multiplication
   *
   * @return A CRSMatrix object
   */
  const CRSMatrix operator* (double scalar, const CRSMatrix& matrix);

}

#endif
