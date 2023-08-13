#ifndef CRSVECTOR_HPP
#define CRSVECTOR_HPP


#include "../declare_lalib.hpp"


namespace lalib {

  class CRSMatrix;  // To avoid circular dependencies
  class DenseVector;  // To avoid circular dependencies

  /**
   * @brief Compressed row storage (CRS) vector class
   *
   * Vector class compatible with CRSMatrix class. As vectors only consist of
   * single column or row (this class doesn't differentiate between them) there
   * is no need for sparse representation. Thus, this class is very close to the
   * DenseVector class, but doesn't utilize SIMD vectors. Dense representation also
   * allows for trivial parallelization using OpenMP threads e.g. in matrix-vector
   * product.
   * @see CRSMatrix
   * @see DenseVector
   *
   * TODO: Vectorize the implementation if CRSMatrix implementation is vectorized.
   */
  class CRSVector {

  protected:
    // Initialize these values to signify an 'empty' matrix
    int _len = 0;
    std::vector<double> data;

  public:
    // ---------- Constructors ------------

    /**
     * @brief Default constructor
     *
     * Constructor that creates an uninitialized CRSVector object.
     */
    CRSVector(void);

    /**
     * @brief Copying constructor
     *
     * Constructor that copies the values from a given CRSVector object.
     *
     * @param that The object to be copied
     */
    CRSVector(const CRSVector& that);

    /**
     * @brief Zeros constructor
     *
     * Constructor that initializes a CRSVector object of wanted size
     * and fills it with zeros.
     *
     * @param len The size of the vector
     */
    CRSVector(int len);

    /**
     * @brief Default value constructor
     *
     * Constructor that initializes a CRSVector object of wanted size
     * and fills it with the wanted value. 
     *
     * @param len The size of the vector
     * @param init_val The value with which the vector is to be filled
     */
    CRSVector(int len, double init_val);

    /**
     * @brief Array copying constructor
     *
     * Constructor that initializes a CRSVector object of wanted size
     * and copies values from a C style array into it.
     *
     * NOTE! As there is no way to verify the length of the C array
     * this constructor might end up reading unwanted memory.
     *
     * @param len The size of the vector
     * @param elems A pointer to the start of the C style array
     */
    CRSVector(int len, double* elems);

    /**
     * @brief Vector copying constructor
     *
     * Constructor that initializes a CRSVector object of wanted size
     * and copies values from a std::vector into it.
     *
     * NOTE! If the length of the passed std::vector doesn't match with
     * the vector size either the extra elements are ignored or the
     * vector is padded with zeros. In either case a warning is printed.
     *
     * @param len The size of the vector
     * @param elems A reference to a std::vector
     */
    CRSVector(int len, std::vector<double>& elems);

    /**
     * @brief Load from file constructor
     *
     * Constructor that initializes a CRSVector with values read from a
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
    CRSVector(const std::string& path, int offset = 0, std::string format = ".dat");

    // ~CRSVector();  // Destructor not needed


    // ------------ Overloaded basic math operators ------------

    // NOTE! The operators will function as elementwise operators

    /**
     * @brief Element-wise addition
     *
     * Method that performs an element-wise addition between this CRSVector
     * object and the one passed as argument.
     *
     * @param that A reference to a CRSVector object used in the sum
     *
     * @return A CRSVector object
     */
    const CRSVector operator+ (const CRSVector& that) const;

    /**
     * @brief Element-wise addition assignment
     *
     * Method that performs an element-wise addition assignment between this CRSVector
     * object and the one passed as argument.
     *
     * @param that A reference to a CRSVector object used in the sum assignment
     *
     * @return A reference to (this) CRSVector object
     */
    CRSVector& operator+= (const CRSVector& that);

    /**
     * @brief Element-wise subtraction
     *
     * Method that performs an element-wise subtraction between this CRSVector
     * object and the one passed as argument.
     *
     * @param that A reference to a CRSVector object used in the difference
     *
     * @return A CRSVector object
     */
    const CRSVector operator- (const CRSVector& that) const;

    /**
     * @brief Element-wise subtraction assignment
     *
     * Method that performs an element-wise subtraction assignment between this CRSVector
     * object and the one passed as argument.
     *
     * @param that A reference to a CRSVector object used in the difference assignment
     *
     * @return A reference to (this) CRSVector object
     */
    CRSVector& operator-= (const CRSVector& that);

    /**
     * @brief Element-wise multiplication
     *
     * Method that performs an element-wise multiplication between this CRSVector
     * object and the one passed as argument.
     *
     * @param that A reference to a CRSVector object used in the multiplication
     *
     * @return A CRSVector object
     */
    const CRSVector operator* (const CRSVector& that) const;

    /**
     * @brief Element-wise multiplication assignment
     *
     * Method that performs an element-wise multiplication assignment between this
     * CRSVector object and the one passed as argument.
     *
     * @param that A reference to a CRSVector object used in the multiplication assignment
     *
     * @return A reference to (this) CRSVector object
     */
    CRSVector& operator*= (const CRSVector& that);

    /**
     * @brief Scalar (right) multiplication
     *
     * Method that performs the standard matrix-scalar multiplication
     *
     * @param that The scalar used in the multiplication
     *
     * @return A CRSVector object
     */
    const CRSVector operator* (const double that) const;

    /**
     * @brief Scalar (right) multiplication assignment
     *
     * Method that performs the standard matrix-scalar multiplication assignment
     *
     * @param that The scalar used in the multiplication
     *
     * @return A reference to (this) CRSVector object
     */
    CRSVector& operator*= (double that);

    /**
     * @brief Element-wise division
     *
     * Method that performs an element-wise division between this CRSVector
     * object and the one passed as argument.
     *
     * @param that A reference to a CRSVector object used in the division
     *
     * @return A CRSVector object
     */
    const CRSVector operator/ (const CRSVector& that) const;

    /**
     * @brief Element-wise division assignment
     *
     * Method that performs an element-wise division assignment between this
     * CRSVector object and the one passed as argument.
     *
     * @param that A reference to a CRSVector object used in the division assignment
     *
     * @return A reference to (this) CRSVector object
     */
    CRSVector& operator/= (const CRSVector& that);

    /**
     * @brief Scalar division
     *
     * Method that performs the standard matrix-scalar division
     *
     * @param that The scalar used in the multiplication
     *
     * @return A CRSVector object
     */
    const CRSVector operator/ (const double that) const;
    
    // ... ?

    // --------- Overloaded indexing operators -----------

    /**
     * @brief Squared bracket indexing method
     *
     * Method that accesses the wanted element in the vector by a single value
     *
     * Alias for CRSVector::operator()
     * @see CRSVector::operator()
     *
     * @param num The index of the element
     *
     * @return The value at specified index
     */
    double operator[] (int num) const;

    /**
     * @brief Standard indexing method
     *
     * Method that accesses the wanted element in the vector
     *
     * @param num The index of the element
     *
     * @return The value at specified index
     */
    double operator() (int num) const;

    /**
     * @brief Named indexing method
     *
     * Method that accesses the wanted element in the vector
     *
     * Alias for CRSVector::operator()
     * @see CRSVector::operator()
     *
     * @param num The index of the element
     *
     * @return The value at specified index
     */
    double get(int num) const;

    /**
     * @brief Standard slicing method
     *
     * Method that slices a wanted sized vector from a CRSVector object
     *
     * NOTE! If the end index is out of bounds only the elements in bounds
     * are returned. In this case a warning is printed.
     *
     * @param start The starting index for the slice
     * @param end The ending index for the slice
     *
     * @return A CRSVector object
     */
    const CRSVector operator() (int start, int end) const;

    /**
     * @brief Named slicing method
     *
     * Method that slices a wanted sized vector from a CRSVector object
     *
     * Alias for CRSVector::operator()
     * @see CRSVector::operator()
     *
     * @param start The starting row index for the slice
     * @param end The ending row index for the slice
     *
     * @return A CRSVector object
     */
    const CRSVector get(int start, int end) const;

    // -------- Placement methods ---------

    /**
     * @brief Standard single value placement
     *
     * Method that places a given value at wanted location in a CRSVector object
     *
     * @param num The index of interest
     * @param val The value to be placed
     */
    void place(int num, double val);

    /**
     * @brief Standard vector placement
     * 
     * Method that places the values in a CRSVector object into the wanted location
     * in another CRSVector object.
     *
     * @param start The starting row index for the placement
     * @param end The ending row index for the placement
     * @param vector A reference to the CRSVector object of which values are to be placed
     */
    void place(int start, int end, CRSVector& vector);
      
    // --------- Other overloaded operators ----------

    /**
     * @brief Default assignment operator
     * 
     * Method that assigns the values in a given CRSVector object into
     * this CRSVector object
     *
     * @param that A reference to the CRSVector object of which values are assigned
     *
     * @return A reference to (this) CRSVector object
     */
    CRSVector& operator= (const CRSVector& that);

    /**
     * @brief Default (equality) comparison operator
     *
     * Method that compares the elements of two CRSVector objects element-wise
     *
     * NOTE! As the elements are stored as double precision floating pointsthere 
     * might be some floating point errors. Thus in some cases it might be better 
     * to use CRSVector::isclose() method.
     * @see CRSVector::isclose()
     *
     * @param that A reference to the CRSVector object of comparison
     *
     * @return A boolean signifying true if equal and false if unequal
     */
    bool operator== (const CRSVector& that);

    /**
     * @brief Default (inequality) comparison operator
     *
     * Method that compares the elements of two CRSVector objects element-wise
     *
     * @param that A reference to the CRSVector object of comparison
     *
     * @return A boolean signifying false if equal and true if unequal
     */
    bool operator!= (const CRSVector& that);

    // --------- Other methods ----------

    /**
     * @brief Approximative equality comparison
     * 
     * Method that compares two CRSVector object element-wise up to a tolerance
     *
     * @param that A reference to the CRSVector object of comparison
     * @param tol OPTIONAL, DEFAULTS TO < double tol = 0.000001 > The tolerance
     * used in the comparison
     *
     * @return A boolean signifying false if equal and true if unequal 
     */
    bool isclose(const CRSVector& that, double tol = 0.000001);

    /**
     * @brief Vector saving
     *
     * Method that saves the CRSVector object in a wanted format
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
     * @brief Vector-matrix multiplication
     *
     * Method that computes the vector-matrix multiplication in a (relatively) efficient way.
     *
     * @param that A reference to the CRSMatrix object used in multiplication
     * @param is_symmetric OPTIONAL, DEFAULTS TO < bool is_symmetric = ".dat" > Boolean flag
     * telling if the CRSMatrix is symmetric
     *
     * @return The resulting CRSVector object
     */
    const CRSVector matmul(const CRSMatrix& that, bool is_symmetric=false) const;
    
    // TODO: const CRSMatrix matmul(const CRSVector& that) const;

    /**
     * @brief Dot (inner) product
     *
     * Method that computes the inner product between two CRSVector objects in an efficient way.
     * This method is crucial in most iterative methods.
     *
     * @param that A reference to the CRSVector object used in the product
     *
     * @return The resulting CRSVector object
     */
    double dot(const CRSVector& that) const;

    /**
     * @brief Convert CRSVector into std::vector
     *
     * Method that returns the vector elements in a std::vector.
     *
     * @return The vector elements in a std::vector
     */
    std::vector<double> toVector() const;

    /**
     * @brief Convert CRSVector object into a CRSMatrix object
     *
     * Method that returns the vector elements as a CRSMatrix object.
     *
     * NOTE! The matrix will be a n x 1 matrix.
     *
     * @return The vector elements as a CRSMatrix object
     */
    const CRSMatrix asCRSMatrix() const;

    /**
     * @brief Convert CRSVector into a double
     *
     * Method that returns the vector element as a double.
     *
     * NOTE! The vector should be a 1 x 1 matrix.
     *
     * @return The vector element as a double
     */
    double asDouble() const;

    /**
     * @brief The l_p norm
     *
     * Method that computes the l_p norm of the CRSVector object
     *
     * @param p OPTIONAL, DEFAULTS TO < int p = 2 > The degree of the norm
     *  
     * @return The computed norm
     */
    double norm(double p=2.0) const;

    // Statistics

    // TODO: double mean();
    // TODO: double sd();

  };

  // TODO: std::ostream& operator<<(std::ostream& os, CRSVector& A);

  // To accomplish commutative property for vector scalar multiplication

  /**
   * @brief Scalar (left) multiplication
   *
   * Method that performs the standard scalar-vector multiplication
   *
   * @param scalar The scalar used in the multiplication
   * @param vector A reference to the CRSVector object used in multiplication
   *
   * @return A CRSVector object
   */
  const CRSVector operator* (double scalar, const CRSVector& vector);

}
  
#endif
