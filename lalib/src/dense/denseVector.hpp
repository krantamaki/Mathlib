#ifndef DENSEVECTOR_HPP
#define DENSEVECTOR_HPP


#include "../declare_lalib.hpp"


namespace lalib {

  class DenseMatrix;  // To avoid circular dependencies
  class CRSVector; // To avoid circular dependencies

  /**
   * @brief Dense representation vector class
   *
   * Vector class compatible with DenseMatrix class. A vector consist of a
   * single column or row (this class doesn't differentiate between them). This, 
   * this class is very close to the CRSVector class, but utilizes SIMD vectors 
   * for increased parallelization. Dense representation also allows for trivial 
   * parallelization using OpenMP threads e.g. in matrix-vector product.
   * @see DenseMatrix
   * @see CRSVector
   */
  class DenseVector {

  protected:
    // Initialize these values to signify an 'empty' matrix
    int _len = 0;
    vect_t* data = NULL;
    int total_vects = 0;

  public:
    // --------- Constructors ----------

    /**
     * @brief Default constructor
     *
     * Constructor that creates an uninitialized DenseVector object.
     */
    DenseVector(void);

    /**
     * @brief Copying constructor
     *
     * Constructor that copies the values from a given DenseVector object.
     *
     * @param that The object to be copied
     */
    DenseVector(const DenseVector& that);

    /**
     * @brief Zeros constructor
     *
     * Constructor that initializes a DenseVector object of wanted size
     * and fills it with zeros.
     *
     * @param len The size of the vector
     */
    DenseVector(int len);

    /**
     * @brief Default value constructor
     *
     * Constructor that initializes a DenseVector object of wanted size
     * and fills it with the wanted value. 
     *
     * @param len The size of the vector
     * @param init_val The value with which the vector is to be filled
     */
    DenseVector(int len, double init_val);

    /**
     * @brief Array copying constructor
     *
     * Constructor that initializes a DenseVector object of wanted size
     * and copies values from a C style array into it.
     *
     * NOTE! As there is no way to verify the length of the C array
     * this constructor might end up reading unwanted memory.
     *
     * @param len The size of the vector
     * @param elems A pointer to the start of the C style array
     */
    DenseVector(int len, double* elems);

    /**
     * @brief Vector copying constructor
     *
     * Constructor that initializes a DenseVector object of wanted size
     * and copies values from a std::vector into it.
     *
     * NOTE! If the length of the passed std::vector doesn't match with
     * the vector size either the extra elements are ignored or the
     * vector is padded with zeros. In either case a warning is printed.
     *
     * @param len The size of the vector
     * @param elems A reference to a std::vector
     */
    DenseVector(int rows, int cols, std::vector<double>& elems);

    /**
     * @brief Default destructor
     *
     * Frees the allocated memory when the DenseVector leaves scope
     */
    ~DenseVector();


    // ------------ Overloaded basic math operators ------------

    // NOTE! The operators will function as elementwise operators

    /**
     * @brief Element-wise addition
     *
     * Method that performs an element-wise addition between this DenseVector
     * object and the one passed as argument.
     *
     * @param that A reference to a DenseVector object used in the sum
     *
     * @return A DenseVector object
     */
    const DenseVector operator+ (const DenseVector& that) const;

    /**
     * @brief Element-wise addition assignment
     *
     * Method that performs an element-wise addition assignment between this DenseVector
     * object and the one passed as argument.
     *
     * @param that A reference to a DenseVector object used in the sum assignment
     *
     * @return A reference to (this) DenseVector object
     */
    DenseVector& operator+= (const DenseVector& that);

    /**
     * @brief Element-wise subtraction
     *
     * Method that performs an element-wise subtraction between this DenseVector
     * object and the one passed as argument.
     *
     * @param that A reference to a DenseVector object used in the difference
     *
     * @return A DenseVector object
     */
    const DenseVector operator- (const DenseVector& that) const;

    /**
     * @brief Element-wise subtraction assignment
     *
     * Method that performs an element-wise subtraction assignment between this DenseVector
     * object and the one passed as argument.
     *
     * @param that A reference to a DenseVector object used in the difference assignment
     *
     * @return A reference to (this) DenseVector object
     */
    DenseVector& operator-= (const DenseVector& that);

    /**
     * @brief Element-wise multiplication
     *
     * Method that performs an element-wise multiplication between this DenseVector
     * object and the one passed as argument.
     *
     * @param that A reference to a DenseVector object used in the multiplication
     *
     * @return A DenseVector object
     */
    const DenseVector operator* (const DenseVector& that) const;
    
    /**
     * @brief Element-wise multiplication assignment
     *
     * Method that performs an element-wise multiplication assignment between this
     * DenseVector object and the one passed as argument.
     *
     * @param that A reference to a DenseVector object used in the multiplication assignment
     *
     * @return A reference to (this) DenseVector object
     */
    DenseVector& operator*= (const DenseVector& that);

    /**
     * @brief Scalar (right) multiplication
     *
     * Method that performs the standard vector-scalar multiplication
     *
     * @param that The scalar used in the multiplication
     *
     * @return A DenseVector object
     */
    const DenseVector operator* (const double that) const;

    /**
     * @brief Element-wise division
     *
     * Method that performs an element-wise division between this DenseVector
     * object and the one passed as argument.
     *
     * @param that A reference to a DenseVector object used in the division
     *
     * @return A DenseVector object
     */
    const DenseVector operator/ (const DenseVector& that) const;

    /**
     * @brief Element-wise division assignment
     *
     * Method that performs an element-wise division assignment between this
     * DenseVector object and the one passed as argument.
     *
     * @param that A reference to a DenseVector object used in the division assignment
     *
     * @return A reference to (this) DenseVector object
     */
    DenseVector& operator/= (const DenseVector& that);

    /**
     * @brief Scalar division
     *
     * Method that performs the standard matrix-scalar division
     *
     * @param that The scalar used in the multiplication
     *
     * @return A DenseVector object
     */
    const DenseVector operator/ (const double that) const;
    
    // ... ?

    // --------- Overloaded indexing operators -----------

    /**
     * @brief Squared bracket indexing method
     *
     * Method that accesses the wanted element in the vector by a single value
     *
     * Alias for DenseVector::operator()
     * @see DenseVector::operator()
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
     * Alias for DenseVector::operator()
     * @see DenseVector::operator()
     *
     * @param num The index of the element
     *
     * @return The value at specified index
     */
    double get(int num) const;

    /**
     * @brief Standard slicing method
     *
     * Method that slices a wanted sized vector from a DenseVector object
     *
     * NOTE! If the end index is out of bounds only the elements in bounds
     * are returned. In this case a warning is printed.
     *
     * @param start The starting index for the slice
     * @param end The ending index for the slice
     *
     * @return A DenseVector object
     */
    const DenseVector operator() (int start, int end) const;

    /**
     * @brief Named slicing method
     *
     * Method that slices a wanted sized vector from a CRSVector object
     *
     * Alias for DenseVector::operator()
     * @see DenseVector::operator()
     *
     * @param start The starting row index for the slice
     * @param end The ending row index for the slice
     *
     * @return A DenseVector object
     */
    const DenseVector get(int start, int end) const;

    /**
     * @brief SIMD accessing method
     *
     * Method that returns the the SIMD vector at specified index in the data array
     *
     * @param num The index of the SIMD vector
     *
     * @return The SIMD vector
     */
    vect_t getSIMD(int num) const;

    // -------- Placement methods ---------

    /**
     * @brief Standard single value placement
     *
     * Method that places a given value at wanted location in a DenseVector object
     *
     * @param num The index of interest
     * @param val The value to be placed
     */
    void place(int num, double val);

    /**
     * @brief Standard vector placement
     * 
     * Method that places the values in a DenseVector object into the wanted location
     * in another DenseVector object.
     *
     * @param start The starting row index for the placement
     * @param end The ending row index for the placement
     * @param vector A reference to the DenseVector object of which values are to be placed
     */
    void place(int start, int end, DenseVector& vector);

    // --------- Other overloaded operators ----------

    /**
     * @brief Default assignment operator
     * 
     * Method that assigns the values in a given DenseVector object into
     * this DenseVector object
     *
     * @param that A reference to the DenseVector object of which values are assigned
     *
     * @return A reference to (this) DenseVector object
     */
    DenseVector& operator= (const DenseVector& that);

    /**
     * @brief Default (equality) comparison operator
     *
     * Method that compares the elements of two DenseVector objects element-wise
     *
     * NOTE! As the elements are stored as double precision floating pointsthere 
     * might be some floating point errors. Thus in some cases it might be better 
     * to use DenseVector::isclose() method.
     * @see DenseVector::isclose()
     *
     * @param that A reference to the DenseVector object of comparison
     *
     * @return A boolean signifying true if equal and false if unequal
     */
    bool operator== (const DenseVector& that);

    /**
     * @brief Default (inequality) comparison operator
     *
     * Method that compares the elements of two DenseVector objects element-wise
     *
     * @param that A reference to the DenseVector object of comparison
     *
     * @return A boolean signifying false if equal and true if unequal
     */
    bool operator!= (const DenseVector& that);

    // --------- Other methods ----------

    /**
     * @brief Approximative equality comparison
     * 
     * Method that compares two DenseVector object element-wise up to a tolerance
     *
     * @param that A reference to the DenseVector object of comparison
     * @param tol OPTIONAL, DEFAULTS TO < double tol = 0.000001 > The tolerance
     * used in the comparison
     *
     * @return A boolean signifying false if equal and true if unequal 
     */
    bool isclose(const DenseVector& that, double tol = 0.000001);

    /**
     * @brief Size of the vector
     *
     * Method that returns the number of elements in the DenseVector object
     *
     * @return The number of elements
     */
    int len() { return _len; }

    /**
     * @brief Size of the vector
     *
     * Method that returns the number of elements in the DenseVector object
     *
     * @return The number of elements as const
     */
    const int len() { return _len; }

    /**
     * @brief Vector-matrix multiplication
     *
     * Method that computes the vector-matrix multiplication in a (relatively) efficient way.
     *
     * @param that A reference to the DenseMatrix object used in multiplication
     *
     * @return The resulting DenseVector object
     */
    const DenseVector matmul(const DenseMatrix& that) const;

    /** 
     * @brief Vector-vector multiplication
     *
     * Method that computes the matrix product vw^T between two vectors v and wt
     * For v^T w multiplication see DenseVector::dot()
     * @see DenseVector::dot()
     *
     * @param that A reference to the DenseVector object used in multiplication
     */
    const DenseMatrix matmul(const DenseVector& that) const;

    /**
     * @brief Dot (inner) product
     *
     * Method that computes the inner product between two DenseVector objects in an efficient way.
     * This method is crucial in most iterative methods.
     *
     * @param that A reference to the DenseVector object used in the product
     *
     * @return The resulting DenseVector object
     */
    double dot(const DenseVector& that) const;

    /**
     * @brief Convert DenseVector into std::vector
     *
     * Method that returns the vector elements in a std::vector.
     *
     * @return The vector elements in a std::vector
     */
    std::vector<double> toVector() const;

    /**
     * @brief Convert DenseVector object into a DenseMatrix object
     *
     * Method that returns the vector elements as a DenseMatrix object.
     *
     * NOTE! The matrix will be a n x 1 matrix.
     *
     * @return The vector elements as a DenseMatrix object
     */
    const DenseMatrix asDenseMatrix() const;
    
    // TODO: CRSVector asCRSVector() const;

    /**
     * @brief Convert DenseVector into a double
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
     * Method that computes the l_p norm of the DenseVector object
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
  std::ostream& operator<<(std::ostream& os, DenseVector& A);

  /**
   * @brief Scalar (left) multiplication
   *
   * Method that performs the standard scalar-vector multiplication
   *
   * @param scalar The scalar used in the multiplication
   * @param vector A reference to the DenseVector object used in multiplication
   *
   * @return A DenseVector object
   */
  const DenseVector operator* (double scalar, const DenseVector& vector);

}
  
#endif
