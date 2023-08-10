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


    // Overload basic math operators

    // NOTE! The operators will function as elementwise operators

    const DenseVector operator+ (const DenseVector& that) const;
    DenseVector& operator+= (const DenseVector& that);
    const DenseVector operator- (const DenseVector& that) const;
    DenseVector& operator-= (const DenseVector& that);
    const DenseVector operator* (const DenseVector& that) const;
    DenseVector& operator*= (const DenseVector& that);
    const DenseVector operator* (const double that) const;
    const DenseVector operator/ (const DenseVector& that) const;
    DenseVector& operator/= (const DenseVector& that);
    const DenseVector operator/ (const double that) const;
    // ... ?


    // Overload indexing operators

    // Additionally for slicing there exists a overloaded operator:
    // DenseVector y = x(start, end)
    // Requires that start < end, but does allow having the end 
    // going out of bounds, but only returns the values that exist. The value start
    // understandably must be in bounds

    double operator[] (int num) const;
    double operator() (int num) const;
    double get(int num) const;  // Alias for operator()
    const DenseVector operator() (int start, int end) const;
    const DenseVector get(int start, int end) const;  // Alias for operator()

    vect_t getSIMD(int num) const;  // Allows user to access the SIMD vectors for further parallelization 


    // Functions for placing values into existing vectors

    void place(int num, double val);
    void place(int start, int end, DenseVector vector);

        
    // Other overloaded operators

    DenseVector& operator= (const DenseVector& that);
    bool operator== (const DenseVector& that);
    bool operator!= (const DenseVector& that);

    // Other methods

    bool isclose(const DenseVector& that, double tol);

    // TODO: int len()  // CHANGE WHOLE STRUCTURE TO BE SIMILAR TO CRSVector

    int ncols() { return _ncols; }
    int nrows() { return _nrows; }
    std::tuple<int, int> shape() { return std::make_tuple(_nrows, _ncols); }

    const int ncols() const { return _ncols; }
    const int nrows() const { return _nrows; }
    const std::tuple<int, int> shape() const { return std::make_tuple(_nrows, _ncols); }

    const DenseVector transpose() const;
    const DenseVector T() const;  // Alias for transpose()
    const DenseVector matmul(const DenseMatrix& that) const;
    const DenseMatrix matmul(const DenseVector& that) const;
    double dot(const DenseVector& that) const;  // Alias for vector vector multiplication which returns double always
    std::vector<double> toVector() const;
    const DenseMatrix asDenseMatrix() const;
    // TODO: CRSVector asCRSVector() const;
    double asDouble() const;
    double norm(double p=2.0) const;

    // Statistics

    // TODO: double mean();
    // TODO: double sd();

  };

  std::ostream& operator<<(std::ostream& os, DenseVector& A);

  // To accomplish commutative property for vector scalar multiplication

  const DenseVector operator* (double scalar, const DenseVector& vector);

}
  
#endif
