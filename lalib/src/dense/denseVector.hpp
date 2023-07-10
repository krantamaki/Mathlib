#ifndef DENSEVECTOR_HPP
#define DENSEVECTOR_HPP

/*
  DenseVector is a special case of DenseMatrix with either just one row or one column.
  Use of DenseVector allows for fast dot products and other useful features.
*/

#include "../declare_lalib.hpp"


namespace lalib {

  class DenseMatrix;  // To avoid circular dependencies

  class DenseVector {

  protected:
    // Initialize these values to signify an 'empty' matrix

    int _ncols = 0;
    int _nrows = 0;
    vect_t* data = NULL;
    int total_vects = 0;

  public:
    // Constructors

    DenseVector(void);
    DenseVector(const DenseVector& that);
    DenseVector(int rows, int cols);
    DenseVector(int rows, int cols, double init_val);
    DenseVector(int rows, int cols, double* elems);
    DenseVector(int rows, int cols, std::vector<double> elems);

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
    double asDouble() const;
    double norm(double p=2.0) const;

    // Statistics

    double mean();
    double sd();

  };

  std::ostream& operator<<(std::ostream& os, DenseVector& A);

  // To accomplish commutative property for vector scalar multiplication

  const DenseVector operator* (double scalar, const DenseVector& vector);

}
  
#endif
