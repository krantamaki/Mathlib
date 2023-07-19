#ifndef CRSVECTOR_HPP
#define CRSVECTOR_HPP

/*
  CRSVector is technically a dense matrix with just a single column/row
  Useful for fast matrix vector multiplication
  As the result of matrix vector multiplication is a CRSVector, which has a
  dense representation can the multiplication be trivially parallelized

  NOTE! The main difference between CRSVector and DenseVector is that 
  CRSVector does not use SIMD
*/

#include "../declare_lalib.hpp"


namespace lalib {

  class CRSMatrix;  // To avoid circular dependencies

  class CRSVector {

  protected:
    // Initialize these values to signify an 'empty' matrix

    int _len = 0;
    std::vector<double> data;

  public:
    // Constructors

    CRSVector(void);
    CRSVector(const CRSVector& that);
    CRSVector(int len);
    CRSVector(int len, double init_val);
    CRSVector(int len, double* elems);
    CRSVector(int len, std::vector<double> elems);

    // ~CRSVector();


    // Overload basic math operators

    // NOTE! The operators will function as elementwise operators

    const CRSVector operator+ (const CRSVector& that) const;
    CRSVector& operator+= (const CRSVector& that);
    const CRSVector operator- (const CRSVector& that) const;
    CRSVector& operator-= (const CRSVector& that);
    const CRSVector operator* (const CRSVector& that) const;
    CRSVector& operator*= (const CRSVector& that);
    const CRSVector operator* (const double that) const;
    const CRSVector operator/ (const CRSVector& that) const;
    CRSVector& operator/= (const CRSVector& that);
    const CRSVector operator/ (const double that) const;
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
    const CRSVector operator() (int start, int end) const;
    const CRSVector get(int start, int end) const;  // Alias for operator() 


    // Functions for placing values into existing vectors

    void place(int num, double val);
    void place(int start, int end, CRSVector vector);

        
    // Other overloaded operators

    CRSVector& operator= (const CRSVector& that);
    bool operator== (const CRSVector& that);
    bool operator!= (const CRSVector& that);

    // Other methods

    bool isclose(const CRSVector& that, double tol);

    int len() { return _len; }

    const int len() const { return _len; }

    // const CRSVector transpose() const;
    // const CRSVector T() const;  // Alias for transpose()
    const CRSVector matmul(const CRSMatrix& that) const;
    const CRSMatrix matmul(const CRSVector& that) const;
    double dot(const CRSVector& that) const;  // Alias for vector vector multiplication which returns double always
    std::vector<double> toVector() const;
    const CRSMatrix asCRSMatrix() const;
    double asDouble() const;
    double norm(double p=2.0) const;

    // Statistics

    double mean();
    double sd();

  };

  std::ostream& operator<<(std::ostream& os, CRSVector& A);

  // To accomplish commutative property for vector scalar multiplication

  const CRSVector operator* (double scalar, const CRSVector& vector);

}
  
#endif
