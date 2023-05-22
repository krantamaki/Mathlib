#ifndef LALIB_H
#define LALIB_H


/*
This is a general linear algebra library for C++. Main functionality consist of different
matrix classes (currently only DenseMatrix is implemented) which override the basic math operators
(+, -, ...) to function as element-wise operations equivalent to Matlab's .* etc and contain methods
like matmul for computing the common matrix operations. Additionally, there are some general template 
functions that work independent of the matrix type (currently only DenseMatrix).

The methods and functions are made as optimal as feasible with parallelization utilizing multi-threading
and SIMD commands as well as making sure memory calls are linear.
*/


#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <float.h>
#include <time.h>
#include <math.h>
#include <omp.h>


// Define a vector for SIMD commands. Currently set up to hold 4 doubles (for 256 bit vector registers)

#define VECT_ELEMS 4
typedef double vect_t __attribute__ ((__vector_size__ (VECT_ELEMS * sizeof(double))));


// Generally useful functions

int _ceil(int a, int b);


// Declare the classes with dense structure

class DenseMatrix {

    protected:
        // Initialize these values to signify an 'empty' matrix

        int _ncols = 0;
        int _nrows = 0;
        vect_t* data = NULL;
        int vects_per_row = 0;
        int total_vects = 0;

    public:
        // Constructors

        DenseMatrix(void);
        DenseMatrix(const DenseMatrix& that);
        DenseMatrix(int rows, int cols);
        DenseMatrix(int rows, int cols, double init_val);
        DenseMatrix(int rows, int cols, double* elems);
        DenseMatrix(int rows, int cols, std::vector<double> elems);

        ~DenseMatrix();


        // Overload basic math operators

        // NOTE! The operators will function as elementwise operators

        const DenseMatrix operator+ (const DenseMatrix& that) const;
        DenseMatrix& operator+= (const DenseMatrix& that);
        const DenseMatrix operator- (const DenseMatrix& that) const;
        DenseMatrix& operator-= (const DenseMatrix& that);
        const DenseMatrix operator* (const DenseMatrix& that) const;
        DenseMatrix& operator*= (const DenseMatrix& that);
        const DenseMatrix operator* (const double that) const;
        const DenseMatrix operator/ (const DenseMatrix& that) const;
        DenseMatrix& operator/= (const DenseMatrix& that);
        const DenseMatrix operator/ (const double that) const;
        // ... ?


        // Overload indexing operators

        // NOTE! To access a single element there are two alternatives:
        // 1. double elem = A[num]
        // 2. double elem = A(row, col)
        // For these to return the same value must hold that:
        //    num = row * num_rows + col
        // Thus it is recommended to use () for indexing

        // Additionally for slicing there exists a overloaded operator:
        // DenseMatrix B = A(rowStart, rowEnd, colStart, colEnd)
        // Requires that *Start < *End and *Start >= 0, but does allow having the *End 
        // going out of bounds, but only returns the values that exist. The values *Start
        // understandably must be in bounds

        double operator[] (int num);
        double operator() (int row, int col);
        double get(int row, int col);  // Alias for operator()
        const DenseMatrix operator() (int rowStart, int rowEnd, int colStart, int colEnd);
        const DenseMatrix get(int rowStart, int rowEnd, int colStart, int colEnd);  // Alias for operator()
        // const DenseVector getCol(int col);
        // const DenseVector getRow(int row);

        vect_t getSIMD(int num);  // Allows user to access the SIMD vectors for further parallelization
        const vect_t getSIMD(int num) const; 

        // Functions for placing values into existing matrices

        void place(int row, int col, double val);
        void place(int rowStart, int rowEnd, int colStart, int colEnd, DenseMatrix matrix);

        
        // Other overloaded operators

        DenseMatrix& operator= (const DenseMatrix& that);
        bool operator== (const DenseMatrix& that);
        bool operator!= (const DenseMatrix& that);

        // Other methods

        int ncols() { return _ncols; }
        int nrows() { return _nrows; }
        std::tuple<int, int> shape() { return std::make_tuple(_nrows, _ncols); }

        const int ncols() const { return _ncols; }
        const int nrows() const { return _nrows; }
        const std::tuple<int, int> shape() const { return std::make_tuple(_nrows, _ncols); }

        const DenseMatrix transpose() const;
        const DenseMatrix T() const;  // Alias for transpose()
        // DenseMatrix inv();
        const DenseMatrix matmul(const DenseMatrix& that) const;
        // const DenseMatrix matmulStrassen(const DenseMatrix& that) const;
        const DenseMatrix matmulNaive(const DenseMatrix& that) const;
        const DenseVector matmul(const DenseVector& that) const;
        // std::vector<double> toVector();
        double asDouble();
        const DenseVector asDenseVector() const;

        // Statistics

        // const DenseVector mean(int dim = 0);
        // const DenseVector sd(int dim = 0);

        // Friend methods
        
        friend std::ostream& operator<<(std::ostream& os, DenseMatrix& A);
};

// To accomplish commutative property for matrix scalar multiplication

const DenseMatrix operator* (double scalar, const DenseMatrix& matrix);


class DenseVector {

    protected:
        // Initialize these values to signify an 'empty' matrix

        int _ncols = 0;
        int _nrows = 0;
        vect_t* data = NULL;
        int total_vects = 0;

    public:
        // Constructors

        // DenseVector(void);
        // DenseVector(const DenseVector& that);
        DenseVector(int rows, int cols);
        // DenseVector(int rows, int cols, double init_val);
        // DenseVector(int rows, int cols, double* elems);
        // DenseVector(int rows, int cols, std::vector<double> elems);

        ~DenseVector();


        // Overload basic math operators

        // NOTE! The operators will function as elementwise operators

        // const DenseVector operator+ (const DenseVector& that) const;
        // DenseVector& operator+= (const DenseVector& that);
        // const DenseVector operator- (const DenseVector& that) const;
        // DenseVector& operator-= (const DenseVector& that);
        // const DenseVector operator* (const DenseVector& that) const;
        // DenseVector& operator*= (const DenseVector& that);
        // const DenseVector operator* (const double that) const;
        // const DenseVector operator/ (const DenseVector& that) const;
        // DenseVector& operator/= (const DenseVector& that);
        // const DenseVector operator/ (const double that) const;
        // ... ?


        // Overload indexing operators

        // Additionally for slicing there exists a overloaded operator:
        // DenseVector y = x(start, end)
        // Requires that start < end, but does allow having the end 
        // going out of bounds, but only returns the values that exist. The value start
        // understandably must be in bounds

        // double operator[] (int num);
        // double operator() (int num);
        // double get(int num);  // Alias for operator()
        // const DenseVector operator() (int start, int end);
        // const DenseVector get(int start, int end);  // Alias for operator()

        vect_t getSIMD(int num);  // Allows user to access the SIMD vectors for further parallelization
        const vect_t getSIMD(int num) const; 


        // Functions for placing values into existing vectors

        void place(int num, double val);
        // void place(int start, int end, DenseVector vector);

        
        // Other overloaded operators

        // DenseVector& operator= (const DenseVector& that);
        // bool operator== (const DenseVector& that);
        // bool operator!= (const DenseVector& that);

        // Other methods

        int ncols() { return _ncols; }
        int nrows() { return _nrows; }
        std::tuple<int, int> shape() { return std::make_tuple(_nrows, _ncols); }

        const int ncols() const { return _ncols; }
        const int nrows() const { return _nrows; }
        const std::tuple<int, int> shape() const { return std::make_tuple(_nrows, _ncols); }

        // const DenseVector transpose() const;
        // const DenseVector T() const;  // Alias for transpose()
        // DenseMatrix inv();
        const DenseVector matmul(const DenseMatrix& that) const;
        const DenseMatrix matmul(const DenseVector& that) const;
        double dot(const DenseVector& that) const;  // Alias for vector vector multiplication which returns double always
        // std::vector<double> toVector();
        const DenseMatrix asDenseMatrix() const;
        double asDouble();

        // Statistics

        // double mean();
        // double sd();

        // Friend methods
        
        // friend std::ostream& operator<<(std::ostream& os, DenseVector& A);
};

// To accomplish commutative property for vector scalar multiplication

// const DenseVector operator* (double scalar, const DenseVector& matrix);


#endif