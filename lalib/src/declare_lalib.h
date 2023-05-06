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
#include <float.h>
#include <time.h>
#include <math.h>
#include <omp.h>


// Define a vector for SIMD commands. Currently set up to hold 4 doubles (for 256 bit vector registers)

#define VECT_ELEMS 4
typedef double vect_t __attribute__ ((__vector_size__ (VECT_ELEMS * sizeof(double))));


// Generally useful functions

int _ceil(int a, int b);


// Declare the matrix classes

class DenseMatrix {

    protected:
        // Initialize these values to signify an 'empty' matrix

        int _ncols = -1;
        int _nrows = -1;
        vect_t* data = NULL;
        int vects_per_row = -1;
        int total_vects = -1;

    public:
        // Constructors

        DenseMatrix(void);
        DenseMatrix(const DenseMatrix& that);
        DenseMatrix(int rows, int cols);
        DenseMatrix(int rows, int cols, double init_val);
        // DenseMatrix(int rows, int cols, double* data);
        // DenseMatrix(int rows, int cols, std::vector<double> data);

        ~DenseMatrix();


        // Overload basic math operators

        // NOTE! The operators will function as elementwise operators

        const DenseMatrix operator+ (const DenseMatrix& that) const;
        DenseMatrix& operator+= (const DenseMatrix& that);
        const DenseMatrix operator- (const DenseMatrix& that) const;
        DenseMatrix& operator-= (const DenseMatrix& that);
        const DenseMatrix operator* (const DenseMatrix& that) const;
        DenseMatrix& operator*= (const DenseMatrix& that);
        // DenseMatrix operator* (double that);
        const DenseMatrix operator/ (const DenseMatrix& that) const;
        DenseMatrix& operator/= (const DenseMatrix& that);
        // DenseMatrix operator/ (double that);
        // ...


        // Overload indexing operators

        // NOTE! To access a single element there are two alternatives:
        // 1. double elem = A[row][col]
        // 2. double elem = A(row, col)
        // Out of these the () operator is recommended as it doesn't create a temporary column vector

        // Additionally for slicing there exists a overloaded operator:
        // DenseMatrix B = A(rowStart, rowEnd, colStart, colEnd)
        // Requires that *Start < *End and *Start >= 0, but does allow having the *End 
        // going out of bounds, but only returns the values that exist. The values *Start
        // understandably must be in bounds

        // DenseMatrix operator[] (int row);
        // double operator[] (int row);  // For vectors
        // DenseMatrix operator() (int rowStart, int rowEnd, int colStart, int colEnd);
        double operator() (int row, int col);

        
        // Other overloaded operators

        DenseMatrix& operator= (const DenseMatrix& that);
        // bool operator== (const DenseMatrix& that);
        // bool operator!= (const DenseMatrix& that);

        // Other methods

        int ncols() { return _ncols; }
        int nrows() { return _nrows; }
        // DenseMatrix transpose();
        // DenseMatrix T();
        // DenseMatrix inv();
        // DenseMatrix matmul(const DenseMatrix& that);
        // std::vector<double> toVector();

        // Friend methods
        
        friend std::ostream& operator<<(std::ostream& os, DenseMatrix& A);
};

#endif