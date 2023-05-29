#ifndef DENSEMATRIX_HPP
#define DENSEMATRIX_HPP

/*
DenseMatrix is the most general form of matrix. The elements of it are stored
in dense format in a single SIMD vector array. This means that with very large
matrices the memory requirements are vast and thus this is not the recommended 
choice. Then use of CRSMatrix (TODO: Implement) is a good choice.
*/

#include "declare_lalib.hpp"


class DenseVector;  // To avoid circular dependencies

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

        double operator[] (int num) const;
        double operator() (int row, int col) const;
        double get(int row, int col) const;  // Alias for operator()
        const DenseMatrix operator() (int rowStart, int rowEnd, int colStart, int colEnd) const;
        const DenseMatrix get(int rowStart, int rowEnd, int colStart, int colEnd) const;  // Alias for operator()
        const DenseVector getCol(int col) const;
        const DenseVector getRow(int row) const;

        vect_t getSIMD(int num);  // Allows user to access the SIMD vectors for further parallelization

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
        const DenseMatrix matmulStrassen(const DenseMatrix& that) const;
        const DenseMatrix matmulNaive(const DenseMatrix& that) const;
        const DenseVector matmul(const DenseVector& that) const;
        std::vector<double> toVector() const;
        double asDouble() const;
        const DenseVector asDenseVector() const;

        // Statistics

        // const DenseVector mean(int dim = 0);
        // const DenseVector sd(int dim = 0);

        // Friend methods
        
        friend std::ostream& operator<<(std::ostream& os, DenseMatrix& A);
};

// To accomplish commutative property for matrix scalar multiplication

const DenseMatrix operator* (double scalar, const DenseMatrix& matrix);

#endif
