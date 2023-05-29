#include "denseMatrix.hpp"
#include "denseVector.hpp"
#include "declare_lalib.hpp"

/*
TODO: PROPER DESCRIPTION HERE
*/

// Generic matrix multiplication

const DenseMatrix DenseMatrix::matmulNaive(const DenseMatrix& that) const {
    if (_ncols != that._nrows) {
        throw std::invalid_argument("Improper dimensions given!");
    }

    // Allocate memory for the resulting matrix
    DenseMatrix ret = DenseMatrix(_nrows, that._ncols);

    // Transpose that for linear memory reads
    DenseMatrix that_T = that.T();

    #pragma omp parallel for schedule(dynamic, 1)
	for (int row = 0; row < _nrows; row++) {
		for (int col = 0; col < that._ncols; col++) {    

			vect_t sum;
            for (int i = 0; i < VECT_ELEMS; i++) {
                sum[i] = 0.0;
            }

			for (int vect = 0; vect < vects_per_row; vect++) {
				sum += data[vects_per_row * row + vect] * that_T.data[vects_per_row * col + vect];
			}

			double val = 0.0;
			for (int elem = 0; elem < VECT_ELEMS; elem++) {
				val += sum[elem];	
			}
			
			ret.place(row, col, val);
		}
	}

    return ret;
}

// TODO: Implement Strassen algorithm
// const DenseMatrix DenseMatrix::matmulStrassen(const DenseMatrix& that) const {}

// Wrapper function that calls either matmulNaive or matmulStrassen based on matrix size

const DenseMatrix DenseMatrix::matmul(const DenseMatrix& that) const {
    if (_ncols != that._nrows) {
        throw std::invalid_argument("Improper dimensions given!");
    }

    // 100 chosen as arbitrary threshold
    if (_ncols > 100 && _nrows > 100 && that._ncols > 100) {
        return this->matmulNaive(that);  // Should call Strassen algorithm, but that is not implemented yet
    }

    return this->matmulNaive(that);
}

// Vector vector multiplication can lead to either a matrix or a scalar depending on if you
// multiply row vector with column vector or column vector with row vector.
// Thus the return type is DenseMatrix that might be a 1 x 1 matrix

const DenseMatrix DenseVector::matmul(const DenseVector& that) const {
    if (_ncols != that._nrows || _nrows != that._ncols) {
        throw std::invalid_argument("Improper dimensions given!");
    }

    return (this->asDenseMatrix()).matmul(that.asDenseMatrix());
}

const DenseVector DenseVector::matmul(const DenseMatrix& that) const {
    if (_ncols != that.nrows()) {
        throw std::invalid_argument("Improper dimensions given!");
    }

    return ((this->asDenseMatrix()).matmul(that)).asDenseVector();
}

// Dot product behaves similarly to matmul, but always returns a scalar value
// (even when multiplying column vector with row vector or column vector with column vector etc.)

double DenseVector::dot(const DenseVector& that) const {
    if (!((_ncols == that._ncols && _nrows == that._nrows) || (_ncols == that._nrows && _nrows == that._ncols))) {
        throw std::invalid_argument("Improper dimensions given!");
    }

    vect_t sum;
    for (int i = 0; i < VECT_ELEMS; i++) {
        sum[i] = 0.0;
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int vect = 0; vect < total_vects; vect++) {
        sum += data[vect] * that.data[vect];
    }

    double ret = 0;
    for (int elem = 0; elem < VECT_ELEMS; elem++) {
		ret += sum[elem];	
	}

    return ret;
}


// Matrix vector multiplications could reuse the matmul method for DenseMatrix
// but as it is crucial for it to be efficient it is implemented from scratch here

const DenseVector DenseMatrix::matmul(const DenseVector& that) const {
    if (_ncols != that.nrows()) {
        throw std::invalid_argument("Improper dimensions given!");
    }

    // Allocate memory for the resulting vector
    DenseVector ret = DenseVector(that.nrows(), 1);

    #pragma omp parallel for schedule(dynamic, 1)
	for (int row = 0; row < _nrows; row++) {

        vect_t sum;
        for (int i = 0; i < VECT_ELEMS; i++) {
            sum[i] = 0.0;
        }

		for (int vect = 0; vect < vects_per_row; vect++) {    

			sum += data[vects_per_row * row + vect] * that.getSIMD(vect);

			double val = 0.0;
			for (int elem = 0; elem < VECT_ELEMS; elem++) {
				val += sum[elem];	
			}
			
			ret.place(row, val);
		}
	}

    return ret;
}
