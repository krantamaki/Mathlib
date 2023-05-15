#include "declare_lalib.h"

/*
TODO: PROPER DESCRIPTION HERE
*/

// Generic matrix multiplication

const DenseMatrix DenseMatrix::matmul(const DenseMatrix& that) const {
    // Check that the dimensions are valid
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

