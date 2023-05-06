#include "declare_lalib.h"


// Function for "dividing up" two integers
int _ceil(int a, int b) {
    return (a + b - 1) / b;
}

// Constructor that doesn't allocate memory
DenseMatrix::DenseMatrix(void) {}

// Constructor that copies the contents of a given matrix
DenseMatrix::DenseMatrix(const DenseMatrix& that) {
    if (that._ncols > 0 && that._nrows > 0) {
        _ncols = that._ncols;
        _nrows = that._nrows;

        vects_per_row = that.vects_per_row;
        total_vects = that.total_vects;

        // Allocate aligned memory
        void* tmp = 0;
        if (posix_memalign(&tmp, sizeof(vect_t), total_vects * sizeof(vect_t))) {
            throw std::bad_alloc();
        }

        data = (vect_t*)tmp;

        // Copy the data from that
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < _nrows; i++) {
            for (int vect = 0; vect < vects_per_row; vect++) {
                data[vects_per_row * i + vect] = that.data[vects_per_row * i + vect];
            }
        } 
    }
}

// Constructor that allocates memory for wanted sized matrix and initializes
// the values as zeros
DenseMatrix::DenseMatrix(int rows, int cols) {
    if (cols < 1 || rows < 1) {
        throw std::invalid_argument("Matrix dimensions must be non-negative!");
    }

    _ncols = cols;
    _nrows = rows;

    vects_per_row = _ceil(cols, VECT_ELEMS);
    total_vects = rows * vects_per_row;

    // Allocate aligned memory
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(vect_t), total_vects * sizeof(vect_t))) {
        throw std::bad_alloc();
    }

    // Initialize the data values as zeros
	data = (vect_t*)tmp;
	const vect_t zeros = {(double)0.0, (double)0.0, (double)0.0, (double)0.0};

	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < rows; i++) {
		for (int vect = 0; vect < vects_per_row; vect++) {
			data[vects_per_row * i + vect] = zeros;
		}
	}
}

// Constructor that allocates memory for wanted sized matrix and initializes
// the values as wanted double
DenseMatrix::DenseMatrix(int rows, int cols, double init_val) {
    if (cols < 1 || rows < 1) {
        throw std::invalid_argument("Matrix dimensions must be non-negative!");
    }

    _ncols = cols;
    _nrows = rows;

    vects_per_row = _ceil(cols, VECT_ELEMS);
    total_vects = rows * vects_per_row;

    // Allocate aligned memory
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(vect_t), total_vects * sizeof(vect_t))) {
        throw std::bad_alloc();
    }

    // Initialize the data values as init_val
	data = (vect_t*)tmp;
	const vect_t init_vals = {init_val, init_val, init_val, init_val};

	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < rows; i++) {
		for (int vect = 0; vect < vects_per_row; vect++) {
			data[vects_per_row * i + vect] = init_vals;
		}
	}
}

DenseMatrix::~DenseMatrix() {
    free(data);
}

DenseMatrix& DenseMatrix::operator+= (const DenseMatrix& that) {
    // Check that the dimensions match
    if (_ncols != that._ncols || _nrows != that._nrows) {
        throw std::invalid_argument("Matrix dimensions must match!");
    } 

	#pragma omp parallel for schedule(dynamic, 1)
	for (int row = 0; row < _nrows; row++) {
		for (int vect = 0; vect < vects_per_row; vect++) {
			data[vects_per_row * row + vect] = data[vects_per_row * row + vect] + that.data[that.vects_per_row * row + vect];
		}
	}

    return *this;
}

const DenseMatrix DenseMatrix::operator+ (const DenseMatrix& that) const {
    return DenseMatrix(*this) += that;
}

DenseMatrix& DenseMatrix::operator-= (const DenseMatrix& that) {
    // Check that the dimensions match
    if (_ncols != that._ncols || _nrows != that._nrows) {
        throw std::invalid_argument("Matrix dimensions must match!");
    } 

	#pragma omp parallel for schedule(dynamic, 1)
	for (int row = 0; row < _nrows; row++) {
		for (int vect = 0; vect < vects_per_row; vect++) {
			data[vects_per_row * row + vect] = data[vects_per_row * row + vect] - that.data[that.vects_per_row * row + vect];
		}
	}

    return *this;
}

const DenseMatrix DenseMatrix::operator- (const DenseMatrix& that) const {
    return DenseMatrix(*this) -= that;
}

DenseMatrix& DenseMatrix::operator*= (const DenseMatrix& that) {
    // Check that the dimensions match
    if (_ncols != that._ncols || _nrows != that._nrows) {
        throw std::invalid_argument("Matrix dimensions must match!");
    } 

	#pragma omp parallel for schedule(dynamic, 1)
	for (int row = 0; row < nrows(); row++) {
		for (int vect = 0; vect < vects_per_row; vect++) {
			data[vects_per_row * row + vect] = data[vects_per_row * row + vect] * that.data[that.vects_per_row * row + vect];
		}
	}

    return *this;
}

const DenseMatrix DenseMatrix::operator* (const DenseMatrix& that) const {
    return DenseMatrix(*this) *= that;
}

DenseMatrix& DenseMatrix::operator/= (const DenseMatrix& that) {
    // Check that the dimensions match
    if (ncols() != that._ncols || nrows() != that._ncols) {
        throw std::invalid_argument("Matrix dimensions must match!");
    } 

	#pragma omp parallel for schedule(dynamic, 1)
	for (int row = 0; row < nrows(); row++) {
		for (int vect = 0; vect < vects_per_row; vect++) {
			data[vects_per_row * row + vect] = data[vects_per_row * row + vect] / that.data[that.vects_per_row * row + vect];
		}
	}

    return *this;
}

const DenseMatrix DenseMatrix::operator/ (const DenseMatrix& that) const {
    return DenseMatrix(*this) /= that;
}

double DenseMatrix::operator() (int row, int col) {
    // Check that the dimensions are in bounds
    if (row < 0 || col < 0 || row >= _nrows || col >= _ncols) {
        throw std::invalid_argument("Given dimensions out of bounds!");
    }

    // Find the proper vector and element in said vector for column col
	const int vect = col / VECT_ELEMS;  // Integer division defaults to floor
	const int elem = col % VECT_ELEMS;

	return data[vects_per_row * row + vect][elem];
}

DenseMatrix& DenseMatrix::operator= (const DenseMatrix& that) {
    // Check for self-assignment ie. case where a = a is called by comparing the pointers of the objects
    if (this == &that) return *this; 

    // Free the existing memory and allocate new one that matches the dimensions of that
    free(data);

    _ncols = that._ncols;
    _nrows = that._nrows;

    vects_per_row = that.vects_per_row;
    total_vects = that.total_vects;

    // Allocate aligned memory
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(vect_t), total_vects * sizeof(vect_t))) {
        throw std::bad_alloc();
    }

    // Initialize the data values as init_val
	data = (vect_t*)tmp;

    // Copy the data from that
    #pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < _nrows; i++) {
		for (int vect = 0; vect < vects_per_row; vect++) {
			data[vects_per_row * i + vect] = that.data[vects_per_row * i + vect];
		}
	}

    return *this;
}

std::ostream& operator<<(std::ostream& os, DenseMatrix& A) {
    if (A.ncols() == -1 || A.nrows() == 1) {
        os << "[]" << std::endl;  // Signifies uninitialized matrix
        
        return os;
    }
    
    os << "[";
    for (int row = 0; row < A.nrows(); row++) {
        if (row > 0) os << ' ';

        os << "[";
        for (int col = 0; col < A.ncols() - 1; col++) {
            os << A(row, col) << ' ';
        }
        os << A(row, A.ncols() - 1) << "]";

        if (row < A.nrows() - 1) os << std::endl; 
    }
    os << "]" << std::endl;

    return os;
}




