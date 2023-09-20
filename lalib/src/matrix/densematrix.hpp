#ifndef DENSEMATRIX_HPP
#define DENSEMATRIX_HPP


#include "matrix_decl.hpp"


using namespace utils;


namespace lalib {

  /**
   * @brief Dense representation matrix class
   *
   * Dense representation of a matrix is the simplest way of storing matrix information.
   * In a dense representation every element of the matrix is held in memory and can be
   * accessed in constant time. This allows for very efficient matrix operations and
   * trivial parallelization. 
   *
   * The current implementation uses SIMD commands and OpenMP threading for speeding up
   * computations.
   */
  template <class type, bool vectorize>
  class Matrix<type, vectorize, false> {

    protected:

      // Alias the used variable type in computations
      using var_t = decltype(_choose_simd<type, vectorize>());


      // Should define a SIMD vector of zeros or scalar zero depending on vectorization
      var_t v_zero = { };

      // Defines a zero of the underlying type
      type t_zero = { };

      // Defines the number of elements in the SIMD vector if vectorized
      int var_size = SIMD_SIZE / (int)sizeof(type);


      // Const function that sums together the elements in a SIMD vector
      inline const type _reduce(const var_t val) const {

        // Check if the implementation is vectorized at compile time
        if constexpr (!vectorize) return val;
        // Otherwise reduce to scalar
        else {
          type ret = t_zero;
          for (int i = 0; i < var_size; i++) {
            ret += val[i];
          }

          return ret;
        }        
      }


      // Const function that fills the SIMD vector with wanted value
      inline const var_t _fill(const type val) const {

        // Check if the implementation is vectorized at compile time
        if constexpr (!vectorize) return val;
        // Otherwise fill a SIMD vector
        else {
          var_t ret;
          for (int i = 0; i < var_size; i++) {
            ret[i] = val;
          }

          return ret;
        }
      }


      // Define the data array
      std::vector<var_t> data;


      // Define the number of SIMD vectors in total in the matrix and per row of the matrix
      int _total_vects = 0;
      int _vects_per_row = 0;


      // Define the size of the matrix
      int _ncols = 0;
      int _nrows = 0;

    public:
    
      // ---------- Constructors ------------
      
      /**
       * @brief Default constuctor
       *
       * Constructor that initializes an empty dense Matrix object
       */
      Matrix(void) { }


      /**
       * @brief Copying constructor
       *
       * Constructor that initializes a dense Matrix object and copies contents
       * of an another dense Matrix object into it
       *
       * @param that A reference to a dense Matrix object wanted to be copied
       */
      Matrix(const Matrix& that) {

        if (that._ncols < 1 || that._nrows < 1) {
          _errorMsg("Matrix dimensions must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        _ncols = that._ncols;
        _nrows = that._nrows;

        if constexpr (vectorize) {
          data = that.data;
          _total_vects = that._total_vects;
          _vects_per_row = that._vects_per_row;
        }
        else {
          data = that.data;
        }
      }


      /**
       * @brief Zeros constructor
       *
       * Constructor that initializes a dense Matrix object of wanted shape and
       * fills it with zeros
       *
       * @param rows The number of rows
       * @param cols The number of columns
       */
      Matrix(int rows, int cols) {

        if (cols < 1 || rows < 1) {
          _errorMsg("Matrix dimensions must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        _ncols = cols;
        _nrows = rows;

        if constexpr (vectorize) {
          _vects_per_row = _ceil(cols, var_size);
          _total_vects = rows * _vects_per_row;

          data = std::vector<var_t>(_total_vects, v_zero);
        }
        else {
          data = std::vector<var_t>(_ncols * _nrows, v_zero);
        }
      }


      /**
       * @brief Default value constructor
       *
       * Constructor that initializes a dense Matrix object of wanted shape and
       * fills it with given value
       *
       * @param rows The number of rows
       * @param cols The number of columns
       * @param init_val The value with which the matrix is to be filled
       */
      Matrix(int rows, int cols, type init_val) {

        if (cols < 1 || rows < 1) {
          _errorMsg("Matrix dimensions must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        _ncols = cols;
        _nrows = rows;

        if constexpr (vectorize) {
          var_t init_vect = _fill(init_val);

          _vects_per_row = _ceil(cols, var_size);
          _total_vects = rows * _vects_per_row;
          data = std::vector<var_t>(_total_vects, init_vect);
        }
        else {
          data = std::vector<var_t>(_ncols * _nrows, init_val);
        }
      }


      /**
       * @brief Array copying constructor
       *
       * Constructor that initializes a dense Matrix object of wanted shape and
       * copies the contents of a C style array into it
       *
       * NOTE! As there is no proper way to verify the length of the C style
       * array this constructor might end up reading unwanted memory.
       *
       * @param rows The number of rows
       * @param cols The number of columns
       * @param elems A pointer to the start of the C style array
       */
      Matrix(int rows, int cols, type* elems) {

        _warningMsg("Initializing a matrix with double array might lead to undefined behaviour!", __func__);

        if (cols < 1 || rows < 1) {
          _errorMsg("Matrix dimensions must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        _ncols = cols;
        _nrows = rows;

        if constexpr (vectorize) {
          _vects_per_row = _ceil(cols, var_size);
          _total_vects = rows * _vects_per_row;

          for (int vect = 0; vect < _total_vects; vect++) {
            var_t tmp_vect;
            for (int elem = 0; elem < var_size; elem++) {
              int i = vect * var_size + elem;
              tmp_vect[elem] = i < _ncols * _nrows ? elems[i] : t_zero;
            }
            data.push_back(tmp_vect);
          }
        }
        else {
          data.assign(elems, elems + _ncols * _nrows);
        }
      }


      /**
       * @brief Vector copying constructor
       *
       * Constructor that initializes a dense Matrix object of wanted shape and
       * copies the contents of a std::vector into it
       *
       * NOTE! If the length of the std::vector doesn't match with the dimensions
       * of the matrix either the extra elements are ignored or the last rows are
       * filled with zeros. In either case a warning is printed.
       *
       * @param rows The number of rows
       * @param cols The number of columns
       * @param elems A reference to a std::vector
       */
      Matrix(int rows, int cols, std::vector<type>& elems) {

        if (cols < 1 || rows < 1) {
          _errorMsg("Matrix dimensions must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if (rows * cols != (int)elems.size()) {
          _warningMsg("Given dimensions don't match with the size of the std::vector!", __func__);
        } 

        _ncols = cols;
        _nrows = rows;

        if constexpr (vectorize) {
          _vects_per_row = _ceil(cols, var_size);
          _total_vects = rows * _vects_per_row;
          for (int vect = 0; vect < _total_vects; vect++) {
            var_t tmp_vect;
            for (int elem = 0; elem < var_size; elem++) {
              int i = vect * var_size + elem;
              tmp_vect[elem] = (i < (int)elems.size()) && (i < _ncols * _nrows) ? elems[i] : t_zero;
            }
            data.push_back(tmp_vect);
          }
        }
        else {
          std::copy(elems.begin(), elems.begin() + _ncols * _nrows, std::back_inserter(data));
        }
      }
    
    
      // ---------  Overloaded basic math operators ----------
    
      // NOTE! The operators will function as elementwise operators

      /**
       * @brief Element-wise addition assignment
       *
       * Method that performs an element-wise addition assignment between this dense Matrix
       * object and the one passed as argument.
       *
       * @param that A reference to a dense Matrix object used in the sum assignment
       *
       * @return A reference to (this) dense Matrix object
       */
      Matrix& operator+= (const Matrix& that) {

        if (_ncols != that._ncols || _nrows != that._nrows) {
          _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        } 

        if constexpr (vectorize) {
          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {
            for (int vect = 0; vect < _vects_per_row - 1; vect++) {
              data[vect] = data[vect] + that.data[vect];
            }
            for (int elem = 0; elem < _ncols % var_size; elem++) {
              data[(row + 1) * _vects_per_row - 1][elem] = data[(row + 1) * _vects_per_row - 1][elem] + that.data[(row + 1) * _vects_per_row - 1][elem];
            }
          }
        }
        else {
          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {
            for (int col = 0; col < _ncols; col++) {
              data[row * _nrows + col] = data[row * _nrows + col] + that.data[row * _nrows + col];
            }
          }
        }
        
        return *this;
      }
      

      /**
       * @brief Element-wise addition
       *
       * Method that performs an element-wise addition between this dense Matrix
       * object and the one passed as argument.
       *
       * @param that A reference to a dense Matrix object used in the sum
       *
       * @return A dense Matrix object
       */
      const Matrix operator+ (const Matrix& that) const {
        return Matrix(*this) += that;
      }


      /**
       * @brief Element-wise subtraction assignment
       *
       * Method that performs an element-wise subtraction assignment between this dense Matrix
       * object and the one passed as argument.
       *
       * @param that A reference to a dense Matrix object used in the difference assignment
       *
       * @return A reference to (this) dense Matrix object
       */
      Matrix& operator-= (const Matrix& that) {

        if (_ncols != that._ncols || _nrows != that._nrows) {
          _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        } 

        if constexpr (vectorize) {
          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {
            for (int vect = 0; vect < _vects_per_row - 1; vect++) {
              data[vect] = data[vect] - that.data[vect];
            }
            for (int elem = 0; elem < _ncols % var_size; elem++) {
              data[(row + 1) * _vects_per_row - 1][elem] = data[(row + 1) * _vects_per_row - 1][elem] - that.data[(row + 1) * _vects_per_row - 1][elem];
            }
          }
        }
        else {
          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {
            for (int col = 0; col < _ncols; col++) {
              data[row * _nrows + col] = data[row * _nrows + col] - that.data[row * _nrows + col];
            }
          }
        }

        return *this;
      }
      

      /**
       * @brief Element-wise subtraction
       *
       * Method that performs an element-wise subtraction between this dense Matrix
       * object and the one passed as argument.
       *
       * @param that A reference to a dense Matrix object used in the difference
       *
       * @return A dense Matrix object
       */
      const Matrix operator- (const Matrix& that) const {
        return Matrix(*this) -= that;
      }


      /**
       * @brief Element-wise multiplication assignment
       *
       * Method that performs an element-wise multiplication assignment between this
       * dense Matrix object and the one passed as argument.
       *
       * @param that A reference to a dense Matrix object used in the multiplication assignment
       *
       * @return A reference to (this) dense Matrix object
       */
      Matrix& operator*= (const Matrix& that) {

        if (_ncols != that._ncols || _nrows != that._nrows) {
          _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        } 

        if constexpr (vectorize) {
          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {
            for (int vect = 0; vect < _vects_per_row - 1; vect++) {
              data[vect] = data[vect] * that.data[vect];
            }
            for (int elem = 0; elem < _ncols % var_size; elem++) {
              data[(row + 1) * _vects_per_row - 1][elem] = data[(row + 1) * _vects_per_row - 1][elem] * that.data[(row + 1) * _vects_per_row - 1][elem];
            }
          }
        }
        else {
          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {
            for (int col = 0; col < _ncols; col++) {
              data[row * _nrows + col] = data[row * _nrows + col] * that.data[row * _nrows + col];
            }
          }
        }

        return *this;
      }


      /**
       * @brief Element-wise multiplication
       *
       * Method that performs an element-wise multiplication between this dense Matrix
       * object and the one passed as argument.
       *
       * @param that A reference to a dense Matrix object used in the multiplication
       *
       * @return A dense Matrix object
       */
      const Matrix operator* (const Matrix& that) const {
        return Matrix(*this) *= that;
      }


      /**
       * @brief Scalar (right) multiplication assignment
       *
       * Method that performs the standard matrix-scalar multiplication assignment
       *
       * @param that The scalar used in the multiplication
       *
       * @return A reference to (this) dense Matrix object
       */
      Matrix& operator*= (type that) {

        if (_nrows < 1 || _ncols < 1) {
          return *this;
        }

        if constexpr (vectorize) {
          var_t mult = _fill(that);

          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {
            for (int vect = 0; vect < _vects_per_row - 1; vect++) {
              data[vect] = data[vect] * mult;
            }
            for (int elem = 0; elem < _ncols % var_size; elem++) {
              data[(row + 1) * _vects_per_row - 1][elem] = data[(row + 1) * _vects_per_row - 1][elem] * that;
            }
          }
        }
        else {
          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {
            for (int col = 0; col < _ncols; col++) {
              data[row * _nrows + col] = data[row * _nrows + col] * that;
            }
          }
        }

        return *this;    
      }


      /**
       * @brief Scalar (right) multiplication
       *
       * Method that performs the standard matrix-scalar multiplication
       *
       * @param that The scalar used in the multiplication
       *
       * @return A dense Matrix object
       */
      const Matrix operator* (const type that) const {
        return Matrix(*this) *= that;
      }


      /**
       * @brief Element-wise division assignment
       *
       * Method that performs an element-wise division assignment between this
       * dense Matrix object and the one passed as argument.
       *
       * @param that A reference to a dense Matrix object used in the division assignment
       *
       * @return A reference to (this) dense Matrix object
       */
      Matrix& operator/= (const Matrix& that) {

        if (ncols() != that._ncols || nrows() != that._ncols) {
          _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        } 

        if constexpr (vectorize) {
          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {
            for (int vect = 0; vect < _vects_per_row - 1; vect++) {
              data[vect] = data[vect] / that.data[vect];
            }
            for (int elem = 0; elem < _ncols % var_size; elem++) {
              data[(row + 1) * _vects_per_row - 1][elem] = data[(row + 1) * _vects_per_row - 1][elem] / that.data[(row + 1) * _vects_per_row - 1][elem];
            }
          }
        }
        else {
          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {
            for (int col = 0; col < _ncols; col++) {
              data[row * _nrows + col] = data[row * _nrows + col] / that.data[row * _nrows + col];
            }
          }
        }

        return *this;
      }
      

      /**
       * @brief Element-wise division
       *
       * Method that performs an element-wise division between this dense Matrix
       * object and the one passed as argument.
       *
       * @param that A reference to a dense Matrix object used in the division
       *
       * @return A dense Matrix object
       */
      const Matrix operator/ (const Matrix& that) const {
        Matrix(*this) /= that;
      }


      /**
       * @brief Scalar division assignment
       *
       * Method that performs the standard matrix-scalar division assignment
       *
       * @param that The scalar used in the division
       *
       * @return A reference to (this) dense Matrix object
       */
      Matrix& operator/= (type that) {

        if (that == t_zero) {
          _errorMsg("Division by zero undefined!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        if (_nrows < 1 || _ncols < 1) {
          return *this;
        }

        if constexpr (vectorize) {
          var_t div = _fill(that);

          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {
            for (int vect = 0; vect < _vects_per_row - 1; vect++) {
              data[vect] = data[vect] / div;
            }
            for (int elem = 0; elem < _ncols % var_size; elem++) {
              data[(row + 1) * _vects_per_row - 1][elem] = data[(row + 1) * _vects_per_row - 1][elem] / that;
            }
          }
        }
        else {
          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {
            for (int col = 0; col < _ncols; col++) {
              data[row * _nrows + col] = data[row * _nrows + col] / that;
            }
          }
        }
        
        return *this;
      }
      

      /**
       * @brief Scalar (right) division
       *
       * Method that performs a matrix-scalar division
       *
       * @param that The scalar used in the division
       *
       * @return A dense Matrix object
       */
      const Matrix operator/ (const type that) const {
        return Matrix(*this) /= that;
      }

      
      // -------- PLACEMENT METHODS ---------

      /**
       * @brief Standard single value placement
       *
       * Method that places a given value at wanted location in a dense Matrix object
       *
       * @param row The row of interest
       * @param col The column of interest
       * @param val The value to be placed
       */
      void place(int row, int col, type val) {

        if (row < 0 || col < 0 || row >= _nrows || col >= _ncols) {
          _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if constexpr (vectorize) {
          const int vect = col / var_size;
          const int elem = col % var_size;

          data[_vects_per_row * row + vect][elem] = val;
        }
        else {
          data[_ncols * row + col] = val;
        }
      }


      /**
       * @brief Standard matrix placement
       * 
       * Method that places the values in a dense Matrix object into the wanted location
       * in another dense Matrix object.
       *
       * @param rowStart The starting row index for the placement
       * @param rowEnd The ending row index for the placement
       * @param colStart The starting column index for the placement
       * @param colEnd The ending column index for the placement
       * @param matrix A reference to the dense Matrix object of which values are to be placed
       */
      void place(int rowStart, int rowEnd, int colStart, int colEnd, const Matrix& matrix) {

        if (_nrows < rowEnd - rowStart || _ncols < colEnd - colStart || matrix._nrows < rowEnd - rowStart || matrix._ncols < colEnd - colStart) {
          _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        #pragma omp parallel for schedule(dynamic, 1)
        for (int row0 = 0; row0 < rowEnd - rowStart; row0++) {
          int row = row0 + rowStart;
          for (int col0 = 0; col0 < colEnd - colStart; col0++) {
            int col = col0 + colStart;
            this->place(row, col, matrix(row0, col0));
          }
        }
      }

      
      // TODO: void placeCol(int col, DenseVector vector);

      // TODO: void placeRow(int row, DenseVector vector);


      // ---------- Overloaded indexing operators -----------

      /**
       * @brief Standard indexing method
       *
       * Method that accesses the wanted element in the matrix
       *
       * @param row The row of the wanted element
       * @param col The column of the wanted element
       *
       * @return The value on row 'row' and column 'col' in the matrix
       */
      type operator() (int row, int col) const {

        if (row < 0 || col < 0 || row >= _nrows || col >= _ncols) {
          _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if constexpr (vectorize) {
          const int vect = col / var_size;
          const int elem = col % var_size;

          return data[_vects_per_row * row + vect][elem];
        }
        else {
          return data[_ncols * row + col];
        }
      }
      

      /**
       * @brief Named indexing method
       *
       * Method that accesses the wanted element in the matrix
       *
       * Alias for Matrix::operator()
       * @see Matrix::operator()
       *
       * @param row The row of the wanted element
       * @param col The column of the wanted element
       *
       * @return The value on row 'row' and column 'col' in the matrix
       */
      type get(int row, int col) const {
        return this->operator() (row, col);
      }


      /**
       * @brief Squared bracket indexing method
       *
       * Method that accesses the wanted element in the matrix by a single value
       *
       * Returns the element on row $num / _ncols$ and column $num % _ncols$. 
       *
       * @param num The index of the element
       *
       * @return The value at specified index
       */
      type operator[] (int num) const {

        if (num >= _ncols * _nrows) {
          _errorMsg("Given index out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        const int row = num / _ncols;
        const int col = num % _ncols;

        return this->operator() (row, col);
      }


      /**
       * @brief Standard slicing method
       *
       * Method that slices a wanted sized matrix from a dense Matrix object
       *
       * NOTE! If the end indeces are out of bounds only the elements in bounds
       * are returned. In this case a warning is printed.
       *
       * @param rowStart The starting row index for the slice
       * @param rowEnd The ending row index for the slice
       * @param colStart The starting column index for the slice
       * @param colEnd The ending column index for the slice
       *
       * @return A dense Matrix object
       */
      const Matrix operator() (int rowStart, int rowEnd, int colStart, int colEnd) const {
        
        if (rowStart >= rowEnd || rowStart < 0 || colStart >= colEnd || colStart < 0) {
          _errorMsg("Improper dimensions given!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if (rowEnd > _nrows || colEnd > _ncols) {
          _warningMsg("End index out of bounds", __func__);
        }

        int _rowEnd = rowEnd > _nrows ? _nrows : rowEnd;
        int _colEnd = colEnd > _ncols ? _ncols : colEnd;

        // Allocate memory for a new matrix
        Matrix ret = Matrix(_rowEnd - rowStart, _colEnd - colStart);

        #pragma omp parallel for schedule(dynamic, 1)
        for (int row0 = 0; row0 < _rowEnd - rowStart; row0++) {
          int row = row0 + rowStart;
          for (int col0 = 0; col0 < _colEnd - colStart; col0++) {
            int col = col0 + colStart;
            ret.place(row0, col0, this->operator()(row, col));
          }
        }

        return ret;
      }
      

      /**
       * @brief Named slicing method
       *
       * Method that slices a wanted sized matrix from a dense Matrix object
       *
       * Alias for Matrix::operator()
       * @see Matrix::operator()
       *
       * @param rowStart The starting row index for the slice
       * @param rowEnd The ending row index for the slice
       * @param colStart The starting column index for the slice
       * @param colEnd The ending column index for the slice
       *
       * @return A dense Matrix object
       */
      const Matrix get(int rowStart, int rowEnd, int colStart, int colEnd) const {
        return this->operator() (rowStart, rowEnd, colStart, colEnd);
      }
      

      /**
       * @brief SIMD accessing method
       *
       * Method that returns the the SIMD vector at specified index in the data array
       * 
       * NOTE! This function returns a pointer to the first element in the SIMD vector
       * that needs to be cast into proper type. 
       * 
       * As this function gives access to pointer it is somewhat scary and should not be
       * used unless absolutely necessary.
       *
       * @param num The index of the SIMD vector
       *
       * @return Pointer that can be cast as the SIMD vector
       */
      type* getSIMD(int num) const {

        if (!vectorize) {
          _errorMsg("To access SIMD vectors implementation must be vectorized", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if (num >= _total_vects) {
          _errorMsg("Index out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        return (type*)&data.data()[num];
      }
      
      
      // TODO: const Vector getCol(int col) const;

      // TODO: const Vector getRow(int row) const;

      
      // -------- Other overloaded operators ----------

      
      /**
       * @brief Default assignment operator
       * 
       * Method that assigns the values in a given a dense Matrix object into
       * this DenseMatrix object
       *
       * @param that A reference to the dense Matrix object of which values are assigned
       *
       * @return A reference to (this) dense Matrix object
       */
      Matrix& operator= (const Matrix& that) {

        // Check for self-assignment ie. case where a = a is called by comparing the pointers of the objects
        if (this == &that) return *this; 

        data = that.data;
        _total_vects = that._total_vects;
        _vects_per_row = that._vects_per_row;
        _ncols = that._ncols;
        _nrows = that._nrows;

        return *this;
      }
      

      /**
       * @brief Default (equality) comparison operator
       *
       * Method that compares the elements of two dense Matrix objects element-wise
       *
       * NOTE! As the elements are stored as double precision floating pointsthere 
       * might be some floating point errors. Thus in some cases it might be better 
       * to use Matrix::isclose() method.
       * @see Matrix::isclose()
       *
       * @param that A reference to the dense Matrix object of comparison
       *
       * @return A boolean signifying true if equal and false if unequal
       */
      bool operator== (const Matrix& that) {

        if (_nrows != that._nrows || _ncols != that._ncols) {
          return false;
        }

        for (int row = 0; row < _nrows; row++) {
          for (int col = 0; col < _ncols; col++) {
            if (this->operator() (row, col) != that(row, col)) {
              return false;
            }
          }
        }

        return true;
      }
      

      /**
       * @brief Default (inequality) comparison operator
       *
       * Method that compares the elements of two dense Matrix objects element-wise
       *
       * @param that A reference to the dense Matrix object of comparison
       *
       * @return A boolean signifying false if equal and true if unequal
       */
      bool operator!= (const Matrix& that) {
        return !(*this == that);
      }

    
      // -------- Other methods ---------

      /**
       * @brief Number of columns
       *
       * Method that returns the number of columns in the dense Matrix object
       *
       * @return The number of columns as const
       */
      const int ncols() const { return _ncols; }


      /**
       * @brief Number of rows
       *
       * Method that returns the number of rows in the dense Matrix object
       *
       * @return The number of rows as const
       */
      const int nrows() const { return _nrows; }


      /**
       * @brief The shape of the matrix
       *
       * Method that returns a tuple containing the number of rows and columns
       * in the dense Matrix object
       *
       * @return Tuple of form < nrows, ncols > as const
       */
      const std::tuple<int, int> shape() const { return std::make_tuple(_nrows, _ncols); }


      /**
       * @brief Approximative equality comparison
       * 
       * Method that compares two dense Matrix object element-wise up to a tolerance
       *
       * @param that A reference to the dense Matrix object of comparison
       * @param tol OPTIONAL, DEFAULTS TO < type tol = 1e-7 > The tolerance
       * used in the comparison
       * @param abs_func OPTIONAL, DEFAULTS TO < std::abs > Function returning the absolute value
       *
       * @return A boolean signifying false if equal and true if unequal 
       */
      bool isclose(const Matrix& that, type tol = (type)1e-7, type (*abs_func)(type) = &std::abs) {
        
        if (_nrows != that._nrows || _ncols != that._ncols) {
          return false;
        }

        for (int row = 0; row < _nrows; row++) {
          for (int col = 0; col < _ncols; col++) {
            if (abs_func(this->operator() (row, col) - that(row, col)) > tol) {
              return false;
            }
          }
        }

        return true;
      }
      

      /**
       * @brief Standard transpose
       *
       * Method that finds the transpose of the dense Matrix object
       *
       * @return The transpose as a dense Matrix object
       */
      const Matrix transpose() const {

        if (_ncols <= 0 || _nrows <= 0) {
          return *this;  // Maybe change to fatal error
        }

        // Allocate memory for needed sized matrix
        Matrix ret = Matrix(_ncols, _nrows);

        #pragma omp parallel for schedule(dynamic, 1)
        for (int row = 0; row < _nrows; row++) {
          for (int col = 0; col < _ncols; col++) {
            ret.place(col, row, this->operator() (row, col));
          }
        }

        return ret;
      }
      

      /**
       * @brief Standard transpose
       *
       * Method that finds the transpose of the dense Matrix object
       *
       * Alias for Matrix::transpose()
       * @see Matrix::transpose()
       *
       * @return The transpose as a dense Matrix object
       */
      const Matrix T() const {
        return this->transpose();
      }
      

      // TODO: Matrix inv();

      // TODO: const Matrix matmulStrassen(const Matrix& that) const;


      /**
       * @brief Matrix-matrix multiplication in the (naive) textbook way
       *
       * Method that computes the matrix-matrix multiplication in the most naive
       * way. That is as a dot product between the rows of left matrix and columns
       * of right matrix.
       *
       * @param that A reference to the dense Matrix object used in multiplication
       *
       * @return The resulting dense Matrix object
       */
      const Matrix matmulNaive(const Matrix& that) const {
        
        if (_ncols != that._nrows) {
          _errorMsg("Improper dimensions given!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        Matrix ret = Matrix(_nrows, that._ncols);

        // Transpose that for linear memory reads
        Matrix that_T = that.T();

        if constexpr (vectorize) {
          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {
            for (int col = 0; col < that._ncols; col++) {    

              var_t sum = v_zero;

              for (int vect = 0; vect < _vects_per_row - 1; vect++) {
                sum += data[_vects_per_row * row + vect] * that_T.data[_vects_per_row * col + vect];
              }

              type val = t_zero;
              for (int elem = 0; elem < (that._ncols % var_size); elem++) {
                val += data[_vects_per_row * (row + 1) - 1][elem] * that_T.data[_vects_per_row * (col + 1) - 1][elem];
              }

              val += _reduce(sum);
              
              ret.place(row, col, val);
            }
          }
        }
        else {
          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {
            for (int col = 0; col < that._ncols; col++) {    

              type val = t_zero;

              for (int i = 0; i < _nrows; i++) {
                val += this->operator() (row, i) * that(i, col);
              }
              
              ret.place(row, col, val);
            }
          }
        }

        return ret;
      }
      

      /**
       * @brief Wrapper for matrix-matrix multiplication
       *
       * Method that efficiently computes the matrix-matrix product between
       * two dense Matrix objects by calling appropriate methods based on matrix
       * properties.
       *
       * NOTE! Currently only calls the Matrix::matmulNaive method as others
       * are not implemented.
       * @see Matrix::matmulNaive
       *
       * @param that A reference to the dense Matrix object used in multiplication
       *
       * @return The resulting dense Matrix object
       */
      const Matrix matmul(const Matrix& that) const {

        if (_ncols != that._nrows) {
          _errorMsg("Improper dimensions given!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        // 100 chosen as arbitrary threshold
        if (_ncols > STRASSEN_THRESHOLD && _nrows > STRASSEN_THRESHOLD && that._ncols > STRASSEN_THRESHOLD) {
          return this->matmulNaive(that);  // Should call Strassen algorithm, but that is not implemented yet
        }

        return this->matmulNaive(that);
      }


      /**
       * @brief Matrix-vector multiplication
       *
       * Method that computes the matrix-vector multiplication in an efficient way.
       * This method is crucial in most iterative methods.
       *
       * @param that A reference to the Vector object used in multiplication
       *
       * @return The resulting Vector object
       */
      const Vector<type, vectorize> matmul(const Vector<type, vectorize>& that) const {

        if (_ncols != that.len()) {
          _errorMsg("Improper dimensions given!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        Vector<type, vectorize> ret = Vector<type, vectorize>(_nrows);

        if constexpr (vectorize) {
          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {

            var_t sum = v_zero;

            for (int vect = 0; vect < _vects_per_row - 1; vect++) {
              sum += data[_vects_per_row * row + vect] * *(var_t*)that.getSIMD(vect);
            }

            type val = t_zero;
            for (int elem = 0; elem < (that.len() % var_size); elem++) {
              val += data[_vects_per_row * (row + 1) - 1][elem] * (*(var_t*)that.getSIMD(_vects_per_row - 1))[elem];
            }

            val += _reduce(sum);
            
            ret.place(row, val);
          }
        }
        else {
          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {

            type val = t_zero;

            for (int col = 0; col < _ncols; col++) {
              val += data[_ncols * row + col] * that(col);
            }

            ret.place(row, val);
          }
        }

        return ret;
      }
      

      /**
       * @brief Convert a dense Matrix into std::vector
       *
       * Method that returns the matrix elements in a std::vector.
       *
       * @return The matrix elements in a std::vector
       */
      std::vector<type> tovector() const {

        if (_ncols <= 0 || _nrows <= 0) {
          _errorMsg("Matrix must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        std::vector<type> ret;
        ret.reserve(_ncols * _nrows);

        for (int row = 0; row < _nrows; row++) {
          for (int col = 0; col < _ncols; col++) {
            ret.push_back(this->operator() (row, col));
          }
        }

        return ret;
      }
      

      /**
       * @brief Convert a dense Matrix into a scalar
       *
       * Method that returns the matrix element as a scalar.
       *
       * NOTE! The matrix should be a 1 x 1 matrix.
       *
       * @return The matrix element as a scalar
       */
      type asScalar() const {

        if (_ncols != 1 || _nrows != 1) {
          _errorMsg("Matrix must be a 1 x 1 matrix!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        return this->operator() (0, 0);
      }
      

      /**
       * @brief Convert a dense Matrix into a Vector
       *
       * Method that returns the matrix elements in a Vector object
       *
       * NOTE! The matrix should be a 1 x n or a n x 1 matrix.
       *
       * @return The matrix elements in a Vector
       */
      const Vector<type, vectorize> asVector() const {

        if (_ncols != 1 && _nrows != 1) {
          _errorMsg("Matrix must be a 1 x n or n x 1 matrix!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
          
        Vector ret = Vector(_nrows * _ncols);

        for (int row = 0; row < _nrows; row++) {
          for (int col = 0; col < _ncols; col++) {
            ret.place(row * _ncols + col, this->operator() (row, col));
          }
        }

        return ret;
      }

      
      /**
       * @brief The Frobenius norm
       *
       * Method that computes the Frobenius norm of the dense Matrix object
       * 
       * @param pow_func OPTIONAL, DEFAULTS TO < std::pow > Pointer to the function used for calculating the powers
       * should be passed if the data type is not some standard library type
       *
       * @return The computed norm
       */
      type norm(type (*pow_func)(type, type) = &std::pow) const {

        if (_ncols <= 0 || _nrows <= 0) {
          _errorMsg("Matrix must be initialized!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        type ret = t_zero;

        for (int row = 0; row < _nrows; row++) {
          for (int col = 0; col < _ncols; col++) {
            ret += pow_func(this->operator() (row, col), (type)2.0);
          }
        }

        return pow_func(ret, (type)(1.0 / 2.0));
      }
      

      // --------- Statistics ----------

      // TODO: const Vector mean(int dim = 0);

      // TODO: const Vector sd(int dim = 0);

  };

}


#endif