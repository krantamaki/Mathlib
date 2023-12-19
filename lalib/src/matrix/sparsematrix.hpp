#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP


#include "matrix_decl.hpp"


using namespace utils;


namespace lalib {

  /**
   * @brief Compressed row storage (CRS) matrix class
   *
   * CRS is a sparse matrix storage format, where only the non-zero values 
   * in the matrix are held in memory. To access these values additional
   * arrays for the column indeces and row pointers are used. This structure
   * allows for a constant time access to rows, which is very useful when
   * defining fast matrix-vector multiplication. However, the access to columns
   * is a linear time operation.
   *
   * TODO: Vectorize the implementation to use SIMD commands. This could be
   * accomplished by filling vectors as if the sparse matrix was a dense one.
   * That is if a non-zero element exists on index 9 (indexing starting from 1)
   * that would be the first element of a 4 double wide vector since in a dense
   * representation two filled vectors would exist before it.
   */
  template <class type, bool vectorize>
  class Matrix<type, vectorize, true> {

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
      std::vector<var_t> vals;
      std::vector<int> colInds;
      std::vector<int> rowPtrs;


      // Define the number of SIMD vectors in total in the matrix and per row of the matrix
      int _total_vects = 0;
      int _vects_per_row = 0;


      // Define the size of the matrix
      int _ncols = 0;
      int _nrows = 0;
  
    public:

      // ------------- Constructors  ---------------

      /**
       * @brief Default constructor
       *
       * Constructor that creates an uninitialized sparse Matrix object.
       */
      Matrix(void) { }


      /**
       * @brief Copying constructor
       *
       * Constructor that copies the values from a given sparse Matrix object.
       *
       * @param that The object to be copied
       */
      Matrix(const Matrix<type, vectorize, true>& that) {

        if (that._ncols < 1 || that._nrows < 1) {
          _errorMsg("Matrix dimensions must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        _ncols = that._ncols;
        _nrows = that._nrows;

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {  
          vals = that.vals;
          colInds = that.colInds;
          rowPtrs = that.rowPtrs;
        }
      }

      
      /**
       * @brief Zeros constructor
       *
       * Constructor that initializes a sparse Matrix object of wanted shape
       * and fills it with zeros.
       *
       * @param rows The number of rows in the matrix
       * @param cols The number of columns in the matrix
       */
      Matrix(int rows, int cols) {

        if (cols < 1 || rows < 1) {
          _errorMsg("Matrix dimensions must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        _ncols = cols;
        _nrows = rows;

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          rowPtrs = std::vector<int>(rows + 1, 0);
        }
      }


      /**
       * @brief Default value constructor
       *
       * Constructor that initializes a sparse Matrix object of wanted shape
       * and fills it with the wanted value. 
       *
       * NOTE! As the point of CRS format is to store sparse matrices in 
       * a memory efficient fashion filling the matrix isn't recommended.
       * Instead, user could use the dense version of Matrix class.
       * @see Matrix
       *
       * @param rows The number of rows in the matrix
       * @param cols The number of columns in the matrix
       * @param init_val The value with which the matrix is to be filled
       */
      Matrix(int rows, int cols, type init_val) {

        if (cols < 1 || rows < 1) {
          _errorMsg("Matrix dimensions must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        _ncols = cols;
        _nrows = rows;

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          if (init_val == t_zero) {
            rowPtrs = std::vector<int>(rows + 1, 0);
          }
          else {
            _warningMsg("Full matrix allocation is not memory efficient! Consider using a dense Matrix object instead.", __func__);

            vals.reserve(_ncols * _nrows);
            colInds.reserve(_ncols * _nrows);
            rowPtrs.reserve(_nrows + 1);

            for (int row = 0; row < _nrows; row++) {
              for (int col = 0; col < _ncols; col++) {
                vals.push_back(init_val);
                colInds.push_back(col);
              }
              rowPtrs.push_back(row * _ncols);
            }
            rowPtrs.push_back(_nrows * _ncols);
          }
        }
      }


      /**
       * @brief Array copying constructor
       *
       * Constructor that initializes a sparse Matrix object of wanted shape
       * and copies values from a C style array into it.
       *
       * NOTE! As there is no way to verify the length of the C array
       * this constructor might end up reading unwanted memory.
       *
       * @param rows The number of rows in the matrix
       * @param cols The number of columns in the matrix
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
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          rowPtrs.push_back(0);

          for (int row = 0; row < _nrows; row++) {
            int elems_on_row = 0;
            for (int col = 0; col < _ncols; col++) {
              type elem = elems[row * _ncols + col];
              if (elem != t_zero) {
                vals.push_back(elem);
                colInds.push_back(col);
                elems_on_row++;
              }
            }
            rowPtrs.push_back(rowPtrs.back() + elems_on_row);
          }
        }
      }


      /**
       * @brief Vector copying constructor
       *
       * Constructor that initializes a sparse Matrix object of wanted shape
       * and copies values from a std::vector into it.
       *
       * NOTE! If the length of the passed std::vector doesn't match with
       * the matrix dimensions either the extra elements are ignored or the
       * last rows of the matrix are filled with zeros. In either case a
       * warning is printed.
       *
       * @param rows The number of rows in the matrix
       * @param cols The number of columns in the matrix
       * @param elems A reference to a std::vector
       */
      Matrix(int rows, int cols, const std::vector<type>& elems) {

        if (cols < 1 || rows < 1) {
          _errorMsg("Matrix dimensions must be positive!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if (rows * cols != (int)elems.size()) {
          _warningMsg("Given dimensions don't match with the size of the std::vector!", __func__);
        } 

        _ncols = cols;
        _nrows = rows;

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          rowPtrs.push_back(0);

          for (int row = 0; row < _nrows; row++) {
            int elems_on_row = 0;
            for (int col = 0; col < _ncols; col++) {
              type elem = (int)elems.size() > row * _ncols + col ? elems[row * _ncols + col] : 0.0;
              if (elem != t_zero) {
                vals.push_back(elem);
                colInds.push_back(col);
                elems_on_row++;
              }
            }
            rowPtrs.push_back(rowPtrs.back() + elems_on_row);
          }
        }
      }
      

      /**
       * @brief CRS array constructor
       *
       * Constructor that initializes a CRSMatrix object of wanted shape
       * and copies the contents of CRS compatible arrays into it.
       *
       * @param rows The number of rows in the matrix
       * @param cols The number of columns in the matrix
       * @param new_vals A reference to a std::vector containing the non-zero
       * elements of the matrix
       * @param new_colInds A reference to a std::vector containing the column
       * indices of the non-zero elements of the matrix
       * @param new_rowPtrs A reference to a std::vector containing the row pointers
       */
      Matrix(int rows, int cols, const std::vector<type>& new_vals, const std::vector<int>& new_colInds, const std::vector<int>& new_rowPtrs) {
        
        if (*std::max_element(new_colInds.begin(), new_colInds.end()) >= cols || *std::max_element(new_rowPtrs.begin(), new_rowPtrs.end()) > (int)new_colInds.size() || *std::min_element(new_colInds.begin(), new_colInds.end()) < 0 || *std::min_element(new_rowPtrs.begin(), new_rowPtrs.end()) < 0) {
          _errorMsg("Matrix dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        
        _ncols = cols;
        _nrows = rows;

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          vals = new_vals;
          colInds = new_colInds;
          rowPtrs = new_rowPtrs;
        }
      }


      /**
       * @brief Load from file constructor
       *
       * Constructor that initializes a sparse Matrix with values read from a
       * file. Supported formats for the files are standard whitespace
       * separated <row col val> tuples or the MTX format. In the whitespace
       * separated format the last row is assumed to contain the bottom left
       * element of the matrix even if it were to be 0.
       *
       * NOTE! The support for MTX format is not yet implemented!
       *
       * @param path A reference to a std::string that defines the path to the 
       * file with the matrix values
       * @param offset OPTIONAL, DEFAULTS TO < int offset = 0 >. The offset between the 
       * indexing conventions. That is if the values in the file are indexed 
       * e.g. starting from 1 that should be passed as the offset.
       * @param format OPTIONAL, DEFAULTS TO < std::string format = ".dat" >. the extension
       * of the used format. Choices are ".dat" and ".mtx".
       * @param safe_indexing OPTIONAL, DEFAULTS TO < bool safe_indexing = false >. 
       * Boolean flag telling if the elements are sorted by rows and columns in the file.
       */
      Matrix(const std::string& path, int offset = 0, const std::string format = ".dat", bool safe_indexing = false) {

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          // Variables to read the line contents to
          int row, col;
          type val;

          if (format != ".dat") {
            _errorMsg("Support for other formats than .dat not implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
          }

          // Read the last line of the file to get the dimensions of the matrix
          std::stringstream lastLine = _lastLine(path);

          std::cout << "Last line: " << lastLine.str() << std::endl;

          int nTokens = _numTokens(lastLine.str());
          
          if (nTokens == 3) {
            lastLine >> row >> col >> val;
            
            _nrows = row + 1 - offset;
            _ncols = col + 1 - offset;

            rowPtrs = std::vector<int>(_nrows + 1, 0);

            // Start reading the lines from the beginning of the file
            std::ifstream file(path);

            if (!file) {
              ERROR("Couldn't open the given file!");
            }

            if (safe_indexing) {
              int n_vals = 0;
              int lastRow = 0;
              while (file >> row >> col >> val) {
                row = row - offset;
                if (row != lastRow) {
                  int n_emptyRows = row - lastRow;
                  for (int i = 1; i < n_emptyRows; i++) {
                    rowPtrs[lastRow + i] = rowPtrs[lastRow];
                  }
                  rowPtrs[row] = n_vals;
                  lastRow = row;
                }
                colInds.push_back(col - offset);
                vals.push_back(val);
                n_vals++;
              }
              rowPtrs[_nrows] = n_vals;
            }
            else {
              while (file >> row >> col >> val) {
                this->place(row - offset, col - offset, val);
              }
            }

            file.close();
          }

          else if (nTokens == 2) {
            lastLine >> row  >> val;
            
            _nrows = row + 1;
            _ncols = 1;

            rowPtrs = std::vector<int>(_nrows + 1, 0);

            // Start reading the lines from the beginning of the file
            std::ifstream file(path);

            if (!file) {
              ERROR("Couldn't open the given file!");
            }

            if (safe_indexing) {
              int n_vals = 0;
              int lastRow = 0;
              while (file >> row >> val) {
                row = row - offset;
                int emptyRows = row - lastRow - 1;
                for (int i = 0; i < emptyRows; i++) {
                  rowPtrs[lastRow + i + 1] = n_vals - 1;
                }
                rowPtrs[row] = n_vals;
                vals.push_back(val);
                n_vals++;
                lastRow = row;
              }
              rowPtrs[_nrows] = n_vals;
            }
            else {
              while (file >> row >> val) {
                this->place(row - offset, 0, val);
              }
            }

            file.close();
          }
          else {
            _errorMsg("Improper data file!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
          }
        }
      }

      
      // ---------  Overloaded basic math operators ----------
    
      // NOTE! The operators will function as elementwise operators

      /**
       * @brief Element-wise addition assignment
       *
       * Method that performs an element-wise addition assignment between this sparse Matrix
       * object and the one passed as argument.
       *
       * @param that A reference to a sparse Matrix object used in the sum assignment
       *
       * @return A reference to (this) sparse Matrix object
       */
      Matrix& operator+= (const Matrix& that) {

        if (_ncols != that._ncols || _nrows != that._nrows) {
          _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          for (int row = 0; row < _nrows; row++) {
            int rowPtr = that.rowPtrs[row];
            int nrowElems = that.rowPtrs[row + 1] - rowPtr;
            for (int col_i = 0; col_i < nrowElems; col_i++) {
              int col = that.colInds[rowPtr + col_i];
              type val = this->operator() (row, col) + that(row, col);
              this->place(row, col, val);
            }
          }
        }
        
        return *this;
      }


      /**
       * @brief Element-wise addition
       *
       * Method that performs an element-wise addition between this sparse Matrix
       * object and the one passed as argument.
       *
       * @param that A reference to a sparse Matrix object used in the sum
       *
       * @return A sparse Matrix object
       */
      const Matrix operator+ (const Matrix& that) const {
        return Matrix<type, vectorize, true>(*this) += that;
      }


      /**
       * @brief Element-wise subtraction assignment
       *
       * Method that performs an element-wise subtraction assignment between this sparse Matrix
       * object and the one passed as argument.
       *
       * @param that A reference to a sparse Matrix object used in the difference assignment
       *
       * @return A reference to (this) sparse Matrix object
       */
      Matrix& operator-= (const Matrix& that) {

        if (_ncols != that._ncols || _nrows != that._nrows) {
          _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          for (int row = 0; row < _nrows; row++) {
            int rowPtr = that.rowPtrs[row];
            int nrowElems = that.rowPtrs[row + 1] - rowPtr;
            for (int col_i = 0; col_i < nrowElems; col_i++) {
              int col = that.colInds[rowPtr + col_i];
              type val = this->operator() (row, col) - that(row, col);
              this->place(row, col, val);
            }
          }
        }
        
        return *this;
      }


      /**
       * @brief Element-wise subtraction
       *
       * Method that performs an element-wise subtraction between this sparse Matrix
       * object and the one passed as argument.
       *
       * @param that A reference to a sparse Matrix object used in the difference
       *
       * @return A sparse Matrix object
       */
      const Matrix operator- (const Matrix& that) const {
        return Matrix<type, vectorize, true>(*this) -= that;
      }


      /**
       * @brief Element-wise multiplication assignment
       *
       * Method that performs an element-wise multiplication assignment between this
       * sparse Matrix object and the one passed as argument.
       *
       * @param that A reference to a sparse Matrix object used in the multiplication assignment
       *
       * @return A reference to (this) sparse Matrix object
       */
      Matrix& operator*= (const Matrix& that) {

        if (_ncols != that._ncols || _nrows != that._nrows) {
          _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          for (int row = 0; row < _nrows; row++) {
            for (int col = 0; col < _ncols; col++) {
              type val = this->operator() (row, col) * that(row, col);
              this->place(row, col, val);
            }
          }
        }
        
        return *this;
      }


      /**
       * @brief Element-wise multiplication
       *
       * Method that performs an element-wise multiplication between this sparse Matrix
       * object and the one passed as argument.
       *
       * @param that A reference to a sparse Matrix object used in the multiplication
       *
       * @return A sparse Matrix object
       */
      const Matrix operator* (const Matrix& that) const {
        return Matrix<type, vectorize, true>(*this) *= that;
      }


      /**
       * @brief Scalar (right) multiplication assignment
       *
       * Method that performs the standard matrix-scalar multiplication
       *
       * @param that The scalar used in the multiplication
       *
       * @return A reference to (this) sparse Matrix object
       */
      Matrix& operator*= (type that) {

        if (_ncols < 1 || _nrows < 1) {
          return *this;
        }

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          for (int i = 0; i < (int)vals.size(); i++) {
            vals[i] *= that;
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
       * @return A sparse Matrix object
       */
      const Matrix operator* (const type that) const {
        return Matrix<type, vectorize, true>(*this) *= that;
      }


      /**
       * @brief Element-wise division assignment
       *
       * Method that performs an element-wise division assignment between this
       * sparse Matrix object and the one passed as argument.
       *
       * @param that A reference to a sparse Matrix object used in the division assignment
       *
       * @return A reference to (this) sparse Matrix object
       */
      Matrix& operator/= (const Matrix& that) {

        if (_ncols != that._ncols || _nrows != that._nrows) {
          _errorMsg("Matrix dimensions must match!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          for (int row = 0; row < _nrows; row++) {
            for (int col = 0; col < _ncols; col++) {
              type val = this->operator() (row, col) / that(row, col);
              this->place(row, col, val);
            }
          }
        }
        
        return *this;
      }


      /**
       * @brief Element-wise division
       *
       * Method that performs an element-wise division between this sparse Matrix
       * object and the one passed as argument.
       *
       * @param that A reference to a sparse Matrix object used in the division
       *
       * @return A sparse Matrix object
       */
      const Matrix operator/ (const Matrix& that) const {
        return Matrix<type, vectorize, true>(*this) /= that;
      }


      /**
       * @brief Scalar division assignment
       *
       * Method that performs the standard matrix-scalar division assignment
       *
       * @param that The scalar used in the division
       *
       * @return A reference to (this) sparse Matrix object
       */
      Matrix& operator/= (type that) {

        if (that == t_zero) {
          _errorMsg("Division by zero undefined!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        if (_nrows < 1 || _ncols < 1) {
          return *this;
        }

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          for (int row = 0; row < _nrows; row++) {
            for (int col = 0; col < _ncols; col++) {
              type val = this->operator() (row, col) / that;
              this->place(row, col, val);
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
       * @return A sparse Matrix object
       */
      const Matrix operator/ (const type that) const {
        return Matrix<type, vectorize, true>(*this) /= that;
      }


      // -------- Placement methods ---------

      /**
       * @brief Standard single value placement
       *
       * Method that places a given value at wanted location in a sparse Matrix object
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
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          // If value is not zero it needs to be placed into the matrix
          if (val != t_zero) {

            // If there are no elements yet in the matrix just add the value
            if (vals.empty()) {
              vals.push_back(val);
              colInds.push_back(col);
            }

            // Otherwise, we need to find the correct location for the value
            else {
          
              int rowPtr = rowPtrs[row];
              int nextRow = rowPtrs[row + 1];

              // If the row is empty just add to the row pointers position
              if (rowPtr == nextRow) {
                vals.insert(vals.begin() + rowPtr, val);
                colInds.insert(colInds.begin() + rowPtr, col);
              }

              // Otherwise, iterate over the columns to find the correct gap
              else {
                int col_i = rowPtr;
                for (; col_i < nextRow; col_i++) {
                  int col0 = colInds[col_i];

                  // If there already is a value at given location replace it
                  if (col0 == col) {
                    vals[col_i] = val;
                    return;
                  }

                  // If the found row is larger than given column then insert value before it
                  else if (col0 > col) {
                    vals.insert(vals.begin() + col_i, val);
                    colInds.insert(colInds.begin() + col_i, col);
                    break;
                  }
                }

                // New column is the largest that has a value in the given row
                if (col_i == nextRow) {
                  vals.insert(vals.begin() + nextRow, val);
                  colInds.insert(colInds.begin() + nextRow, col);
                }
              }
            }

            // Increment the row pointers accordingly
            for (int row_i = row + 1; row_i <= _nrows; row_i++) {
              rowPtrs[row_i] += 1;
            }

          }

          // If input is zero we need to check if it replaces some non-zero value
          else {

            int rowPtr = rowPtrs[row];
            int nextRow = rowPtrs[row + 1];

            // If the row is empty zero cannot replace a non-zero value
            if (rowPtr == nextRow) {
              return;
            }

            // Otherwise, iterate over the columns to check if there is a non-zero value
            else {

              int col_i = rowPtr;
              for (; col_i < nextRow; col_i++) {
                int col0 = colInds[col_i];

                // If there already is a value at given location replace it
                if (col0 == col) {
                  vals.erase(vals.begin() + col_i);
                  colInds.erase(colInds.begin() + col_i);
                  break;
                }

                // If the found column is larger than given column then the zero didn't replace a non-zero
                else if (col0 > col) {
                  return;
                }
              }

              // If the new column is the largest then it cannot replace a non-zero element
              if (col_i == nextRow) {
                return;
              }
            }
              
            // Decrement the row pointers accordingly
            for (int row_i = row + 1; row_i <= _nrows; row_i++) {
              rowPtrs[row_i] -= 1;
            }
          }
        }
      }


      /**
       * @brief Standard matrix placement
       * 
       * Method that places the values in a sparse Matrix object into the wanted location
       * in another sparse Matrix object.
       *
       * @param rowStart The starting row index for the placement
       * @param rowEnd The ending row index for the placement
       * @param colStart The starting column index for the placement
       * @param colEnd The ending column index for the placement
       * @param matrix A reference to the sparse Matrix object of which values are to be placed
       */
      void place(int rowStart, int rowEnd, int colStart, int colEnd, Matrix<type, vectorize, true>& matrix) {

        if (_nrows < rowEnd - rowStart || _ncols < colEnd - colStart || matrix._nrows < rowEnd - rowStart || matrix._ncols < colEnd - colStart) {
          _errorMsg("Given dimensions out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          for (int row0 = 0; row0 < rowEnd - rowStart; row0++) {
            int row = row0 + rowStart;
            for (int col0 = 0; col0 < colEnd - colStart; col0++) {
              int col = col0 + colStart;
              type val = matrix(row0, col0);

              if (val != t_zero) {
                this->place(row, col, val);
              }
              else if (this->operator() (row, col) != t_zero) {
                this->place(row, col, val);
              }
              else {
                continue;
              }
            }
          }
        }
      }


      /**
       * @brief Standard column placement
       *
       * Method that places the values in a Vector object into the wanted column of a
       * sparse Matrix object.
       *
       * @param col The column on to which the elements of the Vector are to be placed
       * @param vector A reference to the Vector object with wanted values
       */
      void placeCol(int col, Vector<type, vectorize>& vector) {

        if (col >= _ncols) {
          _errorMsg("Given column out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if (vector.len() > _nrows) {
          _warningMsg("End index out of bounds", __func__);
        }

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          for (int row = 0; row < _nrows; row++) {
            type val = vector(row);

            if (val != 0.0) {
              this->place(row, col, val);
            }
            else if (this->operator() (row, col) != t_zero) {
              this->place(row, col, val);
            }
            else {
              continue;
            }
          }
        }
      }


      /**
       * @brief Standard row placement
       *
       * Method that places the values in a Vector object into the wanted row of a
       * sparse Matrix object.
       *
       * @param row The row on to which the elements of the Vector are to be placed
       * @param vector A reference to the Vector object with wanted values
       */
      void placeRow(int row, Vector<type, vectorize>& vector) {

        if (row >= _nrows) {
          _errorMsg("Given column out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if (vector.len() > _ncols) {
          _warningMsg("End index out of bounds", __func__);
        }

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          int old_n_row_elems = rowPtrs[row + 1] - rowPtrs[row];

          // Find the new non-zero values for the row
          std::vector<int> new_cols;
          std::vector<var_t> new_vals;

          for (int col = 0; col < _ncols; col++) {
            type val = vector(col);
            
            if (val != t_zero) {
              new_cols.push_back(col);
              new_vals.push_back(val);
            }
          }

          int new_n_row_elems = new_vals.size();

          // Remove existing non-zeros
          vals.erase(vals.begin() + rowPtrs[row], vals.begin() + rowPtrs[row + 1]);
          colInds.erase(colInds.begin() + rowPtrs[row], colInds.begin() + rowPtrs[row + 1]);

          // Add the new non-zeros
          vals.insert(vals.begin() + rowPtrs[row], new_vals.begin(), new_vals.end());
          colInds.insert(colInds.begin() + rowPtrs[row], new_cols.begin(), new_cols.end());

          int n_elem_diff = new_n_row_elems - old_n_row_elems;

          for (int i = row + 1; i <= _nrows; i++) {
            rowPtrs[i] += n_elem_diff;
          }
        }
      }


      // ----------- Overloaded indexing operators ------------
      
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
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          int rowPtr = rowPtrs[row];
          int nextRow = rowPtrs[row + 1];

          if (rowPtr == nextRow) {
            return t_zero;
          }

          else {
            int col_i = rowPtr;
            for (; col_i < nextRow; col_i++) {
              int col0 = colInds[col_i];
              if (col0 == col) {
                return vals[col_i];
              }
              else if (col0 > col) {
                return t_zero;
              }
            }
          }
          
          return t_zero;
        }
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

        int row = num / _nrows;
        int col = num % _nrows;

        return this->operator() (row, col);
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
       * @brief Standard slicing method
       *
       * Method that slices a wanted sized matrix from a sparse Matrix object
       *
       * NOTE! If the end indeces are out of bounds only the elements in bounds
       * are returned. In this case a warning is printed.
       *
       * @param rowStart The starting row index for the slice
       * @param rowEnd The ending row index for the slice
       * @param colStart The starting column index for the slice
       * @param colEnd The ending column index for the slice
       *
       * @return A sparse Matrix object
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

        Matrix ret = Matrix<type, vectorize, true>(_rowEnd - rowStart, _colEnd - colStart);

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          for (int row0 = 0; row0 < _rowEnd - rowStart; row0++) {
            int row = row0 + rowStart;
            for (int col0 = 0; col0 < _colEnd - colStart; col0++) {
              int col = col0 + colStart;
              type val = this->operator()(row, col);
              if (val != t_zero) {
                ret.place(row0, col0, this->operator() (row, col));
              }
            }
          }
        }

        return ret;
      }


      /**
       * @brief Named slicing method
       *
       * Method that slices a wanted sized matrix from a Matrix object
       *
       * Alias for Matrix::operator()
       * @see Matrix::operator()
       *
       * @param rowStart The starting row index for the slice
       * @param rowEnd The ending row index for the slice
       * @param colStart The starting column index for the slice
       * @param colEnd The ending column index for the slice
       *
       * @return A sparse Matrix object
       */
      const Matrix get(int rowStart, int rowEnd, int colStart, int colEnd) const {
        return this->operator() (rowStart, rowEnd, colStart, colEnd);
      }
        

      /**
       * @brief Access a column
       *
       * Method that retrieves a wanted column of a sparse Matrix object
       *
       * @param col The index of the wanted column
       *
       * @return The wanted column as a sparse Vector object
       */
      const Vector<type, vectorize> getCol(int col) const {

        if (col >= _ncols) {
          _errorMsg("Given column out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        Vector ret = Vector<type, vectorize>(_nrows);

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          // As there is no efficient way to access the column elements of a CRSMatrix the implementation is naive
          for (int row = 0; row < _nrows; row++) {
            ret.place(row, this->operator() (row, col));
          }
        }

        return ret;
      }


      /**
       * @brief Access a row
       *
       * Method that retrieves a wanted row of a sparse Matrix object
       *
       * @param row The index of the wanted row
       *
       * @return The wanted row as a Vector object
       */
      const Vector<type, vectorize> getRow(int row) const {

        if (row >= _nrows) {
          _errorMsg("Given row out of bounds!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        Vector ret = Vector<type, vectorize>(_ncols);
        
        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          for (int i = rowPtrs[row]; i < rowPtrs[row + 1]; i++) {
            int col = colInds[i];
            type val = vals[i];
            
            ret.place(col, val);
          }
        }

        return ret;
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

        return (type*)&vals.data()[num];
      }


      // -------- Other overloaded operators ----------

      /**
       * @brief Default assignment operator
       * 
       * Method that assigns the values in a given sparse Matrix object into
       * this sparse Matrix object
       *
       * @param that A reference to the sparse Matrix object of which values are assigned
       *
       * @return A reference to (this) sparse Matrix object
       */
      Matrix& operator= (const Matrix& that) {

        // Check for self-assignment ie. case where a = a is called by comparing the pointers of the objects
        if (this == &that) return *this; 

        _ncols = that._ncols;
        _nrows = that._nrows;

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          vals = that.vals;
          colInds = that.colInds;
          rowPtrs = that.rowPtrs;
        }

        return *this;
      }


      /**
       * @brief Default (equality) comparison operator
       *
       * Method that compares the elements of two sparse Matrix objects element-wise
       *
       * NOTE! As the elements are stored as double precision floating pointsthere 
       * might be some floating point errors. Thus in some cases it might be better 
       * to use Matrix::isclose() method.
       * @see Matrix::isclose()
       *
       * @param that A reference to the sparse Matrix object of comparison
       *
       * @return A boolean signifying true if equal and false if unequal
       */
      bool operator== (const Matrix& that) {

        if (_nrows != that._nrows || _ncols != that._ncols) {
          return false;
        }

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          int this_num_non_zeros = vals.size();
          int that_num_non_zeros = that.vals.size();

          if (this_num_non_zeros != that_num_non_zeros) {
            return false;
          }

          for (int i = 0; i < this_num_non_zeros; i++) {
            if (vals[i] != that.vals[i] || colInds[i] != that.colInds[i]) {
              return false;
            }
          }
        }

        return true;
      }


      /**
       * @brief Default (inequality) comparison operator
       *
       * Method that compares the elements of two sparse Matrix objects element-wise
       *
       * @param that A reference to the sparse Matrix object of comparison
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
       * Method that returns the number of columns in the CRSMatrix object
       *
       * @return The number of columns as const
       */
      const int ncols() const { return _ncols; }


      /**
       * @brief Number of rows
       *
       * Method that returns the number of rows in the CRSMatrix object
       *
       * @return The number of rows as const
       */
      const int nrows() const { return _nrows; }


      /**
       * @brief The shape of the matrix
       *
       * Method that returns a tuple containing the number of rows and columns
       * in the CRSMatrix object
       *
       * @return Tuple of form < nrows, ncols > as const
       */
      const std::tuple<int, int> shape() const { return std::make_tuple(_nrows, _ncols); }



      /**
       * @brief Approximative equality comparison
       * 
       * Method that compares two sparse Matrix object element-wise up to a tolerance
       *
       * @param that A reference to the sparse Matrix object of comparison
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

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          int this_num_non_zeros = vals.size();
          int that_num_non_zeros = that.vals.size();

          if (this_num_non_zeros != that_num_non_zeros) {
            return false;
          }

          for (int i = 0; i < this_num_non_zeros; i++) {
            if (abs_func(vals[i] - that.vals[i]) > tol || colInds[i] != that.colInds[i]) {
              return false;
            }
          }
        }

        return true;
      }


      /**
       * @brief Matrix saving
       *
       * Method that saves the CRSMatrix object in a wanted format
       *
       * NOTE! The support for MTX format is not yet implemented!
       *
       * @param path A reference to a std::string that defines the path to the 
       * file where the matrix is to be stored
       * @param offset OPTIONAL, DEFAULTS TO < int offset = 0 >. The offset between the 
       * indexing conventions. That is if the values should be indexed 
       * e.g. starting from 1 that should be passed as the offset.
       * @param format OPTIONAL, DEFAULTS TO < std::string format ".dat" >. the extension
       * of the used format. Choices are ".dat" and ".mtx".
       */
      bool save(const std::string& path, int offset = 0, std::string format = ".dat") {

        if (_ncols <= 0 || _nrows <= 0) {
          _errorMsg("Cannot save an unitialized matrix!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if (format != ".dat") {
          _errorMsg("Support for other formats than .dat not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        bool success = true;

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          std::ofstream file(path);

          for (int row = 0; row < _nrows; row++) {
            for (int row_p = rowPtrs[row]; row_p < rowPtrs[row + 1]; row_p++) {
              int col = colInds[row_p];
              double val = vals[row_p];

              if (!(file << row + offset << " " << col + offset << " " << val << "\n" << std::flush)) {
                success = false;
              }
            }
          }
          
          file.close();
        }
        
        return success;
      }


      /**
       * @brief CRS format array printing
       *
       * Method that prints the arrays defining the sparse Matrix object into ostream
       *
       * FOR DEBUGGING PURPOSES ONLY!
       */
      void _printArrays() const {
        std::cout << "vals: [";
        for (type val: vals) std::cout << val << " ";
        std::cout << "]\n";

        std::cout << "col_i: [";
        for (int col: colInds) std::cout << col << " ";
        std::cout << "]\n";
        
        std::cout << "row_p: [";
        for (int row_p: rowPtrs) std::cout << row_p << " ";
        std::cout << "]\n";
      }
  

      /**
       * @brief Naive transpose
       *
       * Very inefficient, but certainly functional method for finding
       * the transpose of a sparse Matrix object
       *
       * @return The transpose as a sparse Matrix object
       */
      const Matrix naiveTranspose() const {

        if (_ncols <= 0 || _nrows <= 0) {
          return *this;
        }

        // Initialize the transpose matrix
        Matrix ret = Matrix<type, vectorize, true>(_ncols, _nrows);

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          for (int row = 0; row < _nrows; row++) {
            for (int col = 0; col < _ncols; col++) {
              double val = this->operator() (row, col);

              if (val != t_zero) {
                ret.place(col, row, val);
              }
            }
          }
        }
        
        return ret;
      }


      /**
       * @brief Standard transpose
       *
       * Method that finds the transpose of the sparse Matrix object
       *
       * @return The transpose as a sparse Matrix object
       */
      const Matrix transpose() const {

        if (_ncols <= 0 || _nrows <= 0) {
          return *this;
        }

        // Initialize the transpose matrix
        Matrix ret = Matrix<type, vectorize, true>(_ncols, _nrows);

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          for (int col = 0; col < _ncols; col++) {
            for (int row = 0; row < _nrows; row++) {
              type val = this->operator() (row, col);

              if (val != t_zero) {
                ret.vals.push_back(val);
                ret.colInds.push_back(row);

                for (int col_i = col + 1; col_i <= _ncols; col_i++) {
                  ret.rowPtrs[col_i] += 1;
                }
              }
            }
          }
        }
        
        return ret;
      }


      /**
       * @brief Standard transpose
       *
       * Method that finds the transpose of a sparse Matrix object
       *
       * Alias for Matrix::transpose()
       * @see Matrix::transpose()
       *
       * @return The transpose as a CRSMatrix object
       */
      const Matrix T() const {
        return this->transpose();
      }
      

      // TODO: CRSMatrix inv();

      // TODO: const CRSMatrix matmulStrassen(const CRSMatrix& that) const;


      /**
       * @brief Matrix-matrix multiplication in the (naive) textbook way
       *
       * Method that computes the matrix-matrix multiplication in the most naive
       * way. That is as a dot product between the rows of left matrix and columns
       * of right matrix.
       *
       * @param that A reference to the sparse Matrix object used in multiplication
       *
       * @return The resulting sparse Matrix object
       */
      const Matrix matmulNaive(const Matrix& that) const {

        if (_ncols != that._nrows) {
          _errorMsg("Improper dimensions!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        // Create the matrix that will be filled
        Matrix ret = Matrix<type, vectorize, true>(_nrows, that._ncols);

        // Transpose that to have constant time access to columns
        Matrix that_T = that.T();

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          for (int row = 0; row < _nrows; row++) {
            
            int n_this_row_elems = rowPtrs[row + 1] - rowPtrs[row];
            if (n_this_row_elems == 0) continue;
            
            else {
              
              for (int col = 0; col < that._ncols; col++) {
          
                int n_that_col_elems = that_T.rowPtrs[col + 1] - that_T.rowPtrs[col];
                if (n_that_col_elems == 0) continue;

                else {

                  type sum = t_zero;
                  for (int col_i = rowPtrs[row]; col_i < rowPtrs[row + 1]; col_i++) {
                    for (int row_i = that_T.rowPtrs[col]; row_i < that_T.rowPtrs[col + 1]; row_i++) {
                      if (colInds[col_i] == that_T.colInds[row_i]) {
                        sum += vals[col_i] * that_T.vals[row_i];
                      }
                    }
                  }

                  if (sum != t_zero) {
                    ret.vals.push_back(sum);
                    ret.colInds.push_back(col);

                    for (int row_i = row + 1; row_i <= _nrows; row_i++) {
                      ret.rowPtrs[row_i] += 1;
                    }
                  }
                }
              }
            }
          }
        }
        
        return ret;
      }

      /**
       * @brief Wrapper for matrix-matrix multiplication
       *
       * Method that efficiently computes the matrix-matrix product between
       * two sparse Matrix objects by calling appropriate methods based on matrix
       * properties.
       *
       * NOTE! Currently only calls the Matrix::matmulNaive method as others
       * are not implemented.
       * @see Matrix::matmulNaive
       *
       * @param that A reference to the sparse Matrix object used in multiplication
       *
       * @return The resulting sparse Matrix object
       */
      const Matrix matmul(const Matrix& that) const {

        if (_ncols != that._nrows) {
          _errorMsg("Improper dimensions!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if (_ncols > STRASSEN_THRESHOLD && _nrows > STRASSEN_THRESHOLD && that._ncols > STRASSEN_THRESHOLD) {
          return this->matmulNaive(that);  // Should call Strassen once implemented
        }

        return this->matmulNaive(that);
      }
      

      /**
       * @brief Efficient matrix-vector multiplication
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
          _errorMsg("Improper dimensions!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        Vector ret = Vector<type, vectorize>(_nrows);

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          #pragma omp parallel for schedule(dynamic, 1)
          for (int row = 0; row < _nrows; row++) {
            int n_row_elems = rowPtrs[row + 1] - rowPtrs[row];
            if (n_row_elems == 0) continue;
            else {
              type sum = t_zero;
              for (int col_i = rowPtrs[row]; col_i < rowPtrs[row + 1]; col_i++) {
                int col = colInds[col_i];
                type val = vals[col_i];

                sum += val * that(col);
              }
              ret.place(row, sum);
            }
          }
        }
        
        return ret;
      }


      /**
       * @brief Dot product between a row and given vector
       * 
       * Method that computes the dot product between a given row of a sparse Matrix object
       * and passed Vector object.
       * 
       * @param row Index of the row used for the calculation
       * @param that Vector with which the computation is carried
       * @return The dot product
       */
      const type rowDot(int row, const Vector<type, vectorize>& that) const {

        if (_ncols != that.len()) {
          _errorMsg("Improper dimensions!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if (row < 0 || row >= _nrows) {
          _errorMsg("Improper row index!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        type ret = t_zero;

        int n_row_elems = rowPtrs[row + 1] - rowPtrs[row];
        if (n_row_elems != 0) {
          for (int col_i = rowPtrs[row]; col_i < rowPtrs[row + 1]; col_i++) {
            int col = colInds[col_i];
            type val = vals[col_i];

            ret += val * that(col);
          }
        }

        return ret;
      }


      /**
       * @brief Convert a sparse Matrix object into std::vector
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

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          for (int row = 0; row < _nrows; row++) {
            for (int col = 0; col < _ncols; col++) {
              ret.push_back(this->operator() (row, col));
            }
          }
        }

        return ret;
      }


      /**
       * @brief Convert a sparse Matrix into a double
       *
       * Method that returns the matrix element as a double.
       *
       * NOTE! The matrix should be a 1 x 1 matrix.
       *
       * @return The matrix element as a scalar
       */
      double asDouble() const {

        if (_ncols != 1 || _nrows != 1) {
          _errorMsg("Matrix must be a 1 x 1 matrix!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }

        if constexpr (vectorize) {
          _errorMsg("Vectorized version of sparse matrix not yet implemented!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
        }
        else {
          if (vals.size() > 0) return vals[0];
        }

        return t_zero;
      }
      
      
      /**
       * @brief Convert a sparse Matrix into a Vector
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
          
        Vector ret = Vector<type, vectorize>(_nrows * _ncols);

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
       * Method that computes the Frobenius norm of the sparse Matrix object
       * 
       * @param pow_func OPTIONAL, DEFAULTS TO < std::pow > Pointer to the function used for calculating the powers
       * should be passed if the data type is not some standard library type
       *
       * @return The computed norm
       */
      double norm(type (*pow_func)(type, type) = &std::pow) const {

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
        

      // Statistics

      // TODO: const CRSVector mean(int dim = 0);
      // TODO: const CRSVector sd(int dim = 0);

  };

}


#endif