#include "../lalib/src/vector/Vector.hpp"
#include "../lalib/src/matrix/Matrix.hpp"
#include "../lalib/src/nonstationarySolvers.hpp"
#include "../lalib/src/stationarySolvers.hpp"
#include "../utils/general.hpp"
#include "../utils/messaging.hpp"


/**
 * @brief Example code utilizing the lalib library
 * 
 * This isn't an exhaustive example, but covers the most basic and used
 * functionality. A lot of more specific methods are also available
 * and can be found from the header files sparsematrix.hpp and 
 * densematrix.hpp.
 * 
 * Compile at root mathlib directory with: 
 * > g++ -fopenmp -mavx -std=c++17 -Wall examples/lalib_basics.cpp -lm -o lalib_basics.o
 *
 * Run with:
 * > ./lalib_basics.o
 */


using namespace lalib;


int main() {

  // Firstly, we will set a global verbosity for communication
  // There are 5 different levels. For more information see utils::messaging.hpp

  utils::verbosity(3);


  // Initially generate some basic matrices
  // In this file we use non-vectorized dense matrices. The form
  // of the matrix can be controlled with the template parameters.
  // The signature is <class type, bool vectorized, bool sparse>
  // where type is the data type of the matrix elements. By setting 
  // vectorized as false we end up with unvectorized implementation 
  // and setting sparse as false we end up with a matrix with dense 
  // storage format.

  // To generate a matrix filled with zeros call
  
  Matrix A = Matrix<double, false, false>(3, 3);
  INFO(utils::_format("Matrix filled with zeros: A = \n\n", A));
  INFO(utils::_format("Should be: \n\n", "[[0 0 0]\n [0 0 0]\n [0 0 0]]", "\n"));
  

  // A double array can also be passed as an argument to initialize with it's values
  // Note that this is not recommended as the size of the double array cannot be verified
  // and thus the constructor might end up reading memory it is not supposed to
  
  double tmp[9] = {0.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 0.0, 0.0};
  Matrix B = Matrix<double, false, false>(3, 3, tmp);
  INFO(utils::_format("Matrix filled with values from array: B = \n\n", B));
  INFO(utils::_format("Should be: \n\n", "[[0 2 0]\n [4 5 0]\n [7 0 0]]", "\n"));

  
  // Better option for initializing with existing values is by using std::vector
  
  std::vector<double> tmp_vec(tmp, tmp + sizeof(tmp) / sizeof(tmp[0]));
  Matrix C = Matrix<double, false, false>(3, 3, tmp_vec);
  INFO(utils::_format("Matrix filled with values from std::vector: C =\n\n", C));
  INFO(utils::_format("Should be: \n\n", "[[0 2 0]\n [4 5 0]\n [7 0 0]]", "\n"));


  // Indexing

  // Values in the matrix can be accessed with braces.
  // Note that indexing starts from 0 rather than 1 like is common in
  // most math software (could be changed in the future)

  INFO(utils::_format("Value at index (0, 1) in B: ", B(0, 1), "\n"));
  INFO(utils::_format("Should be: 2", "\n"));

  
  // Values can also be placed in the matrices

  B.place(0, 1, 0.5);
  B.place(0, 0, 1.3);
  B.place(1, 0, 0.0);

  INFO(utils::_format("Updated B is:\n", B));
  INFO(utils::_format("Should be: \n\n", "[[1.3 0.5 0]\n [0 5 0]\n [7 0 0]]", "\n"));

  
  // Basic mathematics

  // The basic math operators +, -, * and / are overloaded to work as elementwise operators
  
  Matrix D = C - B;
  INFO(utils::_format("Matrix formed with math operations: D = C - B = \n\n", D));
  INFO(utils::_format("Should be: \n\n", "[[-1.3 1.5 0]\n [4 0 0]\n [0 0 0]]", "\n"));


  // Matrix multiplication
  Matrix E = D.matmul(B);
  INFO(utils::_format("Matrix formed by matrix multiplication E = DB = \n\n", E));
  INFO(utils::_format("Should be: \n\n", "[[-1.69 6.85 0]\n [5.2 2 0]\n [0 0 0]]", "\n"));


  // Matrix-vector multiplication
  // Note that we don't discern between column and row vectors

  double tmp2[3] = {1.0, 0.0, -0.2};
  Vector v = Vector<double, false>(3, tmp2);
  INFO(utils::_format("Vector initialized from an array: v = \n\n", v));
  INFO(utils::_format("Should be: \n\n", "[1.0\n 0.0\n -0.2]", "\n"));

  Vector w = E.matmul(v);
  INFO(utils::_format("Vector formed by matrix-vector multiplication w = Ev = \n\n", w));
  std::cout << "Should be: \n\n" << "[-1.69\n 5.2\n 0]" << "\n\n";

  // Vector-vector operations can equivalently used

  double a = w.dot(v);
  INFO(utils::_format("Vector dot product a = w^Tv = ", a));
  INFO(utils::_format("Should be: \n\n", "-1.69", "\n"));


  return 0;
}
