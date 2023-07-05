#include "../lalib/src/denseMatrix.hpp"
#include "../lalib/src/denseVector.hpp"

/*
  Compile int the root (mathlib) directory with: g++ -mavx -fopenmp -Wall lalib/src/denseMatrix.cpp lalib/src/denseVector.cpp lalib/src/denseMatmul.cpp examples/denseMatrix_basics.cpp -lm -o denseMatrix_basics.o
  Run with: ./denseMatrix_basics.o
 */


using namespace lalib;

int main() {
  
  // Initially generate some basic matrices

  // To generate a matrix filled with zeros call
  
  DenseMatrix A = DenseMatrix(3, 3);
  std::cout << "Matrix filled with zeros: A = \n\n" << A << "\n";

  std::cout << "Should be: \n\n" << "[[0 0 0]\n [0 0 0]\n [0 0 0]]" << "\n\n";

  
  // To generate a matrix filled with some other double call
  
  DenseMatrix B = DenseMatrix(3, 3, 2.0);
  std::cout << "Matrix filled with some constant (2): B = \n\n" << B << "\n";

  std::cout << "Should be: \n\n" << "[[2 2 2]\n [2 2 2]\n [2 2 2]]" << "\n\n";

  
  // A double array can also be passed as an argument to initialize with it's values
  // Note that this is not recommended as the size of the double array cannot be verified
  // and thus the constructor might end up reading memory it is not supposed to
  
  double tmp[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  DenseMatrix C = DenseMatrix(3, 3, tmp);
  std::cout << "Matrix filled with values from array: C = \n\n" << C << "\n";

  std::cout << "Should be: \n\n" << "[[1 2 3]\n [4 5 6]\n [7 8 9]]" << "\n\n";

  
  // Better option for initializing with existing values is by using std::vector
  
  std::vector<double> tmp_vec(tmp, tmp + sizeof(tmp) / sizeof(tmp[0]));
  DenseMatrix D = DenseMatrix(3, 3, tmp_vec);
  std::cout << "Matrix filled with values from std::vector: D = \n\n" << D << "\n";

  std::cout << "Should be: \n\n" << "[[1 2 3]\n [4 5 6]\n [7 8 9]]" << "\n\n";

  
  // Basic mathematics

  // The basic math operators +, -, * and / are overloaded to work as elementwise operators
  
  DenseMatrix E = C - B * 1.5;
  std::cout << "Matrix formed by math operations: E = C - B * 1.5 = \n\n" << E << "\n";

  std::cout << "Should be \n\n" << "[[-2 -1 0]\n [1 2 3]\n [4 5 6]]" << "\n\n";

  
  // Additionally assignment to existing matrices is allowed
  
  A = C;
  std::cout << "Matrix A after assignment assigning C to it\n\n" << A << "\n";

  
  // And the equality of the matrices can be verified
  
  std::cout << "Check equivalency: A == C: " << (A == C) << "\n\n";

  std::cout << "Should be:" << "1" << "\n\n";


  // Matrix multiplication and DenseVectors

  // In addition to DenseMatrix class a DenseVector class is implemented
  // This allows for efficient matrix vector multiplication and dot products

  // Matrix multiplication itself is accomplished by calling

  DenseMatrix F = E.matmul(A);
  std::cout << "Matrix formed by matrix multiplication: F = E matmul A = \n\n" << F << "\n";

  std::cout << "Should be: \n\n" << "[[-6 -9 -12] [30  36 42]\n [66 81 96]]" << "\n\n";

  
  // DenseVectors can be declared similarly to matrices e.g.
  // Column vector with only constant values can be called as

  DenseVector v = DenseVector(3, 1, 2.0);
  std::cout << "Vector filled with some constant (2): v = \n\n" << v << "\n";

  std::cout << "Should be: \n\n" << "[2\n 2\n 2]" << "\n\n";

  
  // Alternatively, vectors can be initialized from double arrays or std::vectors

  double tmp2[3] = {1.0, 2.0, 3.0};
  DenseVector w = DenseVector(1, 3, tmp2);
  std::cout << "Vector filled with values from array: w = \n\n" << w << "\n";

  std::cout << "Should be: \n\n" << "[1 2 3]" << "\n\n";
  

  // Dot product is called as

  double dotProd = v.dot(w);
  std::cout << "w dot v = " << dotProd << "\n\n";

  std::cout << "Should be: 12" << "\n\n";

  
  // Matrix vector product is called as

  DenseVector x = A.matmul(v);
  std::cout << "Vector can be formed with matrix-vector product: x = A matmul v = \n\n" << x << "\n";

  std::cout << "Should be \n\n" << "[12\n 30\n 48]" << "\n";

  return 0;
}
