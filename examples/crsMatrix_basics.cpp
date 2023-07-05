#include "../lalib/src/crsMatrix.hpp"

/*
  Compile in root (mathlib) directory with: g++ -mavx -fopenmp -Wall lalib/src/crsMatrix.cpp examples/crsMatrix_basics.cpp -lm -o crsMatrix_basics.o
  Run with: ./crsMatrix_basics.o
*/


int main() {
  
  // Initially generate some basic matrices

  // To generate a matrix filled with zeros call
  
  CRSMatrix A = CRSMatrix(3, 3);
  std::cout << "Matrix filled with zeros: A = \n\n" << A << "\n";

  std::cout << "Should be: \n\n" << "[[0 0 0]\n [0 0 0]\n [0 0 0]]" << "\n\n";
  

  // A double array can also be passed as an argument to initialize with it's values
  // Note that this is not recommended as the size of the double array cannot be verified
  // and thus the constructor might end up reading memory it is not supposed to
  
  double tmp[9] = {0.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 0.0, 0.0};
  CRSMatrix B = CRSMatrix(3, 3, tmp);
  std::cout << "Matrix filled with values from array: B = \n\n" << B << "\n";

  std::cout << "Should be: \n\n" << "[[0 2 0]\n [4 5 0]\n [7 0 0]]" << "\n\n";

  
  // Better option for initializing with existing values is by using std::vector
  
  std::vector<double> tmp_vec(tmp, tmp + sizeof(tmp) / sizeof(tmp[0]));
  CRSMatrix C = CRSMatrix(3, 3, tmp_vec);
  std::cout << "Matrix filled with values from std::vector: C =\n\n" << C << "\n";

  std::cout << "Should be: \n\n" << "[[0 2 0]\n [4 5 0]\n [7 0 0]]" << "\n\n";


  // Indexing

  // Values in the matrix can be accessed with braces

  std::cout << "Value at index (0, 1) in B: " << B(0, 1) << "\n\n";

  std::cout << "Should be: 2" << "\n\n";

  
  // Values can also be placed in the matrices

  B.place(0, 1, 0.5);
  B.place(0, 0, 1.3);
  B.place(1, 0, 0.0);

  std::cout << "Updated B is:\n" << B << "\n";

  std::cout << "Should be: \n\n" << "[[1.3 0.5 0]\n [0 5 0]\n [7 0 0]]" << "\n\n";

  
  // Basic mathematics

  // The basic math operators +, -, * and / are overloaded to work as elementwise operators
  
  CRSMatrix D = C - B;
  std::cout << "Matrix formed with math operations: D = C - B = \n\n" << D << "\n";

  std::cout << "Should be: \n\n" << "[[-1.3 1.5 0]\n [4 0 0]\n [0 0 0]]" << "\n\n";

  return 0;
}