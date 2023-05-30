#include "../lalib/src/crsMatrix.hpp"

/*


 */


int main() {
  
  // Initially generate some basic matrices

  // To generate a matrix filled with zeros call
  
  CRSMatrix A = CRSMatrix(3, 3);
  std::cout << "Matrix filled with zeros\n\n" << A << "\n";

  // A double array can also be passed as an argument to initialize with it's values
  // Note that this is not recommended as the size of the double array cannot be verified
  // and thus the constructor might end up reading memory it is not supposed to
  
  double tmp[9] = {0.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 0.0, 0.0};
  CRSMatrix B = CRSMatrix(3, 3, tmp);
  std::cout << "Matrix filled with values from array\n\n" << B << "\n";

  // Better option for initializing with existing values is by using std::vector
  
  std::vector<double> tmp_vec(tmp, tmp + sizeof(tmp) / sizeof(tmp[0]));
  CRSMatrix C = CRSMatrix(3, 3, tmp_vec);
  std::cout << "Matrix filled with values from std::vector\n\n" << C << "\n";


  // Indexing

  // Values in the matrix can be accessed with braces

  std::cout << "At index (0, 1) in B: " << B(0, 1) << "\n\n";

  // Values can also be placed in the matrices

  B.place(0, 1, 0.5);
  B.place(0, 0, 1.3);
  B.place(1, 0, 0.0);

  std::cout << "Updated B is:\n" << B << "\n";

  
  // Basic mathematics

  // The basic math operators +, -, * and / are overloaded to work as elementwise operators
  
  CRSMatrix D = C - B;
  std::cout << "C - B = \n\n" << D << "\n";

  return 0;
}
