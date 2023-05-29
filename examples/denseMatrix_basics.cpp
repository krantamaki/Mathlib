#include "../lalib/src/denseMatrix.hpp"
#include "../lalib/src/denseVector.hpp"

/*


 */


int main() {
  
  // Initially generate some basic matrices

  // To generate a matrix filled with zeros call
  
  DenseMatrix A = DenseMatrix(3, 3);
  std::cout << "Matrix filled with zeros\n\n" << A << "\n";

  // To generate a matrix filled with some other double call
  
  DenseMatrix B = DenseMatrix(3, 3, 2.0);
  std::cout << "Matrix filled with some constant\n\n" << B << "\n";

  // A double array can also be passed as an argument to initialize with it's values
  // Note that this is not recommended as the size of the double array cannot be verified
  // and thus the constructor might end up reading memory it is not supposed to
  
  double tmp[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  DenseMatrix C = DenseMatrix(3, 3, tmp);
  std::cout << "Matrix filled with values from array\n\n" << C << "\n";

  // Better option for initializing with existing values is by using std::vector
  
  std::vector<double> tmp_vec(tmp, tmp + sizeof(tmp) / sizeof(tmp[0]));
  DenseMatrix D = DenseMatrix(3, 3, tmp_vec);
  std::cout << "Matrix filled with values from std::vector\n\n" << D << "\n";

  
  // Basic mathematics

  // The basic math operators +, -, * and / are overloaded to work as elementwise operators
  
  DenseMatrix E = C - B * 1.5;
  std::cout << "C - B * 1.5 = \n\n" << E << "\n";
  
  // Additionally assignment to existing matrices is allowed
  
  A = C;
  std::cout << "Matrix A after assignment\n\n" << A << "\n";

  // And the equality of the matrices can be verified
  
  std::cout << "A == C: " << (A == C) << "\n\n";


  // Matrix multiplication and DenseVectors

  // In addition to DenseMatrix class a DenseVector class is implemented
  // This allows for efficient matrix vector multiplication and dot products

  // Matrix multiplication itself is accomplished by calling

  DenseMatrix F = E.matmul(A);
  std::cout << "E matmul A = \n\n" << F << "\n";

  // DenseVectors can be declared similarly to matrices e.g.
  // Column vector with only constant values can be called as

  DenseVector v = DenseVector(3, 1, 2.0);
  std::cout << "Vector filled with some constant\n\n" << v << "\n";

  // Alternatively, vectors can be initialized from double arrays or std::vectors

  double tmp2[3] = {1.0, 2.0, 3.0};
  DenseVector w = DenseVector(1, 3, tmp2);
  std::cout << "Vector filled with values from array\n\n" << w << "\n";
  

  // Dot product is called as

  std::cout << "w dot v = " << v.dot(w) << "\n\n";

  // Matrix vector product is called as

  DenseVector x = A.matmul(v);
  std::cout << "A matmul v = \n\n" << x << "\n";

  return 0;
}
