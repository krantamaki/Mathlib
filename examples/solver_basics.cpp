#include "../lalib/src/denseMatrix.hpp"
#include "../lalib/src/denseVector.hpp"
#include "../lalib/src/stationarySolvers.hpp"

/*
  Compile at root (mathlib) directory with: g++ -mavx -fopenmp -Wall lalib/src/denseMatrix.cpp lalib/src/denseVector.cpp lalib/src/denseMatmul.cpp examples/solver_basics.cpp -lm -o solver_basics.o
  Run with: ./solver_basics.o
*/


int main() {
  double mat[9] = {3.0, -2.0, 1.0, 1.0, 3.0, 2.0, 1.0, 2.0, 4.0};
  DenseMatrix A = DenseMatrix(3, 3, mat);

  std::cout << "Diagonally dominant coefficient matrix is: A = \n\n" << A << "\n";

  double vec[3] = {2.0, 3.0, 4.0};
  DenseMatrix b = DenseMatrix(3, 1, vec);

  std::cout << "RHS is: b = \n\n" << b << "\n";

  DenseMatrix x0 = DenseMatrix(3, 1);

  DenseMatrix x_k = jacobiSolve<DenseMatrix>(A, x0, b);

  std::cout << "\nFound solution: x = \n\n" << x_k << "\n";

  DenseMatrix b_sol = A.matmul(x_k);

  std::cout << "Ax = \n\n" << b_sol << "\n";

  return 0;
}
