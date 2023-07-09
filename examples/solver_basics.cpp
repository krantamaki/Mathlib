#include "../lalib/src/dense/denseMatrix.hpp"
#include "../lalib/src/dense/denseVector.hpp"
#include "../lalib/src/crs/crsMatrix.hpp"
#include "../lalib/src/nonstationarySolvers.hpp"
#include "../lalib/src/stationarySolvers.hpp"


using namespace lalib;

/*
  Compile at root (mathlib) directory with: g++ -mavx -fopenmp -Wall lalib/src/crs/crsMatrix.cpp lalib/src/crs/crsMatmul.cpp lalib/src/dense/denseMatrix.cpp lalib/src/dense/denseMatmul.cpp lalib/src/dense/denseVector.cpp examples/solver_basics.cpp -lm -o solver_basics.o
  Run with: ./solver_basics.o
*/


int main() {
  double mat[9] = {4.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0};
  DenseMatrix A = DenseMatrix(3, 3, mat);

  std::cout << "s.p.d. coefficient matrix is: A = \n\n" << A << "\n";

  double vec[3] = {2.0, 3.0, 4.0};
  DenseMatrix b = DenseMatrix(3, 1, vec);

  std::cout << "RHS is: b = \n\n" << b << "\n";

  DenseMatrix x0 = DenseMatrix(3, 1, 1.0);

  DenseMatrix x_k = cgSolve<DenseMatrix>(A, x0, b, 100, 0.0001);

  std::cout << "\nFound solution: x = \n\n" << x_k << "\n";

  DenseMatrix b_sol = A.matmul(x_k);

  std::cout << "Ax = \n\n" << b_sol << "\n";

  return 0;
}
