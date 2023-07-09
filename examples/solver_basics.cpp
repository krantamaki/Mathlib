#include "../lalib/src/crsMatrix.hpp"
#include "../lalib/src/nonstationarySolvers.hpp"


using namespace lalib;

/*
  Compile at root (mathlib) directory with: g++ -mavx -fopenmp -Wall lalib/src/crsMatrix.cpp lalib/src/crsMatmul.cpp examples/solver_basics.cpp -lm -o solver_basics.o
  Run with: ./solver_basics.o
*/


int main() {
  double mat[9] = {4.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 4.0};
  CRSMatrix A = CRSMatrix(3, 3, mat);

  std::cout << "s.p.d. coefficient matrix is: A = \n\n" << A << "\n";

  double vec[3] = {2.0, 3.0, 4.0};
  CRSMatrix b = CRSMatrix(3, 1, vec);

  std::cout << "RHS is: b = \n\n" << b << "\n";

  CRSMatrix x0 = CRSMatrix(3, 1, 1.0);

  CRSMatrix x_k = cgSolve<CRSMatrix>(A, x0, b, 100, 0.00001, false);

  std::cout << "\nFound solution: x = \n\n" << x_k << "\n";

  CRSMatrix b_sol = A.matmul(x_k);

  std::cout << "Ax = \n\n" << b_sol << "\n";

  return 0;
}
