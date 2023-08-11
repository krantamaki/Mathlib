#include <chrono>
#include "../lalib/src/crs/crsMatrix.hpp"
#include "../lalib/src/crs/crsVector.hpp"
#include "../lalib/src/nonstationarySolvers.hpp"


/**
 * @brief Finite Difference Method solver for a two dimensional Poisson problem
 *
 * Finite Difference Method is a numerical method used for solving partial differential
 * equations. This example solves a two dimensional Poisson problem of form:
 *
 * \f{eqnarray*}{
 *    -\Delta u &=& f \mbox{ in } \Omega \\
 *            u &=& g \mbox{ on } \partial \Omega
 * \f}
 *
 * This solver assumes the domain: \f$\Omega = (0, 1)^2\f$
 *
 * See eg. [1] for more information
 *
 * [1] Hannukainen, Antti. Numerical matrix computations. Course lecture notes, 2022.
 *
 * Compile at root mathlib directory with: 
 * > g++ -mavx -fopenmp -Wall lalib/src/crs/crsMatrix.cpp lalib/src/crs/crsVector.cpp lalib/src/crs/crsMatmul.cpp examples/poisson_fd.cpp -lm -o poisson_fd.o
 *
 * Run with:
 * > ./poisson_fd.o
 */


using namespace lalib;


// Handy index map
inline int ij(int i, int j, int N) {
  return i * N + j;
}


// Function for interior values
double f(double x_1, double x_2) {
  return (10.0 * x_1) * (10.0 * x_2);
}


// Function for boundary values
double g(double x_1, double x_2) {
  return 0.0;
}


int main() {
  
  int N = 300;  // Number of grid points in each dimension. 2D grid is then N x N
  double h = 1.0 / ((double)N - 1.0);  // The step size
  int N_squared = N * N;
  double h_square = h * h;
  double inv_h_square = 1.0 / h_square;

  std::cout << "\n" << "Solving a two dimensional Poisson problem using Finite Difference Method" << "\n";
  std::cout << "Finite difference mesh consists of " << N << " x " << N << " points" << "\n";
  
  std::cout << "\n" << "Initializing the coefficient matrix and RHS vector. This may take a while..." << "\n";

  auto init_start = std::chrono::high_resolution_clock::now();
  
  // Arrays that will define the CRS structure of the coefficient matrix
  std::vector<double> vals;
  std::vector<int> colInds;
  std::vector<int> rowPtrs;
  rowPtrs.push_back(0);

  // The RHS vector
  CRSVector b = CRSVector(N_squared);
  
  // Form the coefficient matrix
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {      
      int row = ij(i, j, N);
      double x_1 = (double)i / (double)N;
      double x_2 = (double)j / (double)N;

      if ((i > 0) && (i < N - 1) && (j > 0) && (j < N - 1)) {  // Point in the interior
        vals.insert(vals.end(), {-inv_h_square, -inv_h_square, 4 * inv_h_square, -inv_h_square, -inv_h_square});
	colInds.insert(colInds.end(), {ij(i - 1, j, N), ij(i, j - 1, N), row, ij(i, j + 1, N), ij(i + 1, j, N)});
	rowPtrs.push_back(rowPtrs.back() + 5);

	b.place(row, f(x_1, x_2));
      }
      else { // Point on the boundary
        vals.push_back(1.0);
	colInds.push_back(row);
	rowPtrs.push_back(rowPtrs.back() + 1);
	
	b.place(row, g(x_1, x_2));
      }
    }
  }

  CRSMatrix A = CRSMatrix(N_squared, N_squared, vals, colInds, rowPtrs);

  auto init_end = std::chrono::high_resolution_clock::now();

  auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start);

  std::cout << "\n" << "Time taken in preprocessing: " << init_duration.count() << " milliseconds" << "\n";
  
  std::cout << "\n" << "Using a zero vector as the initial guess..." << "\n";
  CRSVector x0 = CRSVector(N_squared);

  std::cout << "\n" << "Solving a " << N_squared << " dimensional system with conjugate gradient method. This may take a while..." << "\n";

  CRSVector ret;

  auto solve_start = std::chrono::high_resolution_clock::now();
  ret = cgSolve(A, x0, b, 10 * N);  // 10 * N chosen as an arbitrary maximum number of iterations
  auto solve_end = std::chrono::high_resolution_clock::now();

  auto solve_duration = std::chrono::duration_cast<std::chrono::milliseconds>(solve_end - solve_start);

  std::cout << "\n" << "Time taken by the solver: " << solve_duration.count() << " milliseconds" << "\n";
  
  std::cout << "Residual norm: " << (A.matmul(ret) - b).norm() << "\n\n";

  std::cout << "DONE" << "\n";

  return 0;
}
