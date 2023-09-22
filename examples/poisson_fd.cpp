#include <chrono>
#include "../lalib/src/matrix/Matrix.hpp"
#include "../lalib/src/vector/Vector.hpp"
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
 * > g++ -mavx -std=c++17 -fopenmp -Wall examples/poisson_fd.cpp -lm -o poisson_fd.o
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
template<class type>
type f(type x_1, type x_2) {
  return x_1 + x_2;
}


// Function for boundary values
template<class type>
type g(type x_1, type x_2) {
  return 0.0;
}


template<class type, bool vectorize, bool sparse>
void poissonFD(int N) {
  type h = 1.0 / ((type)N - 1.0);  // The step size
  int N_squared = N * N;
  type h_square = h * h;
  type inv_h_square = 1.0 / h_square;

  _infoMsg("Solving a two dimensional Poisson problem using Finite Difference Method", __func__);

  std::ostringstream msg1;
  msg1 << "Finite difference mesh consists of " << N << " x " << N << " points";
  _infoMsg(msg1.str(), __func__);

  _infoMsg("Initializing the coefficient matrix and RHS vector. This may take a while...", __func__);

  auto init_start = std::chrono::high_resolution_clock::now();
  
  // Arrays that will define the CRS structure of the coefficient matrix
  std::vector<type> vals;
  std::vector<int> colInds;
  std::vector<int> rowPtrs;
  rowPtrs.push_back(0);

  // The RHS vector
  Vector b = Vector<type, vectorize>(N_squared);

  // Form the coefficient matrix
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {      
      int row = ij(i, j, N);
      type x_1 = (type)i / (type)N;
      type x_2 = (type)j / (type)N;

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

  Matrix A = Matrix<type, vectorize, sparse>(N_squared, N_squared, vals, colInds, rowPtrs);

  auto init_end = std::chrono::high_resolution_clock::now();
  auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start);

  std::ostringstream msg2;
  msg2 << "Time taken in preprocessing: " << init_duration.count() << " milliseconds";
  _infoMsg(msg2.str(), __func__);

  _infoMsg("Using a zero vector as the initial guess", __func__);
  Vector x0 = Vector<type, vectorize>(N_squared);

  std::ostringstream msg3;
  msg3 << "Solving a " << N_squared << " dimensional system with conjugate gradient method. This may take a while...";
  _infoMsg(msg3.str(), __func__);

  Vector<type, vectorize> ret;

  int max_iter = 10 * N;  // 10 * N chosen as an arbitrary maximum number of iterations
  type disc_err = h * h;  // Discretization error should be O(h^2) so use it as convergence tolerance

  std::ostringstream msg4;
  msg4 << "Using discretization error " << disc_err << " as the convergence tolerance.";
  _infoMsg(msg4.str(), __func__);

  auto solve_start = std::chrono::high_resolution_clock::now();
  ret = cgSolve<type, vectorize, sparse>(A, x0, b, max_iter, disc_err);
  auto solve_end = std::chrono::high_resolution_clock::now();

  auto solve_duration = std::chrono::duration_cast<std::chrono::milliseconds>(solve_end - solve_start);

  std::ostringstream msg5;
  msg5 << "Time taken by the solver: " << solve_duration.count() << " milliseconds";
  _infoMsg(msg5.str(), __func__);

  std::ostringstream msg6;
  msg6 << "Residual norm: " << (A.matmul(ret) - b).norm();
  _infoMsg(msg6.str(), __func__);
}


int main() {
  
  int N = 200;  // Number of grid points in each dimension. 2D grid is then N x N
  
  poissonFD<double, false, true>(N);

  _infoMsg("DONE", __func__);

  return 0;
}
