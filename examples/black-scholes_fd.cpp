#include <chrono>
#include <cmath>
#include "../lalib/src/crs/crsMatrix.hpp"
#include "../lalib/src/crs/crsVector.hpp"
#include "../lalib/src/nonstationarySolvers.hpp"


/**
 * @brief Finite Difference Method solver for a single (European) option Black-Scholes equation 
 *
 * Finite Difference Method is a numerical method used for solving partial differential
 * equations. This example solves a single option Black-Scholes equation.
 *
 * See eg. [1] for more information
 *
 * [1] Paul Wilmott. Paul Wilmott Introduces Quantitative Finance
 *
 * Compile at root mathlib directory with: 
 * > g++ -fopenmp -Wall lalib/src/crs/crsMatrix.cpp lalib/src/crs/crsVector.cpp lalib/src/crs/crsMatmul.cpp examples/black-scholes_fd.cpp -lm -o black-scholes_fd.o
 *
 * Run with:
 * > ./black-scholes_fd.o
 */


#define r 0.05  // The risk-free rate
#define vol 0.25  // The implied volatility of the underlying
#define E 100.0  // The strike price
#define T_0 10.0  // Time until expiry (arbitrary unit)


using namespace lalib;


// Handy index map
inline int ij(int i, int j, int N) {
  return i * N + j;
}


// Coefficient for gamma without dividend
inline double A(double S_i, double t_k) {
  return 0.5 * vol * vol * S_i * S_i;
}


// Coefficient for delta without dividend
inline double B(double S_i, double t_k) {
  return r * S_i;
}


// Coefficient for option value without dividend
inline double C(double S_i, double t_k) {
  return -r;
}


// Boundary condition for S -> inf. Set up for standard European call
double u(double S_i, double t_k) {
  return S_i - E * std::exp(-r * (T_0 - t_k));
}


// Boundary condition for S = 0. Set up for standard European call
double d(double S_i, double t_k) {
  return 0.0;
}


// Final condition i.e. the payoff function. Set up for European call
double P(double S_i) {
  return S_i > E ? S_i - E : 0.0;
}


int main() {
  
  int nT = 100;  // Number of grid points in time dimension
  double dt = T_0 / (double)nT;  // The time step. The domain is (0, T_0)
  double inv_dt = 1.0 / dt;

  int nS = 100;  // Number of grid points in underlying value dimension
  double dS = (4 * E) / (double)nS;  // The step of the underlying. The domain is (0, 4E)
  double inv_dS = 1.0 / dS;
  double dS_square = dS * dS;
  double inv_dS_square = 1.0 / dS_square;

  int dim = nS * nT;

  std::cout << "\n" << "Solving the Black-Scholes equation for single option using Finite Difference Method" << "\n";
  std::cout << "Finite difference mesh consists of " << nT << " x " << nS << " points" << "\n";
  
  std::cout << "\n" << "Initializing the coefficient matrix and RHS vector. This may take a while..." << "\n";

  auto init_start = std::chrono::high_resolution_clock::now();
  
  // Arrays that will define the CRS structure of the coefficient matrix
  std::vector<double> vals;
  std::vector<int> colInds;
  std::vector<int> rowPtrs;
  rowPtrs.push_back(0);

  // The RHS vector
  CRSVector b = CRSVector(dim);
  
  // Form the coefficient matrix
  for (int i = 0; i < nT; i++) {
    for (int j = 0; j < nS; j++) {      
      int row = ij(i, j, nT);
      double t_i = i * dt;
      double s_j = j * dS;

      if ((i > 0) && (i < nT - 1) && (j > 0) && (j < nS - 1)) {  // Point in the interior
        vals.insert(vals.end(), {A(s_j, t_i) * inv_dS_square - B(s_j, t_i) * (inv_dS / 2.0),
				 inv_dt - 2 * A(s_j, t_i) * inv_dS_square + C(s_j, t_i),
				 A(s_j, t_i) * inv_dS_square + B(s_j, t_i) * (inv_dS / 2.0),
				 -inv_dt});
	colInds.insert(colInds.end(), {ij(i, j - 1, nT), row, ij(i, j + 1, nT), ij(i + 1, j, nT)});
	rowPtrs.push_back(rowPtrs.back() + 4);
      }
      else if (i == nT - 1) { // Point at final condition
        vals.push_back(1.0);
	colInds.push_back(row);
	rowPtrs.push_back(rowPtrs.back() + 1);
	
	b.place(row, P(s_j));
      }
      else if (j == nS - 1) {  // Point at upper boundary
	vals.push_back(1.0);
	colInds.push_back(row);
	rowPtrs.push_back(rowPtrs.back() + 1);
	
	b.place(row, u(s_j, t_i));
      }
      else if (j == 0) {  // Point at lower boundary
	vals.push_back(1.0);
	colInds.push_back(row);
	rowPtrs.push_back(rowPtrs.back() + 1);
	
	b.place(row, d(s_j, t_i));
      }
      else {  // The initial boundary needs to be specifically considered. Ignore it for now
	vals.push_back(1.0);
	colInds.push_back(row);
	rowPtrs.push_back(rowPtrs.back() + 1);
	
	b.place(row, 1.0);
      }
    }
  }

  CRSMatrix A = CRSMatrix(dim, dim, vals, colInds, rowPtrs);

  auto init_end = std::chrono::high_resolution_clock::now();

  auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start);

  std::cout << "\n" << "Time taken in preprocessing: " << init_duration.count() << " milliseconds" << "\n";
  
  std::cout << "\n" << "Using a zero vector as the initial guess..." << "\n";
  CRSVector x0 = CRSVector(dim);

  std::cout << "\n" << "Solving a " << dim << " dimensional system with conjugate gradient method. This may take a while..." << "\n";

  CRSVector ret;

  auto solve_start = std::chrono::high_resolution_clock::now();
  ret = cgSolve(A, x0, b, 10 * nS);  // 10 * nS chosen as an arbitrary maximum number of iterations
  auto solve_end = std::chrono::high_resolution_clock::now();

  auto solve_duration = std::chrono::duration_cast<std::chrono::milliseconds>(solve_end - solve_start);

  std::cout << "\n" << "Time taken by the solver: " << solve_duration.count() << " milliseconds" << "\n";
  
  std::cout << "Residual norm: " << (A.matmul(ret) - b).norm() << "\n\n";

  std::cout << "DONE" << "\n";

  return 0;
}
