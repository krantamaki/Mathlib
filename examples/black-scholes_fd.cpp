#include <chrono>
#include <cmath>
#include <algorithm>
#include "../lalib/src/vector/Vector.hpp"
#include "../lalib/src/matrix/Matrix.hpp"
#include "../lalib/src/nonstationarySolvers.hpp"
#include "../lalib/src/stationarySolvers.hpp"
#include "../utils/general.hpp"
#include "../utils/messaging.hpp"


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
 * > g++ -fopenmp -std=c++17 -Wall examples/black-scholes_fd.cpp -lm -o black-scholes_fd.o
 *
 * Run with:
 * > ./black-scholes_fd.o
 */


#define RF 0.05  // The risk-free rate
#define VOL 0.25  // The implied volatility of the underlying
#define SQVOL VOL * VOL
#define E 100.0  // The strike price
#define T_0 1.0  // Time until expiry (arbitrary unit)
#define S_MAX_MULT 4.0  // The multiple used to define the upper boundary for asset price


using namespace lalib;


// Function that returns the set value for time step
inline double dt(double _dt = 0.0) {
  static double set_dt = _dt;
  return set_dt;
}


// Function that returns the set value for underlying value step
inline double dS(double _dS = 0.0) {
  static double set_dS = _dS;
  return set_dS;
}


// Coefficient for V(S, t)
double A(int i) {
  // return ((SQVOL * S * S) / dS() - RF * S) * (dt() / (2.0 * dS()));
  return 0.5 * dt() * (RF * i - SQVOL * i * i);
}


// Coefficient for V(S, t + dt)
double B(int i) {
  // return 1 - ((SQVOL * S * S) / (dS() * dS()) + RF) * dt();
  return 1.0 + dt() * (SQVOL * i * i + RF);
}


// Coefficient for V(S - dS, t)
double C(int i) {
  // return (RF * S + (SQVOL * S * S) / dS()) * (dt() / (2.0 * dS()));
  return -0.5 * dt() * (RF * i + SQVOL * i * i);
}


// Boundary condition for S = 0. Set up for standard European call
double l(double t) {
  return 0.0;
}


// Boundary condition for S -> inf. Set up for European call
double u(double t) {
  return S_MAX_MULT * E - E * std::exp(-RF * (T_0 - t));
}


// Final condition i.e. the payoff function. Set up for European call
double P(double S) {
  return S > E ? S - E : 0.0;
}


template<class type, bool vectorize, bool sparse>
void blackScholesFDM(int nS) {

  _infoMsg("Solving the Black-Scholes equation for a single (European) option using Finite Difference Method", __func__);

  // The step of the underlying. The domain is (0, 4E)
  type _dS = (S_MAX_MULT * E) / (type)(nS - 1);
  dS(_dS);


  // The time step. The domain is (0, T_0)
  // Chosen to retain stability
  int nT = nS;
  type _dt = T_0 / (nT);

  std::cout << _dt / (_dS * _dS) << "\n";

  dt(_dt);


  std::ostringstream msg1;
  msg1 << "Finite difference mesh consists of " << nT << " x " << nS << " points";
  _infoMsg(msg1.str(), __func__);

  _infoMsg("Are you sure you want to continue? (yes/no)", __func__);

  std::string input;
  std::cin >> input;

  if (_tolower(input) != "yes") {
    _errorMsg("Exiting program!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }


  _infoMsg("Solving for the system one time step at a time...", __func__);


  // Time the total solver
  auto start = std::chrono::high_resolution_clock::now();

  // Define the solution vector
  Vector V = Vector<type, vectorize>(nT * nS);


  // Solve the system at payoff
  _infoMsg("Solving the system at payoff...", __func__);
  for (int i = 0; i < nS; i++) {
    type s_i = i * _dS;
    V.place(i, P(s_i));
  }


  // Solve the system at each subsequent timepoint
  for (int k = 1; k < nT; k++) {
    type t_k = T_0 - k * _dt;  // Time at which the system is to be solved

    std::ostringstream msg2;
    msg2 << "Solving the system at time: " << t_k << " ...";
    _infoMsg(msg2.str(), __func__);

    // Form the linear system

    // Required arrays for the CRS structure
    std::vector<type> vals;
    std::vector<int> colInds;
    std::vector<int> rowPtrs;
    rowPtrs.push_back(0);

    // Loop over the asset price. Note that at boundaries the coefficient matrix and 
    // the right hand side matrix will just have zeros so those can be ignored in the loop
    for (int i = 0; i < nS; i++) {

      if (i == 0) {
        vals.push_back(1.0);

        colInds.push_back(i);

        rowPtrs.push_back(rowPtrs.back() + 1);
      }
      else if (i == nS - 1) {
        vals.push_back(1.0);

        colInds.push_back(i);

        rowPtrs.push_back(rowPtrs.back() + 1);
      }
      else {
        vals.insert(vals.end(), {A(i),
                                 B(i),
                                 C(i)});

        colInds.insert(colInds.end(), {i - 1,
                                       i, 
                                       i + 1});

        rowPtrs.push_back(rowPtrs.back() + 3);
      }
    }

    // The coefficient matrix is then
    Matrix O = Matrix<type, vectorize, sparse>(nS, nS, vals, colInds, rowPtrs);

    // The right hand side vector will be the solution at previous time point
    Vector v_tmp = V((k - 1) * nS, k * nS);

    // Place the boundary conditions
    v_tmp.place(0, l(t_k));
    v_tmp.place(nS - 1, u(t_k));

    // Define the initial quess as the zero vector
    Vector x_0 = Vector<type, vectorize>(nS);

    // Solve for x
    Vector x = cgnrSolve<type, vectorize, sparse>(O, x_0, v_tmp, 1000, 1e-9);

    // Place to solution vector
    V.place(k * nS, (k + 1) * nS, x);
  }

  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::ostringstream msg3;
  msg3 << "Time taken  to find the solution: " << duration.count() << " milliseconds";
  _infoMsg(msg3.str(), __func__);

  // Save the solution as black_scholes_sol.dat
  _infoMsg("Saving the solution as black_scholes_sol.dat", __func__);
  V.save("black_scholes_sol.dat");
}


int main() {

  int nS = 100;

  verbosity(3);

  blackScholesFDM<double, false, true>(nS);

  _infoMsg("DONE", __func__);

  return 0;
}
