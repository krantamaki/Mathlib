#include "declare_solver.hpp"
#include "../../lalib/src/declare_lalib.hpp"
#include "../../lalib/src/nonstationarySolvers.hpp"
#include "../../lalib/src/stationarySolvers.hpp"
#include "../../lalib/src/crs/crsMatrix.hpp"
#include "../../lalib/src/crs/crsVector.hpp"


using namespace std;
using namespace lalib;


void solver::solve(string coef_path, string rhs_path, string ret_path,
		   string init_path, string method, string verbosityString) {

  int verbosity = stoi(verbosityString);

  if (verbosity != 0) {
    cout << "\n" << " NOTE: Verbosity not yet supported" << "\n\n";
  }

  // Initialize the matrices

  std::cout << "Forming the coefficient matrix ..." << "\n";
  CRSMatrix A = CRSMatrix(coef_path, 1, ".dat", true);

  std::cout << "Forming the right hand side matrix ..." << "\n";
  CRSVector b = CRSVector(rhs_path, 1);
  CRSVector x0;
  
  if (init_path != "") {
    std::cout << "Forming the wanted initial guess ..." << "\n";
    x0 = CRSVector(init_path, 1);
  }
  else {
    std::cout << "Forming a 0 matrix as an initial guess ..." << "\n";
    x0 = CRSVector(b.len());
  }

  std::cout << "\n" << "Solving a " << x0.len() << " dimensional system ..." << "\n";

  CRSVector ret;

  auto start = chrono::high_resolution_clock::now();
  // Call the solver
  if (method == "" || method == "CG" || method == "cg") {
    std::cout << "Calling the Conjugate Gradient method ..." << "\n";
    ret = cgSolve<CRSMatrix, CRSVector>(A, x0, b);
  }
  else if (method == "Jacobi" || method == "jacobi" || method == "JACOBI") {
    std::cout << "Calling the Jacobi method ..." << "\n";
    ret = jacobiSolve<CRSMatrix, CRSVector>(A, x0, b);
  }
  else if (method == "Gauss-Seidel" || method == "gauss-seidel" || method == "GAUSS-SEIDEL" || method == "GS" || method == "gs"){
    std::cout << "Calling the Gauss-Seidel method ..." << "\n";
    ret = gsSolve<CRSMatrix, CRSVector>(A, x0, b);
  }
  else if (method == "SOR" || method == "sor" || method == "Sor"){
    std::cout << "Calling the SOR method ..." << "\n";
    ret = sorSolve<CRSMatrix, CRSVector>(A, x0, b);
  }
  else {
    throw runtime_error("Improper solver provided!");
  }
  auto end = chrono::high_resolution_clock::now();

  auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
  
  std::cout << "\n" << "Time taken by the solver: " << duration.count() << " milliseconds" << "\n";
  
  std::cout << "Residual norm: " << (A.matmul(ret) - b).norm() << "\n\n";

  // NOTE! Save method not yet implemented
  // std::cout << "Saving the solution as " << ret_path << " ..." << "\n";
  // ret.save(ret_path);
}
