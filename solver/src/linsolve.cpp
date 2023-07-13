#include "declare_solver.hpp"
#include "../../lalib/src/declare_lalib.hpp"
#include "../../lalib/src/nonstationarySolvers.hpp"
#include "../../lalib/src/stationarySolvers.hpp"
#include "../../lalib/src/crs/crsMatrix.hpp"


using namespace std;
using namespace lalib;


void solver::solve(string coef_path, string rhs_path, string ret_path,
		   string init_path, string method, string verbosityString) {

  int verbosity = stoi(verbosityString);

  if (verbosity != 0) {
    cout << "\n" << " NOTE: Verbosity not yet supported" << "\n\n";
  }

  // Initialize the matrices
  CRSMatrix A = CRSMatrix(coef_path, 1);
  CRSMatrix b = CRSMatrix(rhs_path, 1);
  CRSMatrix x0;
  
  if (init_path != "") {
    x0 = CRSMatrix(init_path);
  }
  else {
    x0 = CRSMatrix(b.nrows(), b.ncols());
  }

  CRSMatrix ret;
  
  // Call the solver
  if (method == "" || method == "CG" || method == "cg") {
    ret = cgSolve<CRSMatrix>(A, x0, b);
  }
  else if (method == "Jacobi" || method == "jacobi" || method == "JACOBI"){
    ret = jacobiSolve<CRSMatrix>(A, x0, b);
  }
  else if (method == "Gauss-Seidel" || method == "gauss-seidel" || method == "GAUSS-SEIDEL" || method == "GS" || method == "gs"){
    ret = gsSolve<CRSMatrix>(A, x0, b);
  }
  else if (method == "SOR" || method == "sor" || method == "Sor"){
    ret = sorSolve<CRSMatrix>(A, x0, b);
  }
  else {
    throw runtime_error("Improper solver provided!");
  }
  
  ret.save(ret_path);
}
