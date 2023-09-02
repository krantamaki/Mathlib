#include "declare_solver.hpp"
#include "../../lalib/src/declare_lalib.hpp"
#include "../../lalib/src/nonstationarySolvers.hpp"
#include "../../lalib/src/stationarySolvers.hpp"
#include "../../lalib/src/crs/crsMatrix.hpp"
#include "../../lalib/src/crs/crsVector.hpp"


using namespace std;
using namespace lalib;


void solver::linearSolver(string coef_path, string rhs_path, string ret_path,
		   string init_path, string method, string verbosityString) {

  // Initialize the matrices
  _infoMsg("Forming the coefficient matrix ...", __func__);
  CRSMatrix A = CRSMatrix(coef_path, 1, ".dat", true);

  _infoMsg("Forming the right hand side vector ...", __func__);
  CRSVector b = CRSVector(rhs_path, 1);

  // Scale the linear system by ||b||
  // This should be made optional
  double bnorm = b.norm();
  b *= (1 / bnorm);
  
  CRSVector x0;
  
  if (init_path != "") {
    _infoMsg("Forming the wanted initial guess ...", __func__);
    x0 = CRSVector(init_path, 1);
    x0 *= (1 / bnorm);
  }
  else {
    _infoMsg("Forming a 0 vector as an initial guess ...", __func__);
    x0 = CRSVector(b.len());
  }

  std::ostringstream msgStream1;
  msgStream1 << "Solving a " << x0.len() << " dimensional system ...";
  _infoMsg(msgStream1.str(), __func__);

  CRSVector ret;

  auto start = chrono::high_resolution_clock::now();
  // Call the solver
  if (method == "" || method == "CG" || method == "cg") {
    _infoMsg("Calling the Conjugate Gradient method ...", __func__);
    _infoMsg("", __func__);
    ret = cgSolve<CRSMatrix, CRSVector>(A, x0, b);
    _infoMsg("", __func__);
  }
  else if (method == "Jacobi" || method == "jacobi" || method == "JACOBI") {
    _infoMsg("Calling the Jacobi method ...", __func__);
    _infoMsg("", __func__);
    ret = jacobiSolve<CRSMatrix, CRSVector>(A, x0, b);
    _infoMsg("", __func__);
  }
  else if (method == "Gauss-Seidel" || method == "gauss-seidel" || method == "GAUSS-SEIDEL" || method == "GS" || method == "gs"){
    _infoMsg("Calling the Gauss-Seidel method ...", __func__);
    _infoMsg("", __func__);
    ret = gsSolve<CRSMatrix, CRSVector>(A, x0, b);
    _infoMsg("", __func__);
  }
  else if (method == "SOR" || method == "sor" || method == "Sor"){
    _infoMsg("Calling the SOR method ...", __func__);
    _infoMsg("", __func__);
    ret = sorSolve<CRSMatrix, CRSVector>(A, x0, b);
    _infoMsg("", __func__);
  }
  else {
    _errorMsg("Improper solver provided!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
  auto end = chrono::high_resolution_clock::now();

  auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

  std::ostringstream msgStream2;
  msgStream2 << "Time taken by the solver: " << duration.count() << " milliseconds";
  _infoMsg(msgStream2.str(), __func__);

  std::ostringstream msgStream3;
  msgStream3 << "Residual norm: " << (A.matmul(ret) - b).norm();
  _infoMsg(msgStream3.str(), __func__);

  std::ostringstream msgStream4;
  msgStream4 << "Saving the solution as " << ret_path << " ...";
  _infoMsg(msgStream4.str(), __func__);

  ret *= bnorm;
  
  ret.save(ret_path);

  _infoMsg("DONE", __func__);
}
