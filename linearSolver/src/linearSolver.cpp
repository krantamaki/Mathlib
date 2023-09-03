#include "declare_linearSolver.hpp"


using namespace std;
using namespace lalib;


bool linearSolver::linearSolver(map<string, any> config_map) {

  // Get required values from configuration map
  string coef_path = any_cast<string>(config_map["coef_path"]);
  string rhs_path = any_cast<string>(config_map["rhs_path"]);
  string ret_path = any_cast<string>(config_map["ret_path"]);
  string init_path = any_cast<string>(config_map["init_path"]);
  string method = any_cast<string>(config_map["method"]);
  bool linear_system_scaling = any_cast<bool>(config_map["linear_system_scaling"]);
  bool check_symmetric = any_cast<bool>(config_map["check_symmetric"]);
  double convergence_tolerance = any_cast<double>(config_map["convergence_tolerance"]);
  int max_iter = any_cast<int>(config_map["max_iter"]);

  // Load coefficient matrix and rhs vector
  ostringstream msg1;
  msg1 << "Loading the coefficient matrix from " << coef_path;
  _infoMsg(msg1.str(), __func__);
  CRSMatrix A = CRSMatrix(coef_path, 1, ".dat", true);

  ostringstream msg2;
  msg2 << "Loading the right hand side vector from " << rhs_path;
  _infoMsg(msg2.str(), __func__);
  CRSVector b = CRSVector(rhs_path, 1);

  // Scale the linear system by ||b||
  double bnorm = b.norm();
  if (linear_system_scaling) {
    _infoMsg("Applying linear system scaling", __func__);
    b *= (1 / bnorm);
  }
  

  // Define the initial guess
  CRSVector x0;
  
  if (init_path != "__undef__") {
    ostringstream msg3;
    msg3 << "Loading the initial guess vector from " << init_path;
    _infoMsg(msg3.str(), __func__);
    x0 = CRSVector(init_path, 1);
    x0 *= (1 / bnorm);
  }
  else {
    _infoMsg("Forming a 0 vector as an initial guess", __func__);
    x0 = CRSVector(b.len());
  }

  // Solve the system
  std::ostringstream msgStream1;
  msgStream1 << "Solving a " << x0.len() << " dimensional system ...";
  _infoMsg(msgStream1.str(), __func__);

  CRSVector ret;

  auto start = chrono::high_resolution_clock::now();

  if (method == "cg") {
    _infoMsg("Calling the Conjugate Gradient method", __func__);
    _infoMsg("", __func__);
    ret = cgSolve<CRSMatrix, CRSVector>(A, x0, b, max_iter, convergence_tolerance, check_symmetric);
    _infoMsg("", __func__);
  }
  else if (method == "jacobi") {
    _infoMsg("Calling the Jacobi method", __func__);
    _infoMsg("", __func__);
    ret = jacobiSolve<CRSMatrix, CRSVector>(A, x0, b, max_iter, convergence_tolerance);
    _infoMsg("", __func__);
  }
  else if ("gauss-seidel" || method == "gs"){
    _infoMsg("Calling the Gauss-Seidel method", __func__);
    _infoMsg("", __func__);
    ret = gsSolve<CRSMatrix, CRSVector>(A, x0, b, max_iter, convergence_tolerance);
    _infoMsg("", __func__);
  }
  else if (method == "sor"){
    _infoMsg("Calling the SOR method", __func__);
    _infoMsg("", __func__);
    ret = sorSolve<CRSMatrix, CRSVector>(A, x0, b, max_iter, convergence_tolerance);
    _infoMsg("", __func__);
  }
  else {
    _errorMsg("Improper solver provided!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
  auto end = chrono::high_resolution_clock::now();

  // Compute the passed time
  auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

  std::ostringstream msgStream2;
  msgStream2 << "Time taken by the solver: " << duration.count() << " milliseconds";
  _infoMsg(msgStream2.str(), __func__);

  // Compute the residual norm
  double res_norm = (A.matmul(ret) - b).norm();
  std::ostringstream msgStream3;
  msgStream3 << "Residual norm: " << res_norm;
  _infoMsg(msgStream3.str(), __func__);

  // Save the solution
  std::ostringstream msgStream4;
  msgStream4 << "Saving the solution as " << ret_path;
  _infoMsg(msgStream4.str(), __func__);

  if (linear_system_scaling) {
    ret *= bnorm;
  }
  
  ret.save(ret_path);

  // Return boolean signifying if convergence to wanted tolerance was reached
  return (res_norm < convergence_tolerance);
}
