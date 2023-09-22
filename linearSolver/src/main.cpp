#include "declare_linearSolver.hpp"


/**
 * Compile at root mathlib with: 
 * > g++ -std=c++17 -mavx -fopenmp -Wall linearSolver/src/main.cpp utils/parser.cpp -lm -o linearSolver.o
 * Run with: 
 * > ./solver.o <config file>
 */


void linearSolver::welcome(bool always_print) {
  _infoMsg("", __func__, always_print);
  _infoMsg("###########################################################", __func__, always_print);
  _infoMsg("# You are using a solver for large sparse linear systems! #", __func__, always_print);

  ostringstream msgStream1;
  msgStream1 << "# This program was compiled on " << __DATE__ << std::setw(17) << "#";
  _infoMsg(msgStream1.str(), __func__, always_print);

  ostringstream msgStream2;
  msgStream2 << "# Using C++ standard " << _getcppStandard() << " (req. C++17 or later)" << std::setw(11) << "#";
  _infoMsg(msgStream2.str(), __func__, always_print);

  ostringstream msgStream3;
  int thread_int = omp_get_max_threads();
  string thread_str = to_string(thread_int);
  int n_chars = thread_str.size();
  msgStream3 << "# Running with " << thread_str << " openMP threads" << std::setw(29 - n_chars) << "#";
  _infoMsg(msgStream3.str(), __func__, always_print);

  _infoMsg("# Starting the program!                                   #", __func__, always_print);
  _infoMsg("###########################################################", __func__, always_print);
  _infoMsg("", __func__, always_print);
}


int main(int argc, char* argv[]) {

  // Write output into a log file
  freopen("log.txt", "w", stdout);

  if (argc != 2) {
    _errorMsg("Improper number of arguments passed!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  // Verify that the C++ standard is compatible
  if (__cplusplus < 201703L) {
    ostringstream errorMsg;
    errorMsg << "Incompatible C++ version: " << _getcppStandard() << " used! Required C++17 or later.";
    _errorMsg(errorMsg.str(), __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  linearSolver::welcome(true);

  // Parse the configuration file
  string config_path = argv[1];

  ostringstream msg1;
  msg1 << "Configuration file " << config_path << " passed. Parsing it ...";
  _infoMsg(msg1.str(), __func__, true);

  // Define the mandatory keywords for the parser
  vector<string> req_keys = {"coef_path", "rhs_path", "ret_path"};

  // Define the optional keywords for the parser and their associated default values
  map<string, any> opt_keys;

  opt_keys = {{"method", "cg"},
              {"verbosity", 3},
              {"init_path", string("__undef__")},
              {"linear_system_scaling", true},
              {"convergence_tolerance", BASE_TOL},
              {"max_iter", MAX_ITER},
              {"check_symmetric", false}};

  // Parse the input
  map<string, any> config_map = parser::parser(config_path, req_keys, opt_keys);

  // Set the verbosity
  int _verbosity = any_cast<int>(config_map["verbosity"]);

  ostringstream msg2;
  msg2 << "Setting verbosity to: " << _verbosity;
  _infoMsg(msg2.str(), __func__, true);

  verbosity(_verbosity);

  // Solve the system
  _infoMsg("Parsing complete. Advancing to the linear solver ...", __func__);

  bool converged = linearSolver::linearSolver<double, false>(config_map);

  // Check if convergence was reached
  if (converged) _infoMsg("Convergence to wanted tolerance reached", __func__);
  else _infoMsg("Convergence to wanted tolerance NOT reached!", __func__);

  _infoMsg("DONE", __func__, true);

  return converged ? 0 : 1;
}
