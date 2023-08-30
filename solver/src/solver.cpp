#include "declare_solver.hpp"


/*
  Compile at root mathlib with: g++ -std=c++17 -mavx -fopenmp -Wall solver/src/solver.cpp solver/src/parse_file.cpp solver/src/linsolve.cpp lalib/src/crs/crsMatrix.cpp lalib/src/crs/crsVector.cpp lalib/src/crs/crsMatmul.cpp -lm -o solver.o
  Run with: ./solver.o <config file>
*/


using namespace std;


void welcome(bool always_print) {
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
  msgStream3 << "# Running with " << omp_get_max_threads() << " openMP threads" << std::setw(28) << "#";
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

  verbosity(3);

  welcome(true);

  string config_path = argv[1];

  map<string, string> config_map = solver::parse_file(config_path);

  solver::solve(config_map["coef_path"], config_map["rhs_path"], config_map["ret_path"],
		config_map["init_path"], config_map["method"], config_map["verbosity"]);
}
