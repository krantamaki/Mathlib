#include "declare_solver.hpp"


/*
  Compile with at root mathlib with: g++ -mavx -fopenmp -Wall solver/src/solver.cpp solver/src/parse_file.cpp solver/src/linsolve.cpp lalib/src/crs/crsMatrix.cpp lalib/src/crs/crsVector.cpp lalib/src/crs/crsMatmul.cpp -lm -o solver.o
  Run with: ./solver.o <config file>
*/


using namespace std;


int main(int argc, char* argv[]) {

  // Write output into a log file
  // freopen("log.txt", "w", stdout);

  if (argc != 2) {
    throw runtime_error("Improper number of arguments passed!");
  }

  string config_path = argv[1];

  map<string, string> config_map = solver::parse_file(config_path);

  solver::solve(config_map["coef_path"], config_map["rhs_path"], config_map["ret_path"],
		config_map["init_path"], config_map["method"], config_map["verbosity"]);
}
