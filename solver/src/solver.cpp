#include "declare_solver.hpp"


/*
  Compile with: g++ -mavx -fopenmp -Wall solver.cpp parse_file.cpp -lm -o solver.o
  Run with: ./solver.o <config file>
*/


using namespace std;


int main(int argc, char* argv[]) {

  // Write output into a log file
  freopen("log.txt", "w", stdout);

  if (argc != 2) {
    throw runtime_error("Improper number of arguments passed!");
  }

  string config_path = argv[1];

  map<string, string> config_map = solver::parse_file(config_path);

  for (const auto& [key, value] : config_map) {
    cout << "Given key: " << key << " maps to: " << value << "\n";
  }
}
