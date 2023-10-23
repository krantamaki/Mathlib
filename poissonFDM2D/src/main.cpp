#include "declare_poissonFDM2D.hpp"
#include "poissonFDM2D.hpp"


/**
 * Compile at root mathlib with: 
 * > g++ -std=c++17 -mavx -fopenmp -Wall poissonFDM2D/src/main.cpp utils/parser.cpp -lm -o poissonFDM2D.o
 * Run with: 
 * > ./poissonFDM2D.o <config file>
 */


void poissonFDM2D::welcome(bool always_print) {
  _infoMsg("", __func__, always_print);
  _infoMsg("###########################################################", __func__, always_print);
  _infoMsg("# You are using a FDM software for the 2D heat equation!  #", __func__, always_print);

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

  if (argc == 3) {
    // Check if the directory in which the log is to be stored exists
    // If not create it
    vector<string> path_vector = _split(argv[2], '/');
    string dir_path = _join(vector<string>(path_vector.begin(), path_vector.end() - 1), '/');
    
    if (!fs::exists(dir_path)) {
      fs::create_directories(dir_path);
    } 

    // Write output into a log file
    freopen(argv[2], "w", stdout);

    ostringstream msg1;
    msg1 << "Writing output into log file: " << argv[2];
    _infoMsg(msg1.str(), __func__, true);
  }
  else if (argc != 2) {
    _errorMsg("Improper number of arguments passed!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
  else {
    // Write output into a log file
    freopen("log.txt", "w", stdout);
  }

  // Verify that the C++ standard is compatible
  if (__cplusplus < 201703L) {
    ostringstream errorMsg;
    errorMsg << "Incompatible C++ version: " << _getcppStandard() << " used! Required C++17 or later.";
    _errorMsg(errorMsg.str(), __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  poissonFDM2D::welcome(true);

  // Parse the configuration file
  string config_path = argv[1];

  ostringstream msg2;
  msg2 << "Configuration file " << config_path << " passed. Parsing it ...";
  _infoMsg(msg2.str(), __func__, true);

  // Define the mandatory keywords for the parser
  vector<string> req_keys = {"lower_bound",
                             "upper_bound",
                             "left_bound",
                             "right_bound",
                             "height",
                             "width",
                             "duration",
                             "n_height_points",
                             "n_width_points"};

  // Define the optional keywords for the parser and their associated default values
  map<string, any> opt_keys;

  opt_keys = {{"method", "cg"},
              {"verbosity", 3},
              {"convergence_tolerance", BASE_TOL},
              {"max_iter", MAX_ITER},
              {"save_dir", "tmp"},
              {"save_name", "heat_eq"},
              {"stop_unconverged", false},
              {"initial_temp", { }},
              {"n_time_points", -1},
              {"thermal_diffusivity", 1.0}};

  // Parse the input
  map<string, any> config_map = parser::parser(config_path, req_keys, opt_keys);

  // Set the verbosity
  int _verbosity = any_cast<int>(config_map["verbosity"]);

  ostringstream msg3;
  msg3 << "Setting verbosity to: " << _verbosity;
  _infoMsg(msg3.str(), __func__, true);

  verbosity(_verbosity);

  // Solve the system
  _infoMsg("Parsing complete. Advancing to the finite difference method ...", __func__);

  bool converged = poissonFDM2D::poissonFDM2D<double, false, true>(config_map);

  // Check if convergence was reached
  if (converged) _infoMsg("Successfully solved the wanted problem!", __func__);
  else _infoMsg("There was some issue with the problem!", __func__);

  _infoMsg("DONE", __func__, true);

  return converged ? 0 : 1;
}
