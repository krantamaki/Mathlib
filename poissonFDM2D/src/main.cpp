#include "declare_poissonFDM2D.hpp"
#include "poissonFDM2D.hpp"


/**
 * Compile at root mathlib with: 
 * > g++ -std=c++17 -mavx -fopenmp -Wall poissonFDM2D/src/main.cpp utils/parser.cpp -lm -o poissonFDM2D.o
 * Run with: 
 * > ./poissonFDM2D.o <config file>
 */


void poissonFDM2D::welcome(bool always_print) {
  utils::_infoMsg("", __func__, always_print);
  utils::_infoMsg("###########################################################", __func__, always_print);
  utils::_infoMsg("# You are using a FDM software for the 2D heat equation!  #", __func__, always_print);
  utils::_infoMsg(utils::_format("# This program was compiled on ", __DATE__, std::setw(17), "#"), __func__, always_print);
  utils::_infoMsg(utils::_format("# Using C++ standard ", utils::_getcppStandard(), " (req. C++17 or later)", std::setw(11), "#"), __func__, always_print);

  int thread_int = omp_get_max_threads();
  std::string thread_str = std::to_string(thread_int);
  int n_chars = thread_str.size();
  utils::_infoMsg(utils::_format("# Running with ", thread_str, " openMP threads", std::setw(29 - n_chars), "#"), __func__, always_print);

  utils::_infoMsg("# Starting the program!                                   #", __func__, always_print);
  utils::_infoMsg("###########################################################", __func__, always_print);
  utils::_infoMsg("", __func__, always_print);
}


int main(int argc, char* argv[]) {

  if (argc == 3) {
    // Check if the directory in which the log is to be stored exists
    // If not create it
    std::vector<std::string> path_vector = utils::_split(argv[2], '/');
    std::string dir_path = utils::_join(std::vector<std::string>(path_vector.begin(), path_vector.end() - 1), '/');
    
    if (!fs::exists(dir_path)) {
      fs::create_directories(dir_path);
    } 

    // Write output into a log file
    freopen(argv[2], "w", stdout);

    poissonFDM2D::welcome(true);

    utils::_infoMsg(utils::_format("Writing output into log file: ", argv[2]), __func__, true);
  }
  else if (argc != 2) {
    ERROR("Improper number of arguments passed!");
  }

  // Verify that the C++ standard is compatible
  if (__cplusplus < 201703L) {
    ERROR(utils::_format("Incompatible C++ version: ", utils::_getcppStandard(), " used! Required C++17 or later."));
  }

  // Parse the configuration file
  std::string config_path = argv[1];

  utils::_infoMsg(utils::_format("Configuration file ", config_path, " passed. Parsing it ..."), __func__, true);

  // Define the mandatory keywords for the parser
  std::vector<std::string> req_keys = {"lower_bound",
                                       "upper_bound",
                                       "left_bound",
                                       "right_bound",
                                       "height",
                                       "width",
                                       "duration",
                                       "n_height_points",
                                       "n_width_points"};

  // Define the optional keywords for the parser and their associated default values
  std::map<std::string, std::any> opt_keys;

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
  std::map<std::string, std::any> config_map = parser::parser(config_path, req_keys, opt_keys);

  // Set the verbosity
  int _verbosity = std::any_cast<int>(config_map["verbosity"]);
  utils::_infoMsg(utils::_format("Setting verbosity to: ", _verbosity), __func__, true);
  utils::verbosity(_verbosity);

  // Solve the system
  INFO("Parsing complete. Advancing to the finite difference method ...");

  bool converged = poissonFDM2D::poissonFDM2D<double, false, true>(config_map);

  // Check if convergence was reached
  if (converged) { INFO("Successfully solved the wanted problem!"); }
  else { INFO("There was some issue with the problem!"); }

  utils::_infoMsg("DONE", __func__, true);

  return converged ? 0 : 1;
}
