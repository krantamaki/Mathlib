#include "declare_solver.hpp"


using namespace std;


map<string, string> solver::parse_file(const string& filepath) {
  map<string, string> ret;

  vector<string> valid_keys = {"coef_path", "rhs_path", "ret_path", "init_path", "method", "verbosity"};

  ifstream file(filepath);

  string key, value;

  
  while (file >> key >> value) {
    if (ret.find(key) != ret.end()) {
      _errorMsg("Key appears multiple times in config file!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
    }
    
    if (find(valid_keys.begin(), valid_keys.end(), key) != valid_keys.end()) {
      ret.insert({key, value});
    }
    else {
      std::ostringstream msgStream1;
      msgStream1 << "\n" << "Improper key: " << key << " passed. Ignoring...";
      _infoMsg(msgStream1.str(), __func__);
    }
  }

  // Mandatory keys are 'coef_path', 'rhs_path' and 'ret_path' so check that those exists
  if (ret.find("coef_path") == ret.end()) {
    _errorMsg("Path to coefficient matrix not provided!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
  if (ret.find("rhs_path") == ret.end()) {
    _errorMsg("Path to right hand side vector not provided!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }
  if (ret.find("ret_path") == ret.end()) {
    _errorMsg("Path to where the solution is to be stored not provided!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  // Pass empty string to the missing arguments
  if (ret.find("init_path") == ret.end()) {
    ret.insert({"init_path", ""});
  }
  if (ret.find("method") == ret.end()) {
    ret.insert({"method", ""});
  }
  if (ret.find("verbosity") == ret.end()) {
    ret.insert({"verbosity", "0"});
  }

  file.close();
  
  return ret;
}
