#include "parser.hpp"


using namespace std;
using namespace parser;
using namespace utils;


int parser::validate_int_input(const std::string& input, int fileLine) {
  try {
    int ret = stoi(input);
    return ret;
  }
  catch (invalid_argument& e) {
    ostringstream errorMsg;
    errorMsg << "Improper integer input on line " << fileLine << " in input file!";
    _errorMsg(errorMsg.str(), __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  return 0;  // Should never get here
}


double parser::validate_float_input(const std::string& input, int fileLine) {
  try {
    double ret = stod(input);
    return ret;
  }
  catch (invalid_argument& e) {
    ostringstream errorMsg;
    errorMsg << "Improper floating point input on line " << fileLine << " in input file!";
    _errorMsg(errorMsg.str(), __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  return 0.0;  // Should never get here
}


bool parser::validate_bool_input(const std::string& input, int fileLine) {
  if (input == "true") return true;
  else if (input == "false") return false;
  else {
    ostringstream errorMsg;
    errorMsg << "Improper boolean input on line " << fileLine << " in input file!";
    _errorMsg(errorMsg.str(), __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  return false;  // Should never get here
}


map<string, any> parser::parser(const string& filepath, const vector<string>& req_keys, const map<string, any>& opt_key_map) {

  // Combine the required keys and optional keys into a single vector
  vector<string> opt_keys = _mapKeys<string, any>(opt_key_map);
  vector<string> valid_keys = req_keys;
  valid_keys.insert(valid_keys.end(), opt_keys.begin(), opt_keys.end());

  // Open the input file and define a string for the lines
  ifstream file(filepath);
  string fullLine;
  int lineNum = 1;

  map<string, any> ret = opt_key_map;

  // Go over the lines of the file
  while (getline(file, fullLine)) {

    // Empty lines are allowed and just ignored
    if (fullLine.size() == 0) {
      continue;
    }

    string line = _trim(fullLine);

    // Single line comments are allowed and must be started with '#'
    if (line[0] == '#') {
      continue;
    }

    vector<string> equalitySplit = _split(line, '=');
    if (equalitySplit.size() != 2) {
      ostringstream errorMsg;
      errorMsg << "Improper line number " << lineNum << " in input file : " << fullLine;
      _errorMsg(errorMsg.str(), __FILE__, __PRETTY_FUNCTION__, __LINE__);
    }

    string key = _trim(equalitySplit[0]);

    // Unnecessary keys can exist in the input file, but will be ignored
    if (find(valid_keys.begin(), valid_keys.end(), key) == valid_keys.end()) {  
      ostringstream msg;
      msg << "Unnecessary key: " << key << " passed. Ignoring ...";
      _infoMsg(msg.str(), __func__);
      continue;
    }

    vector<string> wspaceSplit = _split(equalitySplit[1], ' ');

    // Assumes no type specifier
    if (wspaceSplit.size() == 1) {
      ostringstream warningMsg;
      warningMsg << "Type definition left out on line " << lineNum << " in input file. This is not recommended!";
      _warningMsg(warningMsg.str(), __func__);
      ret[key] = _trim(wspaceSplit[0]);
    }

    // Assumes type-value pair
    else if (wspaceSplit.size() == 2) {
      string type_lower = _trim(_tolower(wspaceSplit[0]));
      string val = _trim(wspaceSplit[1]);

      if (type_lower == "string") {
        ret[key] = val;
      }
      else if (type_lower == "int") {
        ret[key] = validate_int_input(val, lineNum);
      }
      else if (type_lower == "float") {
        ret[key] = validate_float_input(val, lineNum);
      }
      else if (type_lower == "bool") {
        ret[key] = validate_bool_input(val, lineNum);
      }
      else {
        ostringstream errorMsg;
        errorMsg << "Improper line number " << lineNum << " in input file : " << fullLine;
        _errorMsg(errorMsg.str(), __FILE__, __PRETTY_FUNCTION__, __LINE__);
      }

    }
    else {
      ostringstream errorMsg;
      errorMsg << "Improper line number " << lineNum << " in input file : " << fullLine;
      _errorMsg(errorMsg.str(), __FILE__, __PRETTY_FUNCTION__, __LINE__);
    }

    lineNum++;
  }

  return ret;
}


