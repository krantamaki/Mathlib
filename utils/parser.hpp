#ifndef PARSER_HPP
#define PARSER_HPP


#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <any>
#include <algorithm>

#include "general.hpp"
#include "messaging.hpp"


namespace parser {

  int validate_int_input(const std::string& input, int fileLine);
  double validate_float_input(const std::string& input, int fileLine);
  bool validate_bool_input(const std::string& input, int fileLine);

  std::map<std::string, std::any> parser(const std::string& filepath, const std::vector<std::string>& req_keys, const std::map<std::string, std::any>& opt_keys);

}


#endif