#ifndef MESSAGING_HPP
#define MESSAGING_HPP


#include <iostream>
#include <string>
#include <fstream>
#include <sstream>


#ifndef BASE_VERBOSITY
#define BASE_VERBOSITY 3
#endif


/*
 * Function that defines and returns the verbosity level
 * Note that verbosity will have 3 levels:
 *   1: Error messages
 *   2: Error and warning messages
 *   3: Everything
 */
inline int verbosity(int _verbosity = BASE_VERBOSITY) {
  if (_verbosity < 0 || _verbosity > 3) {
    _verbosity = 1;
  }
  static int set_verbosity = _verbosity;
  return set_verbosity;
}


// Function that generates and throws a more descriptive error message
inline void _errorMsg(const std::string& msg, const char* file, const char* func, int line) {
  std::ostringstream msgStream;

  msgStream << "ERROR: In file " << file << " at function " << func << " on line " << line << " : " << msg;

  std::string msgString = msgStream.str();

  std::cout << msgString << std::endl;

  throw std::runtime_error(msgString);
}


// Function that forms and prints a warning message
inline void _warningMsg(const std::string& msg, const char* func) {
  if (verbosity() > 1) {
    std::cout << func << ": " << "WARNING! " << msg << std::endl;
  }
}


// Function that forms and prints an iteration message
inline void _iterMsg(int iter, double norm, const char* func) {
  if (verbosity() > 2) {
    std::cout << func << ": " << "Iter " << iter << " - Norm " << norm << std::endl;
  }
}


// Function that forms and prints an information message
inline void _infoMsg(const std::string& msg, const char* func, bool always_print=false) {
  if (always_print) {
    std::cout << func << ": " << msg << std::endl;
    return;
  }
  if (verbosity() > 2) {
    std::cout << func << ": " << msg << std::endl;
  }
}


// Function that returns the used C++ standard
inline std::string _getcppStandard() {
  if (__cplusplus == 202101L) return "C++23";
  else if (__cplusplus == 202002L) return "C++20";
  else if (__cplusplus == 201703L) return "C++17";
  else if (__cplusplus == 201402L) return "C++14";
  else if (__cplusplus == 201103L) return "C++11";
  else if (__cplusplus == 199711L) return "C++98";
  else return "pre-standard C++.";
}

#endif
