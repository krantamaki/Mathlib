#ifndef GENERAL_HPP
#define GENERAL_HPP


#include <string>
#include <sstream>


// Function that returns the used C++ standard
inline std::string _getcppStandard() {
  if (__cplusplus == 202101L) return "C++23";
  else if (__cplusplus == 202002L) return "C++20";
  else if (__cplusplus == 201703L) return "C++17";
  else if (__cplusplus == 201402L) return "C++14";
  else if (__cplusplus == 201103L) return "C++11";
  else if (__cplusplus == 199711L) return "C++98";
  else return "C++??";
}


// Function that counts the number of (whitespace separated) tokens in a stringstream
inline int _numTokens(const std::string& str) {
  std::istringstream stream(str);
  std::string token;
  int count = 0;

  while (std::getline(stream, token, ' ')) {
    if (token != "") {
      count++;
    }
  }

  return count;
}


#endif