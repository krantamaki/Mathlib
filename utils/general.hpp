#ifndef GENERAL_HPP
#define GENERAL_HPP


#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <cctype>

#include "messaging.hpp"


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


// Function that counts the number of tokens in a string as divided by a delimeter
inline int _numTokens(const std::string& str, char delim=' ') {
  std::istringstream stream(str);
  std::string token;
  int count = 0;

  while (std::getline(stream, token, delim)) {
    if (token != "") {
      count++;
    }
  }

  return count;
}


// Function that splits a string by a wanted delimeter
inline std::vector<std::string> _split(const std::string& str, char delim=' ') {
  std::istringstream stream(str);
  std::string token;

  std::vector<std::string> ret;

  while (std::getline(stream, token, delim)) {
    if (token != "") {
      ret.push_back(token);
    }
  }

  return ret;
}


// Convert a given string to lowercase
inline std::string _tolower(const std::string& str) {
  std::string ret;
  ret.reserve(str.size());
  for (int i = 0; i < (int)str.size(); i++) {
    char c = (char)tolower(str[i]);
    ret.push_back(c);
  }

  return ret;
} 


// Convert a given string to uppercase
inline std::string _toupper(const std::string& str) {
  std::string ret;
  ret.reserve(str.size());
  for (int i = 0; i < (int)str.size(); i++) {
    char c = (char)toupper(str[i]);
    ret.push_back(c);
  }

  return ret;
} 


// Remove leading and trailing whitespaces from a string
inline std::string _trim(const std::string& str) {

  if (str == "") {
    _errorMsg("Cannot trim an empty string!", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  }

  // Find the index of the first non-whitespace character
  int start;
  for (int i = 0; i < (int)str.size(); i++) {
    if (!isspace(str[i])) {
      start = i;
      break;
    } 
  }

  // Find the index of the last non-whitespace character
  int end;
  for (int i = (int)str.size() - 1; i >= 0; i--) {
    if (!isspace(str[i])) {
      end = i;
      break;
    } 
  }

  return str.substr(start, end - start + 1);
}


// Template function that retrieves all keys in a std::map
// Heavily inspired by: https://www.techiedelight.com/retrieve-all-keys-from-a-map-in-cpp/
template<class keyT, class valT> std::vector<keyT> _mapKeys(std::map<keyT, valT> map) {
  std::vector<keyT> ret;
  for (const auto& [key, val] : map) {
      ret.push_back(key);
  }

  return ret;
}


// Template function that retrieves all values in a std::map
// Heavily inspired by: https://www.techiedelight.com/retrieve-all-keys-from-a-map-in-cpp/
template<class keyT, class valT> std::vector<valT> _mapVals(std::map<keyT, valT> map) {
  std::vector<valT> ret;
  for (const auto& [key, val] : map) {
      ret.push_back(val);
  }

  return ret;
}


#endif