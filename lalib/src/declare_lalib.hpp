#ifndef LALIB_HPP
#define LALIB_HPP


/*
  This is a general linear algebra library for C++. Main functionality consist of different
  matrix classes (currently only DenseMatrix is implemented) which override the basic math operators
  (+, -, ...) to function as element-wise operations equivalent to Matlab's .* etc and contain methods
  like matmul for computing the common matrix operations. Additionally, there are some general template 
  functions that work independent of the matrix type (currently only DenseMatrix).

  The methods and functions are made as optimal as feasible with parallelization utilizing multi-threading
  and SIMD commands as well as making sure memory calls are linear.
*/


#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <iterator>
#include <string>
#include <fstream>
#include <sstream>
#include <float.h>
#include <time.h>
#include <math.h>
#include <omp.h>


// Define constants used by the solvers

#define BASE_TOL 0.0000001
#define MAX_ITER 1000
#define CHECK_SYMMETRIC 0
#define OMEGA 0.75


// Define a vector for SIMD commands. Currently set up to hold 4 doubles (for 256 bit vector registers)

#define VECT_ELEMS 4
typedef double vect_t __attribute__ ((__vector_size__ (VECT_ELEMS * sizeof(double))));


// Generally useful functions

// Function for "dividing up" two integers
inline int _ceil(int a, int b) {
  return (a + b - 1) / b;
}


// Function for reading the last line of a given text file
// Inspired by: https://stackoverflow.com/questions/11876290/c-fastest-way-to-read-only-last-line-of-text-file
inline std::stringstream _lastLine(const std::string& filepath) {
  std::ifstream file(filepath);
  char c;
  file.seekg(-1, std::ios_base::end);  // Go to last non EOF char

  file.get(c);

  if (c == '\n') {  // File ends with a newline char. Ignore it
    file.seekg(-2, std::ios_base::cur);
  }

  bool cont = true;
  while (cont) {
    file.get(c);

    if((int)file.tellg() <= 1) {  // File only has a single line
      file.seekg(0);
      cont = false;
    }
    else if(c == '\n') {  // End of last line found
      cont = false; 
    }
    else {  // Continue
      file.seekg(-2, std::ios_base::cur);
    }
  }

  std::string lastLine;
  std::getline(file, lastLine);

  std::stringstream ret(lastLine);

  file.close();

  return ret;
}


// Function that generates a more descriptive error message that can be thrown
inline std::string _formErrorMsg(const std::string& msg, const char* file, const char* func, int line) {
  std::ostringstream msgStream;

  msgStream << "ERROR: In file " << file << " at function " << func << " on line " << line << " : " << msg;

  return msgStream.str();
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
