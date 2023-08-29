#ifndef LALIB_HPP
#define LALIB_HPP


#include <vector>
#include <tuple>
#include <algorithm>
#include <iterator>
#include <float.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#include "../../utils/messaging.hpp"


// Define a vector for SIMD commands. Currently set up to hold 4 doubles (for 256 bit vector registers)

#define VECT_ELEMS 4
typedef double vect_t __attribute__ ((__vector_size__ (VECT_ELEMS * sizeof(double))));

// Constant expression that should define a wanted sized SIMD vector with only zeros
constexpr vect_t zeros = { };


// Generally useful functions

// Function that sums together the elements in SIMD vector
inline double _reduce(vect_t vector) {
  double ret = 0.0;
  for (int i = 0; i < VECT_ELEMS; i++) {
    ret += vector[i];
  }

  return ret;
}


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
