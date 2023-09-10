#ifndef LALIB_HPP
#define LALIB_HPP


#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <tuple>
#include <algorithm>
#include <iterator>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <float.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#include "../../utils/messaging.hpp"
#include "../../utils/general.hpp"
#include "../../utils/simd.hpp"


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


#endif
