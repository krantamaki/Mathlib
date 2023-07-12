#ifndef SOLVER_HPP
#define SOLVER_HPP


/*
  TODO: DESCRIPTION
*/


#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <map>
#include <float.h>
#include <time.h>
#include <math.h>
#include <omp.h>


namespace solver {
  
  std::map<std::string, std::string> parse_file(const std::string& filepath);
  
  void solve(std::map<std::string, std::string> args);
}


#endif


