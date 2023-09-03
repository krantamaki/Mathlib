#ifndef SOLVER_HPP
#define SOLVER_HPP


/*
  TODO: DESCRIPTION
*/


#include <cstdio>
#include <string>
#include <iomanip>
#include <sstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <map>
#include <chrono>
#include <float.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#include "../../utils/messaging.hpp"
#include "../../utils/general.hpp"
#include "../../utils/parser.hpp"

#include "../../lalib/src/declare_lalib.hpp"
#include "../../lalib/src/nonstationarySolvers.hpp"
#include "../../lalib/src/stationarySolvers.hpp"
#include "../../lalib/src/crs/crsMatrix.hpp"
#include "../../lalib/src/crs/crsVector.hpp"


namespace linearSolver {

  void welcome(bool always_print);
  
  bool linearSolver(std::map<std::string, std::any> config_map);
}


#endif


