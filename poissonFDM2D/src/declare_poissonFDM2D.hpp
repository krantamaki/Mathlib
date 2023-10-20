#ifndef DECL_POISSON2D_HPP


#include <cstdio>
#include <string>
#include <iomanip>
#include <sstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <map>
#include <chrono>
#include <filesystem>
#include <float.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#include "../../utils/messaging.hpp"
#include "../../utils/general.hpp"
#include "../../utils/parser.hpp"

#include "../../lalib/src/declare_lalib.hpp"
#include "../../lalib/src/nonstationarySolvers.hpp"
#include "../../lalib/src/vector/Vector.hpp"
#include "../../lalib/src/matrix/Matrix.hpp"


using namespace utils;
using namespace lalib;
using namespace std;
namespace fs = std::filesystem;


namespace poissonFDM2D {

  void welcome(bool always_print);

}

#endif