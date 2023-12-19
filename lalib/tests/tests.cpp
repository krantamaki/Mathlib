#include <chrono>
#include <cmath>
#include <algorithm>
#include "../src/vector/Vector.hpp"
#include "../src/matrix/Matrix.hpp"
#include "../src/nonstationarySolvers.hpp"
#include "../src/stationarySolvers.hpp"
#include "../../utils/general.hpp"
#include "../../utils/messaging.hpp"
#include "../../utils/testing.hpp"

/*
 * Compile at root mathlib directory with: 
 * > g++ -fopenmp -mavx -std=c++17 -Wall lalib/tests/tests.cpp -lm -o lalib_tests.o
 *
 * Run with:
 * > ./lalib_tests.o
 */


using namespace lalib;
using namespace utils;


#ifndef NUM_TESTS
#define NUM_TESTS 10
#endif


#ifndef PRECISION
#define PRECISION 1e-6
#endif


// Define the tests

template <class type, bool vectorize, bool sparse>
int test1() {
  try {
    type sol_tmp[16] = {2.,  7.,  12., 17.,
                        7.,  12., 17., 22.,
                        12., 17., 22., 27.,
                        17., 22., 27., 32.};

    Matrix sol = Matrix<type, vectorize, sparse>(4, 4, sol_tmp);

    type A_tmp[16] = {1.,  2.,  3.,  4.,
                      5.,  6.,  7.,  8.,
                      9.,  10., 11., 12.,
                      13., 14., 15., 16.};

    Matrix A = Matrix<type, vectorize, sparse>(4, 4, A_tmp);

    Matrix A_T = A.T();
    Matrix ans = A + A_T;

    return TEST(ans.isclose(sol, PRECISION));
  }
  catch (...) {
    return utils::_test(false, __func__, "FAILED - ERROR OCCURED");
  }
}


template <class type, bool vectorize, bool sparse>
int test2() {
  try {
    Matrix sol = Matrix<type, vectorize, sparse>(7, 7, (type)6.);

    Matrix A = Matrix<type, vectorize, sparse>(7, 7, (type)1.);
    Matrix B = Matrix<type, vectorize, sparse>(7, 7, (type)3.);

    Matrix C = A + A;
    Matrix ans = B * C;

    return TEST(ans.isclose(sol, PRECISION));
  }
  catch (...) {
    return utils::_test(false, __func__, "FAILED - ERROR OCCURED");
  }
}


template <class type, bool vectorize, bool sparse>
int test3() {
  try {
    type sol_tmp[16] = {2.,  2.,  6.,  2.,
                        4.,  7.,  4.,  4.,
                        8.,  6.,  6.,  6.,
                        8.,  8.,  8.,  8.};

    Matrix sol = Matrix<type, vectorize, sparse>(4, 4, sol_tmp);

    type A_tmp[16] = {1.,  1.,  1.,  1.,
                      2.,  2.,  2.,  2.,
                      3.,  3.,  3.,  3.,
                      4.,  4.,  4.,  4.};

    Matrix A = Matrix<type, vectorize, sparse>(4, 4, A_tmp);

    Matrix C = 2. * A;
    C.place(0, 2, 6.);
    C.place(1, 1, 7.);
    C.place(2, 0, 8.);

    Matrix ans = C;

    return TEST(ans.isclose(sol, PRECISION));
  }
  catch (...) {
    return utils::_test(false, __func__, "FAILED - ERROR OCCURED");
  }
}


template <class type, bool vectorize, bool sparse>
int test4() {
  // placeRow and placeCol not implemented for dense matrices (yet)
  if constexpr (sparse) {
    try {
      type sol_tmp[25] = {5.,  5.,  5.,  8., 5.,
                          5.,  5.,  5.,  8., 5.,
                          6.,  6.,  6.,  6., 6.,
                          5.,  5.,  5.,  8., 5.,
                          5.,  5.,  5.,  8., 5.};

      Matrix sol = Matrix<type, vectorize, sparse>(5, 5, sol_tmp);

      Matrix A = Matrix<type, vectorize, sparse>(5, 5, (type)5.);

      Vector v1 = Vector<type, vectorize>(5, (type)8.);
      Vector v2 = Vector<type, vectorize>(5, (type)6.);

      A.placeCol(3, v1);
      A.placeRow(2, v2);

      Matrix ans = A;

      return TEST(ans.isclose(sol, PRECISION));
    }
    catch (...) {
      return utils::_test(false, __func__, "FAILED - ERROR OCCURED");
    }
  }
  else {
    return utils::_test(false, __func__, "FAILED - NOT IMPLEMENTED");
  }
}


template <class type, bool vectorize, bool sparse>
int test5() {
  try {
    type sol_tmp[25] = {5.,  5.,  5.,  5., 5.,
                        7.,  7.,  5.,  5., 5.,
                        7.,  7.,  5.,  5., 5.,
                        7.,  7.,  5.,  5., 5.,
                        7.,  7.,  5.,  5., 5.};

    Matrix sol = Matrix<type, vectorize, sparse>(5, 5, sol_tmp);

    Matrix A = Matrix<type, vectorize, sparse>(5, 5, (type)10.);
    Matrix B = Matrix<type, vectorize, sparse>(4, 2, (type)7.);

    Matrix C = A / 2.;
    C.place(1, 5, 0, 2, B);

    Matrix ans = C;

    return TEST(ans.isclose(sol, PRECISION));
  }
  catch (...) {
    return utils::_test(false, __func__, "FAILED - ERROR OCCURED");
  }
}


template <class type, bool vectorize, bool sparse>
int test6() {
  try {
    type sol_tmp[25] = {3.,  0.,  0.,  0., 0.,
                        0.,  3.,  0.,  0., 0.,
                        0.,  0.,  3.,  0., 0.,
                        0.,  0.,  0.,  3., 0.,
                        0.,  0.,  0.,  0., 3.};

    Matrix sol = Matrix<type, vectorize, sparse>(5, 5, sol_tmp);

    Matrix A = Matrix<type, vectorize, sparse>(5, 5, (type)0.);
    Matrix B = Matrix<type, vectorize, sparse>(5, 5, (type)3.);

    for (int i = 0; i < 5; i++) {
      A.place(i, i, (type)1.);
    }

    Matrix ans = A * B;

    return TEST(ans.isclose(sol, PRECISION));
  }
  catch (...) {
    return utils::_test(false, __func__, "FAILED - ERROR OCCURED");
  }
}


template <class type, bool vectorize, bool sparse>
int test7() {
  try {
    Matrix sol = Matrix<type, vectorize, sparse>(5, 5, (type)60.);

    Matrix A = Matrix<type, vectorize, sparse>(5, 5, (type)4.);
    Matrix B = Matrix<type, vectorize, sparse>(5, 5, (type)3.);

    Matrix ans = A.matmul(B);

    return TEST(ans.isclose(sol, PRECISION));
  }
  catch (...) {
    return utils::_test(false, __func__, "FAILED - ERROR OCCURED");
  }
}


template <class type, bool vectorize, bool sparse>
int test8() {
  try {
    type sol_tmp[25] = {3.,  3.,  3.,  3., 3.,
                        1.,  1.,  1.,  1., 1.,
                        2.,  2.,  2.,  2., 2.,
                        4.,  4.,  4.,  4., 4.,
                        0.,  0.,  0.,  0., 0.};

    Matrix sol = Matrix<type, vectorize, sparse>(5, 5, sol_tmp);

    type A_tmp[25] = {0.,  0.,  0.,  0., 0.,
                      1.,  1.,  1.,  1., 1.,
                      2.,  2.,  2.,  2., 2.,
                      3.,  3.,  3.,  3., 3.,
                      4.,  4.,  4.,  4., 4.};

    Matrix A = Matrix<type, vectorize, sparse>(5, 5, A_tmp);

    type B_tmp[25] = {0.,  0.,  0.,  1., 0.,
                      0.,  1.,  0.,  0., 0.,
                      0.,  0.,  1.,  0., 0.,
                      0.,  0.,  0.,  0., 1.,
                      1.,  0.,  0.,  0., 0.};

    Matrix B = Matrix<type, vectorize, sparse>(5, 5, B_tmp);

    Matrix ans = B.matmul(A);

    return TEST(ans.isclose(sol, PRECISION));
  }
  catch (...) {
    return utils::_test(false, __func__, "FAILED - ERROR OCCURED");
  }
}


template <class type, bool vectorize, bool sparse>
int test9() {
  try {
    type sol_tmp[16] = {48., 96., 144., 192.,
                        48., 96., 144., 192.,
                        48., 96., 144., 192.,
                        48., 96., 144., 192.};

    Matrix sol = Matrix<type, vectorize, sparse>(4, 4, sol_tmp);

    Matrix A = Matrix<type, vectorize, sparse>(4, 4, (type)4.);

    Vector v1 = Vector<type, vectorize>(4, (type)3.);

    // Should be changed when vector-vector multiplication is implemented
    type v2_tmp[4] = {1., 2., 3., 4.};
    Matrix v2 = Matrix<type, vectorize, sparse>(1, 4, v2_tmp);

    Vector w = A.matmul(v1);
    std::vector<type> w_vec = w.tovector();
    Matrix w_mat = Matrix<type, vectorize, sparse>(4, 1, w_vec);

    Matrix ans = w_mat.matmul(v2);

    return TEST(ans.isclose(sol, PRECISION));
  }
  catch (...) {
    return utils::_test(false, __func__, "FAILED - ERROR OCCURED");
  }
}


template <class type, bool vectorize, bool sparse>
int test10() {
  // Reading from file not implemented for dense matrices (yet)
  if constexpr (sparse) {
    try {
      type sol_tmp[16] = {13.,   64.,  135.,  180.,
                          24.,   16.,  96.,   96.,
                          52.,   256., 540.,  720.,
                          96.,   576., 1152., 1408.};

      Matrix sol = Matrix<type, vectorize, sparse>(4, 4, sol_tmp);

      Matrix A = Matrix<type, vectorize, sparse>("lalib/tests/matrix_files/mat1_crs.dat");

      Matrix ans = A.matmul(A);

      return TEST(ans.isclose(sol, PRECISION));
    }
    catch (...) {
      return utils::_test(false, __func__, "FAILED - ERROR OCCURED");
    }
  }
  else {
    return utils::_test(false, __func__, "FAILED - NOT IMPLEMENTED");
  }
}


// ...


// Function that runs the tests one by one for the given template parameters
// Returns the number of successful tests
template <class type, bool vectorize, bool sparse>
int run_tests() {

  // Count successful tests
  int successful_tests = 0;

  // Run the tests

  successful_tests += test1<type, vectorize, sparse>();
  successful_tests += test2<type, vectorize, sparse>();
  successful_tests += test3<type, vectorize, sparse>();
  successful_tests += test4<type, vectorize, sparse>();
  successful_tests += test5<type, vectorize, sparse>();
  successful_tests += test6<type, vectorize, sparse>();
  successful_tests += test7<type, vectorize, sparse>();
  successful_tests += test8<type, vectorize, sparse>();
  successful_tests += test9<type, vectorize, sparse>();
  successful_tests += test10<type, vectorize, sparse>();

  return successful_tests;
}


int main() {

  // Set verbosity to provide only necessary information
  verbosity(1);

  int success;
  bool all_passed = true;

  // Run tests for dense matrices

  // Unvectorized
  std::cout << "\nTESTING UNVECTORIZED DENSE IMPLEMENTATION..." << std::endl;
  success = run_tests<double, false, false>();
  std::cout << "PASSED TESTS: " << success << "/" << NUM_TESTS << "\n" << std::endl;

  all_passed = (success == NUM_TESTS) && all_passed;

  // Vectorized
  std::cout << "TESTING VECTORIZED DENSE IMPLEMENTATION..." << std::endl;
  success = run_tests<double, true, false>();
  std::cout << "PASSED TESTS: " << success << "/" << NUM_TESTS << "\n" << std::endl;

  all_passed = (success == NUM_TESTS) && all_passed;

  // Run tests for sparse matrices

  // Unvectorized
  std::cout << "TESTING UNVECTORIZED SPARSE IMPLEMENTATION..." << std::endl;
  success = run_tests<double, false, true>();
  std::cout << "PASSED TESTS: " << success << "/" << NUM_TESTS << "\n" << std::endl;

  all_passed = (success == NUM_TESTS) && all_passed;

  // Vectorized  (NOT YET IMPLEMENTED)
  // std::cout << "TESTING VECTORIZED DENSE IMPLEMENTATION...\n" << std::endl;
  // success = run_tests<double, true, true>();
  // std::cout << "\n" << "PASSED TESTS: " << success << "/" << NUM_TESTS << std::endl;

  // all_passed = (success == NUM_TESTS) && all_passed;

  return all_passed ? 1 : 0;
}
