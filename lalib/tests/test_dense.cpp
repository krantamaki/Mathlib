#include "../src/declare_lalib.hpp"
#include "../src/dense/denseMatrix.hpp"
#include "../src/dense/denseVector.hpp"


#define TOL 0.00001
#define N_TESTS 22


using namespace lalib;


/*
  General tests for the dense representation of matrices
  Compile in lalib directory with: g++ -mavx -fopenmp -Wall src/dense/denseMatrix.cpp src/dense/denseVector.cpp src/dense/denseMatmul.cpp tests/test_dense.cpp -lm -o test_dense.o 
  Run with: ./test_dense.o
*/


int test(DenseMatrix testMatrix, DenseMatrix solMatrix) {
  if (testMatrix.isclose(solMatrix, TOL)) {
    std::cout << "PASSED" << "\n";

    return 1;
  }
  else {
    std::cout << "FAILED" << "\n\n";
    std::cout << testMatrix << "\n";
    std::cout << "Is not equal to:" << "\n\n";
    std::cout << solMatrix << "\n";

    return 0;
  }
}


int main() {

  verbosity(1);

  int passed_tests = 0;

  clock_t start = clock();
  std::cout << "\nTesting the dense matrix implementation" << "\n";
  std::cout << "---------------------------------------" << "\n\n";

  
  // Initialize some matrices and vectors
  
  double tmp[25] =
    {1.0, 3.0, 4.0, 5.0, 7.0,
     2.0, 3.0, 4.0, 5.0, 6.0,
     9.0, 8.0, 7.0, 6.0, 5.0,
     5.0, 6.0, 7.0, 8.0, 9.0,
     8.0, 7.0, 6.0, 5.0, 5.0};
  
  std::vector<double> tmp_vec(tmp, tmp + sizeof(tmp) / sizeof(tmp[0]));
  DenseMatrix A = DenseMatrix(5, 5, tmp_vec);

  double tmp2[25] =
    {0.2, 3.22, 3.01, 23.0, 4.4,
     3.423, 3.2234, -4.234, 5.32, 5.4,
     1.012, -8.4321, 32.432, 42.43, 2.43,
     4.6, 9.89, -3.54, 5.23, 4.26,
     6.89, 4.56, 9.456, 8.36, -7.345};
  
  std::vector<double> tmp_vec2(tmp2, tmp2 + sizeof(tmp2) / sizeof(tmp2[0]));
  DenseMatrix B = DenseMatrix(5, 5, tmp_vec2);
  
  double tmp3[5] =
    {0.1,
     1.0/3.0,
     2.0,
     0.91,
     0.82};

  std::vector<double> tmp_vec3(tmp3, tmp3 + sizeof(tmp3) / sizeof(tmp3[0]));
  DenseVector v = DenseVector(5, tmp_vec3);

  double tmp4[5] =
    {1.02,
     1.0/4.0,
     32.3,
     3.292,
     1.092};

  std::vector<double> tmp_vec4(tmp4, tmp4 + sizeof(tmp4) / sizeof(tmp4[0]));
  DenseVector w = DenseVector(5, tmp_vec4);

  
  // ----------- PLACING AND INDEXING -------------

  std::cout << "Testing placement and indexing:" << "\n\n";

  std::cout << "######### PLACEMENT ##########" << "\n\n";

  std::cout << "TEST 1: ";

  double pi_tmp[25] =
    {1.0, 3.0, 4.0, 6.0, 7.0,
     2.0, 3.0, 4.0, 5.0, 6.0,
     9.0, 8.0, 2.0, 6.0, 5.0,
     5.0, 6.0, 7.0, 8.0, 9.0,
     8.0, 7.0, 6.0, 5.0, 5.0};
  
  std::vector<double> pi_tmp_vec(pi_tmp, pi_tmp + sizeof(pi_tmp) / sizeof(pi_tmp[0]));
  DenseMatrix pi_sol_1 = DenseMatrix(5, 5, pi_tmp_vec);

  DenseMatrix pi_test_1 = DenseMatrix(A);
  pi_test_1.place(0, 3, 6.0);
  pi_test_1.place(2, 2, 2.0);

  passed_tests += test(pi_test_1, pi_sol_1);

  std::cout << "TEST 2: ";

  double pi_tmp2[25] =
    {1.0, 3.0, 4.0, 0.1, 7.0,
     2.0, 3.0, 4.0, 1.0/3.0, 6.0,
     9.0, 8.0, 7.0, 2.0, 5.0,
     5.0, 6.0, 7.0, 0.91, 9.0,
     8.0, 7.0, 6.0, 0.82, 5.0};
  
  std::vector<double> pi_tmp_vec2(pi_tmp2, pi_tmp2 + sizeof(pi_tmp2) / sizeof(pi_tmp2[0]));
  DenseMatrix pi_sol_2 = DenseMatrix(5, 5, pi_tmp_vec2);

  DenseMatrix pi_test_2 = DenseMatrix(A);
  pi_test_2.place(0, 5, 3, 4, v.asDenseMatrix());

  passed_tests += test(pi_test_2, pi_sol_2);
  
  std::cout << "\n######### INDEXING  ##########" << "\n\n";

  std::cout << "TEST 1: ";

  double pi_sol_3 = 8.0;
  double pi_test_3 = A(4, 0);

  if (pi_sol_3 == pi_test_3) {
    std::cout << "PASSED" << "\n";
    passed_tests++;
  }
  else {
    std::cout << "FAILED" << "\n\n";
    std::cout << pi_test_3 << " is not equal to: " << pi_sol_3 << "\n\n";
  }

  
  std::cout << "TEST 2: ";

  double pi_tmp4[5] =
    {3.0,
     3.0,
     8.0,
     6.0,
     7.0};
  
  std::vector<double> pi_tmp_vec4(pi_tmp4, pi_tmp4 + sizeof(pi_tmp4) / sizeof(pi_tmp4[0]));
  DenseVector pi_sol_4 = DenseVector(5, pi_tmp_vec4);

  DenseVector pi_test_4 = A(0, 5, 1, 2).asDenseVector();

  passed_tests += test(pi_test_4.asDenseMatrix(), pi_sol_4.asDenseMatrix());

  std::cout << "---------------------------------------" << "\n";

  // --------- ELEMENT-WISE OPERATIONS ------------

  std::cout << "\nTesting element-wise operations:" << "\n\n";

  std::cout << "########## SUMMATION ##########" << "\n\n";

  std::cout << "TEST 1: ";

  double ew_tmp[25] =
    {1.2, 6.22, 7.01, 28.0, 11.4,
     5.423, 6.2234, -0.234, 10.32, 11.4,
     10.012, -0.4321, 39.432, 48.43, 7.43,
     9.6, 15.89, 3.46, 13.23, 13.26,
     14.89, 11.56, 15.456, 13.36, -2.345};

  std::vector<double> ew_tmp_vec(ew_tmp, ew_tmp + sizeof(ew_tmp) / sizeof(ew_tmp[0]));
  DenseMatrix ew_sol = DenseMatrix(5, 5, ew_tmp_vec);

  DenseMatrix ew_test = A + B;

  passed_tests += test(ew_test, ew_sol);


  std::cout << "TEST 2: ";

  double ew_tmp2[5] =
    {1.12,
     0.583333333,
     34.3,
     4.202,
     1.912};

  std::vector<double> ew_tmp_vec2(ew_tmp2, ew_tmp2 + sizeof(ew_tmp2) / sizeof(ew_tmp2[0]));
  DenseMatrix ew_sol2 = DenseMatrix(5, 1, ew_tmp_vec2);

  DenseVector ew_test2 = v + w;

  passed_tests += test(ew_test2.asDenseMatrix(), ew_sol2);


  std::cout << "\n########## SUBTRACTION ##########" << "\n\n";

  std::cout << "TEST 1: ";

  double ew_tmp3[25] =
    {0.8, -0.22, 0.99, -18.0, 2.6,
     -1.423, -0.2234, 8.234, -0.32, 0.6,
     7.988, 16.4321, -25.432, -36.43, 2.57,
     0.4, -3.89, 10.54, 2.77, 4.74,
     1.11, 2.44, -3.456, -3.36, 12.345};

  std::vector<double> ew_tmp_vec3(ew_tmp3, ew_tmp3 + sizeof(ew_tmp3) / sizeof(ew_tmp3[0]));
  DenseMatrix ew_sol3 = DenseMatrix(5, 5, ew_tmp_vec3);

  DenseMatrix ew_test3 = A - B;

  passed_tests += test(ew_test3, ew_sol3);


  std::cout << "TEST 2: ";

  double ew_tmp4[5] =
    {-0.92,
     0.083333333,
     -30.3,
     -2.382,
     -0.272};

  std::vector<double> ew_tmp_vec4(ew_tmp4, ew_tmp4 + sizeof(ew_tmp4) / sizeof(ew_tmp4[0]));
  DenseMatrix ew_sol4 = DenseMatrix(5, 1, ew_tmp_vec4);

  DenseVector ew_test4 = v - w;

  passed_tests += test(ew_test4.asDenseMatrix(), ew_sol4);


  std::cout << "\n########## MULTIPLICATION ##########" << "\n\n";

  std::cout << "TEST 1: ";

  double ew_tmp5[25] =
    {0.2, 9.66, 12.04, 115.0, 30.8,
     6.846, 9.6702, -16.936, 26.6, 32.4,
     9.108, -67.4568, 227.024, 254.58, 12.15,
     23.0, 59.34, -24.78, 41.84, 38.34,
     55.12, 31.92, 56.736, 41.8, -36.725};

  std::vector<double> ew_tmp_vec5(ew_tmp5, ew_tmp5 + sizeof(ew_tmp5) / sizeof(ew_tmp5[0]));
  DenseMatrix ew_sol5 = DenseMatrix(5, 5, ew_tmp_vec5);

  DenseMatrix ew_test5 = A * B;

  passed_tests += test(ew_test5, ew_sol5);


  std::cout << "TEST 2: ";

  double ew_tmp6[5] =
    {0.102,
     0.083333333,
     64.6,
     2.99572,
     0.89544};

  std::vector<double> ew_tmp_vec6(ew_tmp6, ew_tmp6 + sizeof(ew_tmp6) / sizeof(ew_tmp6[0]));
  DenseMatrix ew_sol6 = DenseMatrix(5, 1, ew_tmp_vec6);

  DenseVector ew_test6 = v * w;

  passed_tests += test(ew_test6.asDenseMatrix(), ew_sol6);

   std::cout << "TEST 3: ";

  double ew_tmp7[25] =
    {0.2, 0.6, 0.8, 1.0, 1.4,
     0.4, 0.6, 0.8, 1.0, 1.2,
     1.8, 1.6, 1.4, 1.2, 1.0,
     1.0, 1.2, 1.4, 1.6, 1.8,
     1.6, 1.4, 1.2, 1.0, 1.0};

  std::vector<double> ew_tmp_vec7(ew_tmp7, ew_tmp7 + sizeof(ew_tmp7) / sizeof(ew_tmp7[0]));
  DenseMatrix ew_sol7 = DenseMatrix(5, 5, ew_tmp_vec7);

  DenseMatrix ew_test7 = A * 0.2;

  passed_tests += test(ew_test7, ew_sol7);


  std::cout << "TEST 4: ";

  double ew_tmp8[5] =
    {0.3,
     1.0,
     6.0,
     2.73,
     2.46};

  std::vector<double> ew_tmp_vec8(ew_tmp8, ew_tmp8 + sizeof(ew_tmp8) / sizeof(ew_tmp8[0]));
  DenseMatrix ew_sol8 = DenseMatrix(5, 1, ew_tmp_vec8);

  DenseVector ew_test8 = v * 3.0;

  passed_tests += test(ew_test8.asDenseMatrix(), ew_sol8);


  std::cout << "\n########## DIVISION ##########" << "\n\n";

  std::cout << "TEST 1: ";

  double ew_tmp9[25] =
    {5.0, 0.931677, 1.328903, 0.217391, 1.590909,
     0.584282, 0.930694, -0.944733, 0.939849, 1.111111,
     8.893280, -0.948755, 0.215836, 0.141409, 2.057613,
     1.086956, 0.606673, -1.977401, 1.529636, 2.112676,
     1.161103, 1.535087, 0.634517, 0.598086, -0.680735};

  std::vector<double> ew_tmp_vec9(ew_tmp9, ew_tmp9 + sizeof(ew_tmp9) / sizeof(ew_tmp9[0]));
  DenseMatrix ew_sol9 = DenseMatrix(5, 5, ew_tmp_vec9);

  DenseMatrix ew_test9 = A / B;

  passed_tests += test(ew_test9, ew_sol9);


  std::cout << "TEST 2: ";

  double ew_tmp10[5] =
    {0.098039,
     1.333333,
     0.061919,
     0.276427,
     0.750915};

  std::vector<double> ew_tmp_vec10(ew_tmp10, ew_tmp10 + sizeof(ew_tmp10) / sizeof(ew_tmp10[0]));
  DenseMatrix ew_sol10 = DenseMatrix(5, 1, ew_tmp_vec10);

  DenseVector ew_test10 = v / w;

  passed_tests += test(ew_test10.asDenseMatrix(), ew_sol10);

  
  std::cout << "TEST 3: ";
  
  DenseMatrix ew_test11 = A / 5.0;

  passed_tests += test(ew_test11, ew_sol7);


  std::cout << "TEST 4: ";

  DenseVector ew_test12 = v / (1.0 / 3.0);

  passed_tests += test(ew_test12.asDenseMatrix(), ew_sol8);

  std::cout << "---------------------------------------" << "\n";

  
  // --------- TRANSPOSE ------------

  std::cout << "\nTesting the transpose operation:" << "\n\n";

  std::cout << "TEST 1: ";

  double t_tmp[25] =
    {1.0, 2.0, 9.0, 5.0, 8.0,
     3.0, 3.0, 8.0, 6.0, 7.0,
     4.0, 4.0, 7.0, 7.0, 6.0,
     5.0, 5.0, 6.0, 8.0, 5.0,
     7.0, 6.0, 5.0, 9.0, 5.0};

  std::vector<double> t_tmp_vec(t_tmp, t_tmp + sizeof(t_tmp) / sizeof(t_tmp[0]));
  DenseMatrix t_sol = DenseMatrix(5, 5, t_tmp_vec);

  DenseMatrix t_test = A.T();

  passed_tests += test(t_test, t_sol);

  /*
  std::cout << "TEST 2: ";

  double t_tmp2[5] =
    {0.1, 1.0 / 3.0, 2.0, 0.91, 0.82};

  std::vector<double> t_tmp_vec2(t_tmp2, t_tmp2 + sizeof(t_tmp2) / sizeof(t_tmp2[0]));
  DenseMatrix t_sol2 = DenseMatrix(1, 5, t_tmp_vec2);

  DenseVector t_test2 = v;

  passed_tests += test(t_test2.asDenseMatrix(), t_sol2);
  */


  std::cout << "---------------------------------------" << "\n";

  
  // --------- MATRIX MULTIPLICATION ------------

  std::cout << "\nTesting matrix multiplication:" << "\n\n";

  std::cout << "TEST 1: ";

  double m_tmp[25] =
    {85.747, 60.5318, 168.528, 293.35, 0.205,
     79.057, 59.1918, 162.082, 307.99, 11.95,
     98.318, 77.8825, 246.282, 619.75, 88.645,
     127.432, 96.5757, 273.454, 561.01, 39.385,
     89.083, 69.9812, 218.614, 543.77, 72.155};

  std::vector<double> m_tmp_vec(m_tmp, m_tmp + sizeof(m_tmp) / sizeof(m_tmp[0]));
  DenseMatrix m_sol = DenseMatrix(5, 5, m_tmp_vec);

  DenseMatrix m_test = A.matmul(B);

  passed_tests += test(m_test, m_sol);


  std::cout << "TEST 2: ";

  double m_sol2 = 68.676493;

  double m_test2 = v.dot(w);

  if (fabs(m_sol2 - m_test2) < TOL) {
    std::cout << "PASSED" << "\n";
    passed_tests++;
  }
  else {
    std::cout << "FAILED" << "\n\n";
    std::cout << m_test2 << " is not equal to: " << m_sol2 << "\n\n";
  }

  /*
  std::cout << "TEST 3: ";

  double m_sol3 = 68.676493;

  DenseMatrix m_test3 = v.matmul(w);

  if (fabs(m_sol3 - m_test3.asDouble()) < TOL) {
    std::cout << "PASSED" << "\n";
    passed_tests++;
  }
  else {
    std::cout << "FAILED" << "\n\n";
    std::cout << m_test3.asDouble() << " is not equal to: " << m_sol3 << "\n\n";
  }
  */

  std::cout << "TEST 5: ";

  double m_tmp5[5] =
    {19.39,
     18.67,
     27.126667,
     31.16,
     23.7833333};

  std::vector<double> m_tmp_vec5(m_tmp5, m_tmp5 + sizeof(m_tmp5) / sizeof(m_tmp5[0]));
  DenseMatrix m_sol5 = DenseMatrix(5, 1, m_tmp_vec5);
  
  DenseVector m_test5 = A.matmul(v);

  passed_tests += test(m_test5.asDenseMatrix(), m_sol5);

  std::cout << "---------------------------------------" << "\n\n";

  std::cout << "TESTS PASSED: " << passed_tests << "/" << N_TESTS << "\n\n";
  
  
  clock_t end = clock();
  std::cout << "Total time spent in tests: " << ((double)(end - start)) / CLOCKS_PER_SEC << " seconds\n";
}
