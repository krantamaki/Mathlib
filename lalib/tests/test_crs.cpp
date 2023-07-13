#include "../src/declare_lalib.hpp"
#include "../src/crs/crsMatrix.hpp"


#define TOL 0.0001
#define N_TESTS 21


using namespace lalib;


/*
  General tests for the dense representation of matrices
  Compile in lalib directory with: g++ -mavx -fopenmp -Wall src/crs/crsMatrix.cpp src/crs/crsMatmul.cpp tests/test_crs.cpp -lm -o test_crs.o 
  Run with: ./test_crs.o
*/


int test(CRSMatrix testMatrix, CRSMatrix solMatrix) {
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

  int passed_tests = 0;

  clock_t start = clock();
  std::cout << "\nTesting the CRS matrix implementation" << "\n";
  std::cout << "---------------------------------------" << "\n\n";


  // Initialize some matrices and vectors
  
  double tmp[25] =
    {1.0, 3.0, 0.0, 0.0, 0.0,
     2.0, 0.0, 4.0, 0.0, 0.0,
     0.0, 8.0, 7.0, 6.0, 0.0,
     0.0, 0.0, 0.0, 8.0, 9.0,
     8.0, 0.0, 6.0, 0.0, 5.0};
  
  std::vector<double> tmp_vec(tmp, tmp + sizeof(tmp) / sizeof(tmp[0]));
  CRSMatrix A = CRSMatrix(5, 5, tmp_vec);

  double tmp2[25] =
    {0.0, 3.22, 0.0, 0.0, 0.0,
     3.423, 3.2234, 0.0, 5.32, 0.0,
     0.0, 0.0, 0.432, 0.0, 0.0,
     0.0, 9.89, -3.54, 0.0, 4.26,
     0.0, 4.56, 0.0, 8.36, -7.345};
  
  std::vector<double> tmp_vec2(tmp2, tmp2 + sizeof(tmp2) / sizeof(tmp2[0]));
  CRSMatrix B = CRSMatrix(5, 5, tmp_vec2);
  
  double tmp3[5] =
    {0.0,
     1.0/3.0,
     0.0,
     0.91,
     0.82};

  std::vector<double> tmp_vec3(tmp3, tmp3 + sizeof(tmp3) / sizeof(tmp3[0]));
  CRSMatrix v = CRSMatrix(5, 1, tmp_vec3);

  double tmp4[5] =
    {0.0,
     1.0/4.0,
     32.3,
     3.292,
     0.0};

  std::vector<double> tmp_vec4(tmp4, tmp4 + sizeof(tmp4) / sizeof(tmp4[0]));
  CRSMatrix w = CRSMatrix(5, 1, tmp_vec4);

  
  // ----------- SAVING AND LOADING --------------

  std::cout << "Testing saving and loading:" << "\n\n";

  std::cout << "######### SAVING ##########" << "\n\n";

  bool success1 = A.save("tmp.txt");
  std::cout << "TEST 1: ";
  if (success1) {
    std::cout << "PASSED" << "\n";
    passed_tests += 1;
  }
  else {
    std::cout << "FAILED" << "\n\n";
    std::cout << "Something went wrong with saving a square matrix" << "\n\n";
  }
  
  
  bool success2 = v.save("tmp2.txt");
  std::cout << "TEST 2: ";
  if (success2) {
    std::cout << "PASSED" << "\n";
    passed_tests += 1;
  }
  else {
    std::cout << "FAILED" << "\n\n";
    std::cout << "Something went wrong with saving a column matrix" << "\n\n";
  }
  
  std::cout << "######### LOADING ##########" << "\n\n";
  CRSMatrix A_loaded = CRSMatrix("tmp.txt");
  std::cout << "TEST 3: ";
  passed_tests += test(A_loaded, A);

  CRSMatrix v_loaded = CRSMatrix("tmp2.txt");
  std::cout << "TEST 4: ";
  passed_tests += test(v_loaded, v);

  std::cout << "---------------------------------------" << "\n";


  // ----------- PLACING AND INDEXING -------------

  std::cout << "Testing placement and indexing:" << "\n\n";

  std::cout << "######### PLACEMENT ##########" << "\n\n";

  std::cout << "TEST 1: ";

  double pi_tmp[25] =
    {1.0, 3.0, 0.0, 0.0, 0.0,
     2.0, 0.0, -1.0, 0.0, 0.0,
     0.0, 8.0, 7.0, 6.0, 0.0,
     0.0, -1.0, 0.0, 8.0, 9.0,
     8.0, 0.0, 6.0, 0.0, 5.0};
  
  std::vector<double> pi_tmp_vec(pi_tmp, pi_tmp + sizeof(pi_tmp) / sizeof(pi_tmp[0]));
  CRSMatrix pi_sol = CRSMatrix(5, 5, pi_tmp_vec);

  CRSMatrix pi_test = CRSMatrix(A);
  pi_test.place(1, 2, -1.0);
  pi_test.place(3, 1, -1.0);

  passed_tests += test(pi_test, pi_sol);

  /* NOT YET IMPLEMENTED
  std::cout << "TEST 2: ";

  double pi_tmp2[25] =
    {1.0, 3.0, 4.0, 0.0, 7.0,
     2.0, 3.0, 4.0, 1.0/3.0, 6.0,
     9.0, 8.0, 7.0, 0.0, 5.0,
     5.0, 6.0, 7.0, 0.91, 9.0,
     8.0, 7.0, 6.0, 0.82, 5.0};
  
  std::vector<double> pi_tmp_vec2(pi_tmp2, pi_tmp2 + sizeof(pi_tmp2) / sizeof(pi_tmp2[0]));
  CRSMatrix pi_sol2 = CRSMatrix(5, 5, pi_tmp_vec2);

  CRSMatrix pi_test2 = CRSMatrix(A);
  pi_test2.place(0, 5, 3, 4, v);

  passed_tests += test(pi_test2, pi_sol2);
  */
  
  std::cout << "\n######### INDEXING  ##########" << "\n\n";

  std::cout << "TEST 1: ";

  double pi_sol3 = 8.0;
  double pi_test3 = A(3, 3);

  if (pi_sol3 == pi_test3) {
    std::cout << "PASSED" << "\n";
    passed_tests++;
  }
  else {
    std::cout << "FAILED" << "\n\n";
    std::cout << pi_test3 << " is not equal to: " << pi_sol3 << "\n\n";
  }

  /* NOT YET IMPLEMENTED
  std::cout << "TEST 2: ";

  double pi_tmp4[5] =
    {3.0,
     0.0,
     8.0,
     0.0,
     0.0};
  
  std::vector<double> pi_tmp_vec4(pi_tmp4, pi_tmp4 + sizeof(pi_tmp4) / sizeof(pi_tmp4[0]));
  CRSMatrix pi_sol4 = CRSMatrix(5, 1, pi_tmp_vec4);

  CRSMatrix pi_test4 = A(0, 5, 1, 2);

  passed_tests += test(pi_test4, pi_sol4);
  */
  
  std::cout << "---------------------------------------" << "\n";


  // --------- ELEMENT-WISE OPERATIONS ------------

  std::cout << "\nTesting element-wise operations:" << "\n\n";

  std::cout << "########## SUMMATION ##########" << "\n\n";

  std::cout << "TEST 1: ";

  double ew_tmp[25] =
    {1.0, 6.22, 0.0, 0.0, 0.0,
     5.423, 3.2234, 4.0, 5.32, 0.0,
     0.0, 8.0, 7.432, 6.0, 0.0,
     0.0, 9.89, -3.54, 8.0, 13.26,
     8.0, 4.56, 6.0, 8.36, -2.345};

  std::vector<double> ew_tmp_vec(ew_tmp, ew_tmp + sizeof(ew_tmp) / sizeof(ew_tmp[0]));
  CRSMatrix ew_sol = CRSMatrix(5, 5, ew_tmp_vec);

  CRSMatrix ew_test = A + B;

  passed_tests += test(ew_test, ew_sol);


  std::cout << "TEST 2: ";

  double ew_tmp2[5] =
    {0.,
     0.583333333,
     32.3,
     4.202,
     0.82};

  std::vector<double> ew_tmp_vec2(ew_tmp2, ew_tmp2 + sizeof(ew_tmp2) / sizeof(ew_tmp2[0]));
  CRSMatrix ew_sol2 = CRSMatrix(5, 1, ew_tmp_vec2);

  CRSMatrix ew_test2 = v + w;

  passed_tests += test(ew_test2, ew_sol2);


  std::cout << "\n########## SUBTRACTION ##########" << "\n\n";

  std::cout << "TEST 1: ";

  double ew_tmp3[25] =
    {1.0, -0.22, 0.0, 0.0, 0.0,
     -1.423, -3.2234, 4.0, -5.32, 0.0,
     0.0, 8.0, 6.568, 6.0, 0.0,
     0.0, -9.89, 3.54, 8.0, 4.74,
     8.0, -4.56, 6.0, -8.36, 12.345};

  std::vector<double> ew_tmp_vec3(ew_tmp3, ew_tmp3 + sizeof(ew_tmp3) / sizeof(ew_tmp3[0]));
  CRSMatrix ew_sol3 = CRSMatrix(5, 5, ew_tmp_vec3);

  CRSMatrix ew_test3 = A - B;

  passed_tests += test(ew_test3, ew_sol3);


  std::cout << "TEST 2: ";

  double ew_tmp4[5] =
    {0.0,
     0.083333333,
     -32.3,
     -2.382,
     0.82};

  std::vector<double> ew_tmp_vec4(ew_tmp4, ew_tmp4 + sizeof(ew_tmp4) / sizeof(ew_tmp4[0]));
  CRSMatrix ew_sol4 = CRSMatrix(5, 1, ew_tmp_vec4);

  CRSMatrix ew_test4 = v - w;

  passed_tests += test(ew_test4, ew_sol4);


  std::cout << "\n########## MULTIPLICATION ##########" << "\n\n";

  std::cout << "TEST 1: ";

  double ew_tmp5[25] =
    {0.0, 9.66, 0.0, 0.0, 0.0,
     6.846, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 3.024, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 38.34,
     0.0, 0.0, 0.0, 0.0, -36.725};

  std::vector<double> ew_tmp_vec5(ew_tmp5, ew_tmp5 + sizeof(ew_tmp5) / sizeof(ew_tmp5[0]));
  CRSMatrix ew_sol5 = CRSMatrix(5, 5, ew_tmp_vec5);

  CRSMatrix ew_test5 = A * B;

  passed_tests += test(ew_test5, ew_sol5);


  std::cout << "TEST 2: ";

  double ew_tmp6[5] =
    {0.0,
     0.083333333,
     0.0,
     2.99572,
     0.0};

  std::vector<double> ew_tmp_vec6(ew_tmp6, ew_tmp6 + sizeof(ew_tmp6) / sizeof(ew_tmp6[0]));
  CRSMatrix ew_sol6 = CRSMatrix(5, 1, ew_tmp_vec6);

  CRSMatrix ew_test6 = v * w;

  passed_tests += test(ew_test6, ew_sol6);

   std::cout << "TEST 3: ";

  double ew_tmp7[25] =
    {0.2, 0.6, 0.0, 0.0, 0.0,
     0.4, 0.0, 0.8, 0.0, 0.0,
     0.0, 1.6, 1.4, 1.2, 0.0,
     0.0, 0.0, 0.0, 1.6, 1.8,
     1.6, 0.0, 1.2, 0.0, 1.0};

  std::vector<double> ew_tmp_vec7(ew_tmp7, ew_tmp7 + sizeof(ew_tmp7) / sizeof(ew_tmp7[0]));
  CRSMatrix ew_sol7 = CRSMatrix(5, 5, ew_tmp_vec7);

  CRSMatrix ew_test7 = A * 0.2;

  passed_tests += test(ew_test7, ew_sol7);


  std::cout << "TEST 4: ";

  double ew_tmp8[5] =
    {0.0,
     1.0,
     0.0,
     2.73,
     2.46};

  std::vector<double> ew_tmp_vec8(ew_tmp8, ew_tmp8 + sizeof(ew_tmp8) / sizeof(ew_tmp8[0]));
  CRSMatrix ew_sol8 = CRSMatrix(5, 1, ew_tmp_vec8);

  CRSMatrix ew_test8 = v * 3.0;

  passed_tests += test(ew_test8, ew_sol8);


  std::cout << "\n########## DIVISION ##########" << "\n\n";

  std::cout << "TEST 1: ";

  double ew_tmp9[25] =
    {1.0 / 0.0, 0.931677, 0.0 / 0.0, 0.0 / 0.0, 0.0 / 0.0,
     0.584282, 0.0, 1.0 / 0.0, 0.0, 0.0 / 0.0,
     0.0 / 0.0, 1.0 / 0.0, 16.203703, 1.0 / 0.0, 0.0 / 0.0,
     0.0 / 0.0, 0.0, 0.0, 1.0 / 0.0, 2.112676,
     1.0 / 0.0, 0.0, 1.0 / 0.0, 0.0, -0.680735};

  std::vector<double> ew_tmp_vec9(ew_tmp9, ew_tmp9 + sizeof(ew_tmp9) / sizeof(ew_tmp9[0]));
  CRSMatrix ew_sol9 = CRSMatrix(5, 5, ew_tmp_vec9);

  CRSMatrix ew_test9 = A / B;

  passed_tests += test(ew_test9, ew_sol9);


  std::cout << "TEST 2: ";

  double ew_tmp10[5] =
    {0.0 / 0.0,
     1.333333,
     0.0,
     0.276427,
     1.0 / 0.0};

  std::vector<double> ew_tmp_vec10(ew_tmp10, ew_tmp10 + sizeof(ew_tmp10) / sizeof(ew_tmp10[0]));
  CRSMatrix ew_sol10 = CRSMatrix(5, 1, ew_tmp_vec10);

  CRSMatrix ew_test10 = v / w;

  passed_tests += test(ew_test10, ew_sol10);

  /* NOT YET IMPLEMENTED
  std::cout << "TEST 3: ";
  
  CRSMatrix ew_test11 = A / 5.0;

  passed_tests += test(ew_test11, ew_sol7);


  std::cout << "TEST 4: ";

  CRSMatrix ew_test12 = v / (1.0 / 3.0);

  passed_tests += test(ew_test12, ew_sol8);
  */
  
  std::cout << "---------------------------------------" << "\n";


  // --------- TRANSPOSE ------------

  std::cout << "\nTesting the transpose operation:" << "\n\n";

  std::cout << "TEST 1: ";

  double t_tmp[25] =
    {1.0, 2.0, 0.0, 0.0, 8.0,
     3.0, 0.0, 8.0, 0.0, 0.0,
     0.0, 4.0, 7.0, 0.0, 6.0,
     0.0, 0.0, 6.0, 8.0, 0.0,
     0.0, 0.0, 0.0, 9.0, 5.0};

  std::vector<double> t_tmp_vec(t_tmp, t_tmp + sizeof(t_tmp) / sizeof(t_tmp[0]));
  CRSMatrix t_sol = CRSMatrix(5, 5, t_tmp_vec);

  CRSMatrix t_test = A.T();

  passed_tests += test(t_test, t_sol);


  std::cout << "TEST 2: ";

  double t_tmp2[5] =
    {0.0, 1.0 / 3.0, 0.0, 0.91, 0.82};

  std::vector<double> t_tmp_vec2(t_tmp2, t_tmp2 + sizeof(t_tmp2) / sizeof(t_tmp2[0]));
  CRSMatrix t_sol2 = CRSMatrix(1, 5, t_tmp_vec2);

  CRSMatrix t_test2 = v.T();

  passed_tests += test(t_test2, t_sol2);

  std::cout << "---------------------------------------" << "\n";


  // --------- MATRIX MULTIPLICATION ------------

  std::cout << "\nTesting matrix multiplication:" << "\n\n";

  std::cout << "TEST 1: ";

  double m_tmp[25] =
    {10.269, 12.8902, 0.0, 15.96, 0.0,
     0.0, 6.44, 1.728, 0.0, 0.0,
     27.384, 85.1272, -18.216, 42.56, 25.56,
     0.0, 120.16, -28.32, 75.24, -32.025,
     0.0, 48.56, 2.592, 41.8, -36.725};

  std::vector<double> m_tmp_vec(m_tmp, m_tmp + sizeof(m_tmp) / sizeof(m_tmp[0]));
  CRSMatrix m_sol = CRSMatrix(5, 5, m_tmp_vec);

  CRSMatrix m_test = A.matmul(B);

  passed_tests += test(m_test, m_sol);


  /* NOT YET IMPLEMENTED
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
  */

  std::cout << "TEST 3: ";

  double m_sol3 = 3.0790533;

  CRSMatrix m_test3 = (v.T()).matmul(w);

  if (fabs(m_sol3 - m_test3.asDouble()) < TOL) {
    std::cout << "PASSED" << "\n";
    passed_tests++;
  }
  else {
    std::cout << "FAILED" << "\n\n";
    std::cout << m_test3.asDouble() << " is not equal to: " << m_sol3 << "\n\n";
  }


  std::cout << "TEST 5: ";

  double m_tmp5[5] =
    {1.0,
     0.0,
     8.126667,
     14.66,
     4.1};

  std::vector<double> m_tmp_vec5(m_tmp5, m_tmp5 + sizeof(m_tmp5) / sizeof(m_tmp5[0]));
  CRSMatrix m_sol5 = CRSMatrix(5, 1, m_tmp_vec5);
  
  CRSMatrix m_test5 = A.matmul(v);

  passed_tests += test(m_test5, m_sol5);

  std::cout << "---------------------------------------" << "\n\n";

  
  std::cout << "TESTS PASSED: " << passed_tests << "/" << N_TESTS << "\n\n";
  
  
  clock_t end = clock();
  std::cout << "Total time spent in tests: " << ((double)(end - start)) / CLOCKS_PER_SEC << " seconds\n";
}
