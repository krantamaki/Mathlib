#ifndef TESTING_HPP
#define TESTING_HPP

#include "messaging.hpp"
#include "general.hpp"


#ifndef TEST_EQ
#define TEST_EQ(sol, ans) { utils::_test(sol == ans, __func__, utils::_format("FAILED - ", sol, " IS NOT EQUAL TO ", ans)) }
#endif

#ifndef TEST_INEQ
#define TEST_INEQ(sol, ans) { utils::_test(sol != ans, __func__, utils::_format("FAILED - ", sol, " IS EQUAL TO ", ans)) }
#endif

#ifndef TEST
#define TEST(statement) { utils::_test(statement, __func__) } 
#endif


namespace utils {
  
  // Function for testing the validity of an input. Meant to be called by TEST* macros
  inline int _test(bool statement, const char* func, std::string _failureMsg = "FAILED") {
    if (statement) {
      _infoMsg("PASSED", func, true);
      
      return 1;
    }
    else {
      _infoMsg(_failureMsg, func, true);

      return 0;
    }
  }

}


#endif