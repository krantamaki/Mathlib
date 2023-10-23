#ifndef MESSAGING_HPP
#define MESSAGING_HPP


#include <iostream>
#include <string>
#include <fstream>
#include <sstream>


#ifndef BASE_VERBOSITY
#define BASE_VERBOSITY 3
#endif

#ifndef EXIT_WITH_WARNING
#define EXIT_WITH_WARNING false
#endif


namespace utils {

  /*
  * Function that defines and returns the verbosity level
  * Note that verbosity will have 3 levels:
  *   1: Error messages
  *   2: 1 warning messages
  *   3: 2 and base info messages
  *   4: 3 and iteration messages
  *   5: Everything
  */
  inline int verbosity(int _verbosity = BASE_VERBOSITY) {
    if (_verbosity < 0 || _verbosity > 5) {
      _verbosity = 1;
    }
    static int set_verbosity = _verbosity;
    return set_verbosity;
  }


  inline bool exit_with_warning(bool _exit_flag = EXIT_WITH_WARNING) {
    static bool exit_flag = _exit_flag;
    return exit_flag;
  } 


  // Function that generates and throws a more descriptive error message
  inline void _errorMsg(const std::string& msg, const char* file, const char* func, int line) {
    std::ostringstream msgStream;

    msgStream << "\n" << "ERROR: In file " << file << " at function " << func << " on line " << line << " : " << msg;
    std::string msgString = msgStream.str();
    std::cout << msgString << "\n\n" << std::endl;

    throw std::runtime_error(msgString);
  }


  // Function that forms and prints a warning message
  inline void _warningMsg(const std::string& msg, const char* func, bool always_print=false) {
    if (always_print) {
      std::cout << func << ": " << "WARNING! " << msg << std::endl;
    }
    if (exit_with_warning()) {
      std::ostringstream msgStream;
      
      msgStream << func << ": " << "WARNING! " << msg << " - EXITING";
      std::string msgString = msgStream.str();
      std::cout << msgString << std::endl;

      throw std::runtime_error(msgString);
    }
    if (verbosity() > 1) {
      std::cout << func << ": " << "WARNING! " << msg << std::endl;
    }
  }


  // Function that forms and prints an iteration message
  inline void _iterMsg(int iter, double norm, const char* func) {
    if (verbosity() > 3) {
      std::cout << func << ": " << "Iter " << iter << " - Norm " << norm << std::endl;
    }
  }


  // Function that forms and prints an information message
  inline void _infoMsg(const std::string& msg, const char* func, bool always_print=false) {
    if (always_print) {
      std::cout << func << ": " << msg << std::endl;
      return;
    }
    if (verbosity() > 2) {
      std::cout << func << ": " << msg << std::endl;
    }
  }


  // Function that forms and prints a low priority info message
  inline void _lowPriorityMsg(const std::string& msg, const char* func) {
    if (verbosity() > 4) {
      std::cout << func << ": " << msg << std::endl;
    }
  }

}


#endif
