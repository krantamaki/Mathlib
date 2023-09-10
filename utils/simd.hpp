#ifndef SIMD_HPP
#define SIMD_HPP


// Define the size of the SIMD vector registers
#define SIMD_SIZE 32  // Bytes


namespace utils {

  // Templated struct for SIMD vectors
  template <class type>
  struct simd {
    typedef type data __attribute__ ((__vector_size__ (SIMD_SIZE)));
  };


  // Template function that chooses either the vectorized or non-vectorized datatype based on a boolean flag.
  // Does the choice at compile time and is thus similar to std::conditional_t, but doesn't evaluate the 
  // unchosen option.
  template <class type, bool vectorize>
  auto _choose_simd() {
    if constexpr (vectorize) {
      typename simd<type>::data a;
      return a;
    }
    else {
      type a;
      return a;
    }
  }

}


#endif