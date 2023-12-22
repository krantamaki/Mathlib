#ifndef POISSON2D_HPP
#define POISSON2D_HPP


#include "declare_poissonFDM2D.hpp"


/**
 * @brief Finite Difference Method solver for two dimensional heat equation
 * 
 * This is an extension of the most basic Poisson solver as given as an example
 * in examples/poisson_fd.cpp
 * 
 * The heat equation is of form:
 * 
 * \f{eqnarray*}{
 *    \Delta u &=& \frac{\partial u}{\partial t} \mbox{ in } \Omega \\
 *           u &=& f \mbox{ on } \partial \Omega
 * \f}
 * 
 * where f is some Dirichlet boundary condition.
 * 
 * We will assue that the domain \Omega is a rectangle with user specified height and
 * width.
 * 
 * Additionally, we assume that the boundary conditions are some constant temperatures
 * on the edges that the user can specify.
 */


namespace poissonFDM2D {

  // Handy index map
  inline int ij(int i, int j, int N) {
    return i * N + j;
  }


  // We will solve the system at each time point. The solution to the system at
  // time t_i is the temperature at time t_{i-1}. As that is known we only need
  // to form the coefficient matrix here
  template<class type, bool vectorize, bool sparse>
  lalib::Matrix<type, vectorize, sparse> formSystem(type height, type width, int nH, int nW, type dt, type alpha) {

    if (height <= (type)0.0 || width <= (type)0.0 || nH <= 0 || nW <= 0 || dt <= (type)0.0) {
      ERROR("The parameter values need to be positive!");
    }

    type dH = height / ((type)nH);  // Step size height-wise
    type dW = width / ((type)nW);  // Step size width-wise

    // Precompute inverses for efficiency
    type isq_dH = 1.0 / (dH * dH);
    type isq_dW = 1.0 / (dW * dW);

    // Arrays that will define the CRS structure of the coefficient matrix
    std::vector<type> vals;
    std::vector<int> colInds;
    std::vector<int> rowPtrs;
    rowPtrs.push_back(0);

    // Form the coefficient matrix
    for (int i = 0; i < nH; i++) {  // Loop over the height
      for (int j = 0; j < nW; j++) {  // Loop over the width

        int row = ij(i, j, nW);  // The row will be placing values on in the coefficient matrix

        // Point on the boundary. We don't need to differentiate between boundaries here
        if (i == 0 || i == nH - 1 || j == 0 || j == nW - 1) {
          vals.push_back((type)1.0);
          colInds.push_back(row);
          rowPtrs.push_back(rowPtrs.back() + 1);
        }
        // Point in the interior
        else {
          vals.insert(vals.end(), {-alpha * dt * isq_dH,
                                   -alpha * dt * isq_dW,
                                   1 + 2 * alpha * dt * isq_dH + 2 * alpha * dt * isq_dW,
                                   -alpha * dt * isq_dW,
                                   -alpha * dt * isq_dH});
          colInds.insert(colInds.end(), {ij(i - 1, j, nW), 
                                         ij(i, j - 1, nW), 
                                         row, 
                                         ij(i, j + 1, nW), 
                                         ij(i + 1, j, nW)});
          rowPtrs.push_back(rowPtrs.back() + 5);
        }
      }
    }

    int dim = nH * nW;
    lalib::Matrix<type, vectorize, sparse> A = lalib::Matrix<type, vectorize, sparse>(dim, dim, vals, colInds, rowPtrs);

    INFO("Successfully formed the linear system");

    return A;
  }


  template<class type, bool vectorize, bool sparse>
  bool poissonFDM2D(std::map<std::string, std::any> config_map) {

    using namespace std;

    // Get required values from configuration map
    string save_dir = any_cast<string>(config_map["save_dir"]);
    string save_name = any_cast<string>(config_map["save_name"]);
    string method = any_cast<string>(config_map["method"]);
    type convergence_tolerance = any_cast<double>(config_map["convergence_tolerance"]);
    bool stop_unconverged = any_cast<bool>(config_map["stop_unconverged"]);
    int max_iter = any_cast<int>(config_map["max_iter"]);
    type lowerBound = any_cast<double>(config_map["lower_bound"]);
    type upperBound = any_cast<double>(config_map["upper_bound"]);
    type rightBound = any_cast<double>(config_map["right_bound"]);
    type leftBound = any_cast<double>(config_map["left_bound"]);
    type initial_temp = any_cast<double>(config_map["initial_temp"]);
    type height = any_cast<double>(config_map["height"]);
    type width = any_cast<double>(config_map["width"]);
    type duration = any_cast<double>(config_map["duration"]);
    int n_height_points = any_cast<int>(config_map["n_height_points"]);
    int n_width_points = any_cast<int>(config_map["n_width_points"]);
    int n_time_points = any_cast<int>(config_map["n_time_points"]);
    type alpha = any_cast<double>(config_map["thermal_diffusivity"]);


    // Check if the directory in which solutions are to be stored exists
    // and if not create it
    if (!fs::exists(save_dir)) {
      fs::create_directories(save_dir);
    } 


    int dim = n_height_points * n_width_points;
    bool success = true;
    
    type dH = height / ((type)n_height_points);  // Step size height-wise
    type dW = width / ((type)n_width_points);  // Step size width-wise


    type dt;

    if (n_time_points != -1) {
      // Check that the time step is stable
      dt = duration / n_time_points;
      if (dt > (dH * dH + dW * dW) / (4 * alpha)) {
        WARNING("The given number of points might not lead to a stable solution!");
      }
    }
    else {
      // Compute a stable time step and associated number steps
      // Solution is stable if dt <= dx^2 / 2 for 1D heat equation
      // Define the time step as:
      dt = (dH * dH + dW * dW) / (5.0 * alpha);
      n_time_points = (int)(duration / dt);
    }

    INFO(utils::_format("Using time step of: ", dt, " with total of: ", n_time_points, " time points"));
    INFO(utils::_format("Dimension of the linear systems will be: ", dim, " x ", dim));

    // Set up the initial solution
    lalib::Vector<type, vectorize> init = lalib::Vector<type, vectorize>(dim, initial_temp);

    // Add boundary conditions to it
    for (int i = 0; i < n_height_points; i++) {
      // Left boundary
      init.place(ij(i, 0, n_width_points), leftBound);
      // Right boundary
      init.place(ij(i, n_width_points - 1, n_width_points), rightBound);
    }

    for (int j = 0; j < n_width_points; j++) {
      // Lower boundary
      init.place(ij(0, j, n_width_points), lowerBound);
      // Upper boundary
      init.place(ij(n_height_points - 1, j, n_width_points), upperBound);
    }


    // Save the initial solution
    INFO(utils::_format("Solutions will be saved in directory ", save_dir, " using name(s) ", save_name, "_ti.dat where i is the time step."));

    if (!init.save(utils::_format(save_dir, "/", save_name, "_t0.dat"))) {
      ERROR("Couldn't save the result vector!");
    }


    // Define the inital RHS vector
    lalib::Vector<type, vectorize> b = lalib::Vector<type, vectorize>(init);

    // Form the coefficient matrix. This needs to be done only once
    INFO("Forming the coefficient matrix. This needs to be done only once...");
    lalib::Matrix<type, vectorize, sparse> A = formSystem<type, vectorize, sparse>(height, width, n_height_points, n_width_points, dt, alpha);


    // Time the solution
    auto start = chrono::high_resolution_clock::now();

    // Loop over the timesteps, form linear systems and solve them
    for (int t_i = 1; t_i <= n_time_points; t_i++) {

      LOWPRIORITY(utils::_format("Solving the system at time: ", t_i * dt, " (", t_i, "/", n_time_points, " done)"));

      // Form the initial guess. We will use a zero vector as initial guess
      lalib::Vector<type, vectorize> x0 = lalib::Vector<type, vectorize>(dim);

      lalib::Vector<type, vectorize> ret;

      // Solve the system
      if (method == "cg") {
        LOWPRIORITY("Calling the Conjugate Gradient method");
        LOWPRIORITY("");
        ret = lalib::cgSolve<type, vectorize, sparse>(A, x0, b, max_iter, convergence_tolerance);
        LOWPRIORITY("");
      }
      else if (method == "cgnr") {
        LOWPRIORITY("Calling the Conjugate Gradient on normal equations method");
        LOWPRIORITY("");
        ret = lalib::cgnrSolve<type, vectorize, sparse>(A, x0, b, max_iter, convergence_tolerance);
        LOWPRIORITY("");
      }
      else {
       ERROR("Improper solver provided!");
      }

      // Compute the residual norm
      type res_norm = (A.matmul(ret) - b).norm();
      LOWPRIORITY(utils::_format("Residual norm: ", res_norm));

      if (res_norm < convergence_tolerance) {
        LOWPRIORITY("Residual within tolerance. Continuing to next time step");
      }
      else {
        if (stop_unconverged) {
          ERROR("Solver did not converge to wanted tolerance. Stopping execution...");
        }
        else {
          WARNING("Solver did not converge to wanted tolerance. Continuing...");
          success = false;
        }
      }

      // Save the solution
      if (!ret.save(utils::_format(save_dir, "/", save_name, "_t", t_i, ".dat"))) {
        ERROR("Couldn't save the result vector!");
      }

      // Update RHS vector
      b = ret;
    }

    auto end = chrono::high_resolution_clock::now();


    // Compute the passed time
    auto time = chrono::duration_cast<chrono::milliseconds>(end - start);

    INFO(utils::_format("Time taken to form the solution: ", time.count(), " milliseconds"));

    return success;
  }

}


#endif