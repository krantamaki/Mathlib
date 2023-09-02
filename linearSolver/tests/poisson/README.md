# FEM system for a Poisson problem on a simple 3D geometry

Included are the coefficient matrix (linsys_a.dat) and RHS vector (linsys_b.dat) formed by Elmer finite element method solver (http://www.elmerfem.org/blog/) for a very simple 3D problem. For more information see e.g. https://github.com/ElmerCSC/elmer-linsys/tree/main/results/Poisson-WinkelStructured

To run this test case compile the solver in root mathlib directory and call:
> ./solver.o solver/tests/poisson/poisson_config.txt