#include "../lalib/src/denseMatrix.hpp"
#include "../lalib/src/denseVector.hpp"
#include "../lalib/src/declare_lalib.hpp"


int main() {
    // Generate initial matrices
    double mat[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    DenseMatrix A = DenseMatrix(3, 3, mat);

    double vec[3] = {9.0, 6.0, 3.0};
    DenseMatrix b = DenseMatrix(3, 1, vec);

    DenseMatrix x0 = DenseMatrix(3, 1);

    DenseMatrix x_k = jacobiSolve<DenseMatrix>(A, x0, b);

    std::cout << "\nFound solution:\n" << x_k << "\n";

    return 0;
}