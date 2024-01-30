#include "matrix.hh"



int main()
{
    auto&& mat = Matrix<int>::rand_mat(3, 4);
    mat.print();

    std::cout << "---------" << std::endl;
    auto&& transposed = mat.transpose();
    transposed.print();
}

