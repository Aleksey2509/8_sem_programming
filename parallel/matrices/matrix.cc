#include "matrix.hh"
#include "omp.h"
#include <chrono>
#include <ctime>
#include <ios>
#include <iostream>
#include <cstdlib>
#include <string>

constexpr int MAT_SIZE = 512;

template <typename Func>
void check_method(Matrix<int>& A, Matrix<int>& B, Func f, Matrix<int>* C_orig = nullptr)
{
    auto begin = std::chrono::high_resolution_clock::now();
    auto result = f(A, B);
    auto end = std::chrono::high_resolution_clock::now();

    // std::cout << "Result" << std::endl;
    // result.print();

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

    if (C_orig)
        std::cout << std::boolalpha << (result == *C_orig) << std::endl;
}
template <typename Func>
void time_method(int size, Func f)
{    
    auto A = Matrix<int>::rand_mat(size, size);
    auto B = Matrix<int>::rand_mat(size, size);
 
    auto begin = std::chrono::high_resolution_clock::now();
    auto result = f(A, B);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
}

void test_everything(const Matrix<int>& A, const Matrix<int>& B)
{
    std::cout << "Baseline" << std::endl;
    auto C_baseline = basic_mult<int>(A, B); 
    std::cout << "Optimized" << std::endl;
    auto C_optimized = optimized_mult<int>(A, B);

    // C_baseline.print();
    // C_optimized.print();

    std::cout << std::boolalpha << (C_baseline == C_optimized) << std::endl;
}

void compare_par_nonpar(int size)
{    
    auto A = Matrix<int>::rand_mat(size, size);
    auto B = Matrix<int>::rand_mat(size, size);
      
    printf("Start\n");
    
    printf("Optimized par mult\n");
    time_method(size, optimized_par_mult<int>);

    
    printf("Done");
}

void test_new_stuff()
{
    auto A = Matrix<int>::rand_mat(MAT_SIZE, MAT_SIZE);
    auto B = Matrix<int>::rand_mat(MAT_SIZE, MAT_SIZE);
    std::cout << "A" << std::endl;
    A.print();
    std::cout << "B" << std::endl;
    B.print();

    (A - B).print();

    std::array<Matrix<int>, 4> mat_arr{A, A, B, B};

    auto AABB = Matrix<int>::from2x2(mat_arr.begin(), mat_arr.end());

    AABB.print();

    auto&& [A1, A2, A3, A4] = AABB.split2x2();

    std::cout << "-------------" << std::endl;
    A1.print();
    std::cout << "-------------" << std::endl;
    A2.print();
    std::cout << "-------------" << std::endl;
    A3.print();
    std::cout << "-------------" << std::endl;
    A4.print();
    std::cout << "-------------" << std::endl;

    return;
    auto C = strassen_mult(A, B);
    std::cout << "C" << std::endl;
    C.print();

    auto true_C = basic_mult(A, B);
    std::cout << "C true" << std::endl;
    C.print();
}

void test_strassen()
{    
    auto A = Matrix<int>::rand_mat(MAT_SIZE, MAT_SIZE);
    auto B = Matrix<int>::rand_mat(MAT_SIZE, MAT_SIZE);
    // std::cout << "A" << std::endl;
    // A.print();
    // std::cout << "B" << std::endl;
    // B.print();

    auto C = strassen_mult(A, B);
    std::cout << "C" << std::endl;
    // C.print();

    auto true_C = optimized_mult(A, B);
    std::cout << "C true" << std::endl;
    // C.print();
    
    std::cout << std::boolalpha << (C == true_C) << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Please give mat size" << std::endl;
        std::exit(1);
    }
    int size = std::stoi(argv[1]);

    if ((size & (size - 1)) != 0)
    {
        std::cout << "mat size should be equal power of 2" << std::endl;
        std::exit(1);
    }
    
    omp_set_num_threads(2);

    time_method(size, strassen_mult<int>);
    compare_par_nonpar(size);
    // time_strassen(size);
}
