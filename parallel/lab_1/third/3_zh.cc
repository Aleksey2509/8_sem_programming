#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>

#include "matrix.hh"
#include "omp.h"
#include <mpi.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int i_size = 5000;
    int j_size = 5000;
    int core_num = 2;

    if (argc == 2)
    {
        core_num = std::atoi(argv[1]);
    }
    std::cerr << "got " << core_num << std::endl;

    omp_set_num_threads(core_num);

    if (argc == 3)
    {
        std::cerr << "give i AND j" << std::endl; 
        return -1;
    }

    if (argc >= 4)
    {
        i_size = std::atoi(argv[2]);
        j_size = std::atoi(argv[3]);
    }

    Matrix<double> a(i_size, j_size);
    Matrix<double> b(i_size, j_size);
    
    for (int i = 0; i < i_size; i++)
        for (int j = 0; j < j_size; j++)
        {
            a(i, j) = 10 * i + j;
            b(i, j) = 0;
        }
    
    auto&& start = std::chrono::high_resolution_clock::now();
    #ifndef PAR
    for (int i = 0; i < i_size; i++)
       for (int j = 0; j < j_size; j++)
       {
           a(i, j) = sin(0.1 * a(i, j));
       }
    for (int i = 0; i < i_size - 1; i++)
       for (int j = 2; j < j_size; j++)
       {
           b(i, j) = a(i + 1, j - 2) * 1.5;
       }

    #else
    int mpi_size{0};
    int mpi_rank{0};
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    #pragma omp parallel for num_threads(core_num)
    for (int j = 0; j < j_size; j++)
    {
        a(0, j) = sin(0.1 * a(0, j));
    }
    #pragma omp parallel for collapse(2) num_threads(core_num)
    for (int i = 1; i < i_size; i++)
        for (int j = j_size - 2; j < j_size; j++)
        {
            a(i, j) = sin(0.1 * a(i, j));
        }
    
    #pragma omp parallel for collapse(2) num_threads(core_num)
    for (int i = 1; i < i_size; i++)
       for (int j = 0; j < j_size - 2; j++)
       {
           a(i, j) = sin(0.1 * a(i, j));
           b(i - 1, j + 2) = a(i, j) * 1.5;
       }


    #endif
    auto&& end = std::chrono::high_resolution_clock::now();
    auto&& passed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "passed " << passed << std::endl;

    a.print();
}
