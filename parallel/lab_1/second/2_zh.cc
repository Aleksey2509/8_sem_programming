#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>

#include "matrix.hh"
#include "omp.h"

int main(int argc, char* argv[])
{
    int i_size = 5000;
    int j_size = 5000;
    int thread_num = omp_get_max_threads();

    if (argc == 2)
    {
        thread_num = std::atoi(argv[1]);
    }
    std::cerr << "got " << thread_num << std::endl;

    omp_set_num_threads(thread_num);

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
    
    for (int i = 0; i < i_size; i++)
        for (int j = 0; j < j_size; j++)
        {
            a(i, j) = 10 * i + j;
        }
    
    auto&& start = std::chrono::high_resolution_clock::now();
    #ifndef PAR
    for (int i = 0; i < i_size - 2; i++)
       for (int j = 3; j < j_size; j++)
       {
           a(i, j) = sin(0.1 * a(i + 2, j - 3));
       }   
    #else
    Matrix<double> b = a;
    #pragma omp parallel for collapse(2) num_threads(thread_num)
    for (int i = 0; i < i_size - 2; i++)
       for (int j = 3; j < j_size; j++)
       {
           a(i, j) = sin(0.1 * b(i + 2, j - 3));
       } 

    #endif
    auto&& end = std::chrono::high_resolution_clock::now();
    auto&& passed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "passed " << passed << std::endl;

    a.print();
}
