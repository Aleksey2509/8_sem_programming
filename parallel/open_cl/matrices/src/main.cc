#include "opencl_resources.hh"
#include <algorithm>
#include <chrono>

int main (int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Please give arr size" << std::endl;
        std::exit(1);
    }
    int size = std::stoi(argv[1]);
 
    if ((size & (size - 1)) != 0)
    {
        std::cout << "size off array to sort should be equal power of 2" << std::endl;
        std::exit(1);
    }

    Matrix<int> A = Matrix<int>::rand_mat(size, size);
    Matrix<int> B = Matrix<int>::rand_mat(size, size);
    Matrix<int> C = Matrix<int>(size, size);
    
	// A.print();
	// B.print();
    
	OpenCLMatrixMultiplier multiplier("./kernels/mult_with_local.cl");
    
    cl::Event event = multiplier.multiply(A, B, C);
	
	// C.print();
    
    auto&& start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    auto&& end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    auto&& delta = end - start;
    std::cout << "GPU time: " << delta / 1000 << "mcs\n" << std::endl;

    auto&& cpu_start = std::chrono::high_resolution_clock::now();
    auto&& true_c = par_strassen_mult(A, B);
    auto&& cpu_end = std::chrono::high_resolution_clock::now();

    auto&& cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();

    std::cout << "CPU time: " << cpu_time << "mcs" << std::endl;

    if (C != true_c)
        std::cout << "Multiplication was not correct" << std::endl;
    else
        std::cout << "Multiplication was correct" << std::endl;

	

    return 0;
}

