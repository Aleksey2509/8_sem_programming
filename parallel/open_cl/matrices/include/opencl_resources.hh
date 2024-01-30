#ifndef OPENCL_RES_HH 
#define OPENCL_RES_HH 

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <concepts>
#include <filesystem>

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 210
#endif

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_QUEUE_PROFILING_ENABLE

#include "CL/cl2.hpp"

#include "matrix.hh"

template <typename it>
void print_container (it begin, it end) {
            
    for (; begin != end; ++begin)
        std::cout << *begin << " ";            
    std::cout << std::endl;
}

#ifndef LOCAL_TILE_SIZE
#define LOCAL_TILE_SIZE 16
#endif

struct OpenCLMatrixMultiplier
{
    using mult_func = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int>;
    cl::Platform     platform_;
    cl::Context      context_;
    cl::CommandQueue queue_;
    std::string      mult_kernel_;

    std::vector<int> sorted;

    OpenCLMatrixMultiplier (const std::filesystem::path& kernel_path)  
    { 
        selectPlatform();
        getContext(platform_());
        getKernel(kernel_path);
		getQueue();
        std::cout << "Will work on: " << platform_.getInfo<CL_PLATFORM_NAME>() << std::endl;
    }

    void selectPlatform ()
	{
    	cl::vector<cl::Platform> platforms;
    	cl::Platform::get (&platforms);

    	for (auto p : platforms)
		{
    	    cl_uint numDevices = 0;
    	    ::clGetDeviceIDs (p(), CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices); // small optimization, do not need device explicitly

    	    if (numDevices > 0)
			{
    	    	platform_ = cl::Platform(p);
				break;
			}
    	}
    }

    void getContext (cl_platform_id id)
	{
        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(id), 0};
        context_ =  cl::Context (CL_DEVICE_TYPE_GPU, properties);
    }

	void getQueue()
	{
        queue_ = cl::CommandQueue(context_, CL_QUEUE_PROFILING_ENABLE);
	}

    void getKernel(const std::filesystem::path& kernel_path)
    {
        std::ifstream input;
        input.exceptions(std::ifstream::failbit);
        input.open(kernel_path);
        std::stringstream dump_buf;
        dump_buf << "#define TILE_SIZE " << LOCAL_TILE_SIZE << "\n\n";
        dump_buf << input.rdbuf();
        input.close();

        mult_kernel_ = dump_buf.str();
    } 

    
    cl::Event multiply (Matrix<int>& A_mat, Matrix<int>& B_mat, Matrix<int>& C_mat)
    {
		int A_rows = A_mat.rows();
		int A_cols = A_mat.cols();
		int B_cols = B_mat.cols();

		int* A_ptr = A_mat.data();
		int* B_ptr = B_mat.data();
		int* C_ptr = C_mat.data();

		size_t A_size = A_rows * A_cols; 
  		size_t B_size = A_cols * B_cols;
  		size_t C_size = A_rows * B_cols;
		
		size_t A_bufsize = A_size * sizeof(int);
		size_t B_bufsize = A_size * sizeof(int);
		size_t C_bufsize = A_size * sizeof(int);

  		cl::Buffer A(context_, CL_MEM_READ_ONLY, A_bufsize);
  		cl::Buffer B(context_, CL_MEM_READ_ONLY, B_bufsize);
  		cl::Buffer C(context_, CL_MEM_WRITE_ONLY, C_bufsize);

  		cl::copy(queue_, A_ptr, A_ptr + A_size, A);
  		cl::copy(queue_, B_ptr, B_ptr + A_size, B);

  		cl::Program program(context_, mult_kernel_, true /* build immediately */);

  		mult_func gemm(program, "matrix_multiply");

  		cl::NDRange GlobalRange(A_rows, B_cols);
  		cl::NDRange LocalRange(LOCAL_TILE_SIZE, LOCAL_TILE_SIZE);
  		cl::EnqueueArgs Args(queue_, GlobalRange, LocalRange);

  		cl::Event Evt = gemm(Args, A, B, C, A_rows, A_cols, B_cols);
  		Evt.wait();

  		cl::copy(queue_, C, C_ptr, C_ptr + C_size);
  		return Evt;
    }   
 
};

#endif
