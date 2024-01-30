#ifndef __OCL_HPP__
#define __OCL_HPP__

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

template <typename it>
void print_container (it begin, it end) {
            
    for (; begin != end; ++begin)
        std::cout << *begin << " ";            
    std::cout << std::endl;
}


struct BitonicOpenCLSorter
{
    using merge_func = cl::KernelFunctor<cl::Buffer, cl_int, cl_int>;
    
    cl::Platform     platform_;
    cl::Context      context_;
    cl::CommandQueue queue_;
    std::string      sort_kernel_;

    std::vector<int> sorted;

    BitonicOpenCLSorter (const std::filesystem::path& kernel_path)  
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
        dump_buf << input.rdbuf();
        input.close();

        sort_kernel_ = dump_buf.str();
    } 

    
    template <typename it>
    requires std::is_same_v<int, typename std::iterator_traits<it>::value_type>
    size_t sort (it begin, it end, const cl::NDRange &group_size)
    {
        std::vector<int> to_sort{begin, end};
        size_t arr_size = to_sort.size();
        size_t sub_arr_size   = 1;
        size_t middle = arr_size / 2;
        size_t gpu_time = 0; 

        int* sequence = to_sort.data();

        cl::Buffer seq_ (context_, CL_MEM_READ_WRITE, arr_size * sizeof(int));
        cl::copy(queue_, sequence, sequence + arr_size, seq_);

        cl::Program program (context_, sort_kernel_, true); 
        cl::NDRange globalRange = arr_size;

        cl::EnqueueArgs args (queue_, globalRange, group_size);
        merge_func merge_seq (program, "bitonic_merge_int");

        for (; sub_arr_size <= middle; sub_arr_size *= 2) 
            for (size_t step = sub_arr_size; step >= 1; step /= 2) 
                gpu_time += offloadBitonicMerge(seq_, sub_arr_size, step, merge_seq, args);
        

        cl::copy (queue_, seq_, sequence, sequence + arr_size);
        sorted = to_sort; 

        return gpu_time; // ns
    }   

    size_t offloadBitonicMerge (cl::Buffer& arr_, size_t sub_seq_size, size_t step,  merge_func& merge_seq, cl::EnqueueArgs& arguments) 
    {
            
            cl::Event event = merge_seq (arguments, arr_, sub_seq_size * 2, step);
            event.wait();

            auto&& start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            auto&& end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            auto&& delta = end - start;
            return delta;
    }
 
};

#endif
