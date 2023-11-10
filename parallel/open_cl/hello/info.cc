#include <iostream>
#include <vector>
#include <string>

#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

void print_device_info(cl::Device dev)
{
    std::cout << "device characteristics:" << std::endl;
	std::cout << "name " << dev.getInfo<CL_DEVICE_NAME>() << std::endl;
	std::cout << "type " << dev.getInfo<CL_DEVICE_TYPE>() << std::endl;
	std::cout << "vendor " << dev.getInfo<CL_DEVICE_VENDOR>() << std::endl;
	std::cout << "maximum amount of compute units " << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;

    auto max_dim_amount = dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
	std::cout << "maximum work item dimensions " << max_dim_amount << std::endl;
    auto work_item_sizes = dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    for (int i = 0; i < max_dim_amount; i++)
        std::cout << "max_work_item size in dim " << i << ": " << work_item_sizes[i] << std::endl;

	std::cout << "maximum size of work group " << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;

    
	std::cout << "images support " << std::boolalpha << dev.getInfo<CL_DEVICE_IMAGE_SUPPORT>() << std::endl;
	
    std::cout << "preferred vector width int " << dev.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>() << std::endl;
	
    std::cout << "global mem size " << dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
    std::cout << "global mem cache type " << dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>() << std::endl;
    std::cout << "global mem cache size " << dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>() << std::endl;
    std::cout << "global mem cacheline size " << dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>() << std::endl;
    std::cout << "local mem type " << dev.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>() << std::endl;
    std::cout << "local mem size " << dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
 	std::cout << "maximum constant buffer size " << dev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() << std::endl;
	std::cout << "maximum amount of constant args " << dev.getInfo<CL_DEVICE_MAX_CONSTANT_ARGS >() << std::endl;
}



int main() {
    const size_t N = 1 << 20;

	// Get list of OpenCL platforms.
	std::vector<cl::Platform> platform;
	cl::Platform::get(&platform);

	if (platform.empty()) {
	    std::cerr << "OpenCL platforms not found." << std::endl;
	    return 1;
	}

	// Get first available GPU device which supports double precision.
	cl::Context context;
	std::vector<cl::Device> device;
	for(auto p = platform.begin(); device.empty() && p != platform.end(); p++) {
	    std::vector<cl::Device> pldev;

	    try {
		p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);

		for(auto d = pldev.begin(); device.empty() && d != pldev.end(); d++) {
		    if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;

		    std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();

		    if (
			    ext.find("cl_khr_fp64") == std::string::npos &&
			    ext.find("cl_amd_fp64") == std::string::npos
		       ) continue;

		    device.push_back(*d);
		    context = cl::Context(device);
		}
	    } catch(...) {
		device.clear();
	    }
	}

	if (device.empty()) {
	    std::cerr << "GPUs with double precision not found." << std::endl;
	    return 1;
	}

    for (auto&& dev : device)
        print_device_info(dev);
    
    
}

