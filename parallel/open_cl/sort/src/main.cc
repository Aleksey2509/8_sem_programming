#include "bitonic_sorter.hh"
#include <algorithm>
#include <pstl/glue_execution_defs.h>
#include <random>
#include <chrono>
#include <execution>

template <typename it>
void check_correctness(it begin, it end)
{
    for (; (begin + 1) < end; ++begin)
    {
        if (*begin > *(begin + 1))
        {
            std::cout << "INCORRECT!!" <<std::endl;
            return;
        }
    }
    std::cout << "All ok" << std::endl;
}


template <typename it>
void cpu_time(it begin, it end)
{
    auto&& time_start = std::chrono::high_resolution_clock::now();
    std::sort(std::execution::par_unseq, begin, end);
    // std::sort(begin, end);
    auto&& time_end = std::chrono::high_resolution_clock::now();

    std::cout << "cpu time: " << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() << " microseconds " << std::endl;
}


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
 
    std::random_device r;
    std::mt19937 gen(r());

    std::uniform_int_distribution<int> dist(0, size);

    std::vector<int> sequence(size, 0);
    for (int i = 0; i < size; i++)
		sequence[i] = dist(gen); 

    #ifdef PRINT
	print_container(sequence.begin(), sequence.end());
    #endif
    BitonicOpenCLSorter sorter("./kernels/kernel.cl");
    
    auto&& seconds = sorter.sort(sequence.begin(), sequence.end(), 64);

    std::vector<int> sorted = sorter.sorted;
    #ifdef PRINT
	print_container(sorted.begin(), sorted.end());
    #endif

    check_correctness(sorted.begin(), sorted.end());
    
    #ifdef COMPARE
    cpu_time(sequence.begin(), sequence.end());
    #endif
    // auto&& time_passed = sorter.getSeq();
    // std::cout << "CPU time: " << seconds.CPUTime << "ms\n";
    std::cout << "GPU time: " << seconds / 1000 << " microseconds" << std::endl;

    return 0;
}

