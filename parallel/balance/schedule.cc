#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>
#include <vector>
#include <unordered_map>
#include <iostream>

#define SCH_TYPE dynamic 

int main(int argc, char* argv[])
{
	// Use 2 threads when creating OpenMP parallel regions
	omp_set_num_threads(4);
    
    std::unordered_map<int, std::vector<int>> id_to_iters;

    int iter_amount = 65;

	// Parallelise the for loop using the given schedule
    #ifdef SCH_TYPE
    #pragma omp parallel for schedule(SCH_TYPE, 1)
    #else
    #pragma omp parallel for
    #endif
    for(int i = 0; i < iter_amount; i++)
	{
        int sleep_amount = rand() % 2000000;
        usleep(sleep_amount);
	    #pragma omp critical
        id_to_iters[omp_get_thread_num()].push_back(i);
    }

	printf("With no chunksize passed:\n");
    
    for (int i = 0; i < 4; i++)
    {
        printf("thread %d took those iterations:\n", i);
        for (auto&& it : id_to_iters[i])
            printf("%d ", it);
        printf("\n");
    }

    #pragma omp barrier

	// Parallelise the for loop using the given schedule and chunks of 4 iterations
    #ifdef SCH_TYPE
	printf("With a chunksize of 4:\n");
    std::unordered_map<int, std::vector<int>> chunk_id_to_iters;
    #pragma omp parallel for schedule(SCH_TYPE, 4)
    for(int i = 0; i < iter_amount; i++)
	{
	    #pragma omp critical
        chunk_id_to_iters[omp_get_thread_num()].push_back(i);
    }

    for (int i = 0; i < 4; i++)
    {
        std::cout << "in chunks: thread " << i << " took folowing iters" << std::endl;
        for (auto&& it : id_to_iters[i])
            std::cout << it << " ";
        std::cout << std::endl;
    }
    #endif
}
