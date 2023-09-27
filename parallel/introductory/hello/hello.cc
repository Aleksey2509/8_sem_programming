#include <stdio.h>
#include <omp.h>


int main()
{
    #pragma omp parallel num_threads(4) 
    {
        #pragma omp critical
        printf("hey, i am omp thread %d, there are %d omp_threads\n", omp_get_thread_num(), omp_get_num_threads());
    }
}
