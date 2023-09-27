#include <atomic>
#include <omp.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
    omp_set_num_threads(4);
    #pragma omp parallel default(none)
    {
        #pragma omp single
        {
            #pragma omp task 
            printf("Doing smth, id = %d\n", omp_get_thread_num());

            #pragma omp task
            printf("Doing other, id = %d\n", omp_get_thread_num());

        }
    }
    return(0);
}
