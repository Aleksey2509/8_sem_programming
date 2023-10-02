#include <atomic>
#include <omp.h>
#include <stdio.h>
#include <thread>
#include <vector>
#include <stdlib.h>
#include <unistd.h>

std::vector<int> data(50);
int acc_ind = 1000; 
#pragma omp threadprivate(acc_ind)

void set_correct_index()
{
    printf("Thread %d set correct index\n", omp_get_thread_num());
    acc_ind = 0;
    std::this_thread::yield();
}

void heavy_task()
{
    sleep(5);
}


void access()
{
    printf("Thread %d accessing at %d\n", omp_get_thread_num(), acc_ind);
    printf("Accessed: %d\n", data[acc_ind]);
}


int main(int argc, char* argv[])
{
    omp_set_num_threads(4);
    #pragma omp parallel default(none)
    {
        #pragma omp single
        {
            #pragma omp task untied 
            {
                set_correct_index();
                heavy_task();
                std::this_thread::yield();
                access();
            }
        }
    }
}
