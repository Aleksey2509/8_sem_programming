#include <stdlib.h>
#include <omp.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
	int a = 10;

	#pragma omp parallel default(none) firstprivate(a)
	{
        printf("Thread %d: a = %d\n", omp_get_thread_num(), a); 

		#pragma omp barrier

		#pragma omp single copyprivate(a)
		{
			a = 100;
            printf("Thread %d: executes single part and changes a to %d\n", omp_get_thread_num(), a);
		}

        printf("Thread %d: a = %d\n", omp_get_thread_num(), a); 
	}
}
