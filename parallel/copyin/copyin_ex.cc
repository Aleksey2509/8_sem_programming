#include <iostream>
#include <omp.h>

int a = 10;
#pragma omp threadprivate(a)

int main(int argc, char* argv[])
{
	// Turn off dynamic threads as required by threadprivate
	omp_set_dynamic(0);

	#pragma omp parallel copyin(a)
	{
		#pragma omp master
		{
			printf("first parallel: master thread changes the value of a to 100.\n");
			a = 100;
		}

		#pragma omp barrier

		printf("first parallel: Thread %d: a = %d.\n", omp_get_thread_num(), a);
	}

	#pragma omp parallel copyin(a)
	{
		printf("second parallel: thread %d: a = %d.\n", omp_get_thread_num(), a);
	}

}
