#include <atomic>
#include <omp.h>
#include <iostream>

void report_num_threads()
{
    int level = omp_get_active_level(); 
    #pragma omp critical
    {
        std::cout << "thread " << omp_get_thread_num()  << " reporting" << " from level " << level << 
        " amount of threads on level " << omp_get_num_threads() << " parent: " <<  omp_get_ancestor_thread_num(level - 1) << std::endl;
        for (int i = 0; i < level - 1; i++)
        {
            std::cout << "parent on level " << i << " is thread " << omp_get_ancestor_thread_num(i) << std::endl;
        }
    }
}

void fork_out()
{
    #pragma omp parallel num_threads(4)
    {
        report_num_threads();
        if (omp_get_num_threads() != 1)
            fork_out();
    }
    return;
}



int main(int argc, char* argv[])
{
    int N = 2;
    if (argc > 1)
        N = std::stoi(argv[1]);

    std::cout << N << std::endl;

    omp_set_max_active_levels(N);

    fork_out();
}
