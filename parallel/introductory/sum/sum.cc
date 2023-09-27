#include <omp.h>
#include <atomic>
#include <iostream>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Please, provide N" << std::endl;
        return -1;
    }
        std::atomic<double> sum;
    int N = std::stoi(argv[1]); 
    #pragma omp parallel for
    for (int i = 1; i < N + 1; i++)
    {
        double expected = sum.load();
        double local_val = 1.0 / i;
        while(!sum.compare_exchange_weak(expected, expected + local_val))
        ;
    }

    std::cout << sum << std::endl;
}
