#include <bitset>
#include <vector>
#include <iostream>
#include <complex>
#include <numbers>
#include <chrono>
#include <omp.h>

unsigned int bitReverse(unsigned int x, int log2n) {
  int n = 0;
  int mask = 0x1;
  for (int i=0; i < log2n; i++) {
    n <<= 1;
    n |= (x & 1);
    x >>= 1;
  }
  return n;
}

void fft(std::complex<float>* input, std::complex<float>* output, int log2n)
{
    int vec_size = 1 << log2n;
    
    for (unsigned int i = 0; i < vec_size; ++i)
    {
        output[bitReverse(i, log2n)] = input[i];
    }
  
	for (size_t m = 2; m <= vec_size; m <<= 1)
	{
        using namespace std::complex_literals;
        auto multiplier = std::complex<float>((2.0 / m) * std::numbers::pi * 1i );
        size_t m2 = m >> 1;
        for (int k = 0; k < vec_size; k += m)
        {
            for (int j=0; j < m2; ++j) 
            {
                auto index1 = k + j;
                auto index2 = index1 + m2;
                auto w = std::exp(multiplier * std::complex<float>(j));
                auto u = output[index1];
                auto t = w * output[index2];
                output[index1] = u + t;
                output[index2] = u - t;
            }
        }
    }
}

void fft_par(std::complex<float>* input, std::complex<float>* output, int log2n)
{
    int vec_size = 1 << log2n;
    
    #pragma omp parallel for
    for (unsigned int i = 0; i < vec_size; ++i)
    {
        output[bitReverse(i, log2n)] = input[i];
    }
  
	for (size_t m = 2; m <= vec_size; m <<= 1)
	{
        using namespace std::complex_literals;
        auto multiplier = std::complex<float>((2.0 / m) * std::numbers::pi * 1i );
        size_t m2 = m >> 1;
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < vec_size; k += m)
        {
            for (int j=0; j < m2; ++j) 
            {
                auto index1 = k + j;
                auto index2 = index1 + m2;
                auto w = std::exp(multiplier * std::complex<float>(j));
                auto u = output[index1];
                auto t = w * output[index2];
                output[index1] = u + t;
                output[index2] = u - t;
            }
        }
    }
}


void test(int size)
{
    std::vector<std::complex<float>> sin_vals (size);
    std::vector<std::complex<float>> out (size);
    std::vector<std::complex<float>> out_par (size);
   
    // for (auto&& it : sin_vals)
    //     std::cout << it << " ";
    // std::cout << std::endl;

    for (int i = 0; i < size; ++i)
    {
        sin_vals[i].real(std::sin(2 * i * 2 * std::numbers::pi / size));
    }
 
    auto time_start = std::chrono::high_resolution_clock::now();
    fft_par(sin_vals.data(), out.data(), std::round(std::log2(size)));
    auto time_end = std::chrono::high_resolution_clock::now();
    auto par_time_start = std::chrono::high_resolution_clock::now();
    fft(sin_vals.data(), out_par.data(), std::round(std::log2(size)));
    auto par_time_end = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < size; i++)
    {
        if ((std::abs(out[i].real() - out_par[i].real()) > 0.001) || 
            (std::abs(out[i].imag() - out_par[i].imag()) > 0.001))
                std::cout << "smth bad at index " << i << std::endl;
    }


    // for (auto&& it : fft_sin_vals)
    //     std::cout << it.real() << " " << it.imag() << std::endl;
        // std::cout << it << " ";

    std::cout << "par time passed " << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() << " mcs" << std::endl;

    std::cout << "non par time passed " << std::chrono::duration_cast<std::chrono::microseconds>(par_time_end - par_time_start).count() << " mcs" << std::endl;

}


int main()
{
    omp_set_num_threads(2);
    test(1048576);
}
