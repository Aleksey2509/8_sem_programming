CXX = g++-10
MPICXX = /usr/bin/mpic++

CXXFLAGS = -O2 -fopenmp -std=c++2a

par_2:
	$(CXX) $(CXXFLAGS) -DPAR 2_zh.cc -o 2_zh

non_par_2:
	$(CXX) $(CXXFLAGS) 2_zh.cc -o 2_zh
