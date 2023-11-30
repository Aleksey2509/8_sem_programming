CXX = g++-10
MPICXX = /usr/bin/mpic++

CXXFLAGS = -O2 -fopenmp -std=c++2a

all: par non_par

par: 2_zh.cc
	$(CXX) $(CXXFLAGS) -DPAR 2_zh.cc -o par

non_par: 2_zh.cc
	$(CXX) $(CXXFLAGS) 2_zh.cc -o non_par
