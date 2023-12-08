CXX = g++-10
MPICXX = /usr/bin/mpic++

CXXFLAGS = -O2 -fopenmp -std=c++2a

all: par non_par

par: 2_zh.cc
	$(CXX) $(CXXFLAGS) -DPAR 2_zh.cc -o par

non_par: 2_zh.cc
	$(CXX) $(CXXFLAGS) 2_zh.cc -o non_par

run_par:
	@echo 'Parallel'
	./par 2 5000 5000 > par.txt

run_non_par:
	@echo 'Non parallel'
	./non_par 2 5000 5000 > non_par.txt

compare:
	@echo 'Comparing'
	diff par.txt non_par.txt

run_all: par non_par run_par run_non_par compare
