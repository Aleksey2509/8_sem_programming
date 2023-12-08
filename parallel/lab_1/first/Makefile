CXX = g++-10
MPICXX = /usr/bin/mpic++
MPIRUN = /usr/bin/mpirun

CXXFLAGS = -O2 -std=c++2a

par:
	$(MPICXX) $(CXXFLAGS) -DPAR 1_zh.cc -o 1_zh_par

non_par:
	$(MPICXX) $(CXXFLAGS) 1_zh.cc -o 1_zh_non_par

run_par:
	@echo 'Parallel:'
	$(MPIRUN) -n 4 --oversubscribe 1_zh_par par.txt 1000 1000

run_non_par:
	@echo 'Non parallel:'
	$(MPIRUN) -n 1 --oversubscribe 1_zh_non_par non_par.txt 1000 1000

compare:
	@echo 'Comparing:'
	diff par.txt non_par.txt

run_all: non_par par run_non_par run_par compare
