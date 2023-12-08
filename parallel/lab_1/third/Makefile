CXX = g++-10
MPICXX = /usr/bin/mpic++
MPIRUN = /usr/bin/mpirun

CXXFLAGS = -std=c++17

par:
	$(MPICXX) $(CXXFLAGS) -DPAR 3_zh.cc -o 3_zh_par

non_par:
	$(MPICXX) $(CXXFLAGS) 3_zh.cc -o 3_zh_non_par

run_par:
	@echo 'Parallel:'
	$(MPIRUN) -n 2 --oversubscribe 3_zh_par par.txt 7000 7000

run_non_par:
	@echo 'Non parallel:'
	$(MPIRUN) -n 1 3_zh_non_par non_par.txt 7000 7000

compare:
	@echo 'Comparing:'
	diff par.txt non_par.txt

run_all: non_par par run_non_par run_par compare
