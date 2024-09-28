CXX = g++-10

RELEASE_OPTS = -O2 -march=native -g 
DEBUG_OPTS = -g

SRC = fft.cc

COMMON_OPTS += -fopenmp -std=c++2a

release:
	$(CXX) $(RELEASE_OPTS) $(COMMON_OPTS) $(SRC) -lm -o fft 
