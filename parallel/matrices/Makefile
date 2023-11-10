CXX = g++-10

RELEASE_OPTS = -O2 -march=native -g --opt-info-vec-missed -fdump-tree-vect
DEBUG_OPTS = -g

SRC = matrix.cc

INCLUDE_DIR = include

COMMON_OPTS += -I$(INCLUDE_DIR) -fopenmp -std=c++2a

all: debug

asm:
	$(CXX) $(DEBUG_OPTS) $(COMMON_OPTS) $(SRC) -S -o matrix.s

debug:
	echo $(CXX)
	$(CXX) $(DEBUG_OPTS) $(COMMON_OPTS) $(SRC) -o matrix

release:
	$(CXX) $(RELEASE_OPTS) $(COMMON_OPTS) $(SRC) -o matrix
