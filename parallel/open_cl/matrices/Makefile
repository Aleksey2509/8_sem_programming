CC=gcc
CXX=g++-10

INCLUDE_DIR = include

CXXFLAGS = -O3 -march=native -I$(INCLUDE_DIR) -std=c++2a -fopenmp

LDFLAGS = -lOpenCL

SRC_DIR = src
SOURCES = main.cc

SRC = $(addprefix $(SRC_DIR)/, $(SOURCES))

EXEC = mult

main: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXEC) $(LDFLAGS)

compare: $(SRC) 
	$(CXX) $(CXXFLAGS) -DCOMPARE $(SRC) -o $(EXEC) $(LDFLAGS)

with_print: $(SRC)
	$(CXX) $(CXXFLAGS) -DPRINT $(SRC) -o $(EXEC) $(LDFLAGS)

test: src/test.cc
	$(CXX) $(CXXFLAGS) -g src/test.cc -o test $(LDFLAGS)
