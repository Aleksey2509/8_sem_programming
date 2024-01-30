CC=gcc
CXX=g++-10

INCLUDE_DIR = include

CXXFLAGS = -O3 -I$(INCLUDE_DIR) -std=c++2a

LDFLAGS = -lOpenCL

SRC_DIR = src
SOURCES = main.cc

SRC = $(addprefix $(SRC_DIR)/, $(SOURCES))

EXEC = sort

main:
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXEC) $(LDFLAGS)

compare:
	$(CXX) $(CXXFLAGS) -DCOMPARE $(SRC) -o $(EXEC) $(LDFLAGS)

with_print:
	$(CXX) $(CXXFLAGS) -DPRINT $(SRC) -o $(EXEC) $(LDFLAGS)
