CC=gcc
COMMON_FLAGS=-g -Wall -O3 -march=native -fopenmp -lm -fstrict-aliasing #-DNDEBUG
CFLAGS=-std=gnu99 $(COMMON_FLAGS)
CXXFLAGS=-std=c++11 $(COMMON_FLAGS)

all: test bench preprocess

test:
	$(CC) test_sparse.c $(CFLAGS) -o test_sparse

bench:
	$(CC) bench_a_mul_b.c $(CFLAGS) -o bench_a_mul_b
	$(CC) bench_csr.c $(CFLAGS) -o bench_csr

bench_cpp:
	$(CXX) bench.cpp $(CXXFLAGS) -lbenchmark -lpthread -o bench

preprocess: preprocess.c csr.h quickSort.h sparse.h utils.h
	$(CC) preprocess.c $(CFLAGS) -o preprocess

clean:
	rm -f test_sparse bench_a_mul_b bench_csr preprocess
