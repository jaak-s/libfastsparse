CC=gcc
CFLAGS=-std=gnu99 -g -Wall -O3 -march=native -fopenmp -lm -fstrict-aliasing #-DNDEBUG

all: test bench preprocess

test:
	$(CC) test_sparse.c $(CFLAGS) -o test_sparse

bench:
	$(CC) bench_a_mul_b.c $(CFLAGS) -o bench_a_mul_b
	$(CC) bench_csr.c $(CFLAGS) -o bench_csr

preprocess: preprocess.c csr.h quickSort.h sparse.h utils.h
	$(CC) preprocess.c $(CFLAGS) -o preprocess

clean:
	rm -f test_sparse bench_a_mul_b bench_csr preprocess
