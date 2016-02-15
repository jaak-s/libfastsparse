CC=gcc
CFLAGS=-std=gnu99 -Wall -DNDEBUG -Wall -O3 -march=native -fopenmp -lm -fstrict-aliasing

all: test bench

test:
	$(CC) test_sparse.c $(CFLAGS) -o test_sparse

bench:
	$(CC) bench_a_mul_b.c $(CFLAGS) -o bench_a_mul_b

clean:
	rm -f test_sparse bench_a_mul_b
