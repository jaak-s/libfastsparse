CFLAGS=-std=gnu99 -Wall -DNDEBUG -Wall -O3 -lm -fstrict-aliasing

all: test bench

test:
	gcc test_sparse.c $(CFLAGS) -o test_sparse

bench:
	gcc bench_a_mul_b.c $(CFLAGS) -o bench_a_mul_b

clean:
	rm test_sparse bench_a_mul_b
