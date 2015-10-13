CFLAGS=-std=gnu99 -Wall -DNDEBUG -Wall -O3 -lm

all: test

test:
	gcc test_sparse.c $(CFLAGS) -o test_sparse

clean:
	rm test_sparse
