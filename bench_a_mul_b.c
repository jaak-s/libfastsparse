#include <stdio.h>
#include <stdlib.h>
#include "sparse.h"
#include "timing.h"

void usage() {
  printf("Usage:\n");
  printf("  bench_a_mul_c <matrix_file>\n");
}

int main(int argc, char **argv) {
  if (argc <= 1) {
    printf("Need matrix file name.\n");
    usage();
    exit(1);
  }
  char* filename = argv[1];
  int nrepeats = 100;

  printf("Benchmarking A*x with '%s'.\n", filename);
  struct SparseBinaryMatrix* A = read_sbm(filename);
  printf("Size of A is %d x %d.\n", A->nrow, A->ncol);
  printf("Number of repeats = %d\n", nrepeats);

  double* y = malloc(A->nrow * sizeof(double));
  double* x = malloc(A->ncol * sizeof(double));

  for (int i = 0; i < A->ncol; i++) {
    x[i] = sin(7.0*i);
  }

  double wall_start, cpu_start;
  double wall_stop,  cpu_stop;

  timing(&wall_start, &cpu_start);
  for (int i = 0; i < nrepeats; i++) {
    A_mul_B(y, A, x);
  }
  timing(&wall_stop, &cpu_stop);

  printf("Wall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / nrepeats, (cpu_stop - cpu_start)/nrepeats);
}
