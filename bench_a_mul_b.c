#include <stdio.h>
#include <stdlib.h>
#include "sparse.h"
#include "timing.h"

void usage() {
  printf("Usage:\n");
  printf("  bench_a_mul_c <matrix_file>\n");
}

void extrema(int* x, long n, int* min, int* max) {
  *min = x[0];
  *max = x[0];

  for (int i = 0; i < n; i++) {
    if (x[i] < min[0]) *min = x[i];
    if (x[i] > max[0]) *max = x[i];
  }
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

  double* y  = malloc(A->nrow * sizeof(double));
  double* y2 = malloc(A->nrow * sizeof(double));
  double* x  = malloc(A->ncol * sizeof(double));

  for (int i = 0; i < A->ncol; i++) {
    x[i] = sin(7.0*i + 0.3);
  }

  double wall_start, cpu_start;
  double wall_stop,  cpu_stop;

  timing(&wall_start, &cpu_start);
  for (int i = 0; i < nrepeats; i++) {
    A_mul_B(y, A, x);
  }
  timing(&wall_stop, &cpu_stop);

  printf("[unsorted]\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / nrepeats, (cpu_stop - cpu_start)/nrepeats);

  // sorting sbm
  int max_row = -1, min_row = -1;
  int max_col = -1, min_col = -1;
  extrema(A->rows, A->nnz, &min_row, &max_row);
  extrema(A->cols, A->nnz, &min_col, &max_col);

  sort_sbm(A);

  extrema(A->rows, A->nnz, &min_row, &max_row);
  extrema(A->cols, A->nnz, &min_col, &max_col);
  // making sure sbm contents is within bounds
  for (int j = 0; j < A->nnz; j++) {
    if (A->rows[j] < 0 || A->rows[j] >= A->nrow) {
      printf("rows[%d] = %d\n", j, A->rows[j]);
      return(1);
    }
    if (A->cols[j] < 0 || A->cols[j] >= A->ncol) {
      printf("cols[%d] = %d\n", j, A->cols[j]);
      return(1);
    }
  }
  A_mul_B(y2, A, x);
  for (int j = 0; j < A->nrow; j++) {
    if (abs(y2[j] - y[j]) > 1e-6) {printf("y2[%d]=%f, y[%d]=%f\n", j, y2[j], j, y[j]); return 1;}
  }

  timing(&wall_start, &cpu_start);
  for (int i = 0; i < nrepeats; i++) {
    A_mul_B(y2, A, x);
  }
  timing(&wall_stop, &cpu_stop);

  printf("[sorted]\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / nrepeats, (cpu_stop - cpu_start)/nrepeats);
}
