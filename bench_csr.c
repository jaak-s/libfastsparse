#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include "sparse.h"
#include "timing.h"
#include "cg.h"
#include "csr.h"

void usage() {
  printf("Usage:\n");
  printf("  bench_csr -f <matrix_file> [-t] [-c]\n");
  printf("  -t  transpose matrix\n");
  printf("  -c  run conjugate gradient\n");
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
  int tflag  = 0;
  char* filename = NULL;
  //int cgflag = 0;
  int c;

  while ((c = getopt(argc, argv, "f:t")) != -1)
    switch (c) {
      //case 'c': cgflag = 1; break;
      case 'f': filename = optarg; break;
      case 't': tflag = 1; break;
      case '?':
        if (optopt == 'f')
          fprintf(stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint(optopt))
          fprintf(stderr, "Unknown option '-%c'.\n", optopt);
        else
          fprintf(stderr, "Unknown option character '\\x%x.\n", optopt);
        usage();
        return 1;
      default:
        usage();
        return 2;
    }
  int nrepeats = 20;
  int cgrepeats = 100;

  if (filename == NULL) {
    fprintf(stderr, "Input matrix file missing.\n");
    usage();
    return 3;
  }

  printf("Benchmarking B*x with '%s'.\n", filename);
  struct SparseBinaryMatrix* A = read_sbm(filename);
  if (tflag) {
    transpose(A);
  }
  printf("Creating BinaryCSR (sorting).\n");
  struct BinaryCSR* B = bcsr_from_sbm(A);
  printf("Size of B is %d x %d.\n", B->nrow, B->ncol);
  printf("Number of repeats = %d\n", nrepeats);
  printf("Number of CG repeats = %d\n", cgrepeats);

  double* y  = (double*)malloc(B->ncol * sizeof(double));
  double* y2 = (double*)malloc(B->ncol * sizeof(double));
  double* x  = (double*)malloc(B->ncol * sizeof(double));

  double* tmp = (double*)malloc(B->nrow * sizeof(double));
  double* yt  = (double*)malloc(B->ncol * sizeof(double));

  for (int i = 0; i < B->ncol; i++) {
    x[i] = sin(7.0*i + 0.3) / 10.0;
  }

  int nthreads;
#pragma omp parallel
  {
#pragma omp single
    {
      nthreads = omp_get_num_threads();
    }
  }
  double* ytmp = (double*)malloc(B->ncol * sizeof(double) * nthreads);

  printf("Nthreads = %d\n", nthreads);

  int min_col = 10000000;
  int max_col = 0;
  for (int i = 0; i < B->nnz; i++) {
    if (B->cols[i] > max_col) max_col = B->cols[i];
    if (B->cols[i] < min_col) min_col = B->cols[i];
  }
  int max_ptr = 0;
  for (int i = 0; i <= B->nrow; i++) {
    max_ptr = B->row_ptr[i];
  }
  printf("Min(cols) = %d, max(cols) = %d\n", min_col, max_col);
  printf("Max(row_ptr) = %d\n", max_ptr);

  double wall_start, cpu_start;
  double wall_stop,  cpu_stop;

  timing(&wall_start, &cpu_start);
  A_mul_B(tmp, A, x);
  At_mul_B(yt, A, tmp);
  timing(&wall_stop, &cpu_stop);
  printf("[A'A x]\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start), (cpu_stop - cpu_start));

  timing(&wall_start, &cpu_start);
  bcsr_AA_mul_B(y2, B, x);
  timing(&wall_stop, &cpu_stop);
  printf("[B'B x]\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start), (cpu_stop - cpu_start));

  timing(&wall_start, &cpu_start);
  parallel_bcsr_AA_mul_B(y, B, x, ytmp);
  for (int i = 0; i < nrepeats; i++) {
    parallel_bcsr_AA_mul_B(y, B, x, ytmp);
  }
  timing(&wall_stop, &cpu_stop);
  printf("[par B'B x]\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / nrepeats, (cpu_stop - cpu_start)/nrepeats);

  for (int j = 0; j < B->ncol; j++) {
    if (abs(yt[j] - y[j]) > 1e-6) { printf("yt[%d]=%f, y[%d]=%f\n", j, yt[j], j, y[j]); return 1; }
  }

  for (int j = 0; j < B->ncol; j++) {
    if (abs(y2[j] - y[j]) > 1e-6) { printf("y2[%d]=%f, y[%d]=%f\n", j, y2[j], j, y[j]); return 1; }
  }
}
