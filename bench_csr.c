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
  struct BinaryCSR* B = bcsr_from_sbm(A);
  printf("Size of B is %d x %d.\n", B->nrow, B->ncol);
  printf("Number of repeats = %d\n", nrepeats);
  printf("Number of CG repeats = %d\n", cgrepeats);

  double* y  = (double*)malloc(B->ncol * sizeof(double));
  //double* y2 = (double*)malloc(B->ncol * sizeof(double));
  double* x  = (double*)malloc(B->ncol * sizeof(double));

  for (int i = 0; i < B->ncol; i++) {
    x[i]     = sin(7.0*i + 0.3);
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

  double wall_start, cpu_start;
  double wall_stop,  cpu_stop;

  timing(&wall_start, &cpu_start);
  for (int i = 0; i < nrepeats; i++) {
    parallel_bcsr_AA_mul_B(y, B, x, ytmp);
  }
  timing(&wall_stop, &cpu_stop);
  printf("[B'B x]\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / nrepeats, (cpu_stop - cpu_start)/nrepeats);
}
