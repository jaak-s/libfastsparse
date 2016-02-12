#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include "sparse.h"
#include "timing.h"
#include "cg.h"

void usage() {
  printf("Usage:\n");
  printf("  bench_a_mul_c -f <matrix_file> [-b block_size] [-t] [-c]\n");
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

/** returns -1 if A is hilbert sorted
 *  otherwise idx of the first non-increasing element */
long check_if_sorted(struct SparseBinaryMatrix *A) {
  int n = ceilPower2(A->nrow > A->ncol ? A->nrow : A->ncol);
  long h = xy2d(n, A->rows[0], A->cols[0]);
  for (long j = 1; j < A->nnz; j++) {
    long h2 = xy2d(n, A->rows[j], A->cols[j]);
    if (h2 <= h) return j;
    h = h2;
  }
  return -1;
}

void randn(double* x, int n) {
  struct drand48_data drand_buf;
  int seed;
#pragma omp parallel private(seed, drand_buf)
  {
    seed = 1202107158 + omp_get_thread_num() * 1999;
    srand48_r (seed, &drand_buf);

#pragma omp for schedule(static)
    for (int i = 0; i < n; i += 2) {
      double x1, x2, w;
      do {
        drand48_r(&drand_buf, &x1);
        drand48_r(&drand_buf, &x2);
        x1 = 2.0 * x1 - 1.0;
        x2 = 2.0 * x2 - 1.0;
        w = x1 * x1 + x2 * x2;
      } while ( w >= 1.0 );

      w = sqrt( (-2.0 * log( w ) ) / w );
      x[i] = x1 * w;
      if (i + 1 < n) {
        x[i+1] = x2 * w;
      }
    }
  }
}

void execute_mul2(double* Y, struct BlockedSBM* B, struct BlockedSBM* Bt, double* X, int cgrepeats, int nthreads) {
  omp_set_num_threads(nthreads);
  for (int i = 0; i < cgrepeats; i++) {
    bsbm_A_mul_B2(Y, B,  X);
    bsbm_A_mul_B2(X, Bt, Y);
  }
}

int main(int argc, char **argv) {
  int cgflag = 0;
  int tflag  = 0;
  char* filename = NULL;
  int block_size = 1024;
  int c;

  opterr = 0;

  while ((c = getopt(argc, argv, "b:cf:t")) != -1)
    switch (c) {
      case 'b': block_size = atoi(optarg); break;
      case 'c': cgflag = 1; break;
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
  int nrepeats = 10;
  int cgrepeats = 100;

  if (filename == NULL) {
    fprintf(stderr, "Input matrix file missing.\n");
    usage();
    return 3;
  }

  printf("Benchmarking A*x with '%s'.\n", filename);
  struct SparseBinaryMatrix* A = read_sbm(filename);
  if (tflag) {
    transpose(A);
  }
  printf("Size of A is %d x %d.\n", A->nrow, A->ncol);
  printf("Number of repeats = %d\n", nrepeats);
  printf("Number of CG repeats = %d\n", cgrepeats);

  double* y  = (double*)malloc(A->nrow * sizeof(double));
  double* y2 = (double*)malloc(A->nrow * sizeof(double));
  double* x  = (double*)malloc(A->ncol * sizeof(double));

  double* Y  = (double*)malloc(2 * A->nrow * sizeof(double));
  double* X  = (double*)malloc(2 * A->ncol * sizeof(double));

  double* Y2  = (double*)malloc(2 * A->nrow * sizeof(double));
  double* X2  = (double*)malloc(2 * A->ncol * sizeof(double));

  double* Y4  = (double*)malloc(4 * A->nrow * sizeof(double));
  double* X4  = (double*)malloc(4 * A->ncol * sizeof(double));

  //double* Y8  = (double*)malloc(8 * A->nrow * sizeof(double));
  //double* X8  = (double*)malloc(8 * A->ncol * sizeof(double));

  for (int i = 0; i < A->ncol; i++) {
    x[i]     = sin(7.0*i + 0.3);
    X[i*2]   = sin(7.0*i + 0.3);
    X[i*2+1] = sin(11*i - 0.2);
    X2[i*2]  = X[i*2];
    X2[i*2+1] = X[i*2 + 1];

    for (int k = 0; k < 4; k++) {
      X4[i*4+k] = sin(7*i + 17*k + 0.3);
    }
    //for (int k = 0; k < 8; k++) {
    //  X8[i*8+k] = sin(7*i + 17*k + 0.3);
    //}
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
    A_mul_B(y, A, x);
  }
  timing(&wall_stop, &cpu_stop);

  printf("[sort]  \tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / nrepeats, (cpu_stop - cpu_start)/nrepeats);

  ////// Blocked SBM //////
  printf("Block size = %d\n", block_size);
  struct BlockedSBM* B = new_bsbm(A, block_size);
  struct SparseBinaryMatrix* At = new_transpose(A);
  struct BlockedSBM* Bt = new_bsbm(At, block_size);
  bsbm_A_mul_B(y2, B, x);
  timing(&wall_start, &cpu_start);
  for (int i = 0; i < nrepeats; i++) {
    bsbm_A_mul_B(y, B, x);
  }
  timing(&wall_stop, &cpu_stop);
  printf("[block]\t\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / nrepeats, (cpu_stop - cpu_start)/nrepeats);

  ////// Blocked SBM 2x //////
  bsbm_A_mul_B2(Y, B, X);
  timing(&wall_start, &cpu_start);
  for (int i = 0; i < nrepeats; i++) {
    bsbm_A_mul_B2(Y, B, X);
  }
  timing(&wall_stop, &cpu_stop);
  printf("[2xblock]\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / nrepeats, (cpu_stop - cpu_start)/nrepeats);
  
  ////// Blocked SBM 2x* //////
  bsbm_A_mul_B2(Y, B, X);
  timing(&wall_start, &cpu_start);
  for (int i = 0; i < nrepeats; i++) {
    bsbm_A_mul_Bn(Y, B, X, 2);
  }
  timing(&wall_stop, &cpu_stop);
  printf("[2xblock*]\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / nrepeats, (cpu_stop - cpu_start)/nrepeats);

  /////// CG with x //////
  timing(&wall_start, &cpu_start);
  for (int i = 0; i < cgrepeats; i++) {
    bsbm_A_mul_B(y, B,  x);
    bsbm_A_mul_B(x, Bt, y);
  }
  timing(&wall_stop, &cpu_stop);
  printf("[cg]\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / cgrepeats, (cpu_stop - cpu_start)/cgrepeats);

  /////// CG with X //////
  timing(&wall_start, &cpu_start);
  for (int i = 0; i < cgrepeats; i++) {
    bsbm_A_mul_B2(Y, B,  X);
    bsbm_A_mul_B2(X, Bt, Y);
  }
  timing(&wall_stop, &cpu_stop);
  printf("[cg2]\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / cgrepeats, (cpu_stop - cpu_start)/cgrepeats);


  /////// Running Macau BlockCG with 2 RHSs //////
  if (cgflag) {
    printf("[BlockCG2]\tGenerating RHSs data.\n");
    double l = 15.0;
    double lsqrt = sqrt(15.0);
    double* N2 = (double*)malloc(2 * A->nrow * sizeof(double));
    double* B2 = (double*)malloc(2 * A->ncol * sizeof(double));
    double* E2 = (double*)malloc(2 * A->ncol * sizeof(double));

    randn(N2, A->nrow * 2);
    randn(E2, A->ncol * 2);
    bsbm_A_mul_B2(B2, Bt, N2);

    int nfeat2 = A->ncol * 2;
    for (int i = 0; i < nfeat2; i++) {
      B2[i] += lsqrt * E2[i];
    }
    printf("[BlockCG2]\tTwo RHSs generated.\n");

    int numIter = 0;
    timing(&wall_start, &cpu_start);
    bsbm_cg2(E2, B, Bt, B2, l, 1e-6, &numIter);
    timing(&wall_stop, &cpu_stop);
    printf("[BlockCG2]\tWall: %.3f\tcpu: %.3f\n", wall_stop - wall_start, cpu_stop - cpu_start);
    printf("[BlockCG2]\tniter: %d\n", numIter);

    free(N2);
    free(B2);
    free(E2);
  }

  ////// Blocked SBM 4x //////
  bsbm_A_mul_B4(Y4, B, X4);
  timing(&wall_start, &cpu_start);
  for (int i = 0; i < nrepeats; i++) {
    bsbm_A_mul_B4(Y4, B, X4);
  }
  timing(&wall_stop, &cpu_stop);
  printf("[4xblock]\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / nrepeats, (cpu_stop - cpu_start)/nrepeats);

  ////// Blocked SBM 8x //////
  /*
  bsbm_A_mul_Bn(Y8, B, X8, 8);
  timing(&wall_start, &cpu_start);
  for (int i = 0; i < nrepeats; i++) {
    bsbm_A_mul_Bn(Y8, B, X8, 8);
  }
  timing(&wall_stop, &cpu_stop);
  printf("[8xblock]\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / nrepeats, (cpu_stop - cpu_start)/nrepeats);
  */

  ////// Sort each block ///////
  sort_bsbm(B);
  timing(&wall_start, &cpu_start);
  for (int i = 0; i < nrepeats; i++) {
    bsbm_A_mul_B(y, B, x);
  }
  timing(&wall_stop, &cpu_stop);
  printf("[sort+block]\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / nrepeats, (cpu_stop - cpu_start)/nrepeats);

  ////// Sort by row ///////
  sort_bsbm_byrow(B);
  timing(&wall_start, &cpu_start);
  for (int i = 0; i < nrepeats; i++) {
    bsbm_A_mul_B(y, B, x);
  }
  timing(&wall_stop, &cpu_stop);
  printf("[rowsort+block]\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / nrepeats, (cpu_stop - cpu_start)/nrepeats);

  ////// CG in 2 separate independent parallel blocks
  int omp_total_threads;
#pragma omp parallel
  {
#pragma omp single
    omp_total_threads = omp_get_num_threads();
  }
  omp_set_nested(1);
  omp_set_dynamic(0);
  omp_set_num_threads(2);
  timing(&wall_start, &cpu_start);
#pragma omp parallel
  {
    if (omp_get_thread_num() % 2 == 0) {
      execute_mul2(Y, B, Bt, X, cgrepeats, omp_total_threads / 2);
    } else {
      execute_mul2(Y2, B, Bt, X2, cgrepeats, omp_total_threads / 2);
    }
  }
  timing(&wall_stop, &cpu_stop);
  printf("[2x cg2]\tWall: %0.5e\tcpu: %0.5e\n", (wall_stop - wall_start) / cgrepeats, (cpu_stop - cpu_start)/cgrepeats);
  omp_set_nested(0);
}
