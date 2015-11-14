/* file minunit_example.c */

#include <stdio.h>
#include <stdlib.h>
#include "minunit.h"
#include "sparse.h"
#include "sparse_blocked.h"
#include "quickSortD.h"
#include "dsparse.h"
#include "linalg.h"

int tests_run = 0;

struct SparseBinaryMatrix* make_sbm() {
  struct SparseBinaryMatrix *A = malloc(sizeof(struct SparseBinaryMatrix));
  A->nrow = 4;
  A->ncol = 3;
  A->nnz  = 5;
  int* rows = malloc(A->nnz * sizeof(int));
  int* cols = malloc(A->nnz * sizeof(int));
  A->rows = rows;
  A->cols = cols;
  rows[0] = 0; rows[1] = 3; rows[2] = 3; rows[3] = 1; rows[4] = 2;
  cols[0] = 0; cols[1] = 2; cols[2] = 0; cols[3] = 2; cols[4] = 1;
  return A;
}

static char * test_A_mul_B() {
  struct SparseBinaryMatrix *A = make_sbm();
  double* x = malloc(A->ncol * sizeof(double));
  double* y = malloc(A->nrow * sizeof(double));
  x[0] = 0.5;
  x[1] = -0.7;
  x[2] = 1.9;
  // multiplication
  A_mul_B(y, A, x);
  mu_assert("error, y[0] != 0.5", y[0] == 0.5);
  mu_assert("error, y[1] != 1.9", y[1] == 1.9);
  mu_assert("error, y[2] !=-0.7", y[2] ==-0.7);
  mu_assert("error, y[3] != 2.4", y[3] == 2.4);
  return 0;
}

static char * test_At_mul_B() {
  struct SparseBinaryMatrix *A = make_sbm();
  double* x = malloc(A->nrow * sizeof(double));
  double* y = malloc(A->ncol * sizeof(double));
  x[0] = 0.2;
  x[1] = 1.3;
  x[2] =-0.7;
  x[3] =-0.5;
  // multiplication
  At_mul_B(y, A, x);
  mu_assert("error, y[0] !=-0.3", y[0] ==-0.3);
  mu_assert("error, y[1] !=-0.7", y[1] ==-0.7);
  mu_assert("error, y[2] != 0.8", y[2] == 0.8);
  return 0;
}

static char * test_randsubseq() {
  srand48(1234567890);
  long* x = malloc(1000 * sizeof(long));
  long nsamples = randsubseq(10000, 1000, 0.05, x);
  mu_assert("error, nsamples < 0", nsamples >= 0);
  mu_assert("error, x[0] outside",   x[0] >= 0 && x[0] < 10000);
  mu_assert("error, x[end] outside", x[nsamples-1] >= 0 && x[nsamples-1] < 10000);
  return 0;
}

static char * test_read_sbm() {
  struct SparseBinaryMatrix *A = read_sbm("data/sbm-100-50.data");
  mu_assert("error, nrow != 100", A->nrow == 100);
  mu_assert("error, ncol != 50",  A->ncol == 50);
  mu_assert("error, nnz != 504",  A->nnz  == 504);
  mu_assert("error, rows[0] != 8", A->rows[0] == 8);
  mu_assert("error, cols[0] != 0", A->cols[0] == 0);
  return 0;
}

static char * test_ceilPower2() {
  mu_assert("error, ceilPower2(16) != 16", ceilPower2(16) == 16);
  mu_assert("error, ceilPower2(15) != 16", ceilPower2(15) == 16);
  mu_assert("error, ceilPower2(17) != 32", ceilPower2(17) == 32);
  mu_assert("error, ceilPower2(1) != 1", ceilPower2(1) == 1);
  mu_assert("error, ceilPower2(2^30) != 2^30", ceilPower2(1073741824) == 1073741824);
  mu_assert("error, ceilPower2(2^30-1) != 2^30", ceilPower2(1073741823) == 1073741824);
  return 0;
}

static char * test_quickSort() {
  long a[] = { 7, 12, 1, -2, 0, 15, 4, 9, 11, 3, -1, 13, 5};
  int N = 13;

  quickSort(a, 0, N - 1);
  for (int i = 1; i < N; ++i) {
    mu_assert("error, quickSort gives wrong order", a[i-1] <= a[i]);
  }
  return 0;
}

static char * test_quickSort1000() {
  int N = 1000;
  long* a = malloc(N*sizeof(long));
  for (int i = 0; i < N; i++) {
    a[i] = (long)(2000*sin(i*17));
  }

  quickSort(a, 0, N - 1);
  for (int i = 1; i < N; ++i) {
    if (a[i-1] > a[i]) {
      printf("i = %d, a[i-1] = %ld, a[i] = %ld\n", i, a[i-1], a[i]);
    }
    mu_assert("error, quickSort for N=1000 gives wrong order", a[i-1] <= a[i]);
  }
  return 0;
}

static char * test_quickSortD() {
  long a[]   = { 7,  12, 1, -2, 0, 15, 4,   9, 11, 3,  -1, 13,  5};
  double v[] = { 1, 0.5, 3, -1, 7,  9, 0.1, 2, -3, 0, 1.2,  8, 10};
  int N = 13;

  double vs[] = { -1.0, 1.2, 7.0, 3.0, 0.0, 0.1, 10.0, 1.0, 2.0, -3.0, 0.5, 8.0, 9.0 };

  quickSortD(a, 0, N - 1, v);
  for (int i = 1; i < N; i++) {
    mu_assert("error, quickSortD gives wrong order", a[i-1] <= a[i]);
  }
  for (int i = 0; i < N; i++) {
    mu_assert("error, quickSortD gives wrong order for double[] v", vs[i] == v[i]);
  }
  return 0;
}

static char * test_hilbert() {
  int N = 4;
  int* rows = malloc(N * sizeof(int));
  int* cols = malloc(N * sizeof(int));
  int nhilbert = 131072;
  long h;

  rows[0] = 5931;
  cols[0] = 91204;
  h = xy2d(nhilbert, rows[0], cols[0]);
  d2xy(nhilbert, h, &rows[1], &cols[1]);
  mu_assert("error, rows[0] != rows[1]", rows[0] == rows[1]);
  mu_assert("error, cols[0] != cols[1]", cols[0] == cols[1]);

  return 0;
}

static char * test_sort_sbm() {
  struct SparseBinaryMatrix *A = read_sbm("data/sbm-100-50.data");
  double* x  = malloc(A->ncol * sizeof(double));
  double* y  = malloc(A->nrow * sizeof(double));
  double* y2 = malloc(A->nrow * sizeof(double));
  for (int i = 0; i < A->ncol; i++) {
    x[i] = sin(i*19 + 0.4) + cos(i*i*3);
  }
  A_mul_B(y, A, x);
  sort_sbm(A);
  A_mul_B(y2, A, x);
  for (int i = 0; i < A->nrow; i++) {
    mu_assert("error, sort_sbm changes A_mul_B", abs(y[i] - y2[i]) < 1e-6);
  }

  // making sure hilbert curve values are sorted
  int n = 128;
  long h = xy2d(n, A->rows[0], A->cols[0]);
  for (long j = 1; j < A->nnz; j++) {
    long h2 = xy2d(128, A->rows[j], A->cols[j]);
    mu_assert("error, sort_sbm does not give right order", h2 > h);
    h = h2;
  }
  return 0;
}

static char * test_blocked_sbm() {
  struct SparseBinaryMatrix *A = read_sbm("data/sbm-100-50.data");
  struct BlockedSBM *B = new_bsbm(A, 8);
  mu_assert("error, B->nrow != A->nrow", B->nrow == A->nrow);
  mu_assert("error, B->ncol != A->ncol", B->ncol == A->ncol);
  mu_assert("error, B->nblocks != 13", B->nblocks == 13);
  mu_assert("error, B->start_row[0] != 100", B->start_row[0] == 0);
  mu_assert("error, B->start_row[1] != 100", B->start_row[1] == 8);
  mu_assert("error, B->start_row[13] != 100", B->start_row[13] == 100);
  double* x  = malloc(B->ncol * sizeof(double));
  double* y  = malloc(B->nrow * sizeof(double));
  double* y2 = malloc(B->nrow * sizeof(double));
  for (int i = 0; i < A->ncol; i++) {
    x[i] = sin(i*17 + 0.2);
  }
  A_mul_B(y, A, x);
  A_mul_B_blocked(y2, B, x);
  double d = dist(y2, y, A->nrow);
  mu_assert("error, dist(y2,y) > 1e-6", d < 1e-6); 
  return 0;
}

static char * test_sort_bsbm() {
  struct SparseBinaryMatrix *A = read_sbm("data/sbm-100-50.data");
  struct BlockedSBM *B = new_bsbm(A, 8);
  double* x  = malloc(B->ncol * sizeof(double));
  double* y  = malloc(B->nrow * sizeof(double));
  double* y2 = malloc(B->nrow * sizeof(double));
  for (int i = 0; i < B->ncol; i++) {
    x[i] = sin(i*19 + 0.4) + cos(i*i*3);
  }
  A_mul_B_blocked(y, B, x);
  sort_bsbm(B);
  A_mul_B_blocked(y2, B, x);
  for (int i = 0; i < B->nrow; i++) {
    mu_assert("error, sort_bsbm changes A_mul_B", abs(y[i] - y2[i]) < 1e-6);
  }

  // making sure hilbert curve values are sorted
  for (int block = 0; block < B->nblocks; block++) {
    int n  = ceilPower2(B->start_row[block+1] - B->start_row[block]);
    long h = row_xy2d(n, B->rows[block][0] - B->start_row[block], B->cols[block][0]);

    for (long j = 1; j < B->nnz[block]; j++) {
      long h2 = row_xy2d(n, B->rows[block][j] - B->start_row[block], B->cols[block][j]);
      mu_assert("error, sort_bsbm does not give right order", h2 > h);
      h = h2;
    }
  }

  return 0;
}

//// tests for SparseDoubleMatrix
struct SparseDoubleMatrix* make_sdm() {
  struct SparseDoubleMatrix *A = malloc(sizeof(struct SparseDoubleMatrix));
  A->nrow = 6;
  A->ncol = 4;
  A->nnz  = 11;
  int *rows = malloc(A->nnz * sizeof(int));
  int *cols = malloc(A->nnz * sizeof(int));
  double *vals = malloc(A->nnz * sizeof(double));
  memcpy(rows, (int []){1, 1, 3, 4, 1, 4, 5, 0, 1, 2, 4}, A->nnz * sizeof(int));
  memcpy(cols, (int []){0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3}, A->nnz * sizeof(int));
  memcpy(vals, (double []){0.65, 0.84, 0.54, 0.59, 0.51, 0.27, 0.23, 0.94, 0.66, 0.31, 0.92}, A->nnz * sizeof(double));
  A->rows = rows;
  A->cols = cols;
  A->vals = vals;
  return A;
}

static char * test_A_mul_B_sdm() {
  struct SparseDoubleMatrix *A = make_sdm();
  double x[] = {0.5, -0.7, 1.9, 2.3};
  double* y = malloc(A->nrow * sizeof(double));
  double yt[] = {2.162, 2.224, 0.713, -0.378, 2.216, 0.437};
  // multiplication
  sdm_A_mul_B(y, A, x);
  for (int i = 0; i < A->nrow; i++) {
    mu_assert("error, sdm_A_mul_B is wrong", y[i] == yt[i]);
  }
  return 0;
}

static char * test_At_mul_B_sdm() {
  struct SparseDoubleMatrix *A = make_sdm();
  double x[] = {0.59, 0.37, 0.14, 0.21, 0.40, 0.81};
  double y[] = {0, 0, 0, 0};
  double yt[] = {0.2405, 0.6548, 0.4807, 1.2102};
  // multiplication
  sdm_At_mul_B(y, A, x);
  for (int i = 0; i < A->nrow; i++) {
    mu_assert("error, sdm_A_mul_B is wrong", abs(y[i] - yt[i]) < 1e-6);
  }
  return 0;
}

static char * test_read_sdm() {
  struct SparseDoubleMatrix *A = read_sdm("data/sdm-100-50.data");
  mu_assert("error, nrow != 100", A->nrow == 100);
  mu_assert("error, ncol != 50",  A->ncol == 50);
  mu_assert("error, nnz != 504",  A->nnz  == 470);
  mu_assert("error, rows[1] != 27", A->rows[1] == 27);
  mu_assert("error, cols[1] != 0", A->cols[1] == 0);
  mu_assert("error, vals[1] != 0.616153", abs(A->vals[1] - 0.616153) < 1e-5);
  mu_assert("error, rows[469] != 40", A->rows[469] == 40);
  mu_assert("error, cols[469] != 49", A->cols[469] == 49);
  mu_assert("error, vals[469] != 0.108172", abs(A->vals[469] - 0.108172) < 1e-5);
  return 0;
}

static char * test_blocked_sdm() {
  struct SparseDoubleMatrix *A = read_sdm("data/sdm-100-50.data");
  struct BlockedSDM *B = new_bsdm(A, 8);
  mu_assert("error, B->nrow != A->nrow", B->nrow == A->nrow);
  mu_assert("error, B->ncol != A->ncol", B->ncol == A->ncol);
  mu_assert("error, B->nblocks != 13", B->nblocks == 13);
  mu_assert("error, B->start_row[0] != 100", B->start_row[0] == 0);
  mu_assert("error, B->start_row[1] != 100", B->start_row[1] == 8);
  mu_assert("error, B->start_row[13] != 100", B->start_row[13] == 100);
  double* x  = malloc(B->ncol * sizeof(double));
  double* y  = malloc(B->nrow * sizeof(double));
  double* y2 = malloc(B->nrow * sizeof(double));
  double* y3 = malloc(B->nrow * sizeof(double));
  for (int i = 0; i < A->ncol; i++) {
    x[i] = sin(i*17 + 0.2);
  }
  sdm_A_mul_B(y, A, x);
  bsdm_A_mul_B(y2, B, x);
  sort_bsdm(B);
  bsdm_A_mul_B(y3, B, x);
  double d  = dist(y2, y, A->nrow);
  double d3 = dist(y3, y, A->nrow);
  mu_assert("error, dist(y2,y) > 1e-6", d < 1e-6); 
  mu_assert("error, dist(y3,y) > 1e-6", d3 < 1e-6); 
  return 0;
}

static char * all_tests() {
    mu_run_test(test_A_mul_B);
    mu_run_test(test_At_mul_B);
    mu_run_test(test_randsubseq);
    mu_run_test(test_read_sbm);
    mu_run_test(test_ceilPower2);
    mu_run_test(test_quickSort);
    mu_run_test(test_quickSort1000);
    mu_run_test(test_quickSortD);
    mu_run_test(test_hilbert);
    mu_run_test(test_sort_sbm);
    mu_run_test(test_blocked_sbm);
    mu_run_test(test_sort_bsbm);
    mu_run_test(test_A_mul_B_sdm);
    mu_run_test(test_At_mul_B_sdm);
    mu_run_test(test_read_sdm);
    mu_run_test(test_blocked_sdm);
    return 0;
}

int main(int argc, char **argv) {
    char *result = all_tests();
    if (result != 0) {
        printf("%s\n", result);
    }
    else {
        printf("ALL TESTS PASSED\n");
    }
    printf("Tests run: %d\n", tests_run);

    return result != 0;
}
