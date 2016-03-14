/* file minunit_example.c */

#include <stdio.h>
#include <stdlib.h>
#include "minunit.h"
#include "sparse.h"
#include "quickSortD.h"
#include "dsparse.h"
#include "linalg.h"
#include "cg.h"
#include "csr.h"

int tests_run = 0;

struct SparseBinaryMatrix* make_sbm() {
  struct SparseBinaryMatrix *A = (struct SparseBinaryMatrix*)malloc(sizeof(struct SparseBinaryMatrix));
  A->nrow = 4;
  A->ncol = 3;
  A->nnz  = 5;
  int* rows = (int*)malloc(A->nnz * sizeof(int));
  int* cols = (int*)malloc(A->nnz * sizeof(int));
  A->rows = rows;
  A->cols = cols;
  rows[0] = 0; rows[1] = 3; rows[2] = 3; rows[3] = 1; rows[4] = 2;
  cols[0] = 0; cols[1] = 2; cols[2] = 0; cols[3] = 2; cols[4] = 1;
  return A;
}

static char * test_A_mul_B() {
  struct SparseBinaryMatrix *A = make_sbm();
  double* x = (double*)malloc(A->ncol * sizeof(double));
  double* y = (double*)malloc(A->nrow * sizeof(double));
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

static char * test_A_mul_B2() {
  struct SparseBinaryMatrix *A = read_sbm("data/sbm-100-50.data");
  double* x  = (double*)malloc(A->ncol * sizeof(double));
  double* y  = (double*)malloc(A->nrow * sizeof(double));
  for (int i = 0; i < A->ncol; i++) {
    x[i] = sin(i*19 + 0.4) + cos(i*i*3);
  }
  A_mul_B(y, A, x);
  mu_assert("error, y[0]  != 1.70095",   fabs(y[0] - 1.70095) < 1e-4);
  mu_assert("error, y[99] != -0.174905", fabs(y[99] +0.174905) < 1e-4);
  return 0;
}

static char * test_bcsr_A_mul_B() {
  struct SparseBinaryMatrix *A = read_sbm("data/sbm-100-50.data");
  struct BinaryCSR B;
  bcsr_from_sbm(&B, A);
  double* x  = (double*)malloc(A->ncol * sizeof(double));
  double* y  = (double*)malloc(A->nrow * sizeof(double));
  double* y2 = (double*)malloc(A->nrow * sizeof(double));
  for (int i = 0; i < A->ncol; i++) {
    x[i] = sin(i*19 + 0.4) + cos(i*i*3);
  }
  A_mul_B(y2, A, x);
  bcsr_A_mul_B(y, &B, x);
  mu_assert("error, y[0]  != 1.70095",   fabs(y[0] - 1.70095) < 1e-4);
  mu_assert("error, y[99] != -0.174905", fabs(y[99] +0.174905) < 1e-4);
  for (int i = 0; i < A->nrow; i++) {
    mu_assert("error, y[i] != ytrue[i]", fabs(y[i] - y2[i]) < 1e-4);
  }
  return 0;
}

static char * test_bcsr_AA_mul_B() {
  struct SparseBinaryMatrix *A = read_sbm("data/sbm-100-50.data");
  struct BinaryCSR B;
  bcsr_from_sbm(&B, A);
  double* x  = (double*)malloc(A->ncol * sizeof(double));
  double* tmp = (double*)malloc(A->nrow * sizeof(double));
  double* y  = (double*)malloc(A->ncol * sizeof(double));
  double* y2 = (double*)malloc(A->ncol * sizeof(double));
  for (int i = 0; i < A->ncol; i++) {
    x[i] = sin(i*19 + 0.4) + cos(i*i*3);
  }
  A_mul_B(tmp, A, x);
  At_mul_B(y2, A, tmp);
  bcsr_AA_mul_B(y, &B, x);
  for (int i = 0; i < A->ncol; i++) {
    mu_assert("error, y[i] != ytrue[i]", fabs(y[i] - y2[i]) < 1e-4);
  }

  // testing parallel version
  int nthreads = 4;
#pragma omp parallel
  {
#pragma omp single
    {
      nthreads = omp_get_num_threads();
    }
  }
  double* ytmp = (double*)malloc(A->ncol * sizeof(double) * nthreads);
  parallel_bcsr_AA_mul_B(y, &B, x, ytmp);
  for (int i = 0; i < A->ncol; i++) {
    mu_assert("error, for parallel y[i] != ytrue[i]", fabs(y[i] - y2[i]) < 1e-4);
  }
  return 0;
}

static char * test_At_mul_B() {
  struct SparseBinaryMatrix *A = make_sbm();
  double* x = (double*)malloc(A->nrow * sizeof(double));
  double* y = (double*)malloc(A->ncol * sizeof(double));
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
  long* x = (long*)malloc(1000 * sizeof(long));
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
  long* a = (long*)malloc(N*sizeof(long));
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
  int* rows = (int*)malloc(N * sizeof(int));
  int* cols = (int*)malloc(N * sizeof(int));
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
  double* x  = (double*)malloc(A->ncol * sizeof(double));
  double* y  = (double*)malloc(A->nrow * sizeof(double));
  double* y2 = (double*)malloc(A->nrow * sizeof(double));
  for (int i = 0; i < A->ncol; i++) {
    x[i] = sin(i*19 + 0.4) + cos(i*i*3);
  }
  A_mul_B(y, A, x);
  sort_sbm(A);
  A_mul_B(y2, A, x);
  for (int i = 0; i < A->nrow; i++) {
    mu_assert("error, sort_sbm changes A_mul_B", fabs(y[i] - y2[i]) < 1e-6);
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
  double* x  = (double*)malloc(B->ncol * sizeof(double));
  double* y  = (double*)malloc(B->nrow * sizeof(double));
  double* y2 = (double*)malloc(B->nrow * sizeof(double));
  for (int i = 0; i < A->ncol; i++) {
    x[i] = sin(i*17 + 0.2);
  }
  A_mul_B(y, A, x);
  bsbm_A_mul_B(y2, B, x);
  double d = dist(y2, y, A->nrow);
  mu_assert("error, dist(y2,y) > 1e-6", d < 1e-6); 

  // test 2 col Y = B * X
  double* X  = (double*)malloc(B->ncol * 2 * sizeof(double));
  double* Y  = (double*)malloc(B->nrow * 2 * sizeof(double));
  for (int i = 0; i < A->ncol; i++) {
    X[i*2]   = sin(i*17 + 0.2);
    X[i*2+1] = sin(i*23 + 0.7);
    x[i]     = X[i*2+1];
  }
  A_mul_B(y2, A, x);
  bsbm_A_mul_B2(Y, B, X);
  for (int row = 0; row < A->nrow; row++) {
    mu_assert("error, |y[row] - Y[row,1]| > 1e-6", fabs(y[row] - Y[row*2]) < 1e-6);
    mu_assert("error, |y2[row] - Y[row,2]| > 1e-6", fabs(y2[row] - Y[row*2+1]) < 1e-6);
  }

  // checking also BinaryCSR
  struct BinaryCSR csr;
  bcsr_from_sbm(&csr, A);
  double* Y2 = (double*) malloc(B->nrow * 2 * sizeof(double));
  bcsr_A_mul_B2(Y2, &csr, X);
  for (int row = 0; row < A->nrow; row++) {
    mu_assert("error, |y[row] - Ycsr[row,1]| > 1e-6", fabs(y[row] - Y2[row*2]) < 1e-6);
    mu_assert("error, |y2[row] - Ycsr[row,2]| > 1e-6", fabs(y2[row] - Y2[row*2+1]) < 1e-6);
  }

  double* Y2n = (double*) malloc(B->nrow * 2 * sizeof(double));
  bcsr_A_mul_Bn(Y2n, &csr, X, 2);
  for (int row = 0; row < A->nrow; row++) {
    mu_assert("error, |y[row] - Ycsr_n[row,1]| > 1e-6", fabs(y[row] - Y2n[row*2]) < 1e-6);
    mu_assert("error, |y2[row] - Ycsr_n[row,2]| > 1e-6", fabs(y2[row] - Y2n[row*2+1]) < 1e-6);
  }
  return 0;
}

static char * test_row_xy2d() {
  mu_assert("row_xy2d(16, 0, 0) != 0", row_xy2d(16, 0, 0) == 0);
  mu_assert("row_xy2d(16, 0, 15) != 255", row_xy2d(16, 0, 15) == 255);
  mu_assert("row_xy2d(16, 0, 16) != 255", row_xy2d(16, 0, 16) == 256);
  mu_assert("row_xy2d(16, 0, 31) != 255", row_xy2d(16, 0, 31) == 511);
  mu_assert("row_xy2d(16, 1, 0) != 0", row_xy2d(16, 1, 0) == 3);

  int x, y;
  row_d2xy(16,   0, &x, &y); mu_assert("row_d2xy(16, 0, x, y) -> (0, 0)", x == 0 && y == 0);
  row_d2xy(16, 255, &x, &y); mu_assert("row_d2xy(16, 255, x, y) -> (0,15)", x == 0 && y == 15);
  row_d2xy(16, 256, &x, &y); mu_assert("row_d2xy(16, 256, x, y) -> (0,16)", x == 0 && y == 16);
  row_d2xy(16, 511, &x, &y); mu_assert("row_d2xy(16, 511, x, y) -> (0,31)", x == 0 && y == 31);
  row_d2xy(16,   3, &x, &y); mu_assert("row_d2xy(16, 3, x, y) -> (1, 0)", x == 1 && y == 0);
  return 0;
}

static char * test_sort_bsbm() {
  struct SparseBinaryMatrix *A = read_sbm("data/sbm-100-50.data");
  struct BlockedSBM *B = new_bsbm(A, 8);
  double* x  = (double*)malloc(B->ncol * sizeof(double));
  double* y  = (double*)malloc(B->nrow * sizeof(double));
  double* y2 = (double*)malloc(B->nrow * sizeof(double));
  for (int i = 0; i < B->ncol; i++) {
    x[i] = sin(i*19 + 0.4) + cos(i*i*3);
  }
  bsbm_A_mul_B(y, B, x);
  sort_bsbm(B);
  bsbm_A_mul_B(y2, B, x);
  for (int i = 0; i < B->nrow; i++) {
    mu_assert("error, sort_bsbm changes A_mul_B", fabs(y[i] - y2[i]) < 1e-6);
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
  struct SparseDoubleMatrix *A = (struct SparseDoubleMatrix*)malloc(sizeof(struct SparseDoubleMatrix));
  A->nrow = 6;
  A->ncol = 4;
  A->nnz  = 11;
  int *rows = (int*)malloc(A->nnz * sizeof(int));
  int *cols = (int*)malloc(A->nnz * sizeof(int));
  double *vals = (double*)malloc(A->nnz * sizeof(double));
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
  double* y = (double*)malloc(A->nrow * sizeof(double));
  double yt[] = {2.162, 2.224, 0.713, -0.378, 2.216, 0.437};
  // multiplication
  sdm_A_mul_B(y, A, x);
  for (int i = 0; i < A->nrow; i++) {
    mu_assert("error, sdm_A_mul_B is wrong", fabs(y[i] - yt[i]) < 1e-6);
  }
  return 0;
}

static char * test_At_mul_B_sdm() {
  struct SparseDoubleMatrix *A = make_sdm();
  double x[] = {0.59, 0.37, 0.14, 0.21, 0.40, 0.81};
  double y[] = {0, 0, 0, 0};
  double yt[] = {0.2405, 0.6602, 0.483, 1.2102};
  // multiplication
  sdm_At_mul_B(y, A, x);
  for (int i = 0; i < A->ncol; i++) {
    mu_assert("error, sdm_At_mul_B is wrong", fabs(y[i] - yt[i]) < 1e-6);
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
  mu_assert("error, vals[1] != 0.616153", fabs(A->vals[1] - 0.616153) < 1e-5);
  mu_assert("error, rows[469] != 40", A->rows[469] == 40);
  mu_assert("error, cols[469] != 49", A->cols[469] == 49);
  mu_assert("error, vals[469] != 0.108172", fabs(A->vals[469] - 0.108172) < 1e-5);
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
  double* x  = (double*)malloc(B->ncol * sizeof(double));
  double* y  = (double*)malloc(B->nrow * sizeof(double));
  double* y2 = (double*)malloc(B->nrow * sizeof(double));
  double* y3 = (double*)malloc(B->nrow * sizeof(double));
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

static char * test_dot_normsq() {
  double x[] = {0.12, -0.82, 1.3, 0.5};
  double y[] = {6.12, 0.19, 3.4, -4.1};

  mu_assert("pnormsq(x,4) != 2.6268", fabs(pnormsq(x,4) - 2.6268) < 1e-8);
  mu_assert("pnormsq(y,4) != 65.8605", fabs(pnormsq(y,4) - 65.8605) < 1e-8);
  mu_assert("pdot(x,y,4)  != 2.9486", fabs(pdot(x, y, 4) - 2.9486) < 1e-8);

  double normsq2[2];
  pnormsq2(normsq2, x, 2);
  mu_assert("pnormsq2(x,2)[0] not right", fabs(normsq2[0] - 0.12*0.12 - 1.3*1.3) < 1e-8);
  mu_assert("pnormsq2(x,2)[1] not right", fabs(normsq2[1] - 0.82*0.82 - 0.5*0.5) < 1e-8);

  double outer[3];
  pouter2(outer, x, 2);
  mu_assert("pouter(x,2)[0] not right", fabs(outer[0] - 0.12*0.12 - 1.3*1.3) < 1e-8);
  mu_assert("pouter(x,2)[1] not right", fabs(outer[1] - 0.82*0.82 - 0.5*0.5) < 1e-8);
  mu_assert("pouter(x,2)[2] not right", fabs(outer[2] - (0.12*(-0.82) + 1.3*0.5)) < 1e-8);
  return 0;
}

static char * test_pdot2() {
  double x[] = {0.95, 0.9, 0.16, 0.46, 0.86, 0.29};
  double y[] = {0.9695, 0.6678, 0.277, 0.1908, 1.108, 0.7632};
  double dot_true[] = {1.918225, 0.910116, 1.32129}; 
  double dot[3];

  pdot2sym(dot, x, y, 3);
  mu_assert("pdot2sym(x, y, 3)[0] not right", fabs(dot[0] - dot_true[0]) < 1e-8);
  mu_assert("pdot2sym(x, y, 3)[1] not right", fabs(dot[1] - dot_true[1]) < 1e-8);
  mu_assert("pdot2sym(x, y, 3)[2] not right", fabs(dot[2] - dot_true[2]) < 1e-8);

  return 0;
}

static char * test_solve2sym() {
  double A[3]   = {0.59,  1.34,  0.86};
  double RHS[4] = {-1.21, 1.91, -0.82, 0.03};
  double Xtrue[4] = {-64.0, 42.5, -22.05098039, 14.1745098};
  double X[4];

  solve2sym(X, A, RHS);
  mu_assert("solve2sym(X, A, RHS)[0] not right", fabs(X[0] - Xtrue[0]) < 1e-8);
  mu_assert("solve2sym(X, A, RHS)[1] not right", fabs(X[1] - Xtrue[1]) < 1e-8);
  mu_assert("solve2sym(X, A, RHS)[2] not right", fabs(X[2] - Xtrue[2]) < 1e-8);
  mu_assert("solve2sym(X, A, RHS)[3] not right", fabs(X[3] - Xtrue[3]) < 1e-8);
  return 0;
}

static char * test_cg() {
  struct SparseBinaryMatrix *A = read_sbm("data/sbm-100-50.data");
  sort_sbm(A);
  struct BlockedSBM *B = new_bsbm(A, 8);
  transpose(A);
  struct BlockedSBM *Bt = new_bsbm(A, 8);

  int F = B->ncol;
  double* x  = (double*)malloc(F * sizeof(double));
  double* b  = (double*)malloc(F * sizeof(double));

  for (int i = 0; i < F; i++) {
    b[i] = sin(i*19 + 0.4) + cos(i*i*3);
  }
  int numIter = 0;

  double lambda = 5.0;
  bsbm_cg(x, B, Bt, b, lambda, 1e-6, &numIter);

  // verifying result:
  mu_assert("x[0] != 0.0638578", fabs(x[0] - 0.0638578) < 1e-4);
  mu_assert("x[1] !=-0.0302702", fabs(x[1] + 0.0302702) < 1e-4);

  struct SparseBinaryMatrix *X = read_sbm("data/sbm-100-50.data");
  double* tmp  = (double*)malloc(X->nrow * sizeof(double));
  double* tmp2 = (double*)malloc(X->ncol * sizeof(double));
  A_mul_B(tmp, X, x);
  At_mul_B(tmp2, X, tmp);
  for (int i = 0; i < B->ncol; i++) {
    tmp2[i] += lambda * x[i];
  }
  double d = dist(tmp2, b, X->ncol);
  mu_assert("resid < 1e-5", d < 1e-5);

  // CG for 2 RHSs
  double* x2 = (double*)malloc(F * 2 * sizeof(double));
  double* b2 = (double*)malloc(F * 2 * sizeof(double));

  for (int i = 0; i < F; i++) {
    b2[2*i]   = sin(i*19 + 0.4) + cos(i*i*3);
    b2[2*i+1] = cos(i*23 + 0.7) + sin(i*i*7);
  }
  bsbm_cg2(x2, B, Bt, b2, lambda, 1e-6, &numIter);

  mu_assert("x2[0] != 0.0638578", fabs(x2[0] - 0.0638578) < 1e-4);
  mu_assert("x2[1] !=-0.0302702", fabs(x2[2] + 0.0302702) < 1e-4);
  
  return 0;
}

static char * test_bcsr_serialization() {
  const char* tmp_file = "/tmp/sbm-100-50.data.csr.bin";
  struct SparseBinaryMatrix *A = read_sbm("data/sbm-100-50.data");
  struct BinaryCSR B_in, B_out;
  bcsr_from_sbm(&B_in, A);
  serialize_to_file(     &B_in,  tmp_file );
  deserialize_from_file( &B_out, tmp_file );
  mu_assert( "serialization of BinaryCSR::nnz not correct",  B_in.nnz  == B_out.nnz);
  mu_assert( "serialization of BinaryCSR::nrow not correct", B_in.nrow == B_out.nrow);
  mu_assert( "serialization of BinaryCSR::nnz not correct",  B_in.ncol == B_out.ncol );

  int eq_cols = 1, eq_row_ptr = 1;
  for ( long i = 0; i < B_out.nnz; i++ ) {
    eq_cols = eq_cols && (B_in.cols[i] == B_out.cols[i]);
  }
  mu_assert( "serialization of BinaryCSR::cols not correct", eq_cols);

  for ( int i = 0; i < B_out.nrow + 1; i++ ) {
    eq_row_ptr = eq_row_ptr && (B_in.row_ptr[i] == B_out.row_ptr[i]);
  }
  mu_assert( "serialization of BinaryCSR::row_ptr not correct", eq_row_ptr);
  free_bcsr(&B_out);
  free_bcsr(&B_in);
  return 0;
}


static char * all_tests() {
    mu_run_test(test_A_mul_B);
    mu_run_test(test_A_mul_B2);
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
    mu_run_test(test_row_xy2d);
    mu_run_test(test_dot_normsq);
    mu_run_test(test_pdot2);
    mu_run_test(test_solve2sym);
    mu_run_test(test_cg);
    mu_run_test(test_bcsr_A_mul_B);
    mu_run_test(test_bcsr_AA_mul_B);
    mu_run_test(test_bcsr_serialization);
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
