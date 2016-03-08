#ifndef CSR_H
#define CSR_H

#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "quickSort.h"

/*** binary CSR ***/
struct BinaryCSR
{
  int nrow;
  int ncol;
  long nnz;
  int* row_ptr; /* points to the row starts for each row */
  int* cols;
};

struct BinaryCSR* bcsr_from_sbm(struct SparseBinaryMatrix* sbm) {
  struct BinaryCSR *A = (struct BinaryCSR*)malloc(sizeof(struct BinaryCSR));
  A->nnz  = sbm->nnz;
  A->nrow = sbm->nrow;
  A->ncol = sbm->ncol;
  A->cols = (int*)malloc(A->nnz * sizeof(int));
  A->row_ptr = (int*)malloc( (A->nrow + 1) * sizeof(int));

  // sorting by row:
  long* h = (long*)malloc(A->nnz * sizeof(long));
#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < A->nnz; i++) {
      h[i] = sbm->rows[i] * (long)sbm->ncol + (long)sbm->cols[i];
  }
  quickSort(h, 0, A->nnz - 1);
  int row_prev = -1;
  for (int i = 0; i < A->nnz; i++) {
    A->cols[i] = h[i] % A->ncol;
    if (A->cols[i] < 0) {
       printf("A->cols[%d] = %d, h[%d]=%ld\n", i, A->cols[i], i, h[i]);
    }
    int row    = h[i] / A->ncol;
    while (row > row_prev) {
      row_prev++;
      A->row_ptr[row_prev] = i;
    }
  }
  for (int row = row_prev + 1; row <= A->nrow; row++) {
    A->row_ptr[row] = A->nnz;
  }
  free(h);

  return A;
}


/** y = A * x */
inline void bcsr_A_mul_B(double* y, struct BinaryCSR *A, double* x) {
  int* row_ptr = A->row_ptr;
  int* cols    = A->cols;
#pragma omp parallel for schedule(dynamic, 256)
  for (int row = 0; row < A->nrow; row++) {
    double tmp = 0;
    for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++) {
      tmp += x[cols[i]];
    }
    y[row] = tmp;
  }
}

/** Y = A * X, where Y and X have two columns and are row-ordered */
inline void bcsr_A_mul_B2(double* Y, struct BinaryCSR *A, double* X) {
  int* row_ptr = A->row_ptr;
  int* cols    = A->cols;
#pragma omp parallel for schedule(dynamic, 256)
  for (int row = 0; row < A->nrow; row++) {
    double tmp1 = 0;
    double tmp2 = 0;
    for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++) {
      int col = cols[i] * 2;
      tmp1 += X[col];
      tmp2 += X[col+1];
    }
    int r = row * 2;
    Y[r]   = tmp1;
    Y[r+1] = tmp2;
  }
}

/** Y = A * X, where Y and X have 8 columns and are row-ordered */
inline void bcsr_A_mul_B8(double* Y, struct BinaryCSR *A, double* X) {
  int* row_ptr = A->row_ptr;
  int* cols    = A->cols;
#pragma omp parallel for schedule(dynamic, 256)
  for (int row = 0; row < A->nrow; row++) {
    double tmp[8] = { 0 };
    for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++) {
      int col = cols[i] << 3; // multiplying with 8
      for (int j = 0; j < 8; j++) {
         tmp[j] += X[col + j];
      }
    }
    int r = row << 3;
    for (int j = 0; j < 8; j++) {
      Y[r + j] = tmp[j];
    }
  }
}

/** y = A'A * x */
inline void bcsr_AA_mul_B(double* y, struct BinaryCSR *A, double* x) {
  int* row_ptr = A->row_ptr;
  int* cols    = A->cols;
  const int nrow = A->nrow;
  memset(y, 0, A->ncol * sizeof(double));
  for (int row = 0; row < nrow; row++) {
    double xv = 0;
    for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++) {
      xv += x[cols[i]];
    }
    for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++) {
      y[cols[i]] += xv;
    }
  }
}

/** y = A'A * x in parallel */
inline void parallel_bcsr_AA_mul_B(double* y, struct BinaryCSR *A, double* x, double* ytmp) {
  int* row_ptr = A->row_ptr;
  int* cols    = A->cols;
  const int nrow    = A->nrow;
  const int ncol    = A->ncol;
#pragma omp parallel
  {
    const int nthreads = omp_get_num_threads();
    const int ithread  = omp_get_thread_num();
    const int ytmp_size = ncol * nthreads;

    double* ytmpi = & ytmp[ncol * ithread];
    memset(ytmpi, 0, ncol * sizeof(double));

#pragma omp for schedule(dynamic, 1024)
    for (int row = 0; row < nrow; row++) {
      double xv = 0;
      for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++) {
        xv += x[cols[i]];
      }
      for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++) {
        ytmpi[cols[i]] += xv;
      }
    }

#pragma omp for schedule(static)
    for (int col = 0; col < ncol; col++) {
      y[col] = 0;
      for (int i = col; i < ytmp_size; i += ncol) {
        y[col] += ytmp[i];
      }
    }

  }
}

#endif /* CSR_H */
