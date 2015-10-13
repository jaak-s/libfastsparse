#ifndef SPARSE_H
#define SPARSE_H

#include <stdlib.h>
#include <math.h>

struct SparseBinaryMatrix
{
  int nrow;
  int ncol;
  long nnz;
  int* rows;
  int* cols;
};

/** y = A * x */
void A_mul_B(double* y, struct SparseBinaryMatrix *A, double* x) {
  int* rows = A->rows;
  int* cols = A->cols;
  for (int i = 0; i < A->nrow; i++) {
    y[i] = 0.0;
  }
  for (long j = 0; j < A->nnz; j++) {
    y[rows[j]] += x[cols[j]];
  }
}

/** y = A' * x */
void At_mul_B(double* y, struct SparseBinaryMatrix *A, double* x) {
  int* rows = A->rows;
  int* cols = A->cols;
  for (int i = 0; i < A->ncol; i++) {
    y[i] = 0.0;
  }
  for (long j = 0; j < A->nnz; j++) {
    y[cols[j]] += x[rows[j]];
  }
}

double exprand() {
  double x = 1 - drand48();
  return log1p(x);
  
}

#endif /* SPARSE_H */
