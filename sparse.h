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

double randexp() {
  return -log(1 - drand48());
}

/** computes random set of samples:
 *  @input N          range from 0 to N-1
 *  @output samples   array for storing sampled values
 *  @returns          number of sampled values
 **/
long randsubseq(long N, long max_samples, double p, long* samples) {
  double L = -1.0 / log1p(-p);
  long i = -1;
  long j = 0;

  while (1) {
    double s = randexp() * L;
    if (s + i >= N - 1) {
      return j;
    }
    i += (long)ceil(s);
    // adding j-th sample:
    samples[j] = i;
    j++;
    if (j >= max_samples) {
      return j;
    }
  }
}

#endif /* SPARSE_H */
