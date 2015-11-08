#ifndef SPARSE_H
#define SPARSE_H

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "hilbert.h"
#include "quickSort.h"

struct SparseBinaryMatrix
{
  int nrow;
  int ncol;
  long nnz;
  int* rows;
  int* cols;
};

/** constructor, computes nrow and ncol from data */
struct SparseBinaryMatrix* new_sbm(long nnz, int* rows, int* cols) {
  struct SparseBinaryMatrix *A = malloc(sizeof(struct SparseBinaryMatrix));
  A->nnz  = nnz;
  A->rows = rows;
  A->cols = cols;
  A->nrow = 0;
  A->ncol = 0;
  for (int i = 0; i < nnz; i++) {
    if (rows[i] >= A->nrow) A->nrow = rows[i] + 1;
    if (cols[i] >= A->ncol) A->ncol = cols[i] + 1;
  }
  return A;
}

/** y = A * x */
void A_mul_B(double* y, struct SparseBinaryMatrix *A, double* x) {
  int* rows = A->rows;
  int* cols = A->cols;
  memset(y, 0, A->nrow * sizeof(double));
  for (long j = 0; j < A->nnz; j++) {
    y[rows[j]] += x[cols[j]];
  }
}

/** y = A' * x */
void At_mul_B(double* y, struct SparseBinaryMatrix *A, double* x) {
  int* rows = A->rows;
  int* cols = A->cols;
  memset(y, 0, A->ncol * sizeof(double));
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

struct SparseBinaryMatrix* read_sbm(const char *filename) {
  FILE* fh = fopen( filename, "r" );
  size_t result1, result2;
  if (fh == NULL) {
    fprintf( stderr, "File error: %s\n", filename );
    exit(1);
  }
  long nnz = 0;
  result1 = fread(&nnz, sizeof(long), 1, fh);
  if (result1 != 1) {
    fprintf( stderr, "File reading error for nnz: %s\n", filename );
    exit(1);
  }
  // reading data
  int* rows = malloc(nnz * sizeof(int));
  int* cols = malloc(nnz * sizeof(int));
  result1 = fread(rows, sizeof(int), nnz, fh);
  result2 = fread(cols, sizeof(int), nnz, fh);
  if (result1 != nnz || result2 != nnz) {
    fprintf( stderr, "File read error: %s\n", filename );
    exit(1);
  }
  fclose(fh);
  // convert data from 1 based to 0 based
  for (long i = 0; i < nnz; i++) {
    rows[i]--;
    cols[i]--;
  }

  return new_sbm(nnz, rows, cols);
} 

int ceilPower2(int x) {
  return 1 << (int)ceil(log2(x));
}

/** sorts SBM according to Hilbert curve */
void sort_sbm(struct SparseBinaryMatrix *A) {
  int* rows = A->rows;
  int* cols = A->cols;

  int maxrc = A->nrow > A->ncol ? A->nrow : A->ncol;
  int n = ceilPower2(maxrc);

  long* h = malloc(A->nnz * sizeof(long));
  for (long j = 0; j < A->nnz; j++) {
    h[j] = xy2d(n, rows[j], cols[j]);
  }
  quickSort(h, 0, A->nnz - 1);
  for (long j = 0; j < A->nnz; j++) {
    d2xy(n, h[j], &rows[j], &cols[j]);
  }

  free(h);
}

double dist(double* x, double* y, int n) {
  double d = 0;
  for (int i = 0; i < n; i++) {
    double diff = x[i] - y[i];
    d += diff*diff;
  }
  return d;
}

#endif /* SPARSE_H */
