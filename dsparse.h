#ifndef SPARSE_H
#define SPARSE_H

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "hilbert.h"
#include "quickSort.h"

struct SparseDoubleMatrix
{
  int nrow;
  int ncol;
  long nnz;
  int* rows;
  int* cols;
  double* vals;
};

/** constructor, computes nrow and ncol from data */
struct SparseDoubleMatrix* new_sdm(long nnz, int* rows, int* cols, double* vals) {
  struct SparseDoubleMatrix *A = malloc(sizeof(struct SparseDoubleMatrix));
  A->nnz  = nnz;
  A->rows = rows;
  A->cols = cols;
  A->vals = vals;
  A->nrow = 0;
  A->ncol = 0;
  for (int i = 0; i < nnz; i++) {
    if (rows[i] >= A->nrow) A->nrow = rows[i] + 1;
    if (cols[i] >= A->ncol) A->ncol = cols[i] + 1;
  }
  return A;
}

void transpose(struct SparseDoubleMatrix *A) {
  int* tmp = A->rows;
  A->rows = A->cols;
  A->cols = tmp;
  int ntmp = A->nrow;
  A->nrow = A->ncol;
  A->ncol = ntmp;
}

/** y = A * x */
void A_mul_B(double* y, struct SparseDoubleMatrix *A, double* x) {
  int* rows = A->rows;
  int* cols = A->cols;
  memset(y, 0, A->nrow * sizeof(double));
  for (long j = 0; j < A->nnz; j++) {
    y[rows[j]] += x[cols[j]];
  }
}

/** y = A' * x */
void At_mul_B(double* y, struct SparseDoubleMatrix *A, double* x) {
  int* rows = A->rows;
  int* cols = A->cols;
  double* vals = A->vals;
  memset(y, 0, A->ncol * sizeof(double));
  for (long j = 0; j < A->nnz; j++) {
    y[cols[j]] += x[rows[j]] * vals[j];
  }
}

long read_long(FILE* fh) {
  long value;
  result1 = fread(&value, sizeof(long), 1, fh);
  if (result1 != 1) {
    fprintf( stderr, "File reading error for a long. File is corrupt.\n");
    exit(1);
  }
  return value;
}

struct SparseDoubleMatrix* read_sdm(const char *filename) {
  FILE* fh = fopen( filename, "r" );
  size_t result1, result2;
  if (fh == NULL) {
    fprintf( stderr, "File error: %s\n", filename );
    exit(1);
  }
  long nrow = read_long(fh);
  long ncol = read_long(fh);
  long nnz  = read_long(fh);
  // reading data
  int* rows = malloc(nnz * sizeof(int));
  int* cols = malloc(nnz * sizeof(int));
  double* vals = malloc(nnz * sizeof(double));
  result1 = fread(rows, sizeof(int), nnz, fh);
  result2 = fread(cols, sizeof(int), nnz, fh);
  result3 = fread(vals, sizeof(double), nnz, fh);
  if (result1 != nnz || result2 != nnz || result3 != nnz) {
    fprintf( stderr, "File read error: %s\n", filename );
    exit(1);
  }
  fclose(fh);
  // convert data from 1 based to 0 based
  for (long i = 0; i < nnz; i++) {
    rows[i]--;
    cols[i]--;
  }

  return new_sdm(nnz, rows, cols, vals);
} 

/** sorts SDM according to Hilbert curve */
void sort_sdm(struct SparseDoubleMatrix *A) {
  int* rows = A->rows;
  int* cols = A->cols;

  int maxrc = A->nrow > A->ncol ? A->nrow : A->ncol;
  int n = ceilPower2(maxrc);

  long* h = malloc(A->nnz * sizeof(long));
  for (long j = 0; j < A->nnz; j++) {
    h[j] = xy2d(n, rows[j], cols[j]);
  }
  quickSort(h, 0, A->nnz - 1);
  ///////// BUG: double values are not sorted!!!!
  for (long j = 0; j < A->nnz; j++) {
    d2xy(n, h[j], &rows[j], &cols[j]);
  }

  free(h);
}

#endif /* SPARSE_H */
