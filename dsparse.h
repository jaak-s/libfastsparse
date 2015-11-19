#ifndef DSPARSE_H
#define DSPARSE_H

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "hilbert.h"
#include "quickSortD.h"

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
struct SparseDoubleMatrix* new_sdm(long nrow, long ncol, long nnz, int* rows, int* cols, double* vals) {
  struct SparseDoubleMatrix *A = (struct SparseDoubleMatrix*)malloc(sizeof(struct SparseDoubleMatrix));
  A->nnz  = nnz;
  A->rows = rows;
  A->cols = cols;
  A->vals = vals;
  A->nrow = nrow;
  A->ncol = ncol;
  return A;
}

void sdm_transpose(struct SparseDoubleMatrix *A) {
  int* tmp = A->rows;
  A->rows = A->cols;
  A->cols = tmp;
  int ntmp = A->nrow;
  A->nrow = A->ncol;
  A->ncol = ntmp;
}

/** y = A * x */
void sdm_A_mul_B(double* y, struct SparseDoubleMatrix *A, double* x) {
  int* rows = A->rows;
  int* cols = A->cols;
  double* vals = A->vals;
  memset(y, 0, A->nrow * sizeof(double));
  for (long j = 0; j < A->nnz; j++) {
    y[rows[j]] += x[cols[j]] * vals[j];
  }
}

/** y = A' * x */
void sdm_At_mul_B(double* y, struct SparseDoubleMatrix *A, double* x) {
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
  size_t result1 = fread(&value, sizeof(long), 1, fh);
  if (result1 != 1) {
    fprintf( stderr, "File reading error for a long. File is corrupt.\n");
    exit(1);
  }
  return value;
}

struct SparseDoubleMatrix* read_sdm(const char *filename) {
  FILE* fh = fopen( filename, "r" );
  size_t result1, result2, result3;
  if (fh == NULL) {
    fprintf( stderr, "File error: %s\n", filename );
    exit(1);
  }
  long nrow = read_long(fh);
  long ncol = read_long(fh);
  long nnz  = read_long(fh);
  // reading data
  int* rows = (int*)malloc(nnz * sizeof(int));
  int* cols = (int*)malloc(nnz * sizeof(int));
  double* vals = (double*)malloc(nnz * sizeof(double));
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

  return new_sdm(nrow, ncol, nnz, rows, cols, vals);
} 

/** sorts SDM according to Hilbert curve */
void sort_sdm(struct SparseDoubleMatrix *A) {
  int* rows = A->rows;
  int* cols = A->cols;

  int maxrc = A->nrow > A->ncol ? A->nrow : A->ncol;
  int n = ceilPower2(maxrc);

  long* h = (long*)malloc(A->nnz * sizeof(long));
  for (long j = 0; j < A->nnz; j++) {
    h[j] = xy2d(n, rows[j], cols[j]);
  }
  quickSortD(h, 0, A->nnz - 1, A->vals);
  for (long j = 0; j < A->nnz; j++) {
    d2xy(n, h[j], &rows[j], &cols[j]);
  }

  free(h);
}

//// Blocked SparseDoubleMatrix ////

struct BlockedSDM {
  int nrow;
  int ncol;
  /** row blocks */
  int nblocks;
  int* start_row;
  int* nnz;
  int** rows;
  int** cols;
  double** vals;
};

/** constructor for blocked rows */
struct BlockedSDM* new_bsdm(struct SparseDoubleMatrix* A, int block_size) {
  struct BlockedSDM *B = (struct BlockedSDM*)malloc(sizeof(struct BlockedSDM));
  B->nrow = A->nrow;
  B->ncol = A->ncol;
  B->nblocks = (int)ceil(A->nrow / (double)block_size);
  B->nnz     = (int*)malloc(B->nblocks * sizeof(int));

  // array of starting rows, including the last
  B->start_row = (int*)malloc((B->nblocks + 1) * sizeof(int));
  for (int i = 0; i < B->nblocks; i++) {
    B->start_row[i] = i * block_size;
    B->nnz[i] = 0;
  }
  B->start_row[B->nblocks] = B->nrow;
  
  B->rows = (int**)malloc(B->nblocks * sizeof(int*));
  B->cols = (int**)malloc(B->nblocks * sizeof(int*));
  B->vals = (double**)malloc(B->nblocks * sizeof(double*));

  // counting nnz in each block
  for (long j = 0; j < A->nnz; j++) {
    int block = A->rows[j] / block_size;
    B->nnz[block]++;
  }
  int* bcounts = (int*)malloc(B->nblocks * sizeof(int));
  for (int i = 0; i < B->nblocks; i++) {
    bcounts[i] = 0;
    B->rows[i] = (int*)malloc(B->nnz[i] * sizeof(int));
    B->cols[i] = (int*)malloc(B->nnz[i] * sizeof(int));
    B->vals[i] = (double*)malloc(B->nnz[i] * sizeof(double));
  }

  for (long j = 0; j < A->nnz; j++) {
    int block = A->rows[j] / block_size;
    B->rows[block][bcounts[block]] = A->rows[j];
    B->cols[block][bcounts[block]] = A->cols[j];
    B->vals[block][bcounts[block]] = A->vals[j];
    bcounts[block]++;
  }

  return B;
}

/** y = B * x */
void bsdm_A_mul_B(double* y, struct BlockedSDM *B, double* x) {
#pragma omp parallel for schedule(dynamic, 1)
  for (int block = 0; block < B->nblocks; block++) {
    int* rows = B->rows[block];
    int* cols = B->cols[block];
    double* vals = B->vals[block];
    int nnz = B->nnz[block];

    // zeroing y:
    memset(y + B->start_row[block], 0, (B->start_row[block+1] - B->start_row[block]) * sizeof(double));

    for (int j = 0; j < nnz; j++) {
      y[rows[j]] += x[cols[j]] * vals[j];
    }
  }
}

void sort_bsdm(struct BlockedSDM *B) {
  for (int block = 0; block < B->nblocks; block++) {
    int* rows = B->rows[block];
    int* cols = B->cols[block];
    int nnz = B->nnz[block];
    int start_row = B->start_row[block];
    int n = ceilPower2(B->start_row[block+1] - B->start_row[block]);

    // convert to hilbert, sort, convert back
    long* h = (long*)malloc(nnz * sizeof(long));
    for (long j = 0; j < nnz; j++) {
      h[j] = row_xy2d(n, rows[j] - start_row, cols[j]);
    }
    quickSortD(h, 0, nnz - 1, B->vals[block]);
    for (long j = 0; j < nnz; j++) {
      row_d2xy(n, h[j], &rows[j], &cols[j]);
      rows[j] += start_row;
    }

    free(h);
  }
}
#endif /* DSPARSE_H */
