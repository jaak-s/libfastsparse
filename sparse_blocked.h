#ifndef SPARSE_BLOCKED_H
#define SPARSE_BLOCKED_H

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "hilbert.h"
#include "quickSort.h"
#include "sparse.h"

struct BlockedSBM {
  int nrow;
  int ncol;
  /** row blocks */
  int nblocks;
  int* start_row;
  int* nnz;
  int** rows;
  int** cols;
};

/** constructor for blocked rows */
struct BlockedSBM* new_bsbm(struct SparseBinaryMatrix* A, int block_size) {
  struct BlockedSBM *B = malloc(sizeof(struct BlockedSBM));
  B->nrow = A->nrow;
  B->ncol = A->ncol;
  B->nblocks = (int)ceil(A->nrow / (double)block_size);
  B->nnz       = malloc(B->nblocks * sizeof(int));

  // array of starting rows, including the last
  B->start_row = malloc((B->nblocks + 1) * sizeof(int));
  for (int i = 0; i < B->nblocks; i++) {
    B->start_row[i] = i * block_size;
    B->nnz[i] = 0;
  }
  B->start_row[B->nblocks] = B->nrow;
  
  B->rows = malloc(B->nblocks * sizeof(int*));
  B->cols = malloc(B->nblocks * sizeof(int*));

  // counting nnz in each block
  for (long j = 0; j < A->nnz; j++) {
    int block = A->rows[j] / block_size;
    B->nnz[block]++;
  }
  int* bcounts = malloc(B->nblocks * sizeof(int));
  for (int i = 0; i < B->nblocks; i++) {
    bcounts[i] = 0;
    B->rows[i] = malloc(B->nnz[i] * sizeof(int));
    B->cols[i] = malloc(B->nnz[i] * sizeof(int));
  }

  for (long j = 0; j < A->nnz; j++) {
    int block = A->rows[j] / block_size;
    B->rows[block][bcounts[block]] = A->rows[j];
    B->cols[block][bcounts[block]] = A->cols[j];
    bcounts[block]++;
  }

  return B;
}

void sort_bsbm(struct BlockedSBM *B) {
  for (int block = 0; block < B->nblocks; block++) {
    int* rows = B->rows[block];
    int* cols = B->cols[block];
    int nnz = B->nnz[block];
    int start_row = B->start_row[block];
    int n = ceilPower2(B->start_row[block+1] - B->start_row[block]);

    // convert to hilbert, sort, convert back
    long* h = malloc(nnz * sizeof(long));
    for (long j = 0; j < nnz; j++) {
      h[j] = row_xy2d(n, rows[j] - start_row, cols[j]);
    }
    quickSort(h, 0, nnz - 1);
    for (long j = 0; j < nnz; j++) {
      row_d2xy(n, h[j], &rows[j], &cols[j]);
      rows[j] += start_row;
    }

    free(h);
  }
}

void sort_bsbm_byrow(struct BlockedSBM *B) {
  for (int block = 0; block < B->nblocks; block++) {
    int* rows = B->rows[block];
    int* cols = B->cols[block];
    int nnz = B->nnz[block];

    long* h = malloc(nnz * sizeof(long));
    for (long j = 0; j < nnz; j++) {
      h[j] = rows[j] * B->ncol + cols[j];
    }
    quickSort(h, 0, nnz - 1);
    for (long j = 0; j < nnz; j++) {
      rows[j] = h[j] / B->ncol;
      cols[j] = h[j] % B->ncol;
    }

    free(h);
  }
}

/** y = B * x */
void A_mul_B_blocked(double* y, struct BlockedSBM *B, double* x) {
#pragma omp parallel for schedule(dynamic, 1)
  for (int block = 0; block < B->nblocks; block++) {
    int* rows = B->rows[block];
    int* cols = B->cols[block];
    int nnz = B->nnz[block];

    // zeroing y:
    memset(y + B->start_row[block], 0, (B->start_row[block+1] - B->start_row[block]) * sizeof(double));

    for (int j = 0; j < nnz; j++) {
      y[rows[j]] += x[cols[j]];
    }
  }
}

#endif /* SPARSE_BLOCKED_H */
