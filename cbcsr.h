#ifndef CBCSR_H
#define CBCSR_H

/*** ColBinaryCSR ***/
struct ColBinaryCSR
{
  int nrow;
  int ncol;
  int nblocks;
  int colblocksize;
  int nnz;      /* nnz */
  int* row_ptr; /* points to the row starts for each row, size: nblocks * nrow + 1 */
  int* cols;    /* list of all column values */
};

static inline void new_cbcsr(
    struct ColBinaryCSR* __restrict__ A,
    int colblocksize, /* number of columns in a block */
    long nnz,
    int nrow,
    int ncol,
    int* rows,
    int* cols)
{
  assert(A);
  A->nnz  = nnz;
  A->nrow = nrow;
  A->ncol = ncol;
  A->nblocks      = ceil(ncol / (double) colblocksize);
  A->colblocksize = colblocksize;
  A->cols    = (int*)malloc(nnz * sizeof(int));
  A->row_ptr = (int*)malloc( (A->nblocks * A->nrow + 1) * sizeof(int));

  //compute number of non-zero entries per row of A
  for (int row = 0; row < A->nrow * A->nblocks + 1; row++) {
    A->row_ptr[row] = 0;
  }

  for (int i = 0; i < nnz; i++) {
    int block = cols[i] / colblocksize;
    A->row_ptr[block * nrow + rows[i]]++;
  }
  // cumsum counts
  for (int row = 0, cumsum = 0, end = A->nrow * A->nblocks; row < end; row++) {
    int temp = A->row_ptr[row];
    A->row_ptr[row] = cumsum;
    cumsum += temp;
  }
  A->row_ptr[A->nrow * A->nblocks] = nnz;

  // writing cols and vals to A->cols and A->vals
  for (int i = 0; i < nnz; i++) {
    int row   = rows[i];
    int block = cols[i] / colblocksize;
    int cell  = block * nrow + row;
    int dest  = A->row_ptr[cell];
    A->cols[dest] = cols[i];
    A->row_ptr[cell]++;
  }
  for (int row = 0, prev = 0, end = A->nrow * A->nblocks; row <= end; row++) {
    int temp        = A->row_ptr[row];
    A->row_ptr[row] = prev;
    prev            = temp;
  }
}

static inline void cbcsr_from_sbm(struct ColBinaryCSR* __restrict__ A,
                                  struct SparseBinaryMatrix* __restrict__ sbm,
                                  int colblocksize) {
  assert(A);
  assert(sbm);
  new_cbcsr(A, colblocksize, sbm->nnz, sbm->nrow, sbm->ncol, sbm->rows, sbm->cols);
}

/** y = A * x */
inline void cbcsr_A_mul_B(double* y, struct ColBinaryCSR *A, double* x) {
  int* row_ptr = A->row_ptr;
  int* cols    = A->cols;
  const int nrow = A->nrow;

  memset(y, 0, nrow * sizeof(double));
#pragma omp parallel
  {
    double* ytmp = (double*)malloc(nrow * sizeof(double));
    memset(ytmp, 0, nrow * sizeof(double));

#pragma omp for collapse(2) schedule(dynamic, 4096) nowait
    for (int block = 0; block < A->nblocks; block++) {
      for (int row = 0; row < nrow; row++) {
        int cell = block * nrow + row;
        double tmp = 0;
        for (int i = row_ptr[cell], end = row_ptr[cell + 1]; i < end; i++) {
          tmp += x[cols[i]];
        }
        ytmp[row] += tmp;
      }
    }
#pragma omp critical
    {
      for (int row = 0; row < nrow; row++) {
        y[row] += ytmp[row];
      }
    }
    free(ytmp);
  }
}
#endif /* CBCSR_H */
