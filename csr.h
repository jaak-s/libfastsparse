#ifndef CSR_H
#define CSR_H

#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "sparse.h"
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

void free_bcsr(struct BinaryCSR* bcsr) {
  assert(bcsr);
  free(bcsr->row_ptr);
  free(bcsr->cols);
}

static inline void new_bcsr(struct BinaryCSR* restrict A, long nnz, int nrow, int ncol, int* rows, int* cols) {
  assert(A);
  //struct BinaryCSR *A = (struct BinaryCSR*)malloc(sizeof(struct BinaryCSR));
  A->nnz  = nnz;
  A->nrow = nrow;
  A->ncol = ncol;
  A->cols = (int*)malloc(nnz * sizeof(int));
  A->row_ptr = (int*)malloc( (nrow + 1) * sizeof(int));

  // sorting by row:
  long* h = (long*)malloc(A->nnz * sizeof(long));
#pragma omp parallel for schedule(static)
  for (int i = 0; i < A->nnz; i++) {
      h[i] = rows[i] * (long)ncol + (long)cols[i];
  }
  quickSort(h, 0, nnz - 1);
  int row_prev = -1;
  for (int i = 0; i < nnz; i++) {
    A->cols[i] = h[i] % ncol;
    if (A->cols[i] < 0) {
       printf("A->cols[%d] = %d, h[%d]=%ld\n", i, A->cols[i], i, h[i]);
    }
    int row = h[i] / ncol;
    while (row > row_prev) {
      row_prev++;
      A->row_ptr[row_prev] = i;
    }
  }
  for (int row = row_prev + 1; row <= nrow; row++) {
    A->row_ptr[row] = nnz;
  }
  free(h);
}

static inline void bcsr_from_sbm(struct BinaryCSR* restrict A,
                                       struct SparseBinaryMatrix* restrict sbm) {
  assert(A);
  assert(sbm);
  new_bcsr(A, sbm->nnz, sbm->nrow, sbm->ncol, sbm->rows, sbm->cols);
}

/**********************************************************************/
/************* SERIALIZATION ******************************************/
/**********************************************************************/

// Quick & dirty serialization code, for now. Consider creating macros
// or using google protobuf if more objects need to be serialized.

#define BINARY_CSR_HEADER "BINARY_CSR: struct BinaryCSR, int[nrow], int[nnz]\n"

static void ser_sanity_check(const char* expected_str, const char* error_str, FILE* file) {
  static char in_buf[256];
  const char* input_str = fgets( in_buf, 256, file );
  assert(input_str);
  if ( strncmp(input_str, expected_str, 256) ) {
    printf("ERROR: could not read data from file, %s\n", error_str);
    printf("  expected: \"%s\"\n", expected_str);
    printf("      read: \"%s\"\n", input_str);
    exit(-1);
  }
}

static inline void serialize_to_file(const struct BinaryCSR* bcsr, const char* filename) {
  FILE* file = fopen(filename, "w+");
  assert(file);
  // write header, used as a kind of version/sanity check
  fputs( BINARY_CSR_HEADER, file );
  // write (shallow copy) struct object
  fputs( "struct BinaryCSR\n", file );
  fwrite(bcsr,          sizeof(struct BinaryCSR), 1, file);
  // write cols and row_ptr array values
  fprintf(file, "int[%d]\n",  bcsr->nrow + 1);
  fwrite(bcsr->row_ptr, sizeof(int), bcsr->nrow + 1, file);
  fprintf(file, "int[%ld]\n", bcsr->nnz );
  fwrite(bcsr->cols,    sizeof(int), bcsr->nnz,      file);
  fclose(file);

  //  printf(" (written bcsr with nnz=%ld and nrow=%d) ", bcsr->nnz, bcsr->nrow);
}

// populates bcsr object with data from file, memory for row_ptr and
// cols arrays is allocated here (overwriting existing pointers)
static inline void deserialize_from_file(struct BinaryCSR* bcsr, const char* filename) {
  static char sbuf[32];
  FILE* file = fopen(filename, "r");
  size_t num_read     __attribute__((unused));
  size_t num_written  __attribute__((unused));
  // read struct object
  ser_sanity_check(BINARY_CSR_HEADER,    "Invalid file format or version", file);
  ser_sanity_check("struct BinaryCSR\n", "struct data corrupted",          file);
  bcsr->row_ptr = NULL; // paranoia
  bcsr->cols    = NULL; //
  num_read = fread(bcsr, sizeof(struct BinaryCSR), 1, file);
    assert( num_read == 1 );
  bcsr->row_ptr = (int*)calloc( bcsr->nrow + 1, sizeof(int) );
  bcsr->cols    = (int*)calloc( bcsr->nnz,      sizeof(int) );
    assert( bcsr->row_ptr );
    assert( bcsr->cols );
  // read row_ptr array content
  num_written = snprintf(sbuf, 32, "int[%d]\n", bcsr->nrow + 1 );
    assert( num_written != 32 );
  ser_sanity_check(sbuf, "nrow data corrupted", file);
  num_read = fread(bcsr->row_ptr, sizeof(int), bcsr->nrow + 1, file);
    assert( num_read == bcsr->nrow + 1 );
  // read cols array content
  num_written = snprintf(sbuf, 32, "int[%ld]\n", bcsr->nnz );
    assert( num_written != 32 );
  ser_sanity_check(sbuf, "cols data corrupted", file);
  num_read = fread(bcsr->cols,    sizeof(int), bcsr->nnz,      file);
    assert( num_read == bcsr->nnz );
  fclose(file);
}

/** y = A * x */
inline void bcsr_A_mul_B(double* y, struct BinaryCSR *A, double* x) {
  int* row_ptr = A->row_ptr;
  int* cols    = A->cols;
#pragma omp parallel for schedule(dynamic, 256)
  for (int row = 0; row < A->nrow; row++) {
    double tmp = 0;
    const int end = row_ptr[row + 1];
    for (int i = row_ptr[row]; i < end; i++) {
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
    const int end = row_ptr[row + 1];
    for (int i = row_ptr[row]; i < end; i++) {
      int col = cols[i] * 2;
      tmp1 += X[col];
      tmp2 += X[col+1];
    }
    int r = row * 2;
    Y[r]   = tmp1;
    Y[r+1] = tmp2;
  }
}

/** Y = A * X, where Y and X have 4 columns and are row-ordered */
inline void bcsr_A_mul_B4(double* Y, struct BinaryCSR *A, double* X) {
  int* row_ptr = A->row_ptr;
  int* cols    = A->cols;
#pragma omp parallel for schedule(dynamic, 256)
  for (int row = 0; row < A->nrow; row++) {
    double tmp[4] = { 0 };
    const int end = row_ptr[row + 1];
    for (int i = row_ptr[row]; i < end; i++) {
      int col = cols[i] << 2; // multiplying with 4
      for (int j = 0; j < 4; j++) {
         tmp[j] += X[col + j];
      }
    }
    int r = row << 2;
    for (int j = 0; j < 4; j++) {
      Y[r + j] = tmp[j];
    }
  }
}

/** Y = A * X, where Y and X have 8 columns and are row-ordered */
inline void bcsr_A_mul_B8(double* Y, struct BinaryCSR *A, double* X) {
  int* row_ptr = A->row_ptr;
  int* cols    = A->cols;
#pragma omp parallel for schedule(dynamic, 256)
  for (int row = 0; row < A->nrow; row++) {
    double tmp[8] = { 0 };
    const int end = row_ptr[row + 1];
    for (int i = row_ptr[row]; i < end; i++) {
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

/** Y = A * X, where Y and X have <ncol> columns and are row-ordered */
inline void bcsr_A_mul_Bn(double* Y, struct BinaryCSR *A, double* X, const int ncol) {
  int* row_ptr = A->row_ptr;
  int* cols    = A->cols;
#pragma omp parallel 
  {
    double* tmp = (double*)malloc(ncol * sizeof(double));
#pragma omp parallel for schedule(dynamic, 256)
    for (int row = 0; row < A->nrow; row++) {
      memset(tmp, 0, ncol * sizeof(double));
      const int end = row_ptr[row + 1];
      for (int i = row_ptr[row]; i < end; i++) {
        int col = cols[i] * ncol;
        for (int j = 0; j < ncol; j++) {
           tmp[j] += X[col + j];
        }
      }
      int r = row * ncol;
      for (int j = 0; j < ncol; j++) {
        Y[r + j] = tmp[j];
      }
    }
    free(tmp);
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
