#ifndef CSR_H
#define CSR_H

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "omp_util.h"
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

inline void free_bcsr(struct BinaryCSR* bcsr) {
  assert(bcsr);
  free(bcsr->row_ptr);
  free(bcsr->cols);
}

static inline void new_bcsr(struct BinaryCSR* __restrict__ A, long nnz, int nrow, int ncol, int* rows, int* cols) {
  assert(A);
  //struct BinaryCSR *A = (struct BinaryCSR*)malloc(sizeof(struct BinaryCSR));
  A->nnz  = nnz;
  A->nrow = nrow;
  A->ncol = ncol;
  A->cols = (int*)malloc(nnz * sizeof(int));
  A->row_ptr = (int*)malloc( (nrow + 1) * sizeof(int));

  //compute number of non-zero entries per row of A
  for (int row = 0; row < nrow; row++) {
    A->row_ptr[row] = 0;
  }

  for (int i = 0; i < nnz; i++) {
    A->row_ptr[rows[i]]++;
  }
  // cumsum counts
  for (int row = 0, cumsum = 0; row < nrow; row++) {
    int temp = A->row_ptr[row];
    A->row_ptr[row] = cumsum;
    cumsum += temp;
  }
  A->row_ptr[nrow] = nnz;

  // writing cols and vals to A->cols and A->vals
  for (int i = 0; i < nnz; i++) {
    int row  = rows[i];
    int dest = A->row_ptr[row];
    A->cols[dest] = cols[i];
    A->row_ptr[row]++;
  }
  for (int row = 0, prev = 0; row <= nrow; row++) {
    int temp        = A->row_ptr[row];
    A->row_ptr[row] = prev;
    prev            = temp;
  }
}

static inline void bcsr_from_sbm(struct BinaryCSR* __restrict__ A,
                                 struct SparseBinaryMatrix* __restrict__ sbm) {
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
    assert( num_read == (unsigned long)bcsr->nrow + 1 );
  // read cols array content
  num_written = snprintf(sbuf, 32, "int[%ld]\n", bcsr->nnz );
    assert( num_written != 32 );
  ser_sanity_check(sbuf, "cols data corrupted", file);
  num_read = fread(bcsr->cols,    sizeof(int), bcsr->nnz,      file);
    assert( num_read == (unsigned long)bcsr->nnz );
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

inline void bcsr_A_mul_B8_auto(double* Y, struct BinaryCSR *A, double* X) {
  int* row_ptr = A->row_ptr;
  int* cols    = A->cols;

#pragma omp parallel for schedule(dynamic, 512)
  for(int row = 0; row < A->nrow; row++) {
    double tmp0=0, tmp1=0, tmp2=0, tmp3=0, 
           tmp4=0, tmp5=0, tmp6=0, tmp7=0;
    for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++) {
      int col = cols[i] * 8;
      tmp0 += X[col];
      tmp1 += X[col+1];
      tmp2 += X[col+2];
      tmp3 += X[col+3];
      tmp4 += X[col+4];
      tmp5 += X[col+5];
      tmp6 += X[col+6];
      tmp7 += X[col+7];
    }
    int r = row * 8;
    Y[r] = tmp0;
    Y[r+1] = tmp1;
    Y[r+2] = tmp2;
    Y[r+3] = tmp3;
    Y[r+4] = tmp4;
    Y[r+5] = tmp5;
    Y[r+6] = tmp6;
    Y[r+7] = tmp7;
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

/** Y = A * X, where Y and X have <ncol> columns and are row-ordered */
inline void bcsr_A_mul_B32n(double* Y, struct BinaryCSR *A, double* X, const int ncol) {
  assert(ncol <= 32);
  int* row_ptr = A->row_ptr;
  int* cols    = A->cols;
#pragma omp parallel for schedule(dynamic, 256)
  for (int row = 0; row < A->nrow; row++) {
    double tmp[32] = { 0 };
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
    const int ytmp_size = ncol * nthreads();

    double* ytmpi = & ytmp[ncol * thread_num()];
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

/*** Double CSR ***/
struct CSR
{
  int nrow;
  int ncol;
  long nnz;
  int* row_ptr; /* points to the row starts for each row */
  int* cols;
  double* vals;
};

inline void free_csr(struct CSR* csr) {
  assert(csr);
  free(csr->row_ptr);
  free(csr->cols);
  free(csr->vals);
}

static inline void new_csr(
    struct CSR* __restrict__ A,
    long nnz,
    int nrow,
    int ncol,
    int* rows,
    int* cols,
    double* vals)
{
  assert(A);
  A->nnz  = nnz;
  A->nrow = nrow;
  A->ncol = ncol;
  A->cols    = (int*)malloc(nnz * sizeof(int));
  A->vals    = (double*)malloc(nnz * sizeof(double));
  A->row_ptr = (int*)malloc( (nrow + 1) * sizeof(int));

  //compute number of non-zero entries per row of A
  for (int row = 0; row < nrow; row++) {
    A->row_ptr[row] = 0;
  }

  for (int i = 0; i < nnz; i++) {
    A->row_ptr[rows[i]]++;
  }
  // cumsum counts
  for (int row = 0, cumsum = 0; row < nrow; row++) {
    int temp = A->row_ptr[row];
    A->row_ptr[row] = cumsum;
    cumsum += temp;
  }
  A->row_ptr[nrow] = nnz;

  // writing cols and vals to A->cols and A->vals
  for (int i = 0; i < nnz; i++) {
    int row = rows[i];
    int dest = A->row_ptr[row];
    A->cols[dest] = cols[i];
    A->vals[dest] = vals[i];

    A->row_ptr[row]++;
  }
  for (int row = 0, prev = 0; row <= nrow; row++) {
    int temp        = A->row_ptr[row];
    A->row_ptr[row] = prev;
    prev            = temp;
  }
}

/** y = A * x */
inline void csr_A_mul_B(double* y, struct CSR *A, double* x) {
  int* row_ptr = A->row_ptr;
  int* cols    = A->cols;
  double* vals = A->vals;
#pragma omp parallel for schedule(dynamic, 256)
  for (int row = 0; row < A->nrow; row++) {
    double tmp = 0;
    const int end = row_ptr[row + 1];
    for (int i = row_ptr[row]; i < end; i++) {
      tmp += x[cols[i]] * vals[i];
    }
    y[row] = tmp;
  }
}

/** Y = A * X, where Y and X have <ncol> columns and are row-ordered */
inline void csr_A_mul_Bn(double* Y, struct CSR *A, double* X, const int ncol) {
  int* row_ptr = A->row_ptr;
  int* cols    = A->cols;
  double* vals = A->vals;
#pragma omp parallel
  {
    double* tmp = (double*)malloc(ncol * sizeof(double));
#pragma omp parallel for schedule(dynamic, 256)
    for (int row = 0; row < A->nrow; row++) {
      memset(tmp, 0, ncol * sizeof(double));
      for (int i = row_ptr[row], end = row_ptr[row + 1]; i < end; i++) {
        int col = cols[i] * ncol;
        double val = vals[i];
        for (int j = 0; j < ncol; j++) {
           tmp[j] += X[col + j] * val;
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

#endif /* CSR_H */
