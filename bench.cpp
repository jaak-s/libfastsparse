#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <valarray>
#include <memory>
#include <tbb/cache_aligned_allocator.h>
#include "benchmark/benchmark.h"

extern "C" {
  #include "csr.h"
}

using std::vector;

void usage() {
  printf("Usage:\n");
  printf("  bench_csr -f <matrix_file> [-t] [-c]\n");
  printf("  -p  matrix file is in preprocessed format\n");
  printf("  -t  transpose matrix\n");
  printf("  -c  run conjugate gradient\n");
}

void extrema(int* x, long n, int* min, int* max) {
  *min = x[0];
  *max = x[0];

  for (int i = 0; i < n; i++) {
    if (x[i] < min[0]) *min = x[i];
    if (x[i] > max[0]) *max = x[i];
  }
}


static BinaryCSR A;
static vector<double>  x;
static vector<double>  y;
static vector<double> X8;
static vector<double> Y8;

static void BM_BCSR_A_mul_B8(benchmark::State& state) {
  while (state.KeepRunning())
    bcsr_A_mul_B8(Y8.data(), &A, X8.data());
}

BENCHMARK(BM_BCSR_A_mul_B8);

int main(int argc, char **argv) {

  benchmark::Initialize(&argc, argv);

  int tflag=0,
      preprocessed_flag=0;
  char* filename = NULL;
  //int cgflag = 0;
  int c;

  while ((c = getopt(argc, argv, "f:tp")) != -1)
    switch (c) {
      //case 'c': cgflag = 1; break;
      case 'f': filename = optarg; break;
      case 't': tflag = 1; break;
      case 'p': preprocessed_flag = 1; break;
      case '?':
        if (optopt == 'f')
          fprintf(stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint(optopt))
          fprintf(stderr, "Unknown option '-%c'.\n", optopt);
        else
          fprintf(stderr, "Unknown option character '\\x%x.\n", optopt);
        usage();
        return 1;
      default:
        usage();
        return 2;
    }
  int nrepeats = 20;
  int cgrepeats = 100;

  if (filename == NULL) {
    fprintf(stderr, "Input matrix file missing.\n");
    usage();
    return 3;
  }
  if ( preprocessed_flag ) {
    printf("Loading preprocessed BinaryCSR from file.\n");
    deserialize_from_file(&A, filename);
    if ( tflag )
      puts("transpose flag '-t' ignored");
  }
  else {
    printf("Loading sparse matrix from '%s'.\n", filename);
    struct SparseBinaryMatrix* matrix_in = read_sbm(filename);
    if (tflag) {
      transpose(matrix_in);
    }
    printf("Processing sparse matrix into BinaryCSR (sorting).\n");
    bcsr_from_sbm(&A, matrix_in);
    free_sbm(matrix_in);
    free(matrix_in);
  }
  printf("Size of A is %d x %d.\n", A.nrow, A.ncol);

  int min_col = 10000000;
  int max_col = 0;
  for (int i = 0; i < A.nnz; i++) {
    if (A.cols[i] > max_col) max_col = A.cols[i];
    if (A.cols[i] < min_col) min_col = A.cols[i];
  }
  int max_ptr = 0;
  for (int i = 0; i <= A.nrow; i++) {
    max_ptr = A.row_ptr[i];
  }
  printf("Min(cols) = %d, max(cols) = %d\n", min_col, max_col);
  printf("Max(row_ptr) = %d\n", max_ptr);

  x  = vector<double>(A.ncol);
  y  = vector<double>(A.nrow, 0);
  X8 = vector<double>(A.ncol*8);
  Y8 = vector<double>(A.nrow*8, 0);

  for (int i = 0; i < A.ncol; i++) {
    x[i] = sin(7.0*i + 0.3) / 10.0;
    for (int k = 0; k < 8; k++)
      X8[i*8+k] = sin(7*i + 17*k + 0.3);
  }

  benchmark::RunSpecifiedBenchmarks();

}
