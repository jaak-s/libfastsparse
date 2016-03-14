#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include <assert.h>
#include <stdbool.h>
#include "sparse.h"
#include "timing.h"
#include "csr.h"

void usage() {
  printf("Usage:\n");
  printf("  preprocess -f <matrix_file> [-c] [-h]\n");
  printf("  -c  generate Compressed Row Storage matrix file (default)\n");
  printf("  -h  generate Hilbert space-filling curve ordered matrix file\n");
}

int main(int argc, char **argv) {
  int csr_flag=1, hilbert_flag=0;
  char* filename = NULL;
  int c;

  while ((c = getopt(argc, argv, "f:t")) != -1)
    switch (c) {
    case 'f': filename = optarg; break;
    case 'c': csr_flag = 1; break;
    case 'h': hilbert_flag = 1; break;
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

  if (filename == NULL) {
    fprintf(stderr, "Input matrix file missing.\n");
    usage();
    return 3;
  }

  printf("Reading input binary matrix from file '%s' ... ", filename);
  struct SparseBinaryMatrix* A = read_sbm(filename);
  printf("done!\n");
  if ( csr_flag ) {
    printf("Creating BinaryCSR (sorting).\n");
    struct BinaryCSR B;
    bcsr_from_sbm(&B, A);
    printf("Size of B is %d x %d.\n", B.nrow, B.ncol);

    static char dest_filename[256];
    strncpy(dest_filename, filename,   256);
    strncat(dest_filename, ".csr.bin", 256);
    printf("Serializing to file '%s' ...", dest_filename);
    serialize_to_file(&B, dest_filename);
    printf(" done!\n");
    // testing serialization
    //
    printf("Testing deserialization ... ");
    struct BinaryCSR T;
    deserialize_from_file(&T, dest_filename);
    assert( B.nnz  == T.nnz &&
            B.nrow == T.nrow &&
            B.ncol == T.ncol );
    bool eq_cols=true, eq_row_ptr=true;
    for ( long i=0; i<T.nnz; i++ )
      eq_cols = eq_cols && (B.cols[i] == T.cols[i]);
    assert(eq_cols);
    for ( int i=0; i<T.nrow+1; i++ )
      eq_row_ptr = eq_row_ptr && (B.row_ptr[i] == T.row_ptr[i]);
    assert(eq_row_ptr);
    free_bcsr(&T);
    printf("done!\n");
    //
    free_bcsr(&B);
  }
  if ( hilbert_flag ) {
    printf("Not yet implemented, sorry!\n");
  }
}
