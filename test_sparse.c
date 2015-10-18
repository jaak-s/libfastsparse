/* file minunit_example.c */

#include <stdio.h>
#include <stdlib.h>
#include "minunit.h"
#include "sparse.h"

int tests_run = 0;

int foo = 7;
int bar = 4;

struct SparseBinaryMatrix* make_sbm() {
  struct SparseBinaryMatrix *A = malloc(sizeof(struct SparseBinaryMatrix));
  A->nrow = 4;
  A->ncol = 3;
  A->nnz  = 5;
  int* rows = malloc(A->nnz * sizeof(int));
  int* cols = malloc(A->nnz * sizeof(int));
  A->rows = rows;
  A->cols = cols;
  rows[0] = 0; rows[1] = 3; rows[2] = 3; rows[3] = 1; rows[4] = 2;
  cols[0] = 0; cols[1] = 2; cols[2] = 0; cols[3] = 2; cols[4] = 1;
  return A;
}

static char * test_A_mul_B() {
  struct SparseBinaryMatrix *A = make_sbm();
  double* x = malloc(A->ncol * sizeof(double));
  double* y = malloc(A->nrow * sizeof(double));
  x[0] = 0.5;
  x[1] = -0.7;
  x[2] = 1.9;
  // multiplication
  A_mul_B(y, A, x);
  mu_assert("error, y[0] != 0.5", y[0] == 0.5);
  mu_assert("error, y[1] != 1.9", y[1] == 1.9);
  mu_assert("error, y[2] !=-0.7", y[2] ==-0.7);
  mu_assert("error, y[3] != 2.4", y[3] == 2.4);
  return 0;
}

static char * test_At_mul_B() {
  struct SparseBinaryMatrix *A = make_sbm();
  double* x = malloc(A->nrow * sizeof(double));
  double* y = malloc(A->ncol * sizeof(double));
  x[0] = 0.2;
  x[1] = 1.3;
  x[2] =-0.7;
  x[3] =-0.5;
  // multiplication
  At_mul_B(y, A, x);
  mu_assert("error, y[0] !=-0.3", y[0] ==-0.3);
  mu_assert("error, y[1] !=-0.7", y[1] ==-0.7);
  mu_assert("error, y[2] != 0.8", y[2] == 0.8);
  return 0;
}

static char * test_randsubseq() {
  srand48(1234567890);
  long* x = malloc(1000 * sizeof(long));
  long nsamples = randsubseq(10000, 1000, 0.05, x);
  mu_assert("error, nsamples < 0", nsamples >= 0);
  mu_assert("error, x[0] outside",   x[0] >= 0 && x[0] < 10000);
  mu_assert("error, x[end] outside", x[nsamples-1] >= 0 && x[nsamples-1] < 10000);
  return 0;
}

static char * all_tests() {
    mu_run_test(test_A_mul_B);
    mu_run_test(test_At_mul_B);
    mu_run_test(test_randsubseq);
    return 0;
}

int main(int argc, char **argv) {
    char *result = all_tests();
    if (result != 0) {
        printf("%s\n", result);
    }
    else {
        printf("ALL TESTS PASSED\n");
    }
    printf("Tests run: %d\n", tests_run);

    return result != 0;
}
