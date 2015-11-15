#ifndef CG_H
#define CG_H

#include <math.h>
#include "sparse.h"

/** y = At*A*x + lambda*x */
void bsbm_AtA(double* y,
              struct BlockedSBM *A,
              struct BlockedSBM *At,
              double* x,
              double* tmp,
              double lambda) {
  bsbm_A_mul_B(tmp, A, x);
  bsbm_A_mul_B(y, At, tmp);
  const int F = At->nrow;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < F; i++) {
    y[i] += lambda * x[i];
  }
}

/** solves (A'A + lambda*I) * x = b */
void bsbm_cg(double* x,
             struct BlockedSBM *A,
             struct BlockedSBM *At,
             double* b,
             double lambda,
             double tol,
             int* out_iter) {
  if ((A->nrow != At->ncol) || (A->ncol != At->nrow)) {
    printf("A (%d x %d) and At (%d x %d) must be transposes of each other.\n",
        A->nrow, A->ncol, At->nrow, At->ncol);
    exit(1);
  }
  int F = A->ncol;
  int N = A->nrow;

  tol = tol * sqrt(pnormsq(b, F));
  // init:
  double* r = malloc(F * sizeof(double));
  double* p = malloc(F * sizeof(double));
#pragma omp parallel for schedule(static)
  for (int i = 0; i < F; i++) {
    x[i] = 0.0;
    r[i] = b[i];
    p[i] = b[i];
  }
  double rsq_old = pnormsq(r, F);
  double* AAp = malloc(F * sizeof(double));
  double* tmp = malloc(N * sizeof(double));

  int iter;
  for (iter = 0; iter < F; iter++) {
    // computing AAp = (At*A + lambda*I) * p
    bsbm_AtA(AAp, A, At, p, tmp, lambda);
    double alpha = rsq_old / pdot(AAp, p, F);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < F; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * AAp[i];
    }

    double rsq_new = pnormsq(r, F);
    if (sqrt(rsq_new) <= tol) break; 

    double beta = rsq_new / rsq_old;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < F; i++) {
      p[i] = r[i] + beta * p[i];
    }
    rsq_old = rsq_new;
  }

  free(AAp);
  free(tmp);
  free(r);
  free(p);
  *out_iter = iter;
}

#endif /* CG_H */
