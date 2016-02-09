#ifndef CG_H
#define CG_H

#include <math.h>
#include "linalg.h"
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
  double* r = (double*)malloc(F * sizeof(double));
  double* p = (double*)malloc(F * sizeof(double));
#pragma omp parallel for schedule(static)
  for (int i = 0; i < F; i++) {
    x[i] = 0.0;
    r[i] = b[i];
    p[i] = b[i];
  }
  double rsq_old = pnormsq(r, F);
  double* AAp = (double*)malloc(F * sizeof(double));
  double* tmp = (double*)malloc(N * sizeof(double));

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

/** solves (A'A + lambda*I) * X = B, where X and B are matrices with two columns */
void bsbm_cg2(double* X,
              struct BlockedSBM *A,
              struct BlockedSBM *At,
              double* B,
              double lambda,
              double tol,
              int* out_iter) {
  if ((A->nrow != At->ncol) || (A->ncol != At->nrow)) {
    printf("A (%d x %d) and At (%d x %d) must be transposes of each other.\n",
        A->nrow, A->ncol, At->nrow, At->ncol);
    exit(1);
  }
  const int F = A->ncol;
  const int F2 = F*2;
  const int N = A->nrow;
  const double tolsq = tol*tol;

  double norms[2], inorms[2];
  pnormsq2(norms, B, F);
  norms[0] = sqrt(norms[0]);
  norms[1] = sqrt(norms[1]);
  inorms[0] = 1.0 / norms[0];
  inorms[1] = 1.0 / norms[1];

  // init:
  double* R = (double*)malloc(F2 * sizeof(double));
  double* P = (double*)malloc(F2 * sizeof(double));
#pragma omp parallel for schedule(static)
  for (int i = 0; i < F2; i+=2) {
    X[i]   = 0.0;
    X[i+1] = 0.0;
    R[i]   = B[i] * inorms[0];
    R[i+1] = B[i+1] * inorms[1];
    P[i]   = R[i];
    P[i+1] = R[i+1];
  }
  double RtR[3], RtR2[3];
  double PtKP[3];
  double Alpha[4];
  double Psi[4];
  pouter2(RtR, R, F);
  double* AAP = (double*)malloc(F * 2 * sizeof(double));
  double* tmp = (double*)malloc(N * 2 * sizeof(double));

  int iter;
  for (iter = 0; iter < F; iter++) {
    // solution update:
    // computing AAp = (At*A + lambda*I) * P
    // TODO: replace by nowait parallel loop
    bsbm_A_mul_B2(tmp, A, P);
    bsbm_A_mul_B2(AAP, At, tmp);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < F2; i++) {
      AAP[i] += lambda * P[i];
    }

    // Alpha = inv(PtKP) * RtR
    pdot2sym(PtKP, P, AAP, F);
    double rhs[4] = {RtR[0], RtR[2], RtR[2], RtR[1]};
    solve2sym(Alpha, PtKP, rhs);

    // X += Alpha' * P
    // R -= Alpha' * AAP
#pragma omp parallel for schedule(static)
    for (int i = 0; i < F2; i+=2) {
      X[i]   += Alpha[0] * P[i] + Alpha[1] * P[i+1];
      X[i+1] += Alpha[2] * P[i] + Alpha[3] * P[i+1];
      R[i]   -= Alpha[0] * AAP[i] + Alpha[1] * AAP[i+1];
      R[i+1] -= Alpha[2] * AAP[i] + Alpha[3] * AAP[i+1];
    }

    // convergence check:
    pouter2(RtR2, R, F);
    if ((RtR2[0] <= tolsq) && (RtR2[1] <= tolsq)) break; 

    // Psi = inv(RtR) * RtR2
    double rhs_psi[4] = {RtR2[0], RtR2[2], RtR2[2], RtR2[1]};
    solve2sym(Psi, RtR, rhs_psi);

    // P = R + Psi' * P
#pragma omp parallel for schedule(static)
    for (int i = 0; i < F2; i+=2) {
      double Pi = P[i], Pi1 = P[i+1];
      P[i]   = R[i]   + Psi[0] * Pi + Psi[1] * Pi1;
      P[i+1] = R[i+1] + Psi[2] * Pi + Psi[3] * Pi1;
    }
    RtR[0] = RtR2[0];
    RtR[1] = RtR2[1];
    RtR[2] = RtR2[2];
  } // end of CG iterations

#pragma omp parallel for schedule(static)
  for (int i = 0; i < F2; i+=2) {
    X[i]   *= norms[0];
    X[i+1] *= norms[1];
  }

  free(AAP);
  free(tmp);
  free(R);
  free(P);
  *out_iter = iter;
}
#endif /* CG_H */
