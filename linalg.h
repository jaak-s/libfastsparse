#ifndef LINALG_H
#define LINALG_H

#include <math.h>

double dist(double* x, double* y, int n) {
  double d = 0;
  for (int i = 0; i < n; i++) {
    double diff = x[i] - y[i];
    d += diff*diff;
  }
  return sqrt(d);
}

double pnormsq(double* x, int n) {
  double normsq = 0.0;
#pragma omp parallel for reduction(+:normsq) schedule(static)
  for (int i = 0; i < n; i++) {
    normsq += x[i] * x[i];
  }
  return normsq;
}

void pnormsq2(double* normsq, double* X, int n) {
  double a = 0.0, b = 0.0;
  const int n2 = n * 2;
#pragma omp parallel for reduction(+:a, b) schedule(static)
  for (int i = 0; i < n2; i += 2) {
    a += X[i]   * X[i];
    b += X[i+1] * X[i+1];
  }
  normsq[0] = a;
  normsq[1] = b;
}

/** computes a'a, b'b, a'b for 2-column matrix X = [a b] */
void pouter2(double* outer, double* X, int n) {
  double aa = 0.0, bb = 0.0, ab = 0.0;
  const int n2 = n * 2;
#pragma omp parallel for reduction(+:aa, bb, ab) schedule(static)
  for (int i = 0; i < n2; i += 2) {
    aa += X[i]   * X[i];
    bb += X[i+1] * X[i+1];
    ab += X[i]   * X[i+1];
  }
  outer[0] = aa;
  outer[1] = bb;
  outer[2] = ab;
}

double pdot(double* x, double* y, int n) {
  double d = 0.0;
#pragma omp parallel for reduction(+:d) schedule(static)
  for (int i = 0; i < n; i++) {
    d += x[i] * y[i];
  }
  return d;
}

// symmetric D = X.dot(Y) for 2-column matrices X and Y
void pdot2sym(double* D, double* X, double* Y, int n) {
  double aa = 0.0, bb = 0.0, ab = 0.0;
  const int n2 = n * 2;
#pragma omp parallel for reduction(+:aa, bb, ab) schedule(static)
  for (int i = 0; i < n2; i += 2) {
    aa += X[i]   * Y[i];
    bb += X[i+1] * Y[i+1];
    ab += X[i]   * Y[i+1];
  }
  D[0] = aa;
  D[1] = bb;
  D[2] = ab;
}

// solves A * X = RHS for X where A is 2x2 symmetric and RHS is 2x2
// X and RHS are column-ordered
inline void solve2sym(double* X, double* A, double* RHS) {
   double dinv = 1.0 / (A[0]*A[1] - A[2]*A[2]);
   double Ainv[3];
   Ainv[0] = dinv * A[1];
   Ainv[1] = dinv * A[0];
   Ainv[2] = -dinv * A[2];
   // computing X = Ainv * RHS
   X[0] = Ainv[0] * RHS[0] + Ainv[2] * RHS[1];
   X[1] = Ainv[2] * RHS[0] + Ainv[1] * RHS[1];
   X[2] = Ainv[0] * RHS[2] + Ainv[2] * RHS[3];
   X[3] = Ainv[2] * RHS[2] + Ainv[1] * RHS[3];
}

#endif /* LINALG_H */
