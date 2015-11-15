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

double pdot(double* x, double* y, int n) {
  double d = 0.0;
#pragma omp parallel for reduction(+:d) schedule(static)
  for (int i = 0; i < n; i++) {
    d += x[i] * y[i];
  }
  return d;
}

#endif /* LINALG_H */
