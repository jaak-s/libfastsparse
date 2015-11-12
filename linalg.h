#ifndef LINALG_H
#define LINALG_H

double dist(double* x, double* y, int n) {
  double d = 0;
  for (int i = 0; i < n; i++) {
    double diff = x[i] - y[i];
    d += diff*diff;
  }
  return d;
}

#endif /* LINALG_H */
