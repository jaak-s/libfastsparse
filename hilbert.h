#ifndef HILBERT_H
#define HILBERT_H

#include <stdlib.h>

long xy2d(int, int, int);
void d2xy(int, long, int*, int*);
void rot(int, int*, int*, int, int);
int ceilPower2(int);

int ceilPower2(int x) {
  return 1 << (int)ceil(log2(x));
}

//convert (x,y) to d
long xy2d (int n, int x, int y) {
    int rx, ry;
    long s;
    long d=0;
    for (s=n/2; s>0; s/=2) {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        rot(s, &x, &y, rx, ry);
    }
    return d;
}

//convert d to (x,y)
void d2xy(int n, long d, int *x, int *y) {
    int rx, ry, s;
    long t = d;
    *x = *y = 0;
    for (s=1; s<n; s*=2) {
        rx = 1 & (t/2);
        ry = 1 & (t ^ rx);
        rot(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        t /= 4;
    }
}

//rotate/flip a quadrant appropriately
void rot(int n, int *x, int *y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n-1 - *x;
            *y = n-1 - *y;
        }

        //Swap x and y
        int t  = *x;
        *x = *y;
        *y = t;
    }
}

// x (row) should be between 0 and n-1
long row_xy2d(int n, int x, int y) {
  int y2 = y % n;
  long nsq = (long)n * (long)n;
  // order of (x,y) is reversed for better cache locality
  return xy2d(n, y2, x) + nsq * (y / n);
}

// returned (x, y) has x within 0 and n - 1
void row_d2xy(int n, long d, int *x, int *y) {
  long nsq  = (long)n * (long)n;
  long h    = d % nsq;
  int block = d / nsq;
  d2xy(n, h, y, x);

  *y += block * n;
}


#endif /* HILBERT_H */
