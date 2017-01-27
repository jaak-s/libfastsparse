#ifndef QUICKSORTD_H
#define QUICKSORTD_H

#include <stdio.h>

// quick sort with added vector of doubles that should be sorted as well

void quickSortD(long[], long, long, double[]);
long partitionD(long[], long, long, double[]);
void insertionSortD(long[], long, long, double[]);

inline void quickSortD(long a[], long l, long r, double v[]) {
  if (r - l < 10) {
    insertionSortD(a, l, r, v);
    return;
  }
  long j;

  if( l < r ) 
  {
    // divide and conquer
    j = partitionD(a, l, r, v);
    quickSortD(a, l, j-1, v);
    quickSortD(a, j+1, r, v);
  }
}


inline long partitionD(long a[], long l, long r, double v[]) {
  long pivot, i, j, t;
  double td;
  t = (l+r) / 2;
  // moving pivot to l, because while(1) assumes it
  pivot = a[t];
  a[t] = a[l];
  a[l] = pivot;
  i = l;
  // moving also v[t] <-> v[l]
  td = v[t];
  v[t] = v[l];
  v[l] = td;
  
  j = r+1;
    
  while(1)
  {
    do ++i; while( a[i] <= pivot && i <= r );
    do --j; while( a[j] > pivot );
    if( i >= j ) break;
    t = a[i]; a[i] = a[j]; a[j] = t;
    td= v[i]; v[i] = v[j]; v[j] = td;
  }
  t = a[l]; a[l] = a[j]; a[j] = t;
  td =v[l]; v[l] = v[j]; v[j] = td;
  return j;
}

inline void insertionSortD(long list[], long start, long end, double v[]) {
  for (long x = start + 1; x <= end; x++) {
    long val = list[x];
    double td = v[x];
    long j = x - 1;
    while (j >= 0 && val < list[j]) {
      list[j + 1] = list[j];
      v[j + 1] = v[j];
      j--;
    }
    list[j + 1] = val;
    v[j + 1] = td;
  }
}

#endif /* QUICKSORTD_H */
