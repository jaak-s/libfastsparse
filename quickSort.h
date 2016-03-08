#ifndef QUICKSORT_H
#define QUICKSORT_H

#include <stdio.h>

void quickSort(long a[], long l, long r);
inline long partition(long[], long, long);
inline void insertionSort(long[], long, long);

inline void quickSort(long a[], long l, long r) {
  if (r - l < 10) {
    insertionSort(a, l, r);
    return;
  }
  long j;

  if( l < r ) 
  {
    // divide and conquer
    j = partition(a, l, r);
    quickSort(a, l, j-1);
    quickSort(a, j+1, r);
  }
}


inline long partition(long a[], long l, long r) {
  long pivot, i, j, t;
  pivot = a[(l+r)/2];
  // moving pivot to l, while(1) assumes it
  a[(l+r)/2] = a[l];
  a[l] = pivot;
  i = l;
  j = r+1;
    
  while(1)
  {
    do ++i; while( a[i] <= pivot && i <= r );
    do --j; while( a[j] > pivot );
    if( i >= j ) break;
    t = a[i]; a[i] = a[j]; a[j] = t;
  }
  t = a[l]; a[l] = a[j]; a[j] = t;
  return j;
}

inline void insertionSort(long list[], long start, long end) {
  for (long x = start + 1; x <= end; x++) {
    long val = list[x];
    long j = x - 1;
    while (j >= 0 && val < list[j]) {
      list[j + 1] = list[j];
      j--;
    }
    list[j + 1] = val;
  }
}

#endif /* QUICKSORT_H */
