#include <sys/time.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/resource.h>

//void timing(double* wcTime, double* cpuTime);

void timing(double* wcTime, double* cpuTime)
{
   struct timeval tp;
   struct rusage ruse;

   gettimeofday(&tp, NULL);
   *wcTime=(double) (tp.tv_sec + tp.tv_usec/1000000.0); 
  
   getrusage(RUSAGE_SELF, &ruse);
   *cpuTime=(double)(ruse.ru_utime.tv_sec+ruse.ru_utime.tv_usec / 1000000.0);
}
