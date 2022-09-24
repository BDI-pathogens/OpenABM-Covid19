#include <stdio.h>

#include "stats.hpp"

#define ERROR 1


double inv_incomplete_gamma_p( double percentile, long n )
{
	if ( n < 1 ) {
		return ERROR;
  }

  return gcem::incomplete_gamma_inv(n,percentile);
}

int main(int argc, char *argv[]) {
  // calculate a range of values for the inverse gamma across a large value of n and write to stdout as csv
  int range = 1024;
  double pct = 0.0;
  double result = 0.0;
  printf("range,pct,result\n");
  for (size_t i = 0;i < 99; ++i) {
    pct += 0.01;
    result = inv_incomplete_gamma_p(pct,range);
    printf("%i,%f,%f\n",range,pct,result);
  }
  return 0;
}