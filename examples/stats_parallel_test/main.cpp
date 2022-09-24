#include <stdio.h>
#include <time.h>

#include "random.h"
#include "stats.hpp"

struct valueobject {
  long riskScore;
  double probFallingIll;
  valueobject* next;
  valueobject* prev;
};

int main(int argc, char *argv[]) {

  const std::size_t elements = 1'000'000'000;

  printf("method,result\n");

  // generate 1 billion random risk scores using two methods:-
  // 1. classic C-like struct allocated on the heap, and processed one at a time in a single thread
  // 2. std::vector (on the heap again), but processed in parallel by the stats library

  clock_t start, end;
  double cpu_time_nsec;

  start = clock();

  valueobject* prev = new valueobject{89,0.7,NULL,NULL};
  valueobject* first = prev;
  for (std::size_t i = 1;i < elements;++i) {
    prev->next = new valueobject{45,0.2,NULL,prev};
    prev = prev->next;
  }

  
  end = clock();
  cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
  printf("classic_alloc,%12f\n",cpu_time_nsec);

  start = clock();

  // Classic method
  generator* gen = rng_alloc();
  valueobject* ptr = first;
  do {
    ptr->probFallingIll = rng_uniform(gen);
    ptr = ptr->next;
  } while (NULL != ptr);
  rng_free(gen);

  end = clock();
  cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
  printf("classic_run,%12f\n",cpu_time_nsec);
  
  start = clock();

  gen = rng_alloc();
  ptr = first;
  do {
    ptr->probFallingIll = stats::punif(ptr->probFallingIll,3.0,2.0,gen);
    ptr = ptr->next;
  } while (NULL != ptr);
  rng_free(gen);

  end = clock();
  cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
  printf("classic_modify,%12f\n",cpu_time_nsec);
  
  start = clock();

  // Clear memory before proceeding
  ptr = first;
  prev = first;
  do {
    ptr = prev->next;
    prev->next = NULL;
    prev->prev = NULL;
    delete(prev);
    prev = ptr;
  } while (NULL != prev);
  ptr = NULL;
  first = NULL;

  end = clock();
  cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
  printf("classic_dealloc,%12f\n",cpu_time_nsec);

  // vector method
  start = clock();
  std::vector<double> probFallingIllVector = stats::runif<std::vector<double>>(elements,1,0.0d,1.0d);
  end = clock();
  cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
  printf("stdvector_combined,%12f\n",cpu_time_nsec);

  
  start = clock();
  auto vecResult = stats::punif(probFallingIllVector,3.0d,2.0d);
  end = clock();
  cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
  printf("stdvector_modify,%12f\n",cpu_time_nsec);

  return 0;
}