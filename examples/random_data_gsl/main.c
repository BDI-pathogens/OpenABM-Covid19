#include <stdio.h>
#include <time.h>

#include "random.h"

int main(int argc, char *argv[]) {

	const char* library = "gsl";

#include "../random_data_stats/run.inc"

  return 0;
}