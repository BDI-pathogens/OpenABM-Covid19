#include <stdio.h>
#include <time.h>

#define USE_STATS 1
#include "random.h"

int main(int argc, char *argv[]) {

	const char* library = "stats";

#include "run.inc"

  return 0;
}