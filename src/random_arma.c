#include "constant.h"
#include "random.h"

// TODO
// #include <armadillo>


void rng_initialise() {
  // TODO
}

struct generator * rng_alloc() {
  struct generator * gen = (struct generator*)malloc( sizeof (struct generator ) );
  if (gen) {
    // TODO
    gen->rng = NULL;
  }
  return gen;
}

void rng_free( generator * gen ) {
  if (gen) {
    // TODO
    // gsl_rng_free( gen->rng );
    free (gen);
  }
}

void rng_set( generator * gen, long seed ) {
  // TODO
}


double rng_uniform( generator * gen) {
  // TODO
  return 0;
}

int rng_uniform_int( generator * gen, long p ) {
  // TODO
  return 0;
}

unsigned int ran_bernoulli( generator * gen, double p ) {
  // TODO
  return 0;
}

double ran_gamma( generator * gen, double a, double b ) {
  // TODO
  return 0;
}

double ran_exponential( generator * gen, double mu ) {
  // TODO
  return 0;
}

void ran_shuffle( generator * gen, void * base, size_t n, size_t size) {
  // TODO
}

unsigned int ran_negative_binomial( generator * gen, double p, double n ) {
  // TODO
  return 0;
}

size_t ran_discrete( generator * gen, size_t K, const double *P ) {
  // TODO
  return 0;
}


double cdf_gamma_P( double P, double a, double b ) {
  // TODO
  return 0;
}

double cdf_gamma_Pinv( double P, double a, double b ) {
  // TODO
  return 0;
}

double cdf_exponential_Pinv( double P, double mu ) {
  // TODO
  return 0;
}

double sf_gamma_inc_P( double a, double x ) {
  // TODO
  return 0;
}


/*****************************************************************************************
*  Name:		incomplete_gamma_p
*  Description: function used for calculating the inverse incomplete gamma gunction
******************************************************************************************/
double incomplete_gamma_p( double x, void *params )
{
  // TODO
  return 0;
}


/*****************************************************************************************
*  Name:		inv_incomplete_gamma_p
*  Description: calculates the inverse of the incomplete gamma p function
******************************************************************************************/
double inv_incomplete_gamma_p( double percentile, long n )
{
	if( n < 1 )
		return ERROR;

  // TODO
  return 0;
}
