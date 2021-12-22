#include "constant.h"
#include "random.h"

// TODO
#include "stats.hpp"
// #include <armadillo>


void rng_initialise() {
  // TODO
}

struct generator * rng_alloc() {
  return new generator; // C++ allocation
}

void rng_free( generator * gen ) {
  if (gen) {
    delete (gen); // C++ deallocation
  }
}

void rng_set( generator * gen, long seed ) {
  gen->rng.seed(seed);
}


double rng_uniform( generator * gen) {
  // TODO ensure 0.0 and 1.0 means <double,double>
  return stats::runif( 0.0, 1.0, gen->rng );
}

int rng_uniform_int( generator * gen, long p ) {
  return stats::runif( 0, 1, gen->rng );
}

unsigned int ran_bernoulli( generator * gen, double p ) {
  return static_cast<unsigned int>(stats::rbern( p, gen->rng ));
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
