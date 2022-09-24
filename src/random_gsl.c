#include "constant.h"
#include "random.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_roots.h>


void rng_initialise() {
  gsl_rng_env_setup();
}

/*
 * From here: https://www.gnu.org/software/gsl/doc/html/rng.html#c.gsl_rng_env_setup
 * Note: "If you don’t specify a generator for GSL_RNG_TYPE then gsl_rng_mt19937 
 * is used as the default. The initial value of gsl_rng_default_seed is zero."
 */
struct generator * rng_alloc() {
  struct generator * gen = (struct generator*)malloc( sizeof (struct generator ) );
  if (gen) {
    gen->rng = gsl_rng_alloc( gsl_rng_default );
  }
  return gen;
}

void rng_free( generator * gen ) {
  if (gen) {
    gsl_rng_free( gen->rng );
    free (gen);
  }
}

void rng_set( generator * gen, long seed ) {
  gsl_rng_set( gen->rng, seed);
}


double rng_uniform( generator * gen) {
  return gsl_rng_uniform( gen->rng );
}

int rng_uniform_int( generator * gen, long p ) {
  return gsl_rng_uniform_int( gen->rng, p );
}

unsigned int ran_bernoulli( generator * gen, double p ) {
  return gsl_ran_bernoulli( gen->rng, p );
}

double ran_gamma( generator * gen, double a, double b ) {
  return gsl_ran_gamma( gen->rng, a, b );
}

double ran_exponential( generator * gen, double mu ) {
  return gsl_ran_exponential( gen->rng, mu );
}

void ran_shuffle( generator * gen, void * base, size_t n, size_t size) {
  gsl_ran_shuffle( gen->rng, base, n, size );
}

unsigned int ran_negative_binomial( generator * gen, double p, double n ) {
  return gsl_ran_negative_binomial( gen->rng, p, n );
}

size_t ran_discrete( generator * gen, size_t K, const double *P ) {
	size_t v;
  gsl_ran_discrete_t *t = gsl_ran_discrete_preproc( K, P );
	v = gsl_ran_discrete( gen->rng, t );
	gsl_ran_discrete_free(t);
  return v;
}


double cdf_gamma_P( double P, double a, double b ) {
  return gsl_cdf_gamma_P( P, a, b);
}

double cdf_gamma_Pinv( double P, double a, double b ) {
  return gsl_cdf_gamma_Pinv( P, a, b);
}

double cdf_exponential_Pinv( double P, double mu ) {
  return gsl_cdf_exponential_Pinv( P, mu );
}

double sf_gamma_inc_P( double a, double x ) {
  return gsl_sf_gamma_inc_P( a, x );
}


/*****************************************************************************************
*  Name:		incomplete_gamma_p
*  Description: function used for calculating the inverse incomplete gamma gunction
******************************************************************************************/
double incomplete_gamma_p( double x, void *params )
{
	struct incomplete_gamma_p_params *p = (struct incomplete_gamma_p_params *) params;
	return(  sf_gamma_inc_P( p->n, x ) - p->percentile );
}


/*****************************************************************************************
*  Name:		inv_incomplete_gamma_p
*  Description: calculates the inverse of the incomplete gamma p function
******************************************************************************************/
double inv_incomplete_gamma_p( double percentile, long n )
{
	if( n < 1 )
		return ERROR;

	// general bids needed for root solving
	const gsl_root_fsolver_type *T;
	gsl_root_fsolver *s;
	gsl_function F;
	int status;
	int iter         = 0;
	int max_iter     = 100;
	double precision = 1e-10;
	double root;

	// specific for this problem
	struct incomplete_gamma_p_params params = { n, percentile };
	double x_lo = 0.0;
	double x_hi = n * 10;
	F.function = &incomplete_gamma_p;
	F.params   = &params;

	T = gsl_root_fsolver_brent;
	s = gsl_root_fsolver_alloc (T);
	gsl_root_fsolver_set (s, &F, x_lo, x_hi );

	do
	{
		iter++;
		status = gsl_root_fsolver_iterate( s );
		root   = gsl_root_fsolver_root( s );
		x_lo   = gsl_root_fsolver_x_lower( s );
		x_hi   = gsl_root_fsolver_x_upper( s );
		status = gsl_root_test_interval(x_lo, x_hi, 0, precision);
	}
	while( status == GSL_CONTINUE && iter < max_iter );

	gsl_root_fsolver_free (s);

	return( root );
}
