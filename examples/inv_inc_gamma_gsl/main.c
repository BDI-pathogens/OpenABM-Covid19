#include <stdio.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_roots.h>

#define ERROR 1

struct incomplete_gamma_p_params { long n; double percentile; };

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