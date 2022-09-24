/*
 * utilities.c
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#ifdef BUILD_RPKG
  #include <Rdefines.h>
  #undef ERROR // mute GCC warning: "ERROR" redefined
#else
  #include <stdio.h>
  #include <stdarg.h>
#endif

#include <stdlib.h>
#include <math.h>
#include "constant.h"
#include "random.h"
#include "utilities.h"

#undef printf

/*****************************************************************************************
*  Name:		setup_rng
*  Description: Setup the random seed so that the utilities functions that use
*               random number generation can be called.  
******************************************************************************************/
void setup_rng(int seed)
{
	rng_initialise();
	rng = rng_alloc();
	rng_set( rng, seed);
}

/*****************************************************************************************
*  Name:		free_rng
*  Description: frees the memory allocated to the rng
*  				USE WITH CARE SINCE IS SESSION SINGLETON
******************************************************************************************/
void free_rng()
{
	rng_free( rng );
}

/*****************************************************************************************
*  Name:		printf_w
*  Description: Wrapper for printf / Rprintf
******************************************************************************************/
int printf_w( const char* format, ... )
{
    int r = 0;
    va_list args;
    va_start(args, format);
#ifdef BUILD_RPKG  /* CRAN packages may not interact with stdout/stderr directly */
    Rvprintf(format, args );
#else
    r = vprintf(format, args );
#endif
    va_end(args);
    return(r);
}

/*****************************************************************************************
*  Name:		print_exit
******************************************************************************************/
void print_exit( char *s, ... )
{
    /* Don't use malloc() to allocate memory here. Rf_error uses longjmp() so we can't
     * reliably call free(). Use a temporary 2 KiB local buffer instead. */
    char buffer[2048] = {0};
    const size_t n = sizeof(buffer) - 1; /* substract 1 for '\n' */

    /* Parse va_list */
    va_list ap;
    va_start(ap, s);
    vsnprintf(buffer, n, s, ap);
    va_end(ap);

#ifdef BUILD_RPKG /* CRAN packages may not call exit(); Rf_error uses longjmp() */
    Rf_error("%s",buffer);
#else
    printf("%s\n", buffer);
    fflush(stdout);
    exit(1);
#endif
}

/*****************************************************************************************
*  Name:		print_now
******************************************************************************************/
void print_now( char *s )
{
    printf_w("%s\n", s );
    fflush_stdout();
}

/*****************************************************************************************
*  Name:		fflush_stdout
******************************************************************************************/
void fflush_stdout()
{
#ifndef BUILD_RPKG /* Referencing `stdout` can raise a NOTE with `R CMD check` */
    fflush(stdout);
#endif
}

/*****************************************************************************************
*  Name:		gamma_draw_list
*  Description: generates a draw list so that we can efficiently sample
*  				from a distribution
*
*  Arguments:	list:	pointer to draw list be filled in
*  				n:		length of draw list
*  				mean:	mean of gamma distribution
*  				sd:		sd of gamma distribution
******************************************************************************************/
void gamma_draw_list(
	int *list,
	int n,
	double mean,
	double sd
)
{
	int idx      = 0;
	double a, b;

	b = sd * sd / mean;
	a = mean / b;

	for( idx = 0; idx < n; idx++ )
		list[idx] = max( round( cdf_gamma_Pinv( ( idx + 1.0 )/( n + 1.0 ), a, b )), 1 );
}

/*****************************************************************************************
*  Name:		bernoulli_draw_list
*  Description: generates a draw list so that we can efficiently sample
*  				from a distribution
*  				the 2 possible outcomes are floor(mean) and floor(mean)+1
*
*  Arguments:	list:	pointer to draw list be filled in
*  				n:		length of draw list
*  				mean:	mean of  distribution
******************************************************************************************/
void bernoulli_draw_list(
	int *list,
	int n,
	double mean
)
{
	int idx = 0;
	int a, b, p;

	a = floor( mean );
	b = a + 1;
	p = round( ( mean - a ) * n );

	for( idx = 0; idx < n; idx++ )
		list[idx] = ifelse( idx < p, b, a );
}

/*****************************************************************************************
*  Name:		geometric_max_draw_list
*  Description: generates a draw list which is geometrically distributed with parameter
*				p and a maximum. Note, since we allow the possibility of immediate
*				drop out then the the first event is 0 not 1 (so if no max we have
*				mean of 1/p-1).
*				Any draws larger than max are given the value max
*
*  Arguments:	list:	pointer to draw list be filled in
*  				n:		length of draw list
*  				mean:	mean of  distribution
******************************************************************************************/
void geometric_max_draw_list(
	int *list,
	int n,
	double p,
	int max
)
{
	int idx;

	if( p == 0 )
	{
		for( idx = 0; idx < n; idx++ )
			list[idx] = max;
		return;
	};

	int day      = 0;
	double cprob = 0;
	double prob  = p;
	int limit   = round( prob * n );

	for( idx = 0; idx < n; idx++ )
	{
		if( idx < limit )
			list[idx] = day;
		else
		{
			day++;
			if( day < max )
			{
				cprob += prob;
				prob  *= ( 1 - p );
				limit = round( ( cprob + prob ) * n );
			}
			else
				limit = n;
			list[idx] = day;
		}
	}
}

/*****************************************************************************************
*  Name:               geometric_draw_list
*  Description: generates a draw list so that we can efficiently sample
*                              from a geometric distribution
*
*  Arguments:  list:   pointer to draw list to be filled in
*                              n:              length of draw list
*                              mean:   mean of  distribution
******************************************************************************************/
void geometric_draw_list(
		int *list,
		int n,
		double mean
)
{
		int idx;
		for( idx = 0; idx < n; idx++ )
				list[idx] = max( round( cdf_exponential_Pinv( ( idx + 1.0 )/( n + 1.0 ), mean)), 1 );
}

/*****************************************************************************************
*  Name:               shifted_geometric_draw_list
*  Description: generates a draw list so that we can efficiently sample
*                              from a geometric distribution, shifted by a certain amount
*
*  Arguments:  list:   pointer to draw list to be filled in
*                              n:              length of draw list
*                              mean:   mean of  distribution
*                              shift:  amount by which the distribution is shifted
******************************************************************************************/
void shifted_geometric_draw_list(
		int *list,
		int n,
		double mean,
		int shift
)
{
		geometric_draw_list( list, n, mean );

		int idx;
		for( idx = 0; idx < n; idx++ )
				list[idx] = shift + list[idx];
}

/*****************************************************************************************
*  Name:		gamma_rate_curve
*  Description: generates a rate curve for how infectious people are based
*  				upon a discrete gamma distribution and a multiplier
*
*  Arguments:	list:	pointer to draw list be filled in
*  				n:		length of draw list
*  				mean:	mean of gamma distribution
*  				sd:		sd of gamma distribution
*  				factor:	multipler of gamma pdf
******************************************************************************************/
void gamma_rate_curve(
	double *list,
	int n,
	double mean,
	double sd,
	double factor
)
{
	int idx = 0;
	double a, b, total;

	b = sd * sd / mean;
	a = mean / b;

	total = 0;
	for( idx = 0; idx < n; idx++ )
	{
		list[idx] = cdf_gamma_P( ( idx + 1 ) * 1.0, a, b ) - total;
		total += list[idx];
	}
	for( idx = 0; idx < n; idx++ )
		list[idx] *= factor / total;
}

/*****************************************************************************************
*  Name:		negative_binomial_draw
*  Description: Draws from a negative binomial distribution with a given mean
*  				and sd
*
*  Arguments:	mean - mean of distrubution
*  				sd	 - standard deviation of distribution
******************************************************************************************/
int negative_binomial_draw( double mean , double sd )
{
	double p, n;

	if( mean == 0 )
		return 0;

	if( mean >= sd * sd )
		print_exit( "negative binomial distirbution must have mean greater than variance" );

	p = mean / sd / sd;
	n = mean * mean / ( sd * sd - mean );

	return ran_negative_binomial( rng, p, n );
}

/*****************************************************************************************
*  Name:		discrete_draw
*  Description: Draws from a discrete set of probabilities
******************************************************************************************/
int discrete_draw( int n, double *p )
{
	int v;
	v = ran_discrete( rng, n, p );
	return(v);
}

/*****************************************************************************************
*  Name:		normalize_array
*  Description: normalizes an array by the the sum of the elements
******************************************************************************************/
void normalize_array( double *array, int N )
{
	int idx;
	double total = 0.0;
	for( idx = 0; idx < N; idx++ )
		total += array[idx];
	for( idx = 0; idx < N; idx++ )
		array[idx] /= total;
}

/*****************************************************************************************
*  Name:		copy_array
*  Description: copies an array of length N
******************************************************************************************/
void copy_array( double *to, double *from, int N )
{
	int idx;
	for( idx = 0; idx < N; idx++ )
		to[idx] = from[idx];
}

/*****************************************************************************************
*  Name:		copy_normalize_array
*  Description: copies an array of length N and then normalizes the copied array
******************************************************************************************/
void copy_normalize_array( double *to, double *from, int N )
{
	copy_array( to, from, N );
	normalize_array( to, N );
}

/*****************************************************************************************
*  Name:		sum_square_diff_array
*  Description: sums the squar of the difference of 2 arrays
******************************************************************************************/
double sum_square_diff_array( double *array, double *array2, int N )
{
	double diff = 0;
	int idx;

	for( idx = 0; idx < N; idx++ )
		diff += ( array[idx] - array2[idx] ) * ( array[idx] -  array2[idx]);
	return diff;
}

/*****************************************************************************************
*  Name:		compare_longs
*  Description: which of 2 longs is largest
******************************************************************************************/
int compare_longs (const void *a, const void *b)
{
    const long *da = (const long *) a;
    const long *db = (const long *) b;
    return (*da > *db) - (*da < *db);
}

/*****************************************************************************************
*  Name:		n_unique_elements
*  Description: how many unique elements in an array of longs
******************************************************************************************/
int n_unique_elements( long* array, int n )
{
	if( n == 0 )
		return 0;

	int idx, n_unique = 1;

	qsort( array, n, sizeof( long ), compare_longs );
	for( idx = 1; idx < n; idx++ )
		if( array[ idx - 1 ] != array[ idx ] )
			n_unique++;

	return n_unique;
}

