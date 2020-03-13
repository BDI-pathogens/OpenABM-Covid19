/*
 * utilities.c
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "constant.h"
#include "utilities.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

/*****************************************************************************************
*  Name:		print_exit
******************************************************************************************/
void print_exit( char *s )
{
    printf("%s\n", s );
    fflush(stdout);
    exit(1);
}

/*****************************************************************************************
*  Name:		print_now
******************************************************************************************/
void print_now( char *s )
{
    printf("%s\n", s );
    fflush(stdout);
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
		list[idx] = ceil( gsl_cdf_gamma_Pinv( ( idx + 1.0 )/( n + 1.0 ), a, b ));
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
		list[idx] = gsl_cdf_gamma_P( ( idx + 1 ) * 1.0, a, b ) - total;
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

	p = mean / sd / sd;
	n = mean * mean / ( sd * sd - mean );

	return gsl_ran_negative_binomial( rng, p, n );
}


