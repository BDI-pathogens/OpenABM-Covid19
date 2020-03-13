/*
 * utilities.h
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

/************************************************************************/
/******************************  Macros     *****************************/
/************************************************************************/

#define max(x,y) ((x) > (y) ? (x) : (y))
#define min(x,y) ((x) < (y) ? (x) : (y))
#define ifelse(x,y,z) ((x) ? (y) : (z))
#define round_random( x ) ( (long int) ( floor( x ) + gsl_ran_bernoulli( rng, x - floor(x) ) ) )
#define ring_inc( x, n ) ( ( x ) = ifelse( ( x ) == ( ( n ) -1 ), 0 , ( x ) + 1 ) )
#define sample_draw_list( x ) ( ( x[ gsl_rng_uniform_int( rng, N_DRAW_LIST ) ] ) )

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void print_now( char* );
void print_exit( char* );
void gamma_draw_list( int*, int, double, double );
void bernoulli_draw_list( int*, int, double );
void gamma_rate_curve( double*, int, double, double, double );
int negative_binomial_draw( double, double );

#endif /* UTILITIES_H_ */
