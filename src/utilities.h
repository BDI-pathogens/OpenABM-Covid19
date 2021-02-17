/*
 * utilities.h
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <math.h>

/************************************************************************/
/******************************  Macros     *****************************/
/************************************************************************/

#define max(x,y) ((x) > (y) ? (x) : (y))
#define min(x,y) ((x) < (y) ? (x) : (y))
#define ifelse(x,y,z) ((x) ? (y) : (z) )
#define round_random( x ) ( (long int) ( floor( x ) + gsl_ran_bernoulli( rng, x - floor(x) ) ) )
#define ring_inc( x, n ) ( ( x ) = ifelse( ( x ) == ( ( n ) -1 ), 0 , ( x ) + 1 ) )
#define ring_dec( x, n ) ( ( x ) = ifelse( ( x ) == 0 , ( n ) -1 , ( x ) - 1  ) )
#define sample_draw_list( x ) ( ( x[ gsl_rng_uniform_int( rng, N_DRAW_LIST ) ] ) )
#define printf(...) printf_w(__VA_ARGS__)

/* Tell GCC to check printf formats for custom print functions.
 * https://gcc.gnu.org/onlinedocs/gcc-3.2/gcc/Function-Attributes.html */
#if __GNUC__ > 3
  #define _AF_PRINTF( _fmtPos, _valistPos) \
   __attribute__((format( printf, _fmtPos, _valistPos )))
#else
  #define _AF_PRINTF( _fmtPos, _valistPos)
#endif

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

int printf_w( const char*, ... ) _AF_PRINTF(1,2);
void print_now( char* );
void print_exit( char*, ... ) _AF_PRINTF(1,2);
void fflush_stdout( void );
void gamma_draw_list( int*, int, double, double );
void bernoulli_draw_list( int*, int, double );
void geometric_max_draw_list( int*, int, double, int );
void shifted_geometric_draw_list( int*, int, double , int);
void geometric_draw_list( int*, int, double );
void gamma_rate_curve( double*, int, double, double, double );
int negative_binomial_draw( double, double );
int discrete_draw( int, double* );
void normalize_array( double*, int );
void copy_array( double*, double*, int );
void copy_normalize_array( double*, double*, int );
double sum_square_diff_array( double*, double*, int );
int n_unique_elements( long*, int );
void setup_gsl_rng( int );
struct incomplete_gamma_p_params { long n; double percentile; };
double incomplete_gamma_p( double, void *params );
double inv_incomplete_gamma_p( double, long );

#endif /* UTILITIES_H_ */
