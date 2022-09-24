/*
 * random.h
 *
 *  Created on: 22 Dec 2021
 *      Author: adamfowleruk
 * Description: abstracts random number funtions from a C library
 */

#ifndef RANDOM_H_
#define RANDOM_H_

#ifdef __cplusplus
extern "C" {
#endif

// definitions - provided to implementations
struct incomplete_gamma_p_params { long n; double percentile; };

// FWD DECLS - filled out by implementations
typedef struct generator generator;

#ifdef __cplusplus
}
#endif

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/

#ifdef USE_STATS
#include "random_stats.h"
#else
#include "random_gsl.h"
#endif


#ifdef __cplusplus
extern "C" {
#endif

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

// DEFINED IN INLUDES or FWD DECL above

// NOTE: Implementation is defined in random_stats.h or random_gsl.h

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void rng_initialise( ); // TODO use ARMA A.randu(1,1);

struct generator * rng_alloc( );
void rng_free( generator * gen );
void rng_set( generator * gen, long seed );



double rng_uniform( generator * gen );
int rng_uniform_int( generator * gen, long p );
unsigned int ran_bernoulli( generator * gen, double p );
double ran_gamma( generator * gen, double a, double b );
double ran_exponential( generator * gen, double mu );
void ran_shuffle( generator * gen, void * base, size_t n, size_t size );
unsigned int ran_negative_binomial( generator * gen, double p, double n );

size_t ran_discrete( generator * gen, size_t K, const double *P );

// Non RNG functions

double cdf_gamma_P( double P, double a, double b );
double cdf_gamma_Pinv( double P, double a, double b );
double cdf_exponential_Pinv( double P, double mu );

// Utility functions abstracted away to random library
double inv_incomplete_gamma_p( double, long );

#ifdef __cplusplus
}
#endif

#endif /* RANDOM_H_ */