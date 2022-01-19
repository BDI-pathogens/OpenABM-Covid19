/*
 * random_arma.h
 *
 *  Created on: 22 Dec 2021
 *      Author: adamfowleruk
 * Description: Uses Keith O'Hara's (Apache-2.0) stats and gcem libraries
 */

#ifndef RANDOM_STATS_H_
#define RANDOM_STATS_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/

#include <random>
#include "stats.hpp"

namespace {

template <typename ArrEl>
void switchElements(ArrEl& i1, ArrEl& i2) {
  ArrEl tmp = i1;
  i1 = i2;
  i2 = tmp;
}

}

#ifdef __cplusplus
extern "C" {
#endif

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

struct generator{
  // Use C++ std::mt19937 with seed of 0 by default (5489u)
#ifdef GSL_COMPAT
  std::mt19937 rng; // requires a 1 line change to stats/misc/options.hpp to use mt19937 and not mt19937_64 in order to compile
#else
  std::mt19937_64 rng; // works by default with the stats library
#endif
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

// STATS specific definitions only (internal API, so likely none)

// fwd decl
int rng_uniform_int( generator * gen, long p );

#ifdef __cplusplus
}


template <typename ArrEl>
void ran_shuffle( generator * gen, ArrEl* base, size_t n, size_t size) {
  size_t i1 = n - 1;
  size_t i2 = 0;
  for ( ; i1 > 0; --i1 ) {
    i2 = rng_uniform_int( gen, i1 + 1 );
    switchElements( base[i1], base[i2] );
  }
}

#endif

#endif /* RANDOM_STATS_H_ */