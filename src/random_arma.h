/*
 * random_arma.h
 *
 *  Created on: 22 Dec 2021
 *      Author: adamfowleruk
 * Description: Armadillo (Apache-2.0) random implementation
 */

#ifndef RANDOM_ARMA_H_
#define RANDOM_ARMA_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/

// TODO
#include <random>
// #include <armadillo>
#include "stats.hpp"


#ifdef __cplusplus
extern "C" {
#endif

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

struct generator{
  // Use C++ std::mt19937_64 with seed of 0 by default (5489u)
  std::mt19937_64 rng;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

// ARMA specific definitions only (internal API, so likely none)

#ifdef __cplusplus
}
#endif

#endif /* RANDOM_ARMA_H_ */