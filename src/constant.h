/*
 * constant.h
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#ifndef CONSTANT_H_
#define CONSTANT_H_

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>


enum DISEASE_STATUS{
	UNINFECTED,
	PRESYMPTOMATIC,
	SYMPTOMATIC,
	HOSPITALIZED,
	RECOVERED,
	DEAD
};

#define MAX_DAILY_INTERACTIONS_KEPT 5
#define MAX_TIME 1000
#define MAX_INFECTIOUS_PERIOD 40
#define N_DRAW_LIST 1000

gsl_rng * rng;

#endif /* CONSTANT_H_ */

