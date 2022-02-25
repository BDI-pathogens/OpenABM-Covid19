/*
 * strain.h
 *
 *  Created on: 31 Mar 2021
 *      Author: nikbaya
 */

#ifndef STRAIN_H_
#define STRAIN_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/

#include "structure.h"
#include "constant.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct strain strain;

struct strain{
	long idx;
	float transmission_multiplier;
	float mean_infectious_period;
	float sd_infectious_period;
	double hospitalised_fraction[N_AGE_GROUPS];
	long total_infected;
	double **infectious_curve;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

short add_new_strain( model*, float, double*, double, double );
void destroy_strain( strain* );
strain* get_strain_by_id( model*, short );

#endif /* STRAIN_H_ */
