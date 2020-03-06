/*
 * model.h
 *
 *  Description: Top level model 'object' c
 *  Created on:  5 Mar 2020
 *      Author:  hinchr
 */

#ifndef MODEL_H_
#define MODEL_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/

#include "individual.h"
#include "params.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct{
	parameters params;
	individual *population;
	int time;

	interaction *interactions;
	long interaction_idx;
	int interaction_day_idx;
	long n_interactions;
	long *possible_interactions;
	long n_possible_interactions;

} model;

gsl_rng * rng;

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

model* new_model();
void set_up_population( model* );
void set_up_interactions( model* );
void destroy_model( model* );

int one_time_step( model* );

#endif /* MODEL_H_ */
