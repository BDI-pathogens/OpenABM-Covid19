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

#include <gsl/gsl_rng.h>
#include "individual.h"
#include "params.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/
typedef struct interaction interaction;

typedef struct{
	gsl_rng *rng;
	parameters params;
	individual *population;

	interaction *interactions;
	long interaction_idx;
	int interaction_day_idx;
	long n_interactions;
	long *possible_interactions;
	long n_possible_interactions;

} model;

struct interaction{
	individual *individual;
	interaction *next;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

model* new_model();
void set_up_gsl( model* );
void set_up_population( model* );
void set_up_interactions( model* );
void destroy_model( model* );

#endif /* MODEL_H_ */
