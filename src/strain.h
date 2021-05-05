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
	float *antigen_phen;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialise_strain( model*, long, float );
void set_antigen_phen_distance( model*, strain*, strain*, float );
float get_antigen_phen_distance( model*, strain*, strain* );

#endif /* STRAIN_H_ */