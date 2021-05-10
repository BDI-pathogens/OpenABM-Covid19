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
	long parent_idx;
	float transmission_multiplier;
	double *antigen_phen;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialise_strain( model*, long, float );
void set_antigen_phen_distance( model*, long, long, float );
float get_antigen_phen_distance( model*, strain*, strain* );
strain* mutate_strain( model*, strain* );

#endif /* STRAIN_H_ */