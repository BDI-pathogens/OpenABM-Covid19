/*
 * strain.h
 *
 *  Created on: 11 Mar 2021
 *      Author: nikbaya
 */

#ifndef STRAIN_H_
#define STRAIN_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/


/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct strain strain;

struct strain{
	long idx;
	strain *parent;
	float transmission_multiplier;
	long n_infected;
};

typedef struct individual individual; // use to avoid having to `#include individual.h`, which also requires strain.h

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialize_strain( strain*, long, strain*, float, long );
// void set_strain_multiplier( strain*, float );
// void set_parent_idx( strain*, long );
void mutate_strain( individual* );


#endif /* STRAIN_H_ */