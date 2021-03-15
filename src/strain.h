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
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialize_strain( strain*, long, strain*, float );
void set_strain_multiplier( strain*, float );
void set_parent_idx( strain*, long );
void mutate_strain( strain* );


#endif /* STRAIN_H_ */