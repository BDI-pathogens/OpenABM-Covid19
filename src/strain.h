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
	long total_infected;
	strain *next;
};

typedef struct model model; // use to avoid having to `#include model.h`, which also requires strain.h
typedef struct individual individual; // use to avoid having to `#include individual.h`, which also requires strain.h

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialize_strain( strain*, long, strain*, float, long, long );
void mutate_strain( model*, individual*, double );
void add_new_strain_to_model( model*, strain* );
void add_to_strain_bin_count( long*, float, int );


#endif /* STRAIN_H_ */