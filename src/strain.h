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
	float mean_time_to_symptoms;
	float sd_time_to_symptoms;
	float mean_asymptomatic_to_recovery;
	float sd_asymptomatic_to_recovery;
	float mean_time_to_recover;
	float sd_time_to_recover;
	float mean_time_hospitalised_recovery;
	float sd_time_hospitalised_recovery;
	float mean_time_critical_survive;
	float sd_time_critical_survive;
	float mean_time_to_death;
	float sd_time_to_death;
	float mean_time_to_hospital;
	float mean_time_to_critical;
	float sd_time_to_critical;
	float mean_time_to_susceptible_after_shift;
	float time_to_susceptible_shift;
	float fraction_asymptomatic[N_AGE_GROUPS]; // fraction who are asymptomatic
	float mild_fraction[N_AGE_GROUPS]; 	       // fraction who just have mild symptoms
	float hospitalised_fraction[N_AGE_GROUPS]; // fraction of symptomatic patients requiring hospitalisation
	float critical_fraction[N_AGE_GROUPS];     // fraction of hospitalised patients who require ICU treatment
	float fatality_fraction[N_AGE_GROUPS];     // fraction of ICU patients who die
	float location_death_icu[N_AGE_GROUPS];    // whether death occurs in the ICU or outside
	float asymptomatic_infectious_factor;
	float mild_infectious_factor;
	long total_infected;
	double **infectious_curve;
	int **transition_time_distributions;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

short add_new_strain( model*, float, double*, double*, double*, double*, double*, double*,
	double, double, double, double, double, double, double,
	double, double, double, double, double, double, double,
	double, double, double, double, double, double, double
);
void destroy_strain( strain* );
strain* get_strain_by_id( model*, short );

#endif /* STRAIN_H_ */
