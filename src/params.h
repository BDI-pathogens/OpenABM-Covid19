/*
 * params.h
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#ifndef PARAMS_H_
#define PARAMS_H_

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct{
	int param_line_number;			// line number to be read from parameter file
	long param_id;					// id of the parameter set
	long n_total;  					// total number of people
	int mean_daily_interactions;    // mean number of daily interactions
	int days_of_interactions;		// the number of days of interactions to keep
	int end_time;				    // maximum end time
	int n_seed_infection;			// number of people seeded with the infections

	double mean_infectious_period;  // mean period in days that people are infectious
	double sd_infectious_period;	// sd of period in days that people are infectious
	double infectious_rate;         // mean total number of people infected for a mean person

	double mean_time_to_symptoms;   // mean time from infection to symptoms
	double sd_time_to_symptoms;		// sd time from infection to symptoms

	double mean_time_to_hospital;   // mean time from symptoms to hospital

	double mean_time_to_recover;	// mean time to recover after hospital
	double sd_time_to_recover;  	// sd time to recover after hospital
	double mean_time_to_death;		// mean time to death after hospital
	double sd_time_to_death;		// sd time to death after hospital
	double cfr;						// case fatality rate

	double fraction_asymptomatic;			// faction who are asymptomatic
	double asymptomatic_infectious_factor;  // relative infectiousness of asymptomatics
	double mean_asymptomatic_to_recovery;   // mean time to recovery for asymptomatics
	double sd_asymptomatic_to_recovery;     // sd of time to recovery for asymptomatics

	int quarantined_daily_interactions; 	// number of interactions a quarantined person has

} parameters;

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void check_params( parameters* );

#endif /* PARAMS_H_ */
