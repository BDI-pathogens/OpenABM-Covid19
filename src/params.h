/*
 * params.h
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#ifndef PARAMS_H_
#define PARAMS_H_

#include "constant.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct{
	long rng_seed; 					// number used to seed the GSL RNG
	char input_param_file[INPUT_CHAR_LEN];	// path to input parameter file
	char output_file_dir[INPUT_CHAR_LEN];	// path to output directory
	int param_line_number;			// line number to be read from parameter file
	long param_id;					// id of the parameter set
	long n_total;  					// total number of people
	int days_of_interactions;		// the number of days of interactions to keep
	int end_time;				    // maximum end time
	int n_seed_infection;			// number of people seeded with the infections

	int mean_random_interactions[N_AGE_TYPES]; // mean number of random interactions each day
	int mean_work_interactions[N_WORK_NETWORKS];// mean number of regular work interactions
	double daily_fraction_work;      			// fraction of daily work interactions without social-distancing
	double daily_fraction_work_used;      		// fraction of daily work interactions with social-distancing
	double child_network_adults;				// fraction of adults in the child network
	double elderly_network_adults;				// fraction of adults in the elderly network

	double mean_infectious_period;  // mean period in days that people are infectious
	double sd_infectious_period;	// sd of period in days that people are infectious
	double infectious_rate;         // mean total number of people infected for a mean person

	double relative_susceptibility_child;	// relative susceptibility of children to adults per day (i.e. after adjust for no. interactions)
	double relative_susceptibility_elderly; // relative susceptibility of elderly to adults per day (i.e. after adjust for no. interactions)
	double adjusted_susceptibility_child;	// adjusted susceptibility of a child per interaction (derived from relative value and no. of interactions)
	double adjusted_susceptibility_elderly; // adjusted susceptibility of an elderly per interaction (derived from relative value and no. of interactions)

	double relative_susceptibility[N_AGE_GROUPS]; // relative susceptibility of an age group
	double adjusted_susceptibility[N_AGE_GROUPS]; // adjusted susceptibility of an age group (normalising for interactions)

	double relative_transmission_by_type[N_INTERACTION_TYPES]; 		// relative transmission rate by the type of interactions (e.g. household/workplace/random) w/o social distance
	double relative_transmission_by_type_used[N_INTERACTION_TYPES]; // relative transmission rate by the type of interactions (e.g. household/workplace/random)

	double mean_time_to_symptoms;   // mean time from infection to symptoms
	double sd_time_to_symptoms;		// sd time from infection to symptoms

	double hospitalised_fraction_type[N_AGE_TYPES];   // fraction of symptomatic patients requiring hospitalisation
	double critical_fraction_type[N_AGE_TYPES];  	  // fraction of hospitalised patients who require ICU treatment
	double fatality_fraction_type[N_AGE_TYPES];  	  // fraction of ICU patients who die

	double hospitalised_fraction[N_AGE_GROUPS];   // fraction of symptomatic patients requiring hospitalisation
	double critical_fraction[N_AGE_GROUPS];  	  // fraction of hospitalised patients who require ICU treatment
	double fatality_fraction[N_AGE_GROUPS];  	  // fraction of ICU patients who die


	double mean_time_to_hospital;   // mean time from symptoms to hospital
	double mean_time_to_critical;   // mean time from hospitalised to critical care

	double mean_time_to_recover;	// mean time to recover after hospital
	double sd_time_to_recover;  	// sd time to recover after hospital
	double mean_time_to_death;		// mean time to death after hospital
	double sd_time_to_death;		// sd time to death after hospital

	double household_size[HOUSEHOLD_N_MAX];// ONS UK number of households with 1-6 person (in thousands)
	double population_group[N_AGE_GROUPS];		// ONS stratification of population (in millions)
	double population_type[N_AGE_TYPES];		// ONS stratification of population (in millions)

	double fraction_asymptomatic;			// faction who are asymptomatic
	double asymptomatic_infectious_factor;  // relative infectiousness of asymptomatics
	double mean_asymptomatic_to_recovery;   // mean time to recovery for asymptomatics
	double sd_asymptomatic_to_recovery;     // sd of time to recovery for asymptomatics

	int quarantined_daily_interactions; 	// number of interactions a quarantined person has
	int hospitalised_daily_interactions; 	// number of interactions a hopsitalised person has

	int quarantine_days;					// number of days of previous contacts to quarantine
	double self_quarantine_fraction;		// fraction of people who self-quarantine when show symptoms

	int quarantine_length_self;				// max length of quarantine if self-quarantine on symptoms
	int quarantine_length_traced;			// max length of quarantine if contact-traced
	int quarantine_length_positive;			// max length of quarantine if receive positive test result
	double quarantine_dropout_self;			// daily dropout rate if self-quarantined
	double quarantine_dropout_traced;		// daily dropout rate if contact-traced
	double quarantine_dropout_positive;     // daily dropout rate if receive positive test result
	int quarantine_on_traced;				// immediately quarantine those who are contact traced
	double traceable_interaction_fraction;  // the proportion of interactions which are traceable even if both users have app
	int tracing_network_depth;				// the number of layers in the interaction network to recursively trace
	int allow_clinical_diagnosis;			// allow a hospital clinical diagnosis to trigger interventions

	int quarantine_household_on_symptoms;   // quarantine other household members when someone shows symptoms
	int quarantine_household_on_positive;   // quarantine other household members when someone tests positive
	int quarantine_household_on_traced;		// quarantine other household members when someone is contact traced
	int quarantine_household_contacts_on_positive; // quarantine the contacts of other household members when someone tests positive

	int test_on_symptoms;					// carry out a test on those with symptoms
	int test_on_traced;						// carry out a test on those with positive test results
	int test_insensititve_period;			// number of days until a test is sensitive (delay test of recent contacts)
	int test_result_wait;					// number of days to wait for a test result
	int test_order_wait;					// minimum number of days to wait for a test to be taken
	
	double app_users_fraction; 				// Proportion of the population that use the apps
	int app_turned_on;						// is the app turned on
	int app_turn_on_time;   				// time after which the app is usable
	double seasonal_flu_rate; 				// Rate of seasonal flu

	double social_distancing_work_network_multiplier;		// during social distancing this multiplier is applied to the fraction of work network connections made
	double social_distancing_random_network_multiplier; 	// during social distancing this multiplier is applied to the fraction of random network connections made
	double social_distancing_house_interaction_multiplier;  // during social distancing this multiplier is applied to the strengin of home connections
	int social_distancing_time_on;							// social distancing turned on at this time
	int social_distancing_time_off;							// social distancing turned off at this time
	int social_distancing_on;								// is social distancing currently on
		
	int sys_write_individual; 		// Should an individual file be written to output?
	int sys_write_timeseries; 		// Should a time series file be written to output?  


} parameters;

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void check_params( parameters* );

#endif /* PARAMS_H_ */
