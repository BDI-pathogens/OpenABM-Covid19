/*
 * disease.h
 *
 * Description: this file contains code relating to the dynamics of the disease
 *				within an individual and transmission events during interactions
 *  Created on: 18 Mar 2020
 *      Author: hinchr
 */

#ifndef DISEASE_H_
#define DISEASE_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/

#include "individual.h"
#include "model.h"
#include "utilities.h"


/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

#define sample_transition_time( model, type ) ( sample_draw_list( model->transition_time_distributions[type] ) )


// set up distributions and infectious curves
void set_up_transition_times( model* );
void set_up_infectious_curves( model* );
double estimate_mean_interactions_by_age( model *pmodel, int age );

// transmission of the virus
void transmit_virus( model* );
void transmit_virus_by_type( model*, int );

// progression of the disease
void new_infection( model*, individual*, individual*, int, strain* );
void new_infection_by_idx( model*, long, long, int, short );
int new_infection_by_idx_check_safe( model*, long, long, int, short );
short seed_infect_by_idx( model*, long, int, int );
long seed_infect_n_people( model*, long, int, int );
void transition_to_symptomatic( model*, individual* );
void transition_to_symptomatic_mild( model*, individual* );
void transition_to_hospitalised( model*, individual* );
void transition_to_hospitalised_recovering( model*, individual* );
void transition_to_critical( model*, individual* );
void transition_to_recovered( model*, individual* );
void transition_to_susceptible( model*, individual* );
void transition_to_death( model*, individual* );
void transition_one_disese_event( model*, individual*, int, int, int );
short apply_cross_immunity( model*, individual*, short, short );

// calculation of R of disease
long n_newly_infected( model*, int time );
double calculate_R_instanteous( model*, int, double );

// cross-immunity
void set_cross_immunity_probability( model*, int, int, float );

#endif /* DISEASE_H_ */
