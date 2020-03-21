/*
 * individual.c
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#include "individual.h"
#include "params.h"
#include "constant.h"
#include "utilities.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

/*****************************************************************************************
*  Name:		initialize_individual
*  Description: initializes and individual at the start of the simulation, note can
*  				only be called once per individual
*  Returns:		void
******************************************************************************************/
void initialize_individual(
	individual *indiv,
	parameters *params,
	long idx
)
{
	int day, jdx;
	if( indiv->idx != 0 )
		print_exit( "Individuals can only be intitialized once!" );

	indiv->idx         = idx;
	indiv->status      = UNINFECTED;
	indiv->quarantined = FALSE;
	indiv->is_case     = FALSE;
	indiv->app_user	   = FALSE;

	for( day = 0; day < params->days_of_interactions; day++ )
	{
		indiv->n_interactions[ day ] = 0;
		indiv->interactions[ day ]   = NULL;
	}

	indiv->time_event = calloc( N_EVENT_TYPES, sizeof(int) );
	for( jdx = 0; jdx <= N_EVENT_TYPES; jdx++ )
		indiv->time_event[jdx] = UNKNOWN;
	
	indiv->quarantine_event         = NULL;
	indiv->quarantine_release_event = NULL;
	indiv->current_disease_event    = NULL;
	indiv->next_disease_event       = NULL;
	indiv->quarantine_test_result   = NO_TEST;
}

/*****************************************************************************************
*  Name:		initialize_hazard
*  Description: gives each individual an initial amount of hazard which they burn through
*  				with each interaction. The value is adjusted by their relative susceptibility
*  				per interaction.
*  Returns:		void
******************************************************************************************/
void initialize_hazard(
	individual *indiv,
	parameters *params
)
{
	double rate = 1.0;
	if( indiv->age_group == AGE_0_17 )
		rate /= params->adjusted_susceptibility_child;
	if( indiv->age_group == AGE_65 )
		rate /= params->adjusted_susceptibility_elderly;

	indiv->hazard = rate * gsl_ran_exponential( rng, 1.0 );
}

/*****************************************************************************************
*  Name:		set_quarantine_status
*  Description: sets the quarantine status of an individual and changes the
*  				number on interactions
*  Returns:		void
******************************************************************************************/
void set_quarantine_status(
	individual *indiv,
	parameters *params,
	int time,
	int status
)
{
	if( status )
	{
		indiv->quarantined             = TRUE;
		indiv->time_event[QUARANTINED] = time;
		indiv->random_interactions = params->quarantined_daily_interactions;
	}
	else
	{
		indiv->quarantined              = FALSE;
		indiv->time_event[QUARANTINED]  = UNKNOWN;
		indiv->quarantine_event         = NULL;
		indiv->quarantine_release_event = NULL;
		if( indiv->status != DEATH && indiv->status != HOSPITALISED )
			indiv->random_interactions = indiv->base_random_interactions;
	}
}

/*****************************************************************************************
*  Name:		set_age_group
*  Description: sets a person's age group and draws other properties based up on this
*
*  				1. The number of random interactions the person has a day which is
*  				   drawn from a negative binomial with an age dependent mean
*				2. Which work network they are a member of - some adults are
*				   assigned to the child/elderly network (i.e. teacher/carers)
*
*  Returns:		void
******************************************************************************************/
void set_age_group( individual *indiv, parameters *params, int group )
{
	double mean, child_net_adults, elderly_net_adults, x;

	indiv->age_group = group;

	mean = params->mean_random_interactions[group];
	indiv->base_random_interactions = negative_binomial_draw( mean, mean );
	indiv->random_interactions      = indiv->base_random_interactions;

	if( group == AGE_18_64 )
	{
		child_net_adults   = params->child_network_adults * params->uk_pop[AGE_0_17] / params->uk_pop[AGE_18_64];
		elderly_net_adults = params->elderly_network_adults * params->uk_pop[AGE_65] / params->uk_pop[AGE_18_64];

		x = gsl_rng_uniform( rng );
		if( x < child_net_adults )
			indiv->work_network = AGE_0_17;
		else if(  x < ( elderly_net_adults + child_net_adults ) )
			indiv->work_network = AGE_65;
		else
			indiv->work_network = AGE_18_64;
	}
	else
		indiv->work_network = group;
}

/*****************************************************************************************
*  Name:		set_dead
*  Description: sets a person as dead
*  Returns:		void
******************************************************************************************/
void set_dead( individual *indiv, int time )
{
	indiv->status        = DEATH;
	indiv->current_disease_event = NULL;
	indiv->random_interactions = 0;
}

/*****************************************************************************************
*  Name:		set_recovered
*  Description: sets a person to recovered
*  Returns:		void
******************************************************************************************/
void set_recovered( individual *indiv, parameters* params, int time )
{
	indiv->status        = RECOVERED;
	indiv->current_disease_event = NULL;
	indiv->random_interactions = indiv->base_random_interactions;
}

/*****************************************************************************************
*  Name:		set_hospitalised
*  Description: sets a person to hospitalised
*  Returns:		void
******************************************************************************************/
void set_hospitalised( individual *indiv, parameters* params, int time )
{
	indiv->status = HOSPITALISED;
	indiv->random_interactions = params->hospitalised_daily_interactions;
}

/*****************************************************************************************
*  Name:		set_critical
*  Description: sets a person to critical
*  Returns:		void
******************************************************************************************/
void set_critical( individual *indiv, parameters* params, int time )
{
	indiv->status = CRITICAL;
}

/*****************************************************************************************
*  Name:		set_case
*  Description: sets a person to be a case
*  Returns:		void
******************************************************************************************/
void set_case( individual *indiv, int time )
{
	indiv->is_case   = TRUE;
	indiv->time_event[CASE] = time;
}

/*****************************************************************************************
*  Name:		destroy_individual
*  Description: Destroys the model structure and releases its memory
******************************************************************************************/
void destroy_individual( individual *indiv )
{
    //free( indiv->interactions );
};

