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
	indiv->hazard = gsl_ran_exponential( rng, 1.0 ) / params->adjusted_susceptibility[indiv->age_group];
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
	}
	else
	{
		indiv->quarantined              = FALSE;
		indiv->time_event[QUARANTINED]  = UNKNOWN;
		indiv->quarantine_event         = NULL;
		indiv->quarantine_release_event = NULL;
	}
	update_random_interactions( indiv, params );
}

/*****************************************************************************************
*  Name:		set_age_type
*  Description: sets a person's age type and draws other properties based up on this
*
*  				1. The number of random interactions the person has a day which is
*  				   drawn from a negative binomial with an age dependent mean
*				2. Which work network they are a member of - some adults are
*				   assigned to the child/elderly network (i.e. teacher/carers)
*
*  Returns:		void
******************************************************************************************/
void set_age_type( individual *indiv, parameters *params, int type )
{
	double mean, child_net_adults, elderly_net_adults, x;

	indiv->age_type = type;

	mean = params->mean_random_interactions[type];
	indiv->base_random_interactions = negative_binomial_draw( mean, mean );
	update_random_interactions( indiv, params );

	if( type == AGE_TYPE_ADULT )
	{
		child_net_adults   = params->child_network_adults * params->population_type[AGE_TYPE_CHILD] / params->population_type[AGE_TYPE_ADULT];
		elderly_net_adults = params->elderly_network_adults * params->population_type[AGE_TYPE_ELDERLY] / params->population_type[AGE_TYPE_ADULT];

		x = gsl_rng_uniform( rng );
		if( x < child_net_adults )
			indiv->work_network = AGE_TYPE_CHILD;
		else if(  x < ( elderly_net_adults + child_net_adults ) )
			indiv->work_network = AGE_TYPE_ELDERLY;
		else
			indiv->work_network = AGE_TYPE_ADULT;
	}
	else
		indiv->work_network = type;

	if( type == AGE_TYPE_CHILD )
	{
		double p_child = params->population_group[AGE_0_9] +  params->population_group[AGE_10_19];
		double p_0_9   = params->population_group[AGE_0_9] / p_child;
		double p_10_19 = params->population_group[AGE_10_19] / p_child;
		double p[2]    = {p_0_9,p_10_19};

		indiv->age_group = discrete_draw( 2, p );
	}
	else
	if( type == AGE_TYPE_ADULT )
	{
		double p_adult = params->population_group[AGE_20_29] +  params->population_group[AGE_30_39]+  params->population_group[AGE_40_49]+  params->population_group[AGE_50_59]+  params->population_group[AGE_60_69];
		double p_20_29 = params->population_group[AGE_20_29] / p_adult;
		double p_30_39 = params->population_group[AGE_30_39] / p_adult;
		double p_40_49 = params->population_group[AGE_40_49] / p_adult;
		double p_50_59 = params->population_group[AGE_50_59] / p_adult;
		double p_60_69 = params->population_group[AGE_60_69] / p_adult;
		double p[5]    = {p_20_29,p_30_39,p_40_49,p_50_59,p_60_69};

		indiv->age_group = discrete_draw( 5, p ) +AGE_20_29;
	}
	else
	{

		double p_elderly = params->population_group[AGE_70_79] +  params->population_group[AGE_80];
		double p_70_79   = params->population_group[AGE_70_79] / p_elderly;
		double p_80      = params->population_group[AGE_80] / p_elderly;
		double p[2]    = {p_70_79,p_80};

		indiv->age_group = discrete_draw( 2, p ) + AGE_70_79;

	}
}

/*****************************************************************************************
*  Name:		update_random_interactions
*  Description: update the number of random interactions for an individual after a
*  				change in status both at the individual level or national policy
*  Returns:		void
******************************************************************************************/
void update_random_interactions( individual *indiv, parameters* params )
{
	double n = indiv->base_random_interactions;


	if( !indiv->quarantined )
	{
		switch( indiv->status )
		{
			case DEATH:			n = 0; 										 break;
			case HOSPITALISED:	n = params->hospitalised_daily_interactions; break;
			case CRITICAL:		n = params->hospitalised_daily_interactions; break;
			default: 			n = ifelse( params->social_distancing_on, n * params->social_distancing_random_network_multiplier, n );
		}
	}
	else
		n = params->quarantined_daily_interactions;

	indiv->random_interactions = round_random( n );
}

/*****************************************************************************************
*  Name:		set_dead
*  Description: sets a person as dead
*  Returns:		void
******************************************************************************************/
void set_dead( individual *indiv, parameters* params, int time )
{
	indiv->status        = DEATH;
	indiv->current_disease_event = NULL;
	update_random_interactions( indiv, params );
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
	update_random_interactions( indiv, params );
}

/*****************************************************************************************
*  Name:		set_hospitalised
*  Description: sets a person to hospitalised
*  Returns:		void
******************************************************************************************/
void set_hospitalised( individual *indiv, parameters* params, int time )
{
	indiv->status = HOSPITALISED;
	update_random_interactions( indiv, params );
}

/*****************************************************************************************
*  Name:		set_house_no
*  Description: sets a person house number
*  Returns:		void
******************************************************************************************/
void set_house_no( individual *indiv, long number )
{
	indiv->house_no = number;
}

/*****************************************************************************************
*  Name:		set_critical
*  Description: sets a person to critical
*  Returns:		void
******************************************************************************************/
void set_critical( individual *indiv, parameters* params, int time )
{
	indiv->status = CRITICAL;
	update_random_interactions( indiv, params );
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

