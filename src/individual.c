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
	for( jdx = 0; jdx < N_EVENT_TYPES; jdx++ )
		indiv->time_event[jdx] = UNKNOWN;
	
	indiv->quarantine_event         = NULL;
	indiv->quarantine_release_event = NULL;
	indiv->current_disease_event    = NULL;
	indiv->next_disease_event       = NULL;
	indiv->quarantine_test_result   = NO_TEST;

	indiv->infector_status  = UNKNOWN;
	indiv->infector_network = UNKNOWN;

    //TODO: kelvin change
    indiv->worker_type = OTHER;
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
*  Name:		set_age_group
*  Description: sets a person's age type and draws other properties based up on this
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
	double mean, sd;

	indiv->age_group = group;
	indiv->age_type  = AGE_TYPE_MAP[group];

	mean = params->mean_random_interactions[indiv->age_type];
	sd   = params->sd_random_interactions[indiv->age_type];
	indiv->base_random_interactions = negative_binomial_draw( mean, sd );
	update_random_interactions( indiv, params );
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
	free( indiv->time_event );
};

