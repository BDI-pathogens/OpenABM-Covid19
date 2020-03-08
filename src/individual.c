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
	int day;
	if( indiv->idx != 0 )
		print_exit( "Individuals can only be intitialized once!" );

	indiv->idx         = idx;
	indiv->status      = UNINFECTED;
	indiv->quarantined = FALSE;
	indiv->hazard      = gsl_ran_exponential( rng, 1.0 );

	indiv->mean_interactions = params->mean_daily_interactions;
	for( day = 0; day < params->days_of_interactions; day++ )
		indiv->n_interactions[ day ] = 0;

	indiv->time_infected     = UNKNOWN;
	indiv->time_symptomatic  = UNKNOWN;
	indiv->time_asymptomatic = UNKNOWN;
	indiv->time_hospitalised = UNKNOWN;
	indiv->time_death	     = UNKNOWN;
	indiv->time_recovered    = UNKNOWN;
	indiv->next_event_type   = UNKNOWN;
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
	int status
)
{
	if( status )
	{
		indiv->quarantined         = TRUE;
		indiv->mean_interactions = params->quarantined_daily_interactions;
	}
	else
	{
		indiv->quarantined         = FALSE;
		indiv->mean_interactions = params->mean_daily_interactions;
	}
}

/*****************************************************************************************
*  Name:		set_dead
*  Description: sets a person as dead
*  Returns:		void
******************************************************************************************/
void set_dead( individual *indiv )
{
	indiv->status = DEATH;
	indiv->mean_interactions = 0;
}

/*****************************************************************************************
*  Name:		set_recovered
*  Description: sets a person to recovered
*  Returns:		void
******************************************************************************************/
void set_recovered( individual *indiv, parameters* params )
{
	indiv->status = RECOVERED;
	indiv->mean_interactions = params->mean_daily_interactions;
}

/*****************************************************************************************
*  Name:		set_hospitalised
*  Description: sets a person to hospitalised
*  Returns:		void
******************************************************************************************/
void set_hospitalised( individual *indiv, parameters* params )
{
	indiv->status = HOSPITALISED;
	indiv->mean_interactions = params->quarantined_daily_interactions;
}

/*****************************************************************************************
*  Name:		destroy_individual
*  Description: Destroys the model structure and releases its memory
******************************************************************************************/
void destroy_individual( individual *indiv )
{
    //free( indiv->interactions );
};

