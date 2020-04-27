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

	indiv->trace_tokens         = NULL;
	indiv->index_trace_token    = NULL;
	indiv->traced_on_this_trace = FALSE;

#if HOSPITAL_ON
    indiv->hospital_state = NOT_IN_HOSPITAL;
	indiv->current_hospital_event = NULL;
	indiv->next_hospital_event = NULL;
    indiv->ward_type = NO_WARD;
    indiv->ward_idx  = NO_WARD;
    indiv->hospital_idx = NO_HOSPITAL;
	indiv->disease_progression_predicted[0] = FALSE;
	indiv->disease_progression_predicted[1] = FALSE;
    indiv->worker_type = NOT_HEALTHCARE_WORKER;
#endif
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

	switch( params->random_interaction_distribution )
	{
		case FIXED:				indiv->base_random_interactions = mean; break;
		case NEGATIVE_BINOMIAL: indiv->base_random_interactions = negative_binomial_draw( mean, sd ); break;
        default:
		print_exit( "random_interaction_distribution not supported" );
	}

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
	int lockdown;

	if( !indiv->quarantined )
	{
		lockdown = params->lockdown_on;
		if( indiv->age_type == AGE_TYPE_ELDERLY )
			lockdown = max( lockdown, params->lockdown_elderly_on );

#if HOSPITAL_ON
        switch( indiv->hospital_state )
		{
            case MORTUARY:		            n = 0; break;
            case WAITING:                   n = params->hospitalised_daily_interactions; break;
            case GENERAL:                   n = params->hospitalised_daily_interactions; break;
            case ICU:                       n = params->hospitalised_daily_interactions; break;
			default: 			            n = ifelse( lockdown, n * params->lockdown_random_network_multiplier, n );
		}
#else
        switch( indiv->status )
        {
            case DEATH:			n = 0; 										 break;
            case HOSPITALISED:	n = params->hospitalised_daily_interactions; break;
            case CRITICAL:		n = params->hospitalised_daily_interactions; break;
            case HOSPITALISED_RECOVERING: n = params->hospitalised_daily_interactions; break;
            default: 			n = ifelse( lockdown, n * params->lockdown_random_network_multiplier, n );
        }
#endif
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
*  Name:		set_hospitalised_recovering
*  Description: sets a person to hospitalised recovering
*  Returns:		void
******************************************************************************************/
void set_hospitalised_recovering( individual *indiv, parameters* params, int time )
{
	indiv->status = HOSPITALISED_RECOVERING;
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

#if HOSPITAL_ON
/*****************************************************************************************
*  Name:		set_waiting
*  Description: sets a person to be added to the hospital waiting list
*  Returns:		void
******************************************************************************************/
void set_waiting( individual *indiv, parameters* params, int time )
{
    indiv->hospital_state = WAITING;
    update_random_interactions( indiv, params );
}

/*****************************************************************************************
*  Name:		set_general_admission
*  Description: sets a person to be added to a general ward
*  Returns:		void
******************************************************************************************/
void set_general_admission( individual *indiv, parameters* params, int time )
{
    indiv->hospital_state = GENERAL;
    update_random_interactions( indiv, params );
}

/*****************************************************************************************
*  Name:		set_icu_admission
*  Description: sets a person to be added to an ICU
*  Returns:		void
******************************************************************************************/
void set_icu_admission( individual *indiv, parameters* params, int time )
{
    indiv->hospital_state = ICU;
    update_random_interactions( indiv, params );
}

/*****************************************************************************************
*  Name:		set_mortuary_admission
*  Description: sets a dead person to be added to the mortuary
*  Returns:		void
******************************************************************************************/
void set_mortuary_admission( individual *indiv, parameters* params, int time )
{
    indiv->hospital_state = MORTUARY;
    indiv->current_hospital_event = NULL;
    update_random_interactions( indiv, params );
}

/*****************************************************************************************
*  Name:		set_discharged
*  Description: sets a recovered person to be discharged from the hospital.
*  Returns:		void
******************************************************************************************/
void set_discharged( individual *indiv, parameters* params, int time )
{
    indiv->hospital_state = DISCHARGED;
    indiv->current_hospital_event = NULL;
    update_random_interactions( indiv, params );
}
#endif

/*****************************************************************************************
*  Name:		destroy_individual
*  Description: Destroys the model structure and releases its memory
******************************************************************************************/
void destroy_individual( individual *indiv )
{
	free( indiv->time_event );
};
