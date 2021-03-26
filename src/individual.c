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
#include "model.h"

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
	indiv->status      = SUSCEPTIBLE;
	indiv->quarantined = FALSE;
	indiv->app_user	   = FALSE;

	indiv->n_interactions = calloc( params->days_of_interactions, sizeof( short ) );
	indiv->interactions   = calloc( params->days_of_interactions, sizeof( interaction* ) );
	for( day = 0; day < params->days_of_interactions; day++ )
	{
		indiv->n_interactions[ day ] = 0;
		indiv->interactions[ day ]   = NULL;
	}

	indiv->infection_events = calloc( 1, sizeof(struct infection_event) );
	indiv->infection_events->times = calloc( N_EVENT_TYPES, sizeof(short	) );
	for( jdx = 0; jdx < N_EVENT_TYPES; jdx++ )
		indiv->infection_events->times[jdx] = UNKNOWN;

	indiv->infection_events->infector_status  = UNKNOWN;
	indiv->infection_events->infector_network = UNKNOWN;
	indiv->infection_events->infector_hospital_state = UNKNOWN;
	indiv->infection_events->time_infected_infector = UNKNOWN;
	indiv->infection_events->next =  NULL;
	indiv->infection_events->is_case     = FALSE;

	indiv->quarantine_event         = NULL;
	indiv->quarantine_release_event = NULL;
	indiv->current_disease_event    = NULL;
	indiv->next_disease_event       = NULL;
	indiv->quarantine_test_result   = NO_TEST;

	indiv->trace_tokens         = NULL;
	indiv->index_trace_token    = NULL;
	indiv->traced_on_this_trace = FALSE;
	indiv->index_token_release_event = NULL;

	indiv->hospital_state = NOT_IN_HOSPITAL;
	indiv->current_hospital_event = NULL;
	indiv->next_hospital_event = NULL;
	indiv->ward_type = NO_WARD;
	indiv->ward_idx  = NO_WARD;
	indiv->hospital_idx = NO_HOSPITAL;
	indiv->disease_progression_predicted[0] = FALSE;
	indiv->disease_progression_predicted[1] = FALSE;
	indiv->worker_type = NOT_HEALTHCARE_WORKER;

	float sigma_x = params->sd_infectiousness_multiplier;
	if ( sigma_x > 0 )
	{
		float b = sigma_x * sigma_x;
		float a = 1 /b;
		indiv->infectiousness_multiplier = gsl_ran_gamma( rng, a, b );
	}
	else
	{
		indiv->infectiousness_multiplier = 1;
	}
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
	int status,
	model *model
)
{
	if( status )
	{
		indiv->quarantined             = TRUE;
		indiv->infection_events->times[QUARANTINED] = time;

		// Increment counters for time series output
		model->n_quarantine_events++;

		if(indiv->app_user == TRUE){
			model->n_quarantine_events_app_user++;
			model->n_quarantine_app_user++;
			if(indiv->status > SUSCEPTIBLE)
				model->n_quarantine_app_user_infected++;
			if(indiv->status == RECOVERED)
				model->n_quarantine_app_user_recovered++;
		}
		if(indiv->status > SUSCEPTIBLE)
			model->n_quarantine_infected++;
		if(indiv->status == RECOVERED)
			model->n_quarantine_recovered++;
	}
	else
	{
		indiv->quarantined              = FALSE;
		indiv->infection_events->times[QUARANTINED]  = UNKNOWN;
		indiv->quarantine_event         = NULL;
		indiv->quarantine_release_event = NULL;

		// Increment counters for time series output
		model->n_quarantine_release_events++;

		if(indiv->app_user == TRUE){
			model->n_quarantine_app_user--;
			model->n_quarantine_release_events_app_user++;
			if(indiv->status > SUSCEPTIBLE)
				model->n_quarantine_app_user_infected--;
			if(indiv->status == RECOVERED)
				model->n_quarantine_app_user_recovered--;
		}
		if(indiv->status > SUSCEPTIBLE)
			model->n_quarantine_infected--;
		if(indiv->status == RECOVERED)
			model->n_quarantine_recovered--;
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

		switch( indiv->status )
		{
			case DEATH:			n = 0; 										 break;
			case HOSPITALISED:	n = params->hospitalised_daily_interactions; break;
			case CRITICAL:		n = params->hospitalised_daily_interactions; break;
			case HOSPITALISED_RECOVERING: n = params->hospitalised_daily_interactions; break;
			default: 			n = ifelse( lockdown, n * params->lockdown_random_network_multiplier, n );
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
*  Name:		transition_vaccine_status
*  Description: moves a person to there next vaccine status
*  Returns:		void
******************************************************************************************/
void transition_vaccine_status( individual* indiv )
{
	if( indiv->vaccine_status_next != NO_EVENT )
		set_vaccine_status( indiv, indiv->vaccine_status_next, NO_EVENT );
}

/*****************************************************************************************
*  Name:		set_vaccine_status
*  Description: sets the vaccine status of an individual
*  Returns:		void
******************************************************************************************/
void set_vaccine_status( individual* indiv, short current_status, short next_status )
{
    // FIXME: need some additional logic to make sure that vaccination status is not made worse by a second vaccine
	indiv->vaccine_status      = current_status;
	indiv->vaccine_status_next = next_status;

	if( current_status == VACCINE_PROTECTED_FULLY )
	{
		if( indiv->status == SUSCEPTIBLE || indiv->status == RECOVERED )
			indiv->status = VACCINE_PROTECT;
	}

	if( ( current_status == VACCINE_WANED ) & ( indiv->status == VACCINE_PROTECT ) )
		indiv->status = SUSCEPTIBLE;
}

/*****************************************************************************************
*  Name:		set_recovered
*  Description: sets a person to recovered
*  Returns:		void
******************************************************************************************/
void set_recovered( individual *indiv, parameters* params, int time, model *model )
{
	if( indiv->quarantined == TRUE){
		model->n_quarantine_recovered++;
		if(indiv->app_user == TRUE)
			model->n_quarantine_app_user_recovered++;
	}

	indiv->status        = RECOVERED;
	indiv->current_disease_event = NULL;
	update_random_interactions( indiv, params );
}

/*****************************************************************************************
*  Name:		set_susceptible
*  Description: sets a person to susceptible
*  Returns:		void
******************************************************************************************/
void set_susceptible( individual *indiv, parameters* params, int time )
{
	indiv->status        = SUSCEPTIBLE;

	infection_event *infection_event_ptr;
	infection_event_ptr = indiv->infection_events;

	int jdx;
	indiv->infection_events = calloc( 1, sizeof(struct infection_event) );
	indiv->infection_events->times = calloc( N_EVENT_TYPES, sizeof(int) );
	for( jdx = 0; jdx < N_EVENT_TYPES; jdx++ )
		indiv->infection_events->times[jdx] = UNKNOWN;

	indiv->infection_events->infector_status  = UNKNOWN;
	indiv->infection_events->infector_network = UNKNOWN;
	indiv->infection_events->time_infected_infector = UNKNOWN;
	indiv->infection_events->next =  infection_event_ptr;
	indiv->infection_events->is_case     = FALSE;

	// Reset the hazard for the newly susceptible individual
	initialize_hazard( indiv, params );
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
	indiv->infection_events->is_case   = TRUE;
	indiv->infection_events->times[CASE] = time;
}

/*****************************************************************************************
*  Name:		set_waiting
*  Description: sets a person to be added to the hospital waiting list
*  Returns:		void
******************************************************************************************/
void set_waiting( individual *indiv, parameters* params, int time )
{
	indiv->hospital_state = WAITING;
}

/*****************************************************************************************
*  Name:		set_general_admission
*  Description: sets a person to be added to a general ward
*  Returns:		void
******************************************************************************************/
void set_general_admission( individual *indiv, parameters* params, int time )
{
	indiv->hospital_state = GENERAL;
}

/*****************************************************************************************
*  Name:		set_icu_admission
*  Description: sets a person to be added to an ICU
*  Returns:		void
******************************************************************************************/
void set_icu_admission( individual *indiv, parameters* params, int time )
{
	indiv->hospital_state = ICU;
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
}

/*****************************************************************************************
*  Name:		destroy_individual
*  Description: Destroys the individual structure and releases its memory
******************************************************************************************/
void destroy_individual( individual *indiv )
{
	infection_event *infection_event, *temporary_infection_event;

	infection_event = indiv->infection_events;

	while( infection_event != NULL )
	{
		temporary_infection_event = infection_event;
		free( infection_event->times );
		infection_event = infection_event->next;
		free( temporary_infection_event );
	}
	free( indiv->n_interactions );
	free( indiv->interactions );
}

/*****************************************************************************************
*  Name:		count_infection_events
*  Description: Count number of times an individual has been infected
******************************************************************************************/

int count_infection_events( individual *indiv )
{
	int infection_count = 0;
    infection_event *infection_event;
	infection_event = indiv->infection_events;

	while(infection_event != NULL){
		if( time_infected_infection_event(infection_event) != UNKNOWN )
			infection_count++;
	infection_event = infection_event->next;
	}

	return infection_count;
}
/*****************************************************************************************
*  Name:		print_individual
******************************************************************************************/

void print_individual( model *model, long idx)
{
        individual *indiv;
        if( idx >= model->params->n_total )
        {
            printf("idx higher than n_total; individual does not exist");
            fflush_stdout();
            return;
        }
        
        indiv = &(model->population[idx]);
        
	printf("indiv->idx: %li\n", indiv->idx );
	printf("indiv->status: %d\n", indiv->status );
	printf("indiv->house_no: %li\n", indiv->house_no );
	printf("indiv->age_group: %d\n", indiv->age_group );
	printf("indiv->age_type: %d\n", indiv->age_type );
	printf("indiv->occupation_network: %d\n", indiv->occupation_network );

	printf("indiv->base_random_interactions: %d\n", indiv->base_random_interactions );
	printf("indiv->random_interactions: %d\n", indiv->random_interactions );

	printf("indiv->hazard: %f\n", indiv->hazard );
	printf("indiv->quarantined: %d\n", indiv->quarantined );
	printf("indiv->quarantine_test_result: %d\n", indiv->quarantine_test_result );
	
	printf("indiv->traced_on_this_trace: %f\n", indiv->traced_on_this_trace );
	printf("indiv->app_user: %d\n", indiv->app_user );
        if(indiv->trace_tokens == NULL){
            printf("indiv->trace_tokens: NULL\n");
        }else{
            printf("indiv->trace_tokens: non-NULL\n");
        }
        if(indiv->index_trace_token == NULL){ 
            printf("indiv->index_trace_token: NULL\n");
        }else{
            printf("indiv->index_trace_token: non-NULL\n");
        }
	fflush_stdout();
}
