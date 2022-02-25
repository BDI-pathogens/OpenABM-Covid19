/*
 * strain.c
 *
 *  Created on: 31 Mar 2021
 *      Author: nikbaya
 */

#include "strain.h"
#include "model.h"
#include "utilities.h"

/*****************************************************************************************
*  Name:		initialise_infectious_curves
*  Returns:		void
******************************************************************************************/
void initialise_infectious_curves(
	strain *strain_ptr,
	parameters *params
)
{
	double infectious_rate = params->infectious_rate_adjusted;
	strain_ptr->infectious_curve = calloc( N_EVENT_TYPES, sizeof(double) );

	strain_ptr->infectious_curve[PRESYMPTOMATIC]          = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	strain_ptr->infectious_curve[PRESYMPTOMATIC_MILD]     = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	strain_ptr->infectious_curve[ASYMPTOMATIC]            = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	strain_ptr->infectious_curve[SYMPTOMATIC]             = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	strain_ptr->infectious_curve[SYMPTOMATIC_MILD]        = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	strain_ptr->infectious_curve[HOSPITALISED]            = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	strain_ptr->infectious_curve[HOSPITALISED_RECOVERING] = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	strain_ptr->infectious_curve[CRITICAL]                = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );

gamma_rate_curve( strain_ptr->infectious_curve[PRESYMPTOMATIC], MAX_INFECTIOUS_PERIOD, strain_ptr->mean_infectious_period,
					  strain_ptr->sd_infectious_period, infectious_rate );

	gamma_rate_curve( strain_ptr->infectious_curve[PRESYMPTOMATIC_MILD], MAX_INFECTIOUS_PERIOD, strain_ptr->mean_infectious_period,
					  strain_ptr->sd_infectious_period, infectious_rate * params->mild_infectious_factor  );

	gamma_rate_curve( strain_ptr->infectious_curve[ASYMPTOMATIC] , MAX_INFECTIOUS_PERIOD, strain_ptr->mean_infectious_period,
			          strain_ptr->sd_infectious_period, infectious_rate * params->asymptomatic_infectious_factor);

	gamma_rate_curve( strain_ptr->infectious_curve[SYMPTOMATIC], MAX_INFECTIOUS_PERIOD, strain_ptr->mean_infectious_period,
			          strain_ptr->sd_infectious_period, infectious_rate );

	gamma_rate_curve( strain_ptr->infectious_curve[SYMPTOMATIC_MILD], MAX_INFECTIOUS_PERIOD, strain_ptr->mean_infectious_period,
			          strain_ptr->sd_infectious_period, infectious_rate * params->mild_infectious_factor );

	gamma_rate_curve( strain_ptr->infectious_curve[HOSPITALISED], MAX_INFECTIOUS_PERIOD, strain_ptr->mean_infectious_period,
			          strain_ptr->sd_infectious_period, infectious_rate );

	gamma_rate_curve( strain_ptr->infectious_curve[HOSPITALISED_RECOVERING], MAX_INFECTIOUS_PERIOD, strain_ptr->mean_infectious_period,
			          strain_ptr->sd_infectious_period, infectious_rate );

	gamma_rate_curve( strain_ptr->infectious_curve[CRITICAL], MAX_INFECTIOUS_PERIOD, strain_ptr->mean_infectious_period,
			          strain_ptr->sd_infectious_period, infectious_rate  );
}

/*****************************************************************************************
*  Name:		initialise_transition_time_distributions
*  Returns:		void
******************************************************************************************/
void initialise_transition_time_distributions(
	strain *strain_ptr,
	parameters *params
)
{
	int **transitions;

	strain_ptr->transition_time_distributions = calloc( N_TRANSITION_TYPES, sizeof( int*) );
	transitions = strain_ptr->transition_time_distributions;

	transitions[ASYMPTOMATIC_RECOVERED]               = calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[PRESYMPTOMATIC_SYMPTOMATIC]           = calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[PRESYMPTOMATIC_MILD_SYMPTOMATIC_MILD] = calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[SYMPTOMATIC_RECOVERED]                = calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[SYMPTOMATIC_MILD_RECOVERED]           = calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[HOSPITALISED_RECOVERED]               = calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[CRITICAL_HOSPITALISED_RECOVERING]     = calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[CRITICAL_DEATH]                       = calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[HOSPITALISED_RECOVERING_RECOVERED]    = calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[SYMPTOMATIC_HOSPITALISED]             = calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[HOSPITALISED_CRITICAL]                = calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[RECOVERED_SUSCEPTIBLE]                = calloc( N_DRAW_LIST, sizeof( int ) );

	gamma_draw_list( transitions[ASYMPTOMATIC_RECOVERED], 	   N_DRAW_LIST, params->mean_asymptomatic_to_recovery, params->sd_asymptomatic_to_recovery );
	gamma_draw_list( transitions[PRESYMPTOMATIC_SYMPTOMATIC],  N_DRAW_LIST, params->mean_time_to_symptoms,         params->sd_time_to_symptoms );
	gamma_draw_list( transitions[PRESYMPTOMATIC_MILD_SYMPTOMATIC_MILD], N_DRAW_LIST, params->mean_time_to_symptoms,params->sd_time_to_symptoms );
	gamma_draw_list( transitions[SYMPTOMATIC_RECOVERED],   	   N_DRAW_LIST, params->mean_time_to_recover,  		   params->sd_time_to_recover );
	gamma_draw_list( transitions[SYMPTOMATIC_MILD_RECOVERED],  N_DRAW_LIST, params->mean_time_to_recover,  		   params->sd_time_to_recover );
	gamma_draw_list( transitions[HOSPITALISED_RECOVERED],      N_DRAW_LIST, params->mean_time_hospitalised_recovery, params->sd_time_hospitalised_recovery);
	gamma_draw_list( transitions[CRITICAL_HOSPITALISED_RECOVERING], N_DRAW_LIST, params->mean_time_critical_survive, params->sd_time_critical_survive);
	gamma_draw_list( transitions[CRITICAL_DEATH],              N_DRAW_LIST, params->mean_time_to_death,    		     params->sd_time_to_death );
	gamma_draw_list( transitions[HOSPITALISED_RECOVERING_RECOVERED], N_DRAW_LIST, params->mean_time_hospitalised_recovery, params->sd_time_hospitalised_recovery);
	bernoulli_draw_list( transitions[SYMPTOMATIC_HOSPITALISED],N_DRAW_LIST, params->mean_time_to_hospital );
	gamma_draw_list( transitions[HOSPITALISED_CRITICAL],       N_DRAW_LIST, params->mean_time_to_critical, params->sd_time_to_critical );
	shifted_geometric_draw_list( transitions[RECOVERED_SUSCEPTIBLE], N_DRAW_LIST, params->mean_time_to_susceptible_after_shift, params->time_to_susceptible_shift );
}

/*****************************************************************************************
*  Name:		add_new_strain
*  Description: Initialises new strain, can only be called once for each strain
*  Arguments:	model  - pointer to model structurture
*  				transmission_multiplier - strain specific transmission multiplier
*  				hospitalised_fraction   - strain specific hospitalised fraction
*  				mean_infectious_period  - strain specific mean_infectious_period (UNKOWN then use default)
*  				sd_infectious_period    - strain specific sd_infectious_period (UNKOWN then use default)
*
*  Returns:		void
******************************************************************************************/
short add_new_strain(
	model *model,
	float transmission_multiplier,
	double *hospitalised_fraction,
	double mean_infectious_period,
	double sd_infectious_period
)
{
	strain *strain_ptr;
	parameters *params = model->params;

	if( model->n_initialised_strains == model->params->max_n_strains )
		return ERROR;

	// if parameters are unspecified then use default values
	if( mean_infectious_period == UNKNOWN )
		mean_infectious_period = params->mean_infectious_period;
	if( sd_infectious_period == UNKNOWN )
		sd_infectious_period = params->sd_infectious_period;

	strain_ptr = &(model->strains[ model->n_initialised_strains ]);
	strain_ptr->idx 					= model->n_initialised_strains;
	strain_ptr->transmission_multiplier = transmission_multiplier;
	strain_ptr->mean_infectious_period  = mean_infectious_period;
	strain_ptr->sd_infectious_period    = sd_infectious_period;
	strain_ptr->total_infected = 0;

	for( int idx = 0; idx < N_AGE_GROUPS; idx++ )
		strain_ptr->hospitalised_fraction[ idx ] = hospitalised_fraction[ idx ];

	initialise_infectious_curves( strain_ptr, params );
	initialise_transition_time_distributions( strain_ptr, params );

	model->n_initialised_strains++;
	return(  model->n_initialised_strains - 1 );
}

/*****************************************************************************************
*  Name:		get_strain_by_id
*  Description: returns a pointer to a strain a given ID
*  Returns:		pointer to vaccine
******************************************************************************************/
strain* get_strain_by_id( model *model, short strain_idx )
{
	if( strain_idx >=  model->n_initialised_strains )
		print_exit( "strain not yet intialised " );

	return &(model->strains[ strain_idx ]);
}

/*****************************************************************************************
*  Name:		destroy_strain
*  Description: Destroys an event list
******************************************************************************************/
void destroy_strain( strain *strain )
{
	free( strain->infectious_curve[PRESYMPTOMATIC] );
	free( strain->infectious_curve[PRESYMPTOMATIC_MILD] );
	free( strain->infectious_curve[ASYMPTOMATIC] );
	free( strain->infectious_curve[SYMPTOMATIC] );
	free( strain->infectious_curve[SYMPTOMATIC_MILD] );
	free( strain->infectious_curve[HOSPITALISED] );
	free( strain->infectious_curve[HOSPITALISED_RECOVERING] );
	free( strain->infectious_curve[CRITICAL] );
	free( strain->infectious_curve );

	free( strain->transition_time_distributions[ASYMPTOMATIC_RECOVERED] );
	free( strain->transition_time_distributions[PRESYMPTOMATIC_SYMPTOMATIC] );
	free( strain->transition_time_distributions[PRESYMPTOMATIC_MILD_SYMPTOMATIC_MILD] );
	free( strain->transition_time_distributions[SYMPTOMATIC_RECOVERED] );
	free( strain->transition_time_distributions[SYMPTOMATIC_MILD_RECOVERED] );
	free( strain->transition_time_distributions[HOSPITALISED_RECOVERED] );
	free( strain->transition_time_distributions[CRITICAL_HOSPITALISED_RECOVERING] );
	free( strain->transition_time_distributions[CRITICAL_DEATH] );
	free( strain->transition_time_distributions[HOSPITALISED_RECOVERING_RECOVERED] );
	free( strain->transition_time_distributions[SYMPTOMATIC_HOSPITALISED] );
	free( strain->transition_time_distributions[HOSPITALISED_CRITICAL] );
	free( strain->transition_time_distributions[RECOVERED_SUSCEPTIBLE] );
	free( strain->transition_time_distributions );
}
