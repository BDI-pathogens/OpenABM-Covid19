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
	strain_ptr->infectious_curve = (double**) calloc( N_EVENT_TYPES, sizeof(double*) );

	strain_ptr->infectious_curve[PRESYMPTOMATIC]          = (double*) calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	strain_ptr->infectious_curve[PRESYMPTOMATIC_MILD]     = (double*) calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	strain_ptr->infectious_curve[ASYMPTOMATIC]            = (double*) calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	strain_ptr->infectious_curve[SYMPTOMATIC]             = (double*) calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	strain_ptr->infectious_curve[SYMPTOMATIC_MILD]        = (double*) calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	strain_ptr->infectious_curve[HOSPITALISED]            = (double*) calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	strain_ptr->infectious_curve[HOSPITALISED_RECOVERING] = (double*) calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	strain_ptr->infectious_curve[CRITICAL]                = (double*) calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );

gamma_rate_curve( strain_ptr->infectious_curve[PRESYMPTOMATIC], MAX_INFECTIOUS_PERIOD, strain_ptr->mean_infectious_period,
					  strain_ptr->sd_infectious_period, infectious_rate );

	gamma_rate_curve( strain_ptr->infectious_curve[PRESYMPTOMATIC_MILD], MAX_INFECTIOUS_PERIOD, strain_ptr->mean_infectious_period,
					  strain_ptr->sd_infectious_period, infectious_rate * strain_ptr->mild_infectious_factor  );

	gamma_rate_curve( strain_ptr->infectious_curve[ASYMPTOMATIC] , MAX_INFECTIOUS_PERIOD, strain_ptr->mean_infectious_period,
			          strain_ptr->sd_infectious_period, infectious_rate * strain_ptr->asymptomatic_infectious_factor);

	gamma_rate_curve( strain_ptr->infectious_curve[SYMPTOMATIC], MAX_INFECTIOUS_PERIOD, strain_ptr->mean_infectious_period,
			          strain_ptr->sd_infectious_period, infectious_rate );

	gamma_rate_curve( strain_ptr->infectious_curve[SYMPTOMATIC_MILD], MAX_INFECTIOUS_PERIOD, strain_ptr->mean_infectious_period,
			          strain_ptr->sd_infectious_period, infectious_rate * strain_ptr->mild_infectious_factor );

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
void initialise_transition_time_distributions( strain *strain_ptr )
{
	int **transitions;

	strain_ptr->transition_time_distributions = (int**) calloc( N_TRANSITION_TYPES, sizeof( int*) );
	transitions = strain_ptr->transition_time_distributions;

	transitions[ASYMPTOMATIC_RECOVERED]               = (int*) calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[PRESYMPTOMATIC_SYMPTOMATIC]           = (int*) calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[PRESYMPTOMATIC_MILD_SYMPTOMATIC_MILD] = (int*) calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[SYMPTOMATIC_RECOVERED]                = (int*) calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[SYMPTOMATIC_MILD_RECOVERED]           = (int*) calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[HOSPITALISED_RECOVERED]               = (int*) calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[CRITICAL_HOSPITALISED_RECOVERING]     = (int*) calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[CRITICAL_DEATH]                       = (int*) calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[HOSPITALISED_RECOVERING_RECOVERED]    = (int*) calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[SYMPTOMATIC_HOSPITALISED]             = (int*) calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[HOSPITALISED_CRITICAL]                = (int*) calloc( N_DRAW_LIST, sizeof( int ) );
	transitions[RECOVERED_SUSCEPTIBLE]                = (int*) calloc( N_DRAW_LIST, sizeof( int ) );

	gamma_draw_list( transitions[ASYMPTOMATIC_RECOVERED], 	   N_DRAW_LIST, strain_ptr->mean_asymptomatic_to_recovery, strain_ptr->sd_asymptomatic_to_recovery );
	gamma_draw_list( transitions[PRESYMPTOMATIC_SYMPTOMATIC],  N_DRAW_LIST, strain_ptr->mean_time_to_symptoms,         strain_ptr->sd_time_to_symptoms );
	gamma_draw_list( transitions[PRESYMPTOMATIC_MILD_SYMPTOMATIC_MILD], N_DRAW_LIST, strain_ptr->mean_time_to_symptoms,strain_ptr->sd_time_to_symptoms );
	gamma_draw_list( transitions[SYMPTOMATIC_RECOVERED],   	   N_DRAW_LIST, strain_ptr->mean_time_to_recover,  		     strain_ptr->sd_time_to_recover );
	gamma_draw_list( transitions[SYMPTOMATIC_MILD_RECOVERED],  N_DRAW_LIST, strain_ptr->mean_time_to_recover,  		     strain_ptr->sd_time_to_recover );
	gamma_draw_list( transitions[HOSPITALISED_RECOVERED],      N_DRAW_LIST, strain_ptr->mean_time_hospitalised_recovery, strain_ptr->sd_time_hospitalised_recovery);
	gamma_draw_list( transitions[CRITICAL_HOSPITALISED_RECOVERING], N_DRAW_LIST, strain_ptr->mean_time_critical_survive, strain_ptr->sd_time_critical_survive);
	gamma_draw_list( transitions[CRITICAL_DEATH],              N_DRAW_LIST, strain_ptr->mean_time_to_death,    		     strain_ptr->sd_time_to_death );
	gamma_draw_list( transitions[HOSPITALISED_RECOVERING_RECOVERED], N_DRAW_LIST, strain_ptr->mean_time_hospitalised_recovery, strain_ptr->sd_time_hospitalised_recovery);
	bernoulli_draw_list( transitions[SYMPTOMATIC_HOSPITALISED],N_DRAW_LIST, strain_ptr->mean_time_to_hospital );
	gamma_draw_list( transitions[HOSPITALISED_CRITICAL],       N_DRAW_LIST, strain_ptr->mean_time_to_critical, strain_ptr->sd_time_to_critical );
	shifted_geometric_draw_list( transitions[RECOVERED_SUSCEPTIBLE], N_DRAW_LIST, strain_ptr->mean_time_to_susceptible_after_shift, strain_ptr->time_to_susceptible_shift );
}

/*****************************************************************************************
*  Name:		add_new_strain
*  Description: Initialises new strain, can only be called once for each strain
*  Arguments:	model  - pointer to model structurture
*  				transmission_multiplier - strain specific transmission multiplier
*  				hospitalised_fraction   - strain specific hospitalised fraction
*  				mean_infectious_period  - strain specific mean_infectious_period (UNKOWN then use default)
*  				sd_infectious_period    - strain specific sd_infectious_period (UNKOWN then use default)
*  				mean_time_to_symptoms   - strain specific mean_time_to_symptoms (UNKOWN then use default)
*
*  Returns:		void
******************************************************************************************/
short add_new_strain(
	model *model,
	float transmission_multiplier,
	double *fraction_asymptomatic,
	double *mild_fraction,
	double *hospitalised_fraction,
	double *critical_fraction,
	double *fatality_fraction,
	double *location_death_icu,
	double mean_infectious_period,
	double sd_infectious_period,
	double asymptomatic_infectious_factor,
	double mild_infectious_factor,
	double mean_time_to_symptoms,
	double sd_time_to_symptoms,
	double mean_asymptomatic_to_recovery,
	double sd_asymptomatic_to_recovery,
	double mean_time_to_recover,
	double sd_time_to_recover,
	double mean_time_hospitalised_recovery,
	double sd_time_hospitalised_recovery,
	double mean_time_critical_survive,
	double sd_time_critical_survive,
	double mean_time_to_death,
	double sd_time_to_death,
	double mean_time_to_hospital,
	double mean_time_to_critical,
	double sd_time_to_critical,
	double mean_time_to_susceptible_after_shift,
	double time_to_susceptible_shift
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
	if( mean_time_to_symptoms == UNKNOWN )
		mean_time_to_symptoms = params->mean_time_to_symptoms;
	if( sd_time_to_symptoms == UNKNOWN )
		sd_time_to_symptoms = params->sd_time_to_symptoms;
	if( asymptomatic_infectious_factor == UNKNOWN )
		asymptomatic_infectious_factor = params->asymptomatic_infectious_factor;
	if( mild_infectious_factor == UNKNOWN )
		mild_infectious_factor = params->mild_infectious_factor;
	if( mean_asymptomatic_to_recovery == UNKNOWN )
		mean_asymptomatic_to_recovery = params->mean_asymptomatic_to_recovery;
	if( sd_asymptomatic_to_recovery == UNKNOWN )
		sd_asymptomatic_to_recovery	= params->sd_asymptomatic_to_recovery;
	if( mean_time_to_recover == UNKNOWN )
		mean_time_to_recover = params->mean_time_to_recover;
	if( sd_time_to_recover == UNKNOWN )
		sd_time_to_recover	= params->sd_time_to_recover;
	if( mean_time_hospitalised_recovery == UNKNOWN )
		mean_time_hospitalised_recovery	= params->mean_time_hospitalised_recovery;
	if( sd_time_hospitalised_recovery == UNKNOWN )
		sd_time_hospitalised_recovery = params->sd_time_hospitalised_recovery;
	if( mean_time_critical_survive == UNKNOWN )
		mean_time_critical_survive = params->mean_time_critical_survive;
	if( sd_time_critical_survive == UNKNOWN )
		sd_time_critical_survive = params->sd_time_critical_survive;
	if( mean_time_to_death == UNKNOWN )
		mean_time_to_death = params->mean_time_to_death;
	if( sd_time_to_death == UNKNOWN )
		sd_time_to_death = params->sd_time_to_death;
	if( mean_time_to_hospital == UNKNOWN )
		mean_time_to_hospital = params->mean_time_to_hospital;
	if( mean_time_to_critical == UNKNOWN )
		mean_time_to_critical = params->mean_time_to_critical;
	if( sd_time_to_critical == UNKNOWN )
		sd_time_to_critical = params->sd_time_to_critical;
	if( mean_time_to_susceptible_after_shift == UNKNOWN )
		mean_time_to_susceptible_after_shift = params->mean_time_to_susceptible_after_shift;
	if( time_to_susceptible_shift== UNKNOWN )
		time_to_susceptible_shift = params->time_to_susceptible_shift;

	strain_ptr = &(model->strains[ model->n_initialised_strains ]);
	strain_ptr->idx 					= model->n_initialised_strains;
	strain_ptr->transmission_multiplier = transmission_multiplier;
	strain_ptr->mean_infectious_period  = mean_infectious_period;
	strain_ptr->sd_infectious_period    = sd_infectious_period;
	strain_ptr->asymptomatic_infectious_factor  = asymptomatic_infectious_factor;
	strain_ptr->mild_infectious_factor          = mild_infectious_factor;
	strain_ptr->mean_time_to_symptoms           = mean_time_to_symptoms;
	strain_ptr->sd_time_to_symptoms				= sd_time_to_symptoms;
	strain_ptr->mean_asymptomatic_to_recovery	= mean_asymptomatic_to_recovery;
	strain_ptr->sd_asymptomatic_to_recovery		= sd_asymptomatic_to_recovery;
	strain_ptr->mean_time_to_recover			= mean_time_to_recover;
	strain_ptr->sd_time_to_recover				= sd_time_to_recover;
	strain_ptr->mean_time_hospitalised_recovery = mean_time_hospitalised_recovery;
	strain_ptr->sd_time_hospitalised_recovery	= sd_time_hospitalised_recovery;
	strain_ptr->mean_time_critical_survive		= mean_time_critical_survive;
	strain_ptr->sd_time_critical_survive		= sd_time_critical_survive;
	strain_ptr->mean_time_to_death				= mean_time_to_death;
	strain_ptr->sd_time_to_death				= sd_time_to_death;
	strain_ptr->mean_time_to_hospital			= mean_time_to_hospital;
	strain_ptr->mean_time_to_critical			= mean_time_to_critical;
	strain_ptr->sd_time_to_critical				= sd_time_to_critical;
	strain_ptr->mean_time_to_susceptible_after_shift = mean_time_to_susceptible_after_shift;
	strain_ptr->time_to_susceptible_shift		= time_to_susceptible_shift;
	strain_ptr->total_infected = 0;

	for( int idx = 0; idx < N_AGE_GROUPS; idx++ ) {
		strain_ptr->hospitalised_fraction[ idx ] = hospitalised_fraction[ idx ];
		strain_ptr->fraction_asymptomatic[ idx ] = fraction_asymptomatic[ idx ];
		strain_ptr->mild_fraction[ idx ]         = mild_fraction[ idx ];
		strain_ptr->critical_fraction[ idx ]     = critical_fraction[ idx ];
		strain_ptr->fatality_fraction[ idx ]     = fatality_fraction[ idx ];
		strain_ptr->location_death_icu[ idx ]    = location_death_icu[ idx ];
 	}

	initialise_infectious_curves( strain_ptr, params );
	initialise_transition_time_distributions( strain_ptr );

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
