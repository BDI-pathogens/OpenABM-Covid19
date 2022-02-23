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
*  Name:		add_new_strain
*  Description: Initialises new strain, can only be called once for each strain
*  Arguments:	model  - pointer to model structurture
*  				transmission_multiplier - strain specific transmission multiplier
*  				hospitalised_fraction   - strain specific hospitalised fraction
*  				mean_infectious_period  - strain specific mean_infectious_period (UNKOWN then use default)
*  Returns:		void
******************************************************************************************/
short add_new_strain(
	model *model,
	float transmission_multiplier,
	double *hospitalised_fraction,
	double mean_infectious_period
)
{
	double infectious_rate;
	strain *strain_ptr;
	parameters *params = model->params;

	if( model->n_initialised_strains == model->params->max_n_strains )
		return ERROR;

	// if parameters are unspecified then use default values
	if( mean_infectious_period == UNKNOWN )
		mean_infectious_period = params->mean_infectious_period;

	strain_ptr = &(model->strains[ model->n_initialised_strains ]);
	strain_ptr->idx 					= model->n_initialised_strains;
	strain_ptr->transmission_multiplier = transmission_multiplier;
	strain_ptr->total_infected = 0;

	for( int idx = 0; idx < N_AGE_GROUPS; idx++ )
		strain_ptr->hospitalised_fraction[ idx ] = hospitalised_fraction[ idx ];

	infectious_rate = params->infectious_rate_adjusted;
	strain_ptr->infectious_curve = calloc( N_EVENT_TYPES, sizeof(double) );

	strain_ptr->infectious_curve[PRESYMPTOMATIC] = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	gamma_rate_curve( strain_ptr->infectious_curve[PRESYMPTOMATIC], MAX_INFECTIOUS_PERIOD, mean_infectious_period,
					  params->sd_infectious_period, infectious_rate );

	strain_ptr->infectious_curve[PRESYMPTOMATIC_MILD] = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	gamma_rate_curve( strain_ptr->infectious_curve[PRESYMPTOMATIC_MILD], MAX_INFECTIOUS_PERIOD, mean_infectious_period,
					  params->sd_infectious_period, infectious_rate * params->mild_infectious_factor  );

	strain_ptr->infectious_curve[ASYMPTOMATIC] = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	gamma_rate_curve( strain_ptr->infectious_curve[ASYMPTOMATIC] , MAX_INFECTIOUS_PERIOD, mean_infectious_period,
					  params->sd_infectious_period, infectious_rate * params->asymptomatic_infectious_factor);

	strain_ptr->infectious_curve[SYMPTOMATIC] = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	gamma_rate_curve( strain_ptr->infectious_curve[SYMPTOMATIC], MAX_INFECTIOUS_PERIOD, mean_infectious_period,
					  params->sd_infectious_period, infectious_rate );

	strain_ptr->infectious_curve[SYMPTOMATIC_MILD] = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	gamma_rate_curve( strain_ptr->infectious_curve[SYMPTOMATIC_MILD], MAX_INFECTIOUS_PERIOD, mean_infectious_period,
					  params->sd_infectious_period, infectious_rate * params->mild_infectious_factor );

	strain_ptr->infectious_curve[HOSPITALISED] = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	gamma_rate_curve( strain_ptr->infectious_curve[HOSPITALISED], MAX_INFECTIOUS_PERIOD, mean_infectious_period,
					  params->sd_infectious_period, infectious_rate );

	strain_ptr->infectious_curve[HOSPITALISED_RECOVERING] = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	gamma_rate_curve( strain_ptr->infectious_curve[HOSPITALISED_RECOVERING], MAX_INFECTIOUS_PERIOD, mean_infectious_period,
						  params->sd_infectious_period, infectious_rate );

	strain_ptr->infectious_curve[CRITICAL] = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	gamma_rate_curve( strain_ptr->infectious_curve[CRITICAL], MAX_INFECTIOUS_PERIOD, mean_infectious_period,
					  params->sd_infectious_period, infectious_rate  );

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
}
