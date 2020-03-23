/*
 * demographics.c
 *
 *  Created on: 23 Mar 2020
 *      Author: hinchr
 */


#include "model.h"
#include "individual.h"
#include "utilities.h"
#include "constant.h"
#include "params.h"
#include "network.h"
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

/*****************************************************************************************
*  Name:		set_up_allocate_work_places
*  Description: sets up to allocate a work place for each person
*  Returns:		void
******************************************************************************************/
void set_up_allocate_work_places( model *model )
{
	int adx, ndx;
	long pdx, n_adult;
	long pop_net_raw[N_WORK_NETWORKS];
	double other;
	double **prob = calloc( N_AGE_GROUPS, sizeof(double*));
	double adult_prop[N_WORK_NETWORK_TYPES] = {
		model->params->child_network_adults,
		1.0,
		model->params->elderly_network_adults
	};

	// get the raw population in each network
	for( ndx = 0; ndx < N_WORK_NETWORKS; ndx++ )
		pop_net_raw[ndx] = 0;
	for( pdx = 0; pdx < model->params->n_total; pdx++ )
		pop_net_raw[ AGE_WORK_MAP[model->population[pdx].age_group] ]++;

	// given the total adults
	n_adult = 0;
	for( ndx = 0; ndx < N_WORK_NETWORKS; ndx++ )
		if( NETWORK_TYPE_MAP[ndx] == NETWORK_TYPE_ADULT )
			n_adult += pop_net_raw[ndx];

	// get the probability of each each age-group going to each network	for
	for( adx = 0; adx < N_AGE_GROUPS; adx++ )
	{
		other = 0.0;
		prob[adx] = calloc( N_WORK_NETWORKS, sizeof(double));
		for( ndx = 0; ndx < N_WORK_NETWORKS; ndx++ )
		{
			prob[adx][ndx] = 0;
			if( NETWORK_TYPE_MAP[AGE_WORK_MAP[adx]] != NETWORK_TYPE_ADULT )
				prob[adx][ndx] = ( ndx == AGE_WORK_MAP[adx] );
			else
			{
				if( NETWORK_TYPE_MAP[ndx]!= NETWORK_TYPE_ADULT )
				{
					prob[adx][ndx] = 1.0 * pop_net_raw[ndx] * adult_prop[NETWORK_TYPE_MAP[ndx]] / n_adult;
					other         += prob[adx][ndx];
				}
			}
		}
		if( NETWORK_TYPE_MAP[AGE_WORK_MAP[adx]] == NETWORK_TYPE_ADULT )
			prob[adx][AGE_WORK_MAP[adx]] = 1.0 - other;
	}

	// randomly assign a work place networks using the probability map
	for( pdx = 0; pdx < model->params->n_total; pdx++ )
		model->population[pdx].work_network_new = discrete_draw( N_WORK_NETWORKS, prob[model->population[pdx].age_group]);

	for( ndx = 0; ndx < N_AGE_GROUPS; ndx++ )
		free(prob[ndx]);

	free(prob);
}

/*****************************************************************************************
*  Name:		calculate_household_distribution
*  Description: Calculates the number of households of each size from the UK
*  				household survey data. The age split of the households is based
*  				upon ONS data for each age group. Then:
*
*  				 1. All elderly are assumed to live in 1 or 2 person household.
*  					We fill them up in equal proportion
*  				 2. The same proportion of household with 3/4/5/6 people in have
*  				    children and then n-2 of the occupents are assumed to be children.
*
*  				 Note there is no age mixing in household between elderly and others.
*
* Argument:		model   		 - pointer to the model
*  				n_house_tot 	 - number of households which this function sets
*  				elderly_frac_1_2 - fraction of 1/2 person households which are elderly
*				child_frac_2_6,	 - fraction of 3/4/5/6 person households which have chlildren
******************************************************************************************/
void calculate_household_distribution(
	model *model,
	long *n_house_tot,
	double *elderly_frac_1_2,
	double *child_frac_3_6
)
{
	long total;
	int idx;
	double survey_tot, max_children, pop_all;
	double n_person_frac[ HOUSEHOLD_N_MAX ];
	double *pop = model->params->population_type;

	pop_all      = pop[AGE_TYPE_CHILD] + pop[AGE_TYPE_ADULT] + pop[AGE_TYPE_ELDERLY];
	survey_tot   = 0;
	max_children = 0;
	for( idx = 0; idx < HOUSEHOLD_N_MAX; idx++)
	{
		n_person_frac[idx] = model->params->household_size[ idx ] * ( idx + 1 );
		survey_tot        += n_person_frac[idx];
		max_children      += model->params->household_size[ idx ] * max( idx -1 , 0 );
	}

	*child_frac_3_6   = pop[AGE_TYPE_CHILD] / pop_all / ( max_children / survey_tot );
	*elderly_frac_1_2 = pop[AGE_TYPE_ELDERLY] / pop_all / ( ( n_person_frac[HH_1] + n_person_frac[HH_2] ) / survey_tot );

	if( *child_frac_3_6 > 1 )
		print_exit( "not sufficient 3-6 person households for all the children" );

	if( *elderly_frac_1_2 > 1 )
		print_exit( "not sufficient 1-2 person households for all the elderly" );

 	total = 0;
	for( idx = 1; idx < HOUSEHOLD_N_MAX; idx++)
	{
		n_house_tot[idx] = (long) round( n_person_frac[idx] / survey_tot / ( idx + 1 ) * model->params->n_total );
		total += n_house_tot[idx] * ( idx + 1 );
	}
	n_house_tot[0] = model->params->n_total - total;
}

/*****************************************************************************************
*  Name:		build_household_network_from_directory
*  Description: Builds a network of household i
******************************************************************************************/
void build_household_network_from_directroy( model *model )
{
	long hdx, edge_idx, h_size;
	int pdx, p2dx;
	network *network     = model->household_network;
	var_array *directory = model->household_directory;

	if( network->n_edges != 0 )
		print_exit( "the household network can only be once" );

	network->n_edges = 0;
	for( hdx = 0; hdx < directory->n_idx; hdx++ )
	{
		h_size   = directory->n_jdx[hdx];
		network->n_edges += h_size  * ( h_size - 1 ) / 2;
	}
	network->edges = calloc( network->n_edges, sizeof( edge ) );

	edge_idx = 0;
	for( hdx = 0; hdx < directory->n_idx; hdx++ )
	{
		h_size   = directory->n_jdx[hdx];
		for( pdx = 0; pdx < h_size; pdx++ )
			for( p2dx = pdx + 1; p2dx < h_size; p2dx++ )
			{
				network->edges[edge_idx].id1 = directory->val[hdx][pdx];
				network->edges[edge_idx].id2 = directory->val[hdx][p2dx];
				edge_idx++;
			}
	}
}



/*****************************************************************************************
*  Name:		build_household_network
*  Description: Builds a network of household interactions based upon the UK household
*  				data. Note this can only be done once since it allocates new memory
*  				to the network structure.
*
*  				As part of the household allocation process we also assign people
*  				to age groups. All elderly are assumed to live in 1-2 person
*  				households and all children are assumed to live in a house with
*  				2 adults.
******************************************************************************************/
void build_household_network( model *model )
{
	long pop_idx, hdx, house_no;
	int ndx, pdx, age;
	long n_house_tot[ HOUSEHOLD_N_MAX ];
	double elderly_frac_1_2;
	double child_frac_3_6;

	calculate_household_distribution( model, n_house_tot, &elderly_frac_1_2, &child_frac_3_6 );

	model->household_directory = calloc( 1, sizeof( var_array ) );
	model->household_directory->n_idx = 0;
	for( ndx = 0; ndx < HOUSEHOLD_N_MAX; ndx++ )
		model->household_directory->n_idx += n_house_tot[ndx];
	model->household_directory->n_jdx = calloc( model->household_directory->n_idx, sizeof( int ) );
	model->household_directory->val   = calloc( model->household_directory->n_idx, sizeof( long* ) );

	pop_idx  = 0;
	house_no = 0;
	for( ndx = 0; ndx < HOUSEHOLD_N_MAX; ndx++ )
		for( hdx = 0; hdx < n_house_tot[ndx]; hdx++ )
		{
			model->household_directory->n_jdx[house_no] = ndx + 1;
			model->household_directory->val[house_no]   = calloc( ndx + 1, sizeof( long ) );
			for( pdx = 0; pdx < ( ndx + 1 ); pdx++ )
			{
				age = AGE_TYPE_ADULT;
				if( ndx <= HH_2 && ( 1.0 * hdx / n_house_tot[ndx] ) < elderly_frac_1_2 )
					age = AGE_TYPE_ELDERLY;
				if( ndx >= HH_3 && ( 1.0 * hdx / n_house_tot[ndx] ) < child_frac_3_6 && pdx < ( ndx - 1 ) )
					age = AGE_TYPE_CHILD;

				set_age_type( &(model->population[pop_idx]), model->params, age );
				set_house_no( &(model->population[pop_idx]), house_no );
				model->household_directory->val[house_no][pdx] = pop_idx;
				pop_idx++;
			}
			house_no++;
		}

	build_household_network_from_directroy( model );
}

