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
#include "demographics.h"

#define REJECTION_MULT 1.0001
#define ACCEPTANCE_MULT 0.99
#define INTIAL_ACCEPTANCE_FACTOR 0.00001
#define SAMPLE_BATCH 4
#define POPULATION_PREF 2
#define MAX_ALLOWABLE_ERROR 1e-5

/*****************************************************************************************
*  Name:		assign_household_distribution
*  Description: Given a table of individuals with ages and house numbers, assigns
*  				them to the objects in the model and generates the household lookup
*  				directroy.
*  Returns:		void
******************************************************************************************/
void assign_household_distribution( model *model, demographic_household_table *demo_house )
{
	long hdx, pdx;
	int housesize, age, idx;
	directory *dir;
	individual *indiv;

	// check to see that demo_house has the correct number of people
	if( demo_house->n_total != model->params->n_total )
	{
		printf( "demo_house n_tota = %li", demo_house->n_total);
		print_exit( "The demographic-household table has a different n_total to the parameters");
	}

	// now allocate people to households and set up the household directory
	dir = calloc( 1, sizeof( directory ) );
	dir->n_idx = demo_house->n_households;
	dir->n_jdx = calloc( demo_house->n_households, sizeof( int ) );
	dir->val   = calloc( demo_house->n_households, sizeof( long* ) );
	model->household_directory = dir;

	pdx = 0;
	hdx = 0;
	while( hdx < demo_house->n_households )
	{
		if( demo_house->idx[ pdx ] != pdx )
			print_exit( "Demographic_household idx column must be 0 to n_total-1 in order" );

		if( demo_house->house_no[ pdx ] != hdx )
		{
			printf( "pdx = %li; hdx = %li; demo_house_no = %li\n", pdx, hdx, demo_house->house_no[ pdx ]);
			print_exit( "Demographic_household house_no column must contains ordered values 0 to n_household-1 (repeated for some household)" );
		}

		housesize = 0;
		while( demo_house->house_no[ pdx ] == hdx )
		{
			// check the house column is correct
			age = demo_house->age_group[ pdx ];
			if( (age < 0) | (age >= N_AGE_GROUPS) )
				print_exit( "Demographic_household age column values must be between 0 and N_AGE_GROUPS" );

			// set up the individual
			indiv = &(model->population[pdx]);
			set_age_group( indiv, model->params, age );
			set_house_no( indiv, hdx );
			model->n_population_by_age[age]++;

			housesize++;
			pdx++;
			if( pdx == demo_house->n_total )
				break;
		};

		// update the household directory
		dir->n_jdx[hdx] = housesize;
		dir->val[hdx] = calloc( housesize, sizeof( long ) );
		for( idx = 0; idx < housesize; idx++ )
			dir->val[hdx][idx] = pdx - housesize + idx;

		// check the final values are correct
		hdx++;
		if( (hdx == demo_house->n_households) & (pdx != demo_house->n_total) )
			print_exit( "The person with index n_total-1 is not in house with index n_household-1!" );

		if( (hdx != demo_house->n_households) & (pdx == demo_house->n_total) )
			print_exit( "The person with index n_total-1 is not in house with index n_household-1!" );
	}

}

/*****************************************************************************************
*  Name:		set_up_household_distribution
*  Description: Either generates the household and age of everybody in the model OR
*  				use a pre-calculated table.
*  				Assigns the households to the individuals.
*
*  Returns:		void
******************************************************************************************/
void set_up_household_distribution( model *model )
{
	if( model->params->demo_house == NULL )
		generate_household_distribution( model );

	// now set to the individuals and create the household directory
	assign_household_distribution( model, model->params->demo_house );
}

/*****************************************************************************************
*  Name:		set_up_allocate_work_places
*  Description: sets up to allocate a work place for each person
*  Returns:		void
******************************************************************************************/
void set_up_allocate_work_places( model *model )
{
    if ( model->use_custom_occupation_networks == 0 )
        set_up_allocate_default_work_places( model );
    else
        set_up_allocate_custom_work_places( model );

}

/*****************************************************************************************
*  Name:		set_up_allocate_custom_work_places
*  Description: Sets allocations of work place to individual from ths user specifided
*  				custom network table
*  Returns:		void
******************************************************************************************/
void set_up_allocate_custom_work_places( model *model )
{
    long pdx;
    for( pdx = 0; pdx < model->params->n_total; pdx++ )
        model->population[pdx].occupation_network = model->params->occupation_network_table->network_no[pdx];
}

/*****************************************************************************************
*  Name:		set_up_allocate_default_work_places
*  Description: Allocates people to occupation networks
*  Returns:		void
******************************************************************************************/
void set_up_allocate_default_work_places( model *model )
{
	int adx, ndx;
	long pdx, n_adult;
	long pop_net_raw[N_DEFAULT_OCCUPATION_NETWORKS];
	double other;
	double **prob = calloc( N_AGE_GROUPS, sizeof(double*));
	double adult_prop[N_OCCUPATION_NETWORK_TYPES] = {
		model->params->child_network_adults,
		1.0,
		model->params->elderly_network_adults
	};

	// get the raw population in each network
	for( ndx = 0; ndx < N_DEFAULT_OCCUPATION_NETWORKS; ndx++ )
		pop_net_raw[ndx] = 0;
	for( pdx = 0; pdx < model->params->n_total; pdx++ )
		pop_net_raw[ AGE_OCCUPATION_MAP[model->population[pdx].age_group] ]++;

	// given the total adults
	n_adult = 0;
	for( ndx = 0; ndx < N_DEFAULT_OCCUPATION_NETWORKS; ndx++ )
		if( NETWORK_TYPE_MAP[ndx] == NETWORK_TYPE_ADULT )
			n_adult += pop_net_raw[ndx];

	// get the probability of each each age-group going to each network	for
	for( adx = 0; adx < N_AGE_GROUPS; adx++ )
	{
		other = 0.0;
		prob[adx] = calloc( N_DEFAULT_OCCUPATION_NETWORKS, sizeof(double));
		for( ndx = 0; ndx < N_DEFAULT_OCCUPATION_NETWORKS; ndx++ )
		{
			prob[adx][ndx] = 0;
			if( NETWORK_TYPE_MAP[AGE_OCCUPATION_MAP[adx]] != NETWORK_TYPE_ADULT )
				prob[adx][ndx] = ( ndx == AGE_OCCUPATION_MAP[adx] );
			else
			{
				if( NETWORK_TYPE_MAP[ndx]!= NETWORK_TYPE_ADULT )
				{
					prob[adx][ndx] = 1.0 * pop_net_raw[ndx] * adult_prop[NETWORK_TYPE_MAP[ndx]] / n_adult;
					other         += prob[adx][ndx];
				}
			}
		}
		if( NETWORK_TYPE_MAP[AGE_OCCUPATION_MAP[adx]] == NETWORK_TYPE_ADULT )
			prob[adx][AGE_OCCUPATION_MAP[adx]] = 1.0 - other;
	}

	// randomly assign a work place networks using the probability map
	for( pdx = 0; pdx < model->params->n_total; pdx++ )
		model->population[pdx].occupation_network = discrete_draw( N_DEFAULT_OCCUPATION_NETWORKS, prob[model->population[pdx].age_group]);

	for( ndx = 0; ndx < N_AGE_GROUPS; ndx++ )
		free(prob[ndx]);

	free(prob);
}

/*****************************************************************************************
*  Name:		add_reference_household
*  Description: given a set of population totals by age_group
*  				add a single addition reference household
*  Returns:		void
******************************************************************************************/
void add_reference_household( double *array, long hdx, int **REFERENCE_HOUSEHOLDS)
{
	int idx;
	for( idx = 0; idx < N_AGE_GROUPS; idx++ )
		array[idx] += (double) REFERENCE_HOUSEHOLDS[hdx][idx];
}

/*****************************************************************************************
*  Name:		generate_household_distribution
*  Description: sets up the initial household distribution and allocates people to them
*  				method matches both population structure and household structure using
*  				a rejection sampling method.
*  				1. Have reference panels of sample households
*  				2. Sample from panel (in batches) and accept if the differences
*  				  between the desired population structure and household structure
*  				  is below a threshold
*  			   3. Alter the acceptance threshold dynamically to get better and better fits
*
*  Returns:		void
******************************************************************************************/
void generate_household_distribution( model *model )
{
	int idx, housesize, age;
	long hdx, n_households, pdx, sample;
	double error, last_error, acceptance;
	demographic_household_table *demo_house = calloc( 1, sizeof( demographic_household_table ) );
	double *population_target      = calloc( N_AGE_GROUPS, sizeof(double));
	double *population_total       = calloc( N_AGE_GROUPS, sizeof(double));
	double *population_trial       = calloc( N_AGE_GROUPS, sizeof(double));
	double *household_target       = calloc( N_HOUSEHOLD_MAX, sizeof(double));
	double *household_total        = calloc( N_HOUSEHOLD_MAX, sizeof(double));
	double *household_trial        = calloc( N_HOUSEHOLD_MAX, sizeof(double));
	long *trial_samples		       = calloc( SAMPLE_BATCH, sizeof(long));
	int *REFERENCE_HOUSEHOLD_SIZE  = calloc( model->params->N_REFERENCE_HOUSEHOLDS, sizeof(int));
	long *households               = calloc( model->params->n_total, sizeof(long));

	// assign targets
	copy_normalize_array( population_target, model->params->population, N_AGE_GROUPS );
	copy_normalize_array( household_target, model->params->household_size, N_HOUSEHOLD_MAX );

	// get number of people in household for each group
	for( hdx = 0; hdx < model->params->N_REFERENCE_HOUSEHOLDS; hdx++ )
	{
		REFERENCE_HOUSEHOLD_SIZE[hdx] = -1;
		for( idx = 0; idx < N_AGE_GROUPS; idx++ )
			REFERENCE_HOUSEHOLD_SIZE[hdx] += model->params->REFERENCE_HOUSEHOLDS[hdx][idx];
	}

	// always accept the first sample
	pdx = 0;
	n_households = 0;
	for( idx = 0; idx < SAMPLE_BATCH; idx++ )
	{
		sample       = gsl_rng_uniform_int( rng, model->params->N_REFERENCE_HOUSEHOLDS );
		households[n_households++] = sample;
		add_reference_household( population_total, sample, model->params->REFERENCE_HOUSEHOLDS);
		household_total[ REFERENCE_HOUSEHOLD_SIZE[sample]]++;
		pdx += ( REFERENCE_HOUSEHOLD_SIZE[sample] + 1 );
	}

	// calculate the intital error
	copy_normalize_array( population_trial, population_total, N_AGE_GROUPS    );
	copy_normalize_array( household_trial,  household_total,  N_HOUSEHOLD_MAX );
	last_error  = sum_square_diff_array( population_trial, population_target, N_AGE_GROUPS ) * POPULATION_PREF;
	last_error += sum_square_diff_array( household_trial,  household_target,  N_HOUSEHOLD_MAX );
	acceptance  = last_error * INTIAL_ACCEPTANCE_FACTOR;

	while( pdx < model->params->n_total )
	{
		copy_array( population_trial, population_total, N_AGE_GROUPS    );
		copy_array( household_trial,  household_total,  N_HOUSEHOLD_MAX );

		for( idx = 0; idx < SAMPLE_BATCH; idx++ )
		{
			trial_samples[idx] = gsl_rng_uniform_int( rng, model->params->N_REFERENCE_HOUSEHOLDS );
			add_reference_household( population_trial, trial_samples[idx], model->params->REFERENCE_HOUSEHOLDS );
			household_trial[REFERENCE_HOUSEHOLD_SIZE[ trial_samples[idx]] ]++;
		}

		// calculate the error of the total with the proposed sample
		normalize_array( population_trial, N_AGE_GROUPS    );
		normalize_array( household_trial,  N_HOUSEHOLD_MAX );
		error  = sum_square_diff_array( population_trial, population_target, N_AGE_GROUPS ) * POPULATION_PREF;
		error += sum_square_diff_array( household_trial,  household_target,  N_HOUSEHOLD_MAX );

		// accept better than previous or within the acceptance threshold, then reduce the acceptance threshold
		// reject if error is worse than the acceptance threshold, then increase the the acceptance threshold
		if( error < last_error + acceptance )
		{
			for( idx = 0; idx < SAMPLE_BATCH; idx++ )
			{
				households[n_households++] = trial_samples[idx];
				add_reference_household( population_total, trial_samples[idx] , model->params->REFERENCE_HOUSEHOLDS);
				household_total[REFERENCE_HOUSEHOLD_SIZE[ trial_samples[idx]] ]++;
				pdx += REFERENCE_HOUSEHOLD_SIZE[trial_samples[idx]] + 1;
			}
			acceptance *= ACCEPTANCE_MULT;
			last_error  = min( error, last_error );
		}
		else
			acceptance *= REJECTION_MULT;
	}

	free( population_target );
	free( population_total );
	free( population_trial );
	free( household_target );
	free( household_total );
	free( household_trial );
	free( trial_samples );

	if( error > MAX_ALLOWABLE_ERROR )
		print_exit( "Household rejection sampling failed to accurately converge" );

	// now generate the demographic household table
	demo_house->idx          = calloc( model->params->n_total, sizeof( long ) );
	demo_house->age_group    = calloc( model->params->n_total, sizeof( int ) );
	demo_house->house_no     = calloc( model->params->n_total, sizeof( long ) );
	demo_house->n_total      = model->params->n_total;

	pdx = 0;
	for( hdx = 0; hdx < n_households; hdx++ )
	{
		housesize = 0;
		for( age = N_AGE_GROUPS - 1; age >= 0; age-- )
		{
			for( idx = 0; idx < model->params->REFERENCE_HOUSEHOLDS[households[hdx]][age]; idx++ )
			{
				demo_house->idx[pdx]       = pdx;
				demo_house->age_group[pdx] = age;
				demo_house->house_no[pdx]  = hdx;
				pdx++;

				if( pdx == model->params->n_total )
				{
					break;
				};
			}
			if( pdx == model->params->n_total )
				break;
		}
		if( pdx == model->params->n_total )
			break;

	}

	demo_house->n_households = hdx + 1;
	model->params->demo_house = demo_house;

	free( households );
	free( REFERENCE_HOUSEHOLD_SIZE );
}

/*****************************************************************************************
*  Name:		build_household_network_from_directory
*  Description: Builds a network of household i
******************************************************************************************/
void build_household_network_from_directroy( network *network, directory *directory )
{
	long hdx, edge_idx, h_size;
	int pdx, p2dx;

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

