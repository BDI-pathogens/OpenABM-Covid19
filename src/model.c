/*
 * model.c
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#include "model.h"
#include "individual.h"
#include "utilities.h"
#include "constant.h"
#include "params.h"
#include "network.h"
#include "disease.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

/*****************************************************************************************
*  Name:		new_model
*  Description: Builds a new model object from a parameters object and returns a
*  				pointer to it.
*  				 1. Creates memory for it
*  				 2. Initialises the gsl random numbers generator
*  Returns:		pointer to model
******************************************************************************************/
model* new_model( parameters *params )
{
	int type;

	model *model_ptr = NULL;
	model_ptr = calloc( 1, sizeof( model ) );
	if( model_ptr == NULL )
	    print_exit("calloc to model failed\n");
	
	model_ptr->params = params;
	model_ptr->time   = 0;

	model_ptr->event_lists = calloc( N_EVENT_TYPES, sizeof( event_list ) );
	for( type = 0; type < N_EVENT_TYPES;  type++ )
		set_up_event_list( model_ptr, params, type );

	set_up_population( model_ptr );
	set_up_app_users( model_ptr );
	set_up_networks( model_ptr );
	set_up_interactions( model_ptr );
	set_up_events( model_ptr );
	set_up_distributions( model_ptr );
	set_up_individual_hazard( model_ptr );
	set_up_seed_infection( model_ptr );

	model_ptr->n_quarantine_days = 0;

	return model_ptr;
};

/*****************************************************************************************
*  Name:		destroy_model
*  Description: Destroys the model structure and releases its memory
******************************************************************************************/
void destroy_model( model *model )
{
	long idx;

	for( idx = 0; idx < model->params->n_total; idx++ )
		destroy_individual( &(model->population[idx] ) );
    free( model->population );
    free( model->possible_interactions );
    free( model->interactions );
    free( model->events );
	for( idx = 0; idx < N_TRANSITION_TYPES; idx++ )
		free( model->transition_time_distributions[ idx ] );
	free( model->transition_time_distributions );

    destroy_network( model->random_network);
    destroy_network( model->household_network );
    for( idx = 0; idx < N_AGE_GROUPS; idx++ )
    	destroy_network( model->work_network[idx] );
    for( idx = 0; idx < N_EVENT_TYPES; idx++ )
    	destroy_event_list( model, idx );
};

/*****************************************************************************************
*  Name:		set_up_event_list
*  Description: sets up an event_list
*  Returns:		void
******************************************************************************************/
void set_up_event_list( model *model, parameters *params, int type )
{

	int day, age;
	event_list *list = &(model->event_lists[ type ]);
	list->type       = type;

	list->n_daily          = calloc( MAX_TIME, sizeof(long) );
	list->n_daily_current  = calloc( MAX_TIME, sizeof(long) );
	list->infectious_curve = calloc( MAX_INFECTIOUS_PERIOD, sizeof(double) );
	list->n_total_by_age   = calloc( N_AGE_GROUPS, sizeof(long) );
	list->n_daily_by_age   = calloc( MAX_TIME, sizeof(long*) );
	list->events		   = calloc( MAX_TIME, sizeof(event*));

	list->n_current = 0;
	list->n_total   = 0;
	for( day = 0; day < MAX_TIME; day++ )
	{
		list->n_daily_by_age[day] = calloc( N_AGE_GROUPS, sizeof(long) );
		for( age = AGE_0_17; age <= AGE_65; age++ )
			list->n_daily_by_age[day][age] = 0;

		list->n_daily[day] = 0;
		list->n_daily_current[day] = 0;
	}
}

/*****************************************************************************************
*  Name:		destroy_event_list
*  Description: Destroys an event list
******************************************************************************************/
void destroy_event_list( model *model, int type )
{
	int day;
	free( model->event_lists[type].n_daily );

	for( day = 0; day < MAX_TIME; day++ )
		free( model->event_lists[type].n_daily_by_age[day]);

	free( model->event_lists[type].n_daily_current );
	free( model->event_lists[type].infectious_curve );
	free( model->event_lists[type].n_total_by_age );
	free( model->event_lists[type].n_daily_by_age );
	free( model->event_lists[type].events );
};

/*****************************************************************************************
*  Name:		set_up_networks
*  Description: sets up then networks
*  Returns:		void
******************************************************************************************/
void set_up_networks( model *model )
{
	long idx;
	long n_total 			  = model->params->n_total;
	long n_random_interactions;
	double mean_interactions  = 0;

	for( idx = AGE_0_17; idx <= AGE_65; idx++ )
		mean_interactions = max( mean_interactions, model->params->mean_random_interactions[idx] );
	n_random_interactions = (long) round( n_total * ( 1.0 + mean_interactions ) );

	model->random_network        = new_network( n_total, RANDOM );
	model->random_network->edges = calloc( n_random_interactions, sizeof( edge ) );

	model->household_network = new_network( n_total, HOUSEHOLD );
	build_household_network( model );

	model->work_network = calloc( N_AGE_GROUPS, sizeof( network* ) );
	for( idx = AGE_0_17; idx <= AGE_65; idx++ )
		set_up_work_network( model, idx );
}

/*****************************************************************************************
*  Name:		set_up_work_network
*  Description: sets up the work network for a particular age group
*  Returns:		void
******************************************************************************************/
void set_up_work_network( model *model, int age )
{
	long idx;
	long n_people = 0;
	long *people;
	int n_interactions;

	people = calloc( model->params->n_total, sizeof( long ) );
	for( idx = 0; idx < model->params->n_total; idx++ )
		if( model->population[idx].work_network == age )
			people[n_people++] = idx;

	model->work_network[age] = new_network( n_people, WORK );
	n_interactions           = (int) round( model->params->mean_work_interactions[age] / model->params->daily_fraction_work );
	build_watts_strogatz_network( model->work_network[age], n_people, n_interactions, 0.1, TRUE );
	relabel_network( model->work_network[age], people );

	free( people );
}


/*****************************************************************************************
*  Name:		set_up_events
*  Description: sets up the event tags
*  Returns:		void
******************************************************************************************/
void set_up_events( model *model )
{
	long idx;
	int types = 6;
	parameters *params = model->params;

	model->events     = calloc( types * params->n_total, sizeof( event ) );
	model->next_event = &(model->events[0]);
	for( idx = 1; idx < types * params->n_total; idx++ )
	{
		model->events[idx-1].next = &(model->events[idx]);
		model->events[idx].last   = &(model->events[idx-1]);
	}
	model->events[types * params->n_total - 1].next = model->next_event;
	model->next_event->last = &(model->events[types * params->n_total - 1] );
}

/*****************************************************************************************
*  Name:		set_up_population
*  Description: sets up the initial population
*  Returns:		void
******************************************************************************************/
void set_up_population( model *model )
{
	parameters *params = model->params;
	long idx;

	model->population = calloc( params->n_total, sizeof( individual ) );
	for( idx = 0; idx < params->n_total; idx++ )
		initialize_individual( &(model->population[idx]), params, idx );
}

/*****************************************************************************************
*  Name:		set_up_individual_hazard
*  Description: sets the initial hazard for each individual
*  Returns:		void
******************************************************************************************/
void set_up_individual_hazard( model *model )
{
	parameters *params = model->params;
	long idx;

	for( idx = 0; idx < params->n_total; idx++ )
		initialize_hazard( &(model->population[idx]), params );
}

/*****************************************************************************************
*  Name:		estimate_total_interactions
*  Description: estimates the total number of interactions from the networks
*  Returns:		void
******************************************************************************************/
double estimate_total_interactions( model *model )
{
	long idx;
	double n_interactions;
	n_interactions = 0;

	n_interactions += model->household_network->n_edges;
	for( idx = 0; idx < model->params->n_total; idx++ )
		n_interactions += model->population[idx].base_random_interactions * 0.5;
	for( idx = AGE_0_17; idx <= AGE_65 ; idx++ )
		n_interactions += model->work_network[idx]->n_edges * model->params->daily_fraction_work;

	return n_interactions;
}

/*****************************************************************************************
*  Name:		estimate_mean_interactions_by_age
*  Description: estimates the mean number of interactions by age
*  Returns:		void
******************************************************************************************/
double estimate_mean_interactions_by_age( model *model, int age )
{
	long pdx, ndx;
	long people = 0;
	double inter  = 0;

	for( pdx = 0; pdx < model->params->n_total; pdx++ )
		if( model->population[pdx].age_group == age )
		{
			people++;
			inter += model->population[pdx].base_random_interactions;
		}
	for( pdx = 0; pdx < model->household_network->n_edges; pdx++ )
	{
		if( model->population[model->household_network->edges[pdx].id1].age_group == age )
			inter++;
		if( model->population[model->household_network->edges[pdx].id2].age_group == age )
			inter++;
	}

	for( ndx = AGE_0_17; ndx <= AGE_65 ; ndx++ )
		for( pdx = 0; pdx < model->work_network[ndx]->n_edges; pdx++ )
		{
			if( model->population[model->work_network[ndx]->edges[pdx].id1].age_group == age )
				inter += model->params->daily_fraction_work;
			if( model->population[model->work_network[ndx]->edges[pdx].id2].age_group == age )
				inter  += model->params->daily_fraction_work;
		}

	return 1.0 * inter / people;
}

/*****************************************************************************************
*  Name:		set_up_interactions
*  Description: sets up the stock of interactions, note that these get recycled once we
*  				move to a later date
*  Returns:		void
******************************************************************************************/
void set_up_interactions( model *model )
{
	parameters *params = model->params;
	individual *indiv;
	long idx, n_idx, indiv_idx, n_daily_interactions, n_interactions;

	n_daily_interactions = (long) round( 2 * 1.1 * estimate_total_interactions( model ) );
	n_interactions       = n_daily_interactions * params->days_of_interactions;

	model->interactions          = calloc( n_interactions, sizeof( interaction ) );
	model->n_interactions        = n_interactions;
	model->interaction_idx       = 0;
	model->interaction_day_idx   = 0;

	model->possible_interactions = calloc( n_daily_interactions, sizeof( long ) );
	idx = 0;
	for( indiv_idx = 0; indiv_idx < params->n_total; indiv_idx++ )
	{
		indiv = &(model->population[ indiv_idx ]);
		for( n_idx = 0; n_idx < indiv->random_interactions; n_idx++ )
			model->possible_interactions[ idx++ ] = indiv_idx;
	}

	model->n_possible_interactions = idx;
	model->n_total_intereactions   = 0;
}

/*****************************************************************************************
*  Name:		set_up_distributions
*  Description: sets up discrete distributions and functions which are used to
*  				model events and calculates infectious rate per interaction.
*
*  				The infectious rate per interaction adult is the infectious rate divided
*  				by the mean number of daily interactions of an adult.
*
*  				Adjustments are calculated for children and elderly based upon their
*  				difference in the number of daily interactions and the relative overall
*  				suscpetibility.
*
*  Returns:		void
******************************************************************************************/
void set_up_distributions( model *model )
{
	parameters *params = model->params;
	double infectious_rate;
	double mean_interactions[N_AGE_GROUPS];
	int idx;
	int **transitions;

	model->transition_time_distributions = calloc( N_TRANSITION_TYPES, sizeof( int*) );
	for( idx = 0; idx < N_TRANSITION_TYPES; idx++ )
		model->transition_time_distributions[idx] = calloc( N_DRAW_LIST, sizeof( int ) );
	transitions = model->transition_time_distributions;

	gamma_draw_list( transitions[ASYMPTOMATIC_RECOVERED], 	   N_DRAW_LIST, params->mean_asymptomatic_to_recovery, params->sd_asymptomatic_to_recovery );
	gamma_draw_list( transitions[PRESYMPTOMATIC_SYMPTOMATIC],  N_DRAW_LIST, params->mean_time_to_symptoms,         params->sd_time_to_symptoms );
	gamma_draw_list( transitions[SYMPTOMATIC_RECOVERED],   	   N_DRAW_LIST, params->mean_time_to_recover,  		   params->sd_time_to_recover );
	gamma_draw_list( transitions[HOSPITALISED_RECOVERED],      N_DRAW_LIST, params->mean_time_to_recover,  		   params->sd_time_to_recover );
	gamma_draw_list( transitions[HOSPITALISED_DEATH],          N_DRAW_LIST, params->mean_time_to_death,    		   params->sd_time_to_death );
	bernoulli_draw_list( transitions[SYMPTOMATIC_HOSPITALISED],N_DRAW_LIST, params->mean_time_to_hospital );

	mean_interactions[AGE_0_17]  = estimate_mean_interactions_by_age( model, AGE_0_17 );
	mean_interactions[AGE_18_64] = estimate_mean_interactions_by_age( model, AGE_18_64 );
	mean_interactions[AGE_65]    = estimate_mean_interactions_by_age( model, AGE_65 );

	infectious_rate   = params->infectious_rate / mean_interactions[AGE_18_64];
	params->adjusted_susceptibility_child   = params->relative_susceptibility_child * mean_interactions[AGE_18_64] / mean_interactions[AGE_0_17];
	params->adjusted_susceptibility_elderly = params->relative_susceptibility_elderly * mean_interactions[AGE_18_64] / mean_interactions[AGE_65];

	gamma_rate_curve( model->event_lists[PRESYMPTOMATIC].infectious_curve, MAX_INFECTIOUS_PERIOD, params->mean_infectious_period,
					  params->sd_infectious_period, infectious_rate );

	gamma_rate_curve( model->event_lists[ASYMPTOMATIC].infectious_curve, MAX_INFECTIOUS_PERIOD, params->mean_infectious_period,
				      params->sd_infectious_period, infectious_rate * params->asymptomatic_infectious_factor);

	gamma_rate_curve( model->event_lists[SYMPTOMATIC].infectious_curve, MAX_INFECTIOUS_PERIOD, params->mean_infectious_period,
				      params->sd_infectious_period, infectious_rate );

	gamma_rate_curve( model->event_lists[HOSPITALISED].infectious_curve, MAX_INFECTIOUS_PERIOD, params->mean_infectious_period,
					  params->sd_infectious_period, infectious_rate );
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
	double n_person_frac[ UK_HOUSEHOLD_N_MAX ];
	double *pop = model->params->uk_pop;

	pop_all      = pop[AGE_0_17] + pop[AGE_18_64] + pop[AGE_65];
	survey_tot   = 0;
	max_children = 0;
	for( idx = 0; idx < UK_HOUSEHOLD_N_MAX; idx++)
	{
		n_person_frac[idx] = model->params->uk_house[ idx ] * ( idx + 1 );
		survey_tot        += n_person_frac[idx];
		max_children      += model->params->uk_house[ idx ] * max( idx -1 , 0 );
	}

	*child_frac_3_6   = pop[AGE_0_17] / pop_all / ( max_children / survey_tot );
	*elderly_frac_1_2 = pop[AGE_65] / pop_all / ( ( n_person_frac[HH_1] + n_person_frac[HH_2] ) / survey_tot );

	if( *child_frac_3_6 > 1 )
		print_exit( "not sufficient 3-6 person households for all the children" );

	if( *elderly_frac_1_2 > 1 )
		print_exit( "not sufficient 1-2 person households for all the elderly" );

 	total = 0;
	for( idx = 1; idx < UK_HOUSEHOLD_N_MAX; idx++)
	{
		n_house_tot[idx] = (long) round( n_person_frac[idx] / survey_tot / ( idx + 1 ) * model->params->n_total );
		total += n_house_tot[idx] * ( idx + 1 );
	}
	n_house_tot[0] = model->params->n_total - total;
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
	long pop_idx, hdx, edge_idx;
	int ndx, pdx, p2dx, age;
	long n_house_tot[ UK_HOUSEHOLD_N_MAX ];
	double elderly_frac_1_2;
	double child_frac_3_6;
	network *network   = model->household_network;

	if( network->n_edges != 0 )
		print_exit( "the household network can only be once" );

	calculate_household_distribution( model, n_house_tot, &elderly_frac_1_2, &child_frac_3_6 );
	for( ndx = 0; ndx < UK_HOUSEHOLD_N_MAX; ndx++ )
		network->n_edges += n_house_tot[ndx] * ndx * ( ndx + 1 ) / 2;
	network->edges = calloc( network->n_edges, sizeof( edge ) );

	edge_idx = 0;
	pop_idx = 0;
	for( ndx = 0; ndx < UK_HOUSEHOLD_N_MAX; ndx++ )
		for( hdx = 0; hdx < n_house_tot[ndx]; hdx++ )
			for( pdx = 0; pdx < ( ndx + 1 ); pdx++ )
			{
				age = AGE_18_64;
				if( ndx <= HH_2 && ( 1.0 * hdx / n_house_tot[ndx] ) < elderly_frac_1_2 )
					age = AGE_65;
				if( ndx >= HH_3 && ( 1.0 * hdx / n_house_tot[ndx] ) < child_frac_3_6 && pdx < ( ndx - 1 ) )
					age = AGE_0_17;

				set_age_group( &(model->population[pop_idx]), model->params, age );

				for( p2dx = pdx+1; p2dx < ( ndx + 1 ); p2dx++ )
				{
					network->edges[edge_idx].id1 = pop_idx;
					network->edges[edge_idx].id2 = pop_idx + p2dx - pdx;

					edge_idx++;
				}
				pop_idx++;
			}
}

/*****************************************************************************************
*  Name:		new_event
*  Description: gets a new event tag
*  Returns:		void
******************************************************************************************/
event* new_event( model *model )
{
	event *event = model->next_event;

	model->next_event       = event->next;
	model->next_event->last = event->last;
	event->last->next       = model->next_event;

	event->next = NULL;
	event->last = NULL;

	return event;
}



/*****************************************************************************************
*  Name:		flu_infections
*  Description: Randomly pick people from the population to go down with flu, they
*  				will then self-quarantine (same fraction of true infected people)
*  				and request a test.
*
*  Returns:		void
******************************************************************************************/
void flu_infections( model *model )
{
	long idx, pdx, n_infected;
	individual *indiv;

	n_infected = round( model->params->n_total * model->params->seasonal_flu_rate );

	idx = 0;
	while( idx < n_infected )
	{
		pdx   = gsl_rng_uniform_int( rng, model->params->n_total );
		indiv = &(model->population[pdx]);

		if( !(indiv->status == UNINFECTED && indiv->quarantined == FALSE ) )
			continue;

		if( gsl_ran_bernoulli( rng, model->params->self_quarantine_fraction ) )
		{
			set_quarantine_status( indiv, model->params, model->time, TRUE );
			indiv->quarantine_event = add_individual_to_event_list( model, QUARANTINED, indiv, model->time );
			add_individual_to_event_list( model, TEST_TAKE, indiv, model->time+1 );
		}

		idx++;
	}
}

/*****************************************************************************************
*  Name:		quarantine_contracts
*  Description: Quarantine contacts
*  Returns:		void
******************************************************************************************/
void quarantine_contacts( model *model, individual *indiv )
{
	interaction *inter;
	individual *contact;
	int idx, ddx, day, n_contacts;
	int time_event;

	day = model->interaction_day_idx;
	for( ddx = 0; ddx < model->params->quarantine_days; ddx++ )
	{
		n_contacts = indiv->n_interactions[day];
		time_event = model->time + max( model->params->test_insensititve_period - ddx, 1 );

		if( n_contacts > 0 )
		{
			inter = indiv->interactions[day];
			for( idx = 1; idx < n_contacts; idx++ )
			{
				if( inter->type != HOUSEHOLD )
					continue;

				contact = inter->individual;
				if( contact->status != HOSPITALISED && contact->status != DEATH && contact->quarantined == FALSE )
				{
					if( gsl_ran_bernoulli( rng, model->params->quarantine_fraction ) )
					{
						set_quarantine_status( contact, model->params, model->time, TRUE );
						contact->quarantine_event = add_individual_to_event_list( model, QUARANTINED, contact, model->time );
						add_individual_to_event_list( model, TEST_TAKE, contact, time_event );
					}
				}
				inter = inter->next;
			}
		}
		day = ifelse( day == 0, model->params->days_of_interactions -1, day-1 );
	}
}


/*****************************************************************************************
*  Name:		release_individual_from_quarantine
*  Description: Release an individual held in quarantine
*  Returns:		void
******************************************************************************************/
void release_individual_from_quarantine( model *model, individual* indiv )
{
	remove_event_from_event_list( model, indiv->quarantine_event );
	if( indiv->quarantine_release_event != NULL )
		remove_event_from_event_list( model, indiv->quarantine_release_event );
	set_quarantine_status( indiv, model->params, model->time, FALSE );
}

/*****************************************************************************************
*  Name:		release_from_quarantine
*  Description: Release those held in quarantine
*  Returns:		void
******************************************************************************************/
void release_from_quarantine( model *model )
{
	long idx, n_quarantined;
	event *event, *next_event;

	n_quarantined = model->event_lists[QUARANTINE_RELEASE].n_daily_current[ model->time ];
	next_event    = model->event_lists[QUARANTINE_RELEASE].events[ model->time ];
	for( idx = 0; idx < n_quarantined; idx++ )
	{
		event      = next_event;
		next_event = event->next;
		release_individual_from_quarantine( model, event->individual );
	}
}

/*****************************************************************************************
*  Name:		quarantined_test_take
*  Description: Take a test
*  Returns:		void
******************************************************************************************/
void quarantined_test_take( model *model )
{
	long idx, n_test_take;
	event *event, *next_event;
	individual *indiv;

	n_test_take = model->event_lists[TEST_TAKE].n_daily_current[ model->time ];
	next_event  = model->event_lists[TEST_TAKE].events[ model->time ];

	for( idx = 0; idx < n_test_take; idx++ )
	{
		event      = next_event;
		next_event = event->next;
		indiv      = event->individual;

		// add test
		if( indiv->status == UNINFECTED || indiv->status == RECOVERED )
			indiv->quarantine_test_result = FALSE;
		else
			indiv->quarantine_test_result = TRUE;

		add_individual_to_event_list( model, TEST_RESULT, indiv, model->time + model->params->test_result_wait );
		remove_event_from_event_list( model, event );
	}
}

/*****************************************************************************************
*  Name:		quarantined_test_result
*  Description: Receive test result
*  Returns:		void
******************************************************************************************/
void quarantined_test_result( model *model )
{
	long idx, n_test_result;
	event *event, *next_event;
	individual *indiv;

	n_test_result = model->event_lists[TEST_RESULT].n_daily_current[ model->time ];
	next_event    = model->event_lists[TEST_RESULT].events[ model->time ];

	for( idx = 0; idx < n_test_result; idx++ )
	{
		event      = next_event;
		next_event = event->next;
		indiv      = event->individual;

		if( indiv->quarantine_test_result == FALSE )
			indiv->quarantine_release_event = add_individual_to_event_list( model, QUARANTINE_RELEASE, indiv, model->time );
		else
		{
			if( indiv->is_case == FALSE )
			{
				set_case( indiv, model->time );
				add_individual_to_event_list( model, CASE, indiv, model->time );
			}

			if( indiv->status != HOSPITALISED )
			{
				indiv->quarantine_release_event = add_individual_to_event_list( model, QUARANTINE_RELEASE, indiv, model->time + 14 );
				quarantine_contacts( model, indiv );
			}
		}

		remove_event_from_event_list( model, event );
	}
}

/*****************************************************************************************
*  Name:		add_indiv_to_event_list
*  Description: adds an individual to an event list at a particular time
*
*  Arguments:	list:	pointer to the event list
*  				indiv:	pointer to the individual
*  				time:	time of the event (int)
*  				model:	pointer to the model
*
*  Returns:		a pointer to the newly added event
******************************************************************************************/
event* add_individual_to_event_list(
	model *model,
	int type,
	individual *indiv,
	int time
)
{
	event_list *list    = &(model->event_lists[ type ]);
	event *event        = new_event( model );
	event->individual   = indiv;
	event->type         = type;
	event->time         = time;

	if( list->n_daily_current[time] >0  )
	{
		list->events[ time ]->last = event;
		event->next  = list->events[ time ];
	}

	list->events[time ] = event;
	list->n_daily[time]++;
	list->n_daily_by_age[time][indiv->age_group]++;
	list->n_daily_current[time]++;

	return event;
}

/*****************************************************************************************
*  Name:		remove_event_from_event_list
*  Description: removes an event from an list at a particular time
*
*  Arguments:	list:	pointer to the event list
*  				event:	pointer to the event
*  				time:	time of the event (int)
*
*  Returns:		a pointer to the newly added event
******************************************************************************************/
void remove_event_from_event_list(
	model *model,
	event *event
)
{
	int type = event->type;
	int time = event->time;
	event_list *list = &(model->event_lists[ type ]);

	if( list->n_daily_current[ time ] > 1 )
	{
		if( event != list->events[ time ] )
		{
			if( event->next == NULL )
				event->last->next = NULL;
			else
			{
				event->last->next = event->next;
				event->next->last = event->last;
			}
		}
		else
			list->events[ time ] = event->next;
	}
	else
		list->events[time] = NULL;

	model->next_event->last->next = event;
	event->last = model->next_event->last;
	event->next = model->next_event;
	model->next_event->last = event;

	list->n_current--;
	list->n_daily_current[ time ]--;
}

/*****************************************************************************************
*  Name:		update_event_list_counters
*  Description: updates the event list counters, called at the end of a time step
*  Returns:		void
******************************************************************************************/
void update_event_list_counters( model *model, int type )
{
	model->event_lists[type].n_current += model->event_lists[type].n_daily_current[ model->time ];
	model->event_lists[type].n_total   += model->event_lists[type].n_daily[ model->time ];

	for( int age = AGE_0_17; age <= AGE_65; age++ )
		model->event_lists[type].n_total_by_age[age] += model->event_lists[type].n_daily_by_age[ model->time ][ age ];
}

/*****************************************************************************************
*  Name:		set_up_seed_infection
*  Description: sets up the initial population
*  Returns:		void
******************************************************************************************/
void set_up_seed_infection( model *model )
{
	parameters *params = model->params;
	int idx;
	unsigned long int person;

	for( idx = 0; idx < params->n_seed_infection; idx ++ )
	{
		person = gsl_rng_uniform_int( rng, params->n_total );
		new_infection( model, &(model->population[ person ]), &(model->population[ person ]) );
	}

	update_event_list_counters( model, PRESYMPTOMATIC );
	update_event_list_counters( model, ASYMPTOMATIC );
}


/*****************************************************************************************
*  Name:		build_random_newtork
*  Description: Builds a new random network
******************************************************************************************/
void build_random_network( model *model )
{
	long idx, n_pos, person;
	int jdx;
	long *interactions = model->possible_interactions;
	network *network   = model->random_network;

	network->n_edges = 0;
	n_pos            = 0;
	for( person = 0; person < model->params->n_total; person++ )
		for( jdx = 0; jdx < model->population[person].random_interactions; jdx++ )
			interactions[n_pos++]=person;

	if( n_pos == 0 )
		return;

	gsl_ran_shuffle( rng, interactions, n_pos, sizeof(long) );

	idx = 0;
	n_pos--;
	while( idx < n_pos )
	{
		if( interactions[ idx ] == interactions[ idx + 1 ] )
		{
			idx++;
			continue;
		}
		network->edges[network->n_edges].id1 = interactions[ idx++ ];
		network->edges[network->n_edges].id2 = interactions[ idx++ ];
		network->n_edges++;
	}
}

/*****************************************************************************************
*  Name:		add_interactions_from_network
*  Description: Adds the daily interactions to all individual from a network
******************************************************************************************/
void add_interactions_from_network(
	model *model,
	network *network,
	int skip_hospitalised,
	int skip_quarantined,
	double prob_drop
)
{
	long idx     = 0;
	long all_idx = model->interaction_idx;
	int day      = model->interaction_day_idx;

	interaction *inter1, *inter2;
	individual *indiv1, *indiv2;

	while( idx < network->n_edges )
	{
		indiv1 = &(model->population[ network->edges[idx].id1 ] );
		indiv2 = &(model->population[ network->edges[idx++].id2 ] );

		if( indiv1->status == DEATH || indiv2 ->status == DEATH )
			continue;
		if( skip_hospitalised && ( indiv1->status == HOSPITALISED || indiv2->status == HOSPITALISED ) )
			continue;
		if( skip_quarantined && ( indiv1->quarantined || indiv2->quarantined ) )
			continue;
		if( prob_drop > 0 && gsl_ran_bernoulli( rng, prob_drop ) )
			continue;

		inter1 = &(model->interactions[ all_idx++ ]);
		inter2 = &(model->interactions[ all_idx++ ]);

		inter1->type       = network->type;
		inter1->individual = indiv2;
		inter1->next       = indiv1->interactions[ day ];
		indiv1->interactions[ day ] = inter1;
		indiv1->n_interactions[ day ]++;

		inter2->type       = network->type;
		inter2->individual = indiv1;
		inter2->next       = indiv2->interactions[ day ];
		indiv2->interactions[ day ] = inter2;
		indiv2->n_interactions[ day ]++;

		model->n_total_intereactions++;

		if( all_idx > model->n_interactions )
			all_idx = 0;
	}

	model->interaction_idx =  all_idx;
}

/*****************************************************************************************
*  Name:		build_daily_newtork
*  Description: Builds a new interaction network
******************************************************************************************/
void build_daily_newtork( model *model )
{
	int idx, day;

	day = model->interaction_day_idx;
	for( idx = 0; idx < model->params->n_total; idx++ )
		model->population[ idx ].n_interactions[ day ] = 0;

	build_random_network( model );
	add_interactions_from_network( model, model->random_network, FALSE, FALSE, 0 );
	add_interactions_from_network( model, model->household_network, TRUE, FALSE, 0 );

	for( idx = AGE_0_17; idx <= AGE_65; idx++ )
		add_interactions_from_network( model, model->work_network[idx], TRUE, TRUE, 1.0 - model->params->daily_fraction_work );

};

/*****************************************************************************************
*  Name:		one_time_step
*  Description: Move the model through one time step
******************************************************************************************/
int one_time_step( model *model )
{
	(model->time)++;
	update_event_list_counters( model, SYMPTOMATIC );
	update_event_list_counters( model, HOSPITALISED );
	update_event_list_counters( model, RECOVERED );
	update_event_list_counters( model, DEATH );
	update_event_list_counters( model, TEST_TAKE );
	update_event_list_counters( model, TEST_RESULT );

	build_daily_newtork( model );
	transmit_virus( model );

	transition_to_symptomatic( model );
	transition_to_hospitalised( model );
	transition_to_recovered( model );
	transition_to_death( model );
	flu_infections( model );
	quarantined_test_take( model );
	quarantined_test_result( model );
	release_from_quarantine( model );

	update_event_list_counters( model, PRESYMPTOMATIC );
	update_event_list_counters( model, ASYMPTOMATIC );
	update_event_list_counters( model, QUARANTINED );
	update_event_list_counters( model, CASE );
	model->n_quarantine_days += model->event_lists[QUARANTINED].n_current;

	ring_inc( model->interaction_day_idx, model->params->days_of_interactions );

	return 1;
};


/*****************************************************************************************
*  Name:		set_up_app_users
*  Description: Set up the proportion of app users in the population (default is FALSE)
******************************************************************************************/
void set_up_app_users( model *model )
{
	long idx;
	long* app_users_idx = calloc(model->params->n_total, sizeof(long));
	
	for( idx = 0; idx < model->params->n_total; idx++ )
		app_users_idx[idx] = idx;
	
	gsl_ran_shuffle( rng, app_users_idx, model->params->n_total, sizeof(long) );
	
	long n_app_users = (long) (model->params->app_users_fraction * model->params->n_total);
	
	for( idx = 0; idx < n_app_users; idx++ )
		model->population[ app_users_idx[idx] ].app_user = TRUE;
};

