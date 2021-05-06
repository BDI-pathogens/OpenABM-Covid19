/*
 * params.c
 *
 *  Created on: 7 Mar 2020
 *      Author: hinchr
 */

#include "params.h"
#include "constant.h"
#include "utilities.h"
#include "model.h"
#include "disease.h"
#include "individual.h"
#include "interventions.h"
#include "demographics.h"
#include <string.h>


/*****************************************************************************************
*  Name: 		initialize_params
*  Description: initializes the params structure
******************************************************************************************/
void initialize_params( parameters *params )
{
	params->demo_house = NULL;
	params->occupation_network_table = NULL;
}

/*****************************************************************************************
*  Name: 		set_demographic_house_table
*  Description: sets ups a clean demographic house table
*  Arguments:	params:	  	the parameter structure
*  				n_total:    the total number of people
*  				n_household:the total number of households
*  				people:		the person idx of all the people (length n_total)
*  				ages:		the age group of all the people (length n_total)
*  				house_nos:  the house no of all the people (lnength n_total)
******************************************************************************************/
int set_demographic_house_table(
	parameters* params,
	long n_total,
	long n_households,
	long* people,
	long* ages,
	long* house_nos
)
{
	long pdx;

	if( params->demo_house != NULL )
	{
		free( params->demo_house->idx );
		free( params->demo_house->age_group );
		free( params->demo_house->house_no );
		free( params->demo_house );
	}
	params->demo_house = calloc( 1, sizeof( demographic_household_table ) );

	if( n_total != params->n_total )
	{
		print_now( "Total number of people must be the same as n_total in params" );
		return FALSE;
	}

	params->demo_house->n_total      = n_total;
	params->demo_house->n_households = n_households;
	params->demo_house->age_group    = calloc( n_total, sizeof( int ) );
	params->demo_house->idx          = calloc( n_total, sizeof( long ) );
	params->demo_house->house_no     = calloc( n_total, sizeof( long ) );

	for( pdx = 0; pdx < n_total; pdx++ )
	{
		if( (people[pdx] < 0) | (people[pdx] >= params->n_total) )
		{
			print_now( "The person index must be between 0 and n_total -1" );
			return FALSE;
		}
		if( (ages[pdx] < 0) | (ages[pdx] >= N_AGE_GROUPS) )
		{
			print_now( "The person's age must be between 0 and N_AGE_GROUPS-1" );
			return FALSE;
		}
		if( (house_nos[pdx] < 0) | (house_nos[pdx] >= params->demo_house->n_households) )
		{
			print_now( "The person's nouse_no must be between 0 and n_households" );
			return FALSE;
		}

		params->demo_house->idx[ pdx ]       = people[pdx];
		params->demo_house->age_group[ pdx ] = ages[pdx];
		params->demo_house->house_no[ pdx ]  = house_nos[pdx];
	}

	return TRUE;
}

/*****************************************************************************************
*  Name: 		set_app_users
*  Description: sets specific users to have or not have the app
*  Arguments:	model:	  	the model object
*  				users:    	array of users IDs to change
*  				n_users:    length of array of users IDs
*  				on_off:		TRUE turn app on for them, FALSE turn app off for them
******************************************************************************************/
int set_app_users(
	model *model,
	long *users,
	long n_users,
	int on_off
)
{
	long n_total = model->params->n_total;
	long idx;

	if( n_users < 1 )
	{
		print_now( "n_users must be positive" );
		return FALSE;
	}

	for( idx = 0; idx < n_users; idx++ )
	{
		if( users[ idx ] < 0 || users[ idx ] >= n_total )
		{
			print_now( "users must between 0 and n_total" );
			return FALSE;
		}
	}

	if( ( on_off != FALSE ) && ( on_off > TRUE ) )
	{
		print_now( "on_off must be TRUE or FALSE" );
		return FALSE;
	}

	for( idx = 0; idx < n_users; idx++ )
		model->population[ users[ idx ] ].app_user = on_off;

	return TRUE;
}


/*****************************************************************************************
*  Name: 		get_app_users
*  Description: returns all app users
*  Arguments:	model:	  	the model object
*  				users:    	array of users IDs to change
******************************************************************************************/
int get_app_users(
	model *model,
	long *users
)
{
	long n_total = model->params->n_total;
	long idx;

	for( idx = 0; idx < n_total; idx++ )
		users[ idx ] = model->population[ idx ].app_user;

	return TRUE;
}


/*****************************************************************************************
*  Name: 		get_app_user_by_index
*  Description: returns a specific app user by index
*  Arguments:	model:	  	the model object
*  				users:    	array of users IDs to change
******************************************************************************************/
int get_app_user_by_index(
	model *model,
	int idx
)
{
	long n_total = model->params->n_total;
  if (idx < 0 || idx >= n_total) {
    print_exit("idx (=%i) is out of bound. Allowed range: [0,%li[", idx, n_total);
  }

  return model->population[ idx ].app_user;
}

/*****************************************************************************************
*  Name: 		get_individuals
*  Description: populate input arrays with the population (including current status)
*  Arguments:	model:					the model object
*  				ids:					array of users IDs
*  				statuses:				array of users current status
*  				age_groups:				array of users age group
*  				occupation_networks:	array of users occupation networks
*  				house_ids:				array of users house id

******************************************************************************************/
long get_individuals(
	model *model,
	long *ids,
	int *statuses,
	int *age_groups,
	int *occupation_networks,
	long *house_ids,
	int *infection_counts,
	short *vaccine_statuses
)
{
	long n_total = model->params->n_total;
	long idx;
	
	for( idx = 0; idx < n_total; idx++ ){
		ids[ idx ] = model->population[ idx ].idx;
		statuses[ idx ] = model->population[ idx ].status;
		age_groups[ idx ] = model->population[ idx ].age_group;
		occupation_networks[ idx ] = model->population[ idx ].occupation_network;
		house_ids[ idx ] = model->population[ idx ].house_no;
		infection_counts[ idx ] = count_infection_events( &(model->population[ idx ]) );
		vaccine_statuses[ idx ] = model->population[ idx ].vaccine_status;
	}
	return idx;
}


/*****************************************************************************************
*  Name: 		set_indiv_occupation_network_property
*  Description: Sets the values of a single occupational network by index
******************************************************************************************/
int set_indiv_occupation_network_property(
	parameters* params,
	long network,
	int age_type,
	double mean_interaction,
	double lockdown_multiplier,
	long network_id,
	const char *network_name
)
{
	demographic_occupation_network_table *table = params->occupation_network_table;

	table->age_type[network]                        = age_type;
    table->mean_interactions[network]               = mean_interaction;
    table->lockdown_occupation_multipliers[network] = lockdown_multiplier;
    table->network_ids[network]                     = network_id;

    if( strlen( network_name ) < 128 )
    {
        strcpy( table->network_names[network], network_name );
    }
    else
        print_exit( "Network name cannot exceed 127 characters." );

    return TRUE;
}

/*****************************************************************************************
*  Name: 		set_occupation_network_table
*  Description: Allocates memory for the occupational networks
******************************************************************************************/
int set_occupation_network_table(
	parameters* params,
	long n_total,
	long n_networks
)
{
	if( params->occupation_network_table != NULL )
		destroy_occupation_network_table( params );

	params->occupation_network_table = calloc( 1, sizeof(demographic_occupation_network_table) );
	params->occupation_network_table->n_networks = n_networks;
	params->occupation_network_table->network_no = calloc( n_total, sizeof(long) );
	params->occupation_network_table->age_type = calloc( n_networks, sizeof(int) );
	params->occupation_network_table->mean_interactions = calloc( n_networks, sizeof(double) );
	params->occupation_network_table->lockdown_occupation_multipliers = calloc( n_networks, sizeof(double) );
	params->occupation_network_table->network_ids = calloc( n_networks, sizeof(long) );

	params->occupation_network_table->network_names = calloc( n_networks, sizeof(char *) );
    for( int i = 0; i != n_networks; ++i )
    	params->occupation_network_table->network_names[i] = calloc( 128, sizeof(char) );

    return TRUE;
}

/*****************************************************************************************
*  Name: 		set_indiv_occupation_network
*  Description: Assigns occupation network to each individual.
******************************************************************************************/
int set_indiv_occupation_network(
	parameters* params,
	long n_total,
	long *people,
	long *network
)
{
    if( params->occupation_network_table == NULL )
    {
        print_exit("Occupation_network_table is not initialized.");
    }

    if ( n_total != params->n_total )
    {
        print_exit("Each individual must have one occupation network assignment.");
    }

    for ( int pdx = 0; pdx != n_total; ++pdx)
    {
        if( (network[pdx] < 0) | (network[pdx] >= params->occupation_network_table->n_networks) )
        {
            print_now( "The person's network_no must be between 0 and n_occupation_networks." );
            return FALSE;
        }
        params->occupation_network_table->network_no[pdx] = network[pdx];
    }

    return TRUE;
}

/*****************************************************************************************
*  Name: 		set_up_default_occupation_network_table
*  Description: Generate the occupation network table from the default parameters if
*  				a custom network table has not been supplied
******************************************************************************************/
void set_up_default_occupation_network_table( parameters *params )
{
    set_occupation_network_table( params, params->n_total, N_DEFAULT_OCCUPATION_NETWORKS );

    for( int network = 0; network != N_DEFAULT_OCCUPATION_NETWORKS; ++network )
    {
        set_indiv_occupation_network_property( params, network, NETWORK_TYPE_MAP[network],
                                               params->mean_work_interactions[NETWORK_TYPE_MAP[network]],
                                               params->lockdown_occupation_multiplier[network],
                                               OCCUPATION_DEFAULT_MAP[network],
                                               DEFAULT_NETWORKS_NAMES[OCCUPATION_DEFAULT_MAP[network]] );
    }
}



/*****************************************************************************************
*  Name: 		get_model_param_hospital_on
*  Description: Gets the value of a parameter
******************************************************************************************/
int get_model_param_hospital_on(model *model)
{

    return model->params->hospital_on;
}


/*****************************************************************************************
*  Name: 		get_model_param_fatality_fraction
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_model_param_fatality_fraction(model * model, int age_group)
{

    if ( age_group >= N_AGE_GROUPS ) return ERROR;

	return model->params->fatality_fraction[age_group];
}



/*****************************************************************************************
*  Name: 		get_model_param_daily_fraction_work_used
*  Description: Gets the value of a parameter
******************************************************************************************/
double get_model_param_daily_fraction_work_used(model *model, int idx)
{
    if (idx >= N_DEFAULT_OCCUPATION_NETWORKS) return -1;

    return model->occupation_network[idx]->daily_fraction;
}

/*****************************************************************************************
*  Name:        get_model_param_quarantine_days
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_quarantine_days(model *model)
{
    return model->params->quarantine_days;
}

/*****************************************************************************************
*  Name:		get_model_param_self_quarantine_fraction
*  Description: Gets the value of an int parameter
******************************************************************************************/
double get_model_param_self_quarantine_fraction(model *model)
{
    return model->params->self_quarantine_fraction;
}

/*****************************************************************************************
*  Name:		get_model_param_trace_on_symptoms
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_trace_on_symptoms(model *model)
{
    return model->params->trace_on_symptoms;
}

/*****************************************************************************************
*  Name:		get_model_param_trace_on_positive
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_trace_on_positive(model *model)
{
    return model->params->trace_on_positive;
}

/*****************************************************************************************
*  Name:		get_model_param_quarantine_on_traced
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_quarantine_on_traced(model *model)
{
    return model->params->quarantine_on_traced;
}

/*****************************************************************************************
*  Name:		get_model_param_traceable_interaction_fraction
*  Description: Gets the value of an int parameter
******************************************************************************************/
double get_model_param_traceable_interaction_fraction(model *model)
{
    return model->params->traceable_interaction_fraction;
}

/*****************************************************************************************
*  Name:		get_model_param_tracing_network_depth
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_tracing_network_depth(model *model)
{
    return model->params->tracing_network_depth;
}

/*****************************************************************************************
*  Name:		get_model_param_allow_clinical_diagnosis
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_allow_clinical_diagnosis(model *model)
{
    return model->params->allow_clinical_diagnosis;
}

/*****************************************************************************************
*  Name:		get_model_param_quarantine_household_on_symptoms
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_quarantine_household_on_symptoms(model *model)
{
    return model->params->quarantine_household_on_symptoms;
}

/*****************************************************************************************
*  Name:		get_model_param_quarantine_household_on_positive
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_quarantine_household_on_positive(model *model)
{
    return model->params->quarantine_household_on_positive;
}

/*****************************************************************************************
*  Name:		get_model_param_quarantine_household_on_traced_positive
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_quarantine_household_on_traced_positive(model *model)
{
    return model->params->quarantine_household_on_traced_positive;
}

/*****************************************************************************************
*  Name:		get_model_param_quarantine_household_on_traced_symptoms
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_quarantine_household_on_traced_symptoms(model *model)
{
    return model->params->quarantine_household_on_traced_symptoms;
}

/*****************************************************************************************
*  Name:		get_model_param_quarantine_household_contacts_on_positive
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_quarantine_household_contacts_on_positive(model *model)
{
    return model->params->quarantine_household_contacts_on_positive;
}

/*****************************************************************************************
*  Name:                get_model_param_quarantine_household_contacts_on_symptoms
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_quarantine_household_contacts_on_symptoms(model *model)
{
    return model->params->quarantine_household_contacts_on_symptoms;
}

/*****************************************************************************************
*  Name:		get_model_param_test_on_symptoms
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_test_on_symptoms(model *model)
{
    return model->params->test_on_symptoms;
}

/*****************************************************************************************
*  Name:		get_model_param_test_release_on_negative
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_test_release_on_negative(model *model)
{
    return model->params->test_release_on_negative;
}

/*****************************************************************************************
*  Name:		get_model_param_test_on_traced
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_test_on_traced(model *model)
{
    return model->params->test_on_traced;
}

/*****************************************************************************************
*  Name:		get_model_param_test_result_wait
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_test_result_wait(model *model)
{
    return model->params->test_result_wait;
}

/*****************************************************************************************
*  Name:		get_model_param_test_order_wait
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_test_order_wait(model *model)
{
    return model->params->test_order_wait;
}

/*****************************************************************************************
*  Name:		get_model_param_test_result_wait_priority
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_test_result_wait_priority(model *model)
{
    return model->params->test_result_wait_priority;
}

/*****************************************************************************************
*  Name:		get_model_param_test_order_wait_priority
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_test_order_wait_priority(model *model)
{
    return model->params->test_order_wait_priority;
}

/*****************************************************************************************
*  Name: 		get_model_param_priority_test_contacts
*  Description: Gets the value of a parameter
******************************************************************************************/
int get_model_param_priority_test_contacts(model *model, int idx)
{
    if (idx >= N_AGE_GROUPS) return -1;
    return model->params->priority_test_contacts[idx];
}

/*****************************************************************************************
*  Name:		get_model_param_app_users_fraction
*  Description: Gets the value of double parameter
******************************************************************************************/
double get_model_param_app_users_fraction(model *model)
{
    int age;
	double t_pop, frac;

	t_pop = 0;
	frac  = 0;
	for( age = 0; age < N_AGE_GROUPS; age++ )
	{
		t_pop += model->params->population[ age ];
		frac  += model->params->app_users_fraction[ age ] * model->params->population[ age ];
	}

	return frac / t_pop;
}

/*****************************************************************************************
*  Name:		get_model_param_app_turned_on
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_app_turned_on(model *model)
{
    return model->params->app_turned_on;
}

/*****************************************************************************************
*  Name:		get_model_param_lockdown_on
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_lockdown_on(model *model)
{
	return model->params->lockdown_on;
}

/*****************************************************************************************
*  Name:        get_model_param_lockdown_house_interaction_multiplier
*  Description: Gets the value of a double parameter
******************************************************************************************/
double get_model_param_lockdown_house_interaction_multiplier(model *model)
{
	return model->params->lockdown_house_interaction_multiplier;
}

/*****************************************************************************************
*  Name:        get_model_param_lockdown_random_network_multiplier
*  Description: Gets the value of a double parameter
******************************************************************************************/
double get_model_param_lockdown_random_network_multiplier(model *model)
{
	return model->params->lockdown_random_network_multiplier;
}

/*****************************************************************************************
*  Name:        get_model_param_lockdown_occupation_multiplier
*  Description: Gets the value of a double parameter
******************************************************************************************/
double get_model_param_lockdown_occupation_multiplier(model *model, int idx)
{
	if ( idx >= N_DEFAULT_OCCUPATION_NETWORKS)  return FALSE;
	return model->params->lockdown_occupation_multiplier[idx];
}

/*****************************************************************************************
*  Name:        get_model_param_manual_trace_on_hospitalization
*  Description: Gets the value of parameter
******************************************************************************************/
int get_model_param_manual_trace_on_hospitalization( model* model )
{
	return model->params->manual_trace_on_hospitalization;
}

/*****************************************************************************************
*  Name:        get_model_param_manual_trace_on_positive
*  Description: Gets the value of parameter
******************************************************************************************/
int get_model_param_manual_trace_on_positive( model* model )
{
	return model->params->manual_trace_on_positive;
}

/*****************************************************************************************
*  Name:        get_model_param_manual_trace_on
*  Description: Gets the value of parameter
******************************************************************************************/
int get_model_param_manual_trace_on( model* model )
{
	return model->params->manual_trace_on;
}

/*****************************************************************************************
*  Name:        get_model_param_manual_trace_delay
*  Description: Gets the value of parameter
******************************************************************************************/
int get_model_param_manual_trace_delay( model* model )
{
	return model->params->manual_trace_delay;
}

/*****************************************************************************************
*  Name:        get_model_param_manual_trace_exclude_app_users
*  Description: Gets the value of parameter
******************************************************************************************/
int get_model_param_manual_trace_exclude_app_users( model* model )
{
	return model->params->manual_trace_exclude_app_users;
}

/*****************************************************************************************
*  Name:        get_model_param_manual_trace_n_workers
*  Description: Gets the value of parameter
******************************************************************************************/
int get_model_param_manual_trace_n_workers( model* model )
{
	return model->params->manual_trace_n_workers;
}

/*****************************************************************************************
*  Name:        get_model_param_manual_trace_interviews_per_worker_day
*  Description: Gets the value of parameter
******************************************************************************************/
int get_model_param_manual_trace_interviews_per_worker_day( model* model )
{
	return model->params->manual_trace_interviews_per_worker_day;
}

/*****************************************************************************************
*  Name:        get_model_param_manual_trace_notifications_per_worker_day
*  Description: Gets the value of parameter
******************************************************************************************/
int get_model_param_manual_trace_notifications_per_worker_day( model* model )
{
	return model->params->manual_trace_notifications_per_worker_day;
}

/*****************************************************************************************
*  Name:        get_model_param_manual_traceable_fraction_occupation
*  Description: Gets the value of parameter
******************************************************************************************/
double get_model_param_manual_traceable_fraction( model* model, int type )
{
	return model->params->manual_traceable_fraction[type];
}

/*****************************************************************************************
*  Name:        set_model_param_quarantine_days
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_quarantine_days(model *model, int value )
{
    model->params->quarantine_days = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_self_quarantine_fraction
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_self_quarantine_fraction(model *model, double value)
{
    model->params->self_quarantine_fraction = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_trace_on_symptoms
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_trace_on_symptoms(model *model, int value) {
   model->params->trace_on_symptoms = value;
   return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_trace_on_positive
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_trace_on_positive(model *model, int value) {
   model->params->trace_on_positive = value;
   return TRUE;
}

/*****************************************************************************************
*  Name: 		set_model_param_quarantine_on_traced
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_quarantine_on_traced( model *model, int value )
{
    model->params->quarantine_on_traced = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_traceable_interaction_fractio
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_traceable_interaction_fraction( model *model, double value )
{
    model->params->traceable_interaction_fraction = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_tracing_network_depth
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_tracing_network_depth( model *model, int value )
{
    model->params->tracing_network_depth = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_allow_clinical_diagnosis
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_allow_clinical_diagnosis( model *model, int value )
{
    model->params->allow_clinical_diagnosis = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_quarantine_household_on_symptoms
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_quarantine_household_on_symptoms( model *model, int value )
{
    model->params->quarantine_household_on_symptoms = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_quarantine_household_on_positive
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_quarantine_household_on_positive( model *model, int value )
{
    model->params->quarantine_household_on_positive = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_quarantine_household_on_traced_positive
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_quarantine_household_on_traced_positive( model *model, int value )
{
    model->params->quarantine_household_on_traced_positive = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_quarantine_household_on_traced_symptoms
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_quarantine_household_on_traced_symptoms( model *model, int value )
{
    model->params->quarantine_household_on_traced_symptoms = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_quarantine_household_contacts_on_positive
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_quarantine_household_contacts_on_positive( model *model, int value )
{
    model->params->quarantine_household_contacts_on_positive = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:                set_model_param_quarantine_household_contacts_on_symptoms
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_quarantine_household_contacts_on_symptoms( model *model, int value )
{
    model->params->quarantine_household_contacts_on_symptoms = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_test_on_symptoms
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_test_on_symptoms(model *model, int value) {
   model->params->test_on_symptoms = value;
   return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_test_on_traced
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_test_on_traced( model *model, int value )
{
    model->params->test_on_traced = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_test_release_on_negative
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_test_release_on_negative( model *model, int value )
{
    model->params->test_release_on_negative = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_test_result_wait
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_test_result_wait( model *model, int value )
{
    model->params->test_result_wait = value;
	check_params( model->params );
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_test_order_wait
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_test_order_wait( model *model, int value )
{
    model->params->test_order_wait = value;
	check_params( model->params );
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_test_result_wait_priority
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_test_result_wait_priority( model *model, int value )
{
    model->params->test_result_wait_priority = value;

    if( model->params->test_order_wait_priority == NO_PRIORITY_TEST )
    	model->params->test_order_wait_priority = model->params->test_order_wait;

    if( value == NO_PRIORITY_TEST )
    	model->params->test_order_wait_priority = NO_PRIORITY_TEST;

    check_params( model->params );

    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_test_order_wait_priority
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_test_order_wait_priority( model *model, int value )
{
    model->params->test_order_wait_priority = value;

    if( model->params->test_result_wait_priority == NO_PRIORITY_TEST )
     	model->params->test_result_wait_priority = model->params->test_result_wait;

     if( value == NO_PRIORITY_TEST )
     	model->params->test_result_wait_priority = NO_PRIORITY_TEST;

 	check_params( model->params );

    return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_priority_test_contacts
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_priority_test_contacts( model *model, int value, int idx )
{
	if (idx >= N_AGE_GROUPS) return FALSE;
	model->params->priority_test_contacts[idx] = value;
	return TRUE;
}
/*****************************************************************************************
*  Name:		set_model_param_app_users_fraction
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_app_users_fraction( model *model, double value )
{
    if( value > 1 || value < 0 )
    	return FALSE;

    int age;
    for( age = 0; age < N_AGE_GROUPS; age++ )
        model->params->app_users_fraction[ age ] = value;

    set_up_app_users( model );
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_relative_transmission
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_relative_transmission( model *model, double value, int type )
{
	double old = model->params->relative_transmission[ type ];

	// ignore very small changes
	if( fabs( old - value ) < 1e-8 )
		return TRUE;

	model->params->relative_transmission[ type ]      = value;
	model->params->relative_transmission_used[ type ] = value;

	if( type == HOUSEHOLD && model->params->lockdown_on )
		model->params->relative_transmission_used[ type ] = value * model->params->lockdown_house_interaction_multiplier;

	set_up_infectious_curves( model );
	return TRUE;
}


/*****************************************************************************************
*  Name:		set_model_param_app_turned_on
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_app_turned_on( model *model, int value )
{
    model->params->app_turned_on = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		get_model_param_risk_score
*  Description: Gets the value of the risk score parameter
******************************************************************************************/
double get_model_param_risk_score(
	model *model,
	int day,
	int age_infector,
	int age_susceptible
)
{
	if( (day < 0) | (day >= model->params->days_of_interactions ) )
		return UNKNOWN;

	if( (age_infector < 0) | (age_infector >= N_AGE_GROUPS) )
		return UNKNOWN;

	if( (age_susceptible < 0) | (age_susceptible >= N_AGE_GROUPS) )
		return UNKNOWN;

	return model->params->risk_score[ day ][ age_infector ][ age_susceptible ];
}

/*****************************************************************************************
*  Name:		get_model_param_risk_score_household
*  Description: Gets the value of the risk score household parameter
******************************************************************************************/
double get_model_param_risk_score_household(
	model *model,
	int age_infector,
	int age_susceptible
)
{
	if( (age_infector < 0) | (age_infector >= N_AGE_GROUPS) )
		return UNKNOWN;

	if( (age_susceptible < 0) | (age_susceptible >= N_AGE_GROUPS) )
		return UNKNOWN;

	return model->params->risk_score_household[ age_infector ][ age_susceptible ];
}

/*****************************************************************************************
*  Name:		set_model_param_risk_score
*  Description: Sets the value of the risk score parameter
******************************************************************************************/
int set_model_param_risk_score(
	model *model,
	int day,
	int age_infector,
	int age_susceptible,
	double value
)
{
	if( (day < 0) | (day >= model->params->days_of_interactions) )
		return FALSE;

	if( (age_infector < 0) | (age_infector >= N_AGE_GROUPS) )
		return FALSE;

	if( (age_susceptible < 0) | (age_susceptible >= N_AGE_GROUPS) )
		return FALSE;

	model->params->risk_score[ day ][ age_infector ][ age_susceptible ] = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_risk_score_household
*  Description: Sets the value of the risk score household parameter
******************************************************************************************/
int set_model_param_risk_score_household(
	model *model,
	int age_infector,
	int age_susceptible,
	double value
)
{
	if( (age_infector < 0) | (age_infector >= N_AGE_GROUPS) )
		return FALSE;

	if( (age_susceptible < 0) | (age_susceptible >= N_AGE_GROUPS) )
		return FALSE;

	model->params->risk_score_household[ age_infector ][ age_susceptible ] = value;
    return TRUE;
}


/*****************************************************************************************
*  Name:		set_model_param_lockdown_on
*  Description: Carries out checks on the input parameters
******************************************************************************************/
void update_work_intervention_state(model *model, int value)
{
	int network;
	parameters *params = model->params;

	if (value == TRUE) {
		// Turn intervetions on
        if (model->use_custom_occupation_networks == 0)
        {
            for (network = 0; network < N_DEFAULT_OCCUPATION_NETWORKS; network++ )
            {
                model->occupation_network[network]->daily_fraction = params->daily_fraction_work * params->lockdown_occupation_multiplier[network];
            }
        }
        else
            for ( network = 0; network < model->n_occupation_networks; network++ )
            {
                model->occupation_network[network]->daily_fraction = params->daily_fraction_work *
                                                                     model->params->occupation_network_table->lockdown_occupation_multipliers[network];
            }

	}
	else {
		for ( network = 0; network < model->n_occupation_networks; network++ )
		{
			model->occupation_network[network]->daily_fraction= params->daily_fraction_work;
		}
	}
}

/*****************************************************************************************
*  Name:		update_household_intervention_stat
*  Description: updates the
******************************************************************************************/
void update_household_intervention_state(model *model, int value)
{
	if (value == TRUE)
	{
		// Turn household multipliers on
		model->params->relative_transmission_used[HOUSEHOLD] = model->params->relative_transmission[HOUSEHOLD] *
															   model->params->lockdown_house_interaction_multiplier;
	}
	else
	{
		//Set household transmission to non multiplied state
		model->params->relative_transmission_used[HOUSEHOLD] = model->params->relative_transmission[HOUSEHOLD];
	}
}

/*****************************************************************************************
*  Name:		set_model_param_lockdown_on
*  Description: turns lockdown on and off
******************************************************************************************/
int set_model_param_lockdown_on( model *model, int value )
{
	long pdx;
	parameters *params = model->params;
	// If lockdown is off and we're setting it off again, return
	if( value == FALSE && !params->lockdown_on ){
			return TRUE;
	}
	else {
		update_work_intervention_state(model, value);
		update_household_intervention_state(model, value);
	}
	params->lockdown_on = value;
	set_up_infectious_curves( model );

	for( pdx = 0; pdx < params->n_total; pdx++ )
		update_random_interactions( &(model->population[pdx]), params );

	model->rebuild_networks = TRUE;

	return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_lockdown_elderly_on
*  Description: Carries out checks on the input parameters
******************************************************************************************/
int set_model_param_lockdown_elderly_on( model *model, int value )
{
	long pdx;
	int network;
	parameters *params = model->params;
	individual *indiv;

    if (value == TRUE)
    {
        if (model->use_custom_occupation_networks == 0)
        {
            for ( network = 0; network < N_DEFAULT_OCCUPATION_NETWORKS; network++ )
            {
                if ( NETWORK_TYPE_MAP[network] == NETWORK_TYPE_ELDERLY )
                    model->occupation_network[network]->daily_fraction = params->daily_fraction_work *
                                                                         params->lockdown_occupation_multiplier[network];
            }
        }
        else
        {
            for ( network = 0; network < model->n_occupation_networks; network++ )
            {
                if ( params->occupation_network_table->age_type[network] == NETWORK_TYPE_ELDERLY )
                    model->occupation_network[network]->daily_fraction = params->daily_fraction_work *
                                                                         params->occupation_network_table->lockdown_occupation_multipliers[network];
            }

        }
    }
	else if( value == FALSE )
	{
		if( !params->lockdown_elderly_on )
			return TRUE;

		if( !params->lockdown_on )
		{
            for( network = 0; network < model->n_occupation_networks; network++ )
                model->occupation_network[network]->daily_fraction = params->daily_fraction_work;
		}

	}
	else
	{
		return FALSE;
	}
	params->lockdown_elderly_on = value;
	set_up_infectious_curves( model );

	for( pdx = 0; pdx < params->n_total; pdx++ )
	{
		indiv = &(model->population[pdx]);
		if( indiv->age_type == AGE_TYPE_ELDERLY )
			update_random_interactions( indiv, params );
	}

	model->rebuild_networks = TRUE;

	return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_lockdown_house_interaction_multiplier
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_lockdown_house_interaction_multiplier( model *model, double value )
{
	model->params->lockdown_house_interaction_multiplier = value;

	if( model->params->lockdown_on )
		return set_model_param_lockdown_on( model, TRUE );

	if( model->params->lockdown_elderly_on )
		return set_model_param_lockdown_elderly_on( model, TRUE );

	return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_lockdown_random_network_multiplier
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_lockdown_random_network_multiplier( model *model, double value )
{
	model->params->lockdown_random_network_multiplier = value;

	if( model->params->lockdown_on )
		return set_model_param_lockdown_on( model, TRUE );

	if( model->params->lockdown_elderly_on )
		return set_model_param_lockdown_elderly_on( model, TRUE );

	return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_lockdown_occupation_multiplier
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_lockdown_occupation_multiplier( model *model, double value, int idx )
{
	if ( idx >= model->n_occupation_networks ) return FALSE;

	if ( model->use_custom_occupation_networks == 0 )
		model->params->lockdown_occupation_multiplier[idx] = value;
	else
		model->params->occupation_network_table->lockdown_occupation_multipliers[idx] = value;

	if( model->params->lockdown_on )
		return set_model_param_lockdown_on( model, TRUE );

	if( model->params->lockdown_elderly_on )
		return set_model_param_lockdown_elderly_on( model, TRUE );

	return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_manual_traceable_fraction
*  Description: Sets the value of parameter for given type
******************************************************************************************/
int set_model_param_manual_traceable_fraction( model *model, double value, int type )
{
	model->params->manual_traceable_fraction[type] = value;

	return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_manual_trace_on_hospitalization
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_manual_trace_on_hospitalization( model* model, int value )
{
	model->params->manual_trace_on_hospitalization = value;

	return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_manual_trace_on_positive
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_manual_trace_on_positive( model* model, int value )
{
	model->params->manual_trace_on_positive = value;

	return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_manual_trace_on
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_manual_trace_on( model* model, int value )
{
	model->params->manual_trace_on = value;

	return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_manual_trace_delay
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_manual_trace_delay( model* model, int value )
{
	model->params->manual_trace_delay = value;

	return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_manual_trace_exclude_app_users
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_manual_trace_exclude_app_users( model* model, int value )
{
	model->params->manual_trace_exclude_app_users = value;

	return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_manual_trace_n_workers
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_manual_trace_n_workers( model* model, int value )
{
	model->params->manual_trace_n_workers = value;

	return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_manual_trace_interviews_per_worker_day
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_manual_trace_interviews_per_worker_day( model* model, int value )
{
	model->params->manual_trace_interviews_per_worker_day = value;

	return TRUE;
}

/*****************************************************************************************
*  Name:        set_model_param_manual_trace_notifications_per_worker_day
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_manual_trace_notifications_per_worker_day( model* model, int value )
{
	model->params->manual_trace_notifications_per_worker_day = value;

	return TRUE;
}

/*****************************************************************************************
*  Name:		check_params
*  Description: Carries out checks on the input parameters
******************************************************************************************/
void check_params( parameters *params )
{
	int idx;

    if( params->end_time > MAX_TIME )
     	print_exit( "BAD PARAM end_time - can't be greater than MAX_TIME " );

    if( params->quarantine_days > params->days_of_interactions )
    	print_exit( "BAD PARAM quarantine_days - can't be greater than days_of_interactions" );

    if( params->lockdown_time_on < 1 )
      	print_exit( "BAD PARAM lockdown_time_on - can only be turned on at the first time step" );

    if( params->lockdown_elderly_time_on < 1 )
        print_exit( "BAD PARAM lockdown_elderly_time_on - can only be turned on at the first time step" );

    if( params->random_interaction_distribution != FIXED && params->random_interaction_distribution != NEGATIVE_BINOMIAL )
 	   print_exit( "BAR_PARAM - random_interaction_distribution - only fixed and negative-binomial distributions are supported" );

    if( params->random_interaction_distribution == NEGATIVE_BINOMIAL )
		for( idx = 0; idx < N_AGE_TYPES; idx++ )
			if( params->mean_random_interactions[idx] >= params->sd_random_interactions[idx] * params->sd_random_interactions[idx] )
				print_exit( "BAD_PARAM - sd_random_interations_xxxx - variance must be greater than the mean for (negative binomial distribution");
}

/*****************************************************************************************
*  Name:        set_model_param_fatality_fraction
*  Description: Allow updates of the fatality fraction
******************************************************************************************/
int set_model_param_fatality_fraction(model * model, double value, int age_group)
{
	if (age_group >= N_AGE_GROUPS) return ERROR;
	model->params->fatality_fraction[age_group] = value;
	return TRUE;
}



/*****************************************************************************************
*  Name:        check_hospital_params
*  Description: Carries out checks on the input parameters
******************************************************************************************/
void check_hospital_params( parameters *params )
{
    if( params->n_hcw_per_ward[COVID_GENERAL][DOCTOR] < 1 )
        print_exit( "BAD PARAM n_doctors_covid_general_ward cant be less than 1");

    if( params->n_hcw_per_ward[COVID_GENERAL][NURSE] < 1 )
        print_exit( "BAD PARAM n_nurses_covid_general_ward cant be less than 1");

    if( params->n_hcw_per_ward[COVID_ICU][DOCTOR] < 1 )
        print_exit( "BAD PARAM n_doctors_covid_icu_ward cant be less than 1");

    if( params->n_hcw_per_ward[COVID_ICU][NURSE] < 1 )
        print_exit( "BAD PARAM n_nurses_covid_icu_ward cant be less than 1");

    int general_doctors = params->n_hcw_per_ward[COVID_GENERAL][DOCTOR] * params->n_wards[COVID_GENERAL] * params->n_hospitals;
    int general_nurses = params->n_hcw_per_ward[COVID_GENERAL][NURSE] * params->n_wards[COVID_GENERAL] * params->n_hospitals;
    int icu_doctors = params->n_hcw_per_ward[COVID_ICU][DOCTOR] * params->n_wards[COVID_ICU] * params->n_hospitals;
    int icu_nurses = params->n_hcw_per_ward[COVID_ICU][NURSE] * params->n_wards[COVID_ICU] * params->n_hospitals;

    int total_number_hcw = general_nurses + general_doctors + icu_nurses + icu_doctors;

    if( total_number_hcw > (params->n_total - params->n_seed_infection)/2 )
        print_exit( "BAD PARAMS number of healthcare workers is greater than half the total population. Change number of wards / worker per ward" );

    if( params->hcw_mean_work_interactions > total_number_hcw/2 )
        print_exit( "BAD PARAM hcw_mean_work_interactions must be less than or equal to half the total number of healthcare workers" );
}

void destroy_occupation_network_table(parameters *params)
{
    for (int i = 0; i != params->occupation_network_table->n_networks; ++i)
        free(params->occupation_network_table->network_names[i]);
    free(params->occupation_network_table->network_names);
    free(params->occupation_network_table->network_ids);
    free(params->occupation_network_table->mean_interactions);
    free(params->occupation_network_table->age_type);
    free(params->occupation_network_table->network_no);
}

/*****************************************************************************************
*  Name:        destroy_params
*  Description: Destroy the parameters
******************************************************************************************/
void destroy_params( parameters *params )
{
	int idx;
	for( idx = 0; idx < params->N_REFERENCE_HOUSEHOLDS; idx++ )
		free( params->REFERENCE_HOUSEHOLDS[idx] );
	free( params->REFERENCE_HOUSEHOLDS );

	if( params->demo_house != NULL )
	{
		free( params->demo_house->age_group );
		free( params->demo_house->house_no );
		free( params->demo_house->idx );
	}

	if ( params->occupation_network_table != NULL )
	    destroy_occupation_network_table(params);

}
