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
*  Name:		get_model_param_quarantine_household_on_traced
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_quarantine_household_on_traced(model *model)
{
    return model->params->quarantine_household_on_traced;
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
*  Name:		get_model_param_test_on_symptoms
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_model_param_test_on_symptoms(model *model)
{
    return model->params->test_on_symptoms;
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
*  Name:        get_model_param_lockdown_work_network_multiplier
*  Description: Gets the value of a double parameter
******************************************************************************************/
double get_model_param_lockdown_work_network_multiplier(model *model)
{
	return model->params->lockdown_work_network_multiplier;
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
*  Name:		set_model_param_quarantine_household_on_traced
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_quarantine_household_on_traced( model *model, int value )
{
    model->params->quarantine_household_on_traced = value;
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
*  Name:		set_model_param_test_result_wait
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_test_result_wait( model *model, int value )
{
    model->params->test_result_wait = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_test_order_wait
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_test_order_wait( model *model, int value )
{
    model->params->test_order_wait = value;
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
*  Name:		set_model_param_app_turned_on
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_app_turned_on( model *model, int value )
{
    model->params->app_turned_on = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_model_param_lockdown_on
*  Description: Carries out checks on the input parameters
******************************************************************************************/
int set_model_param_lockdown_on( model *model, int value )
{
	long pdx;
	int network;
	parameters *params = model->params;

	if( value == TRUE )
	{
		for( network = 0; network < N_WORK_NETWORKS; network++ )
			params->daily_fraction_work_used[network] = params->daily_fraction_work *
														params->lockdown_work_network_multiplier;


        params->relative_transmission_used[HOUSEHOLD] = params->relative_transmission[HOUSEHOLD] * params->lockdown_house_interaction_multiplier;
	}
	else
	if( value == FALSE )
	{
		if( !params->lockdown_on )
			return TRUE;

		for( network = 0; network < N_WORK_NETWORKS; network++ )
			if( !( NETWORK_TYPE_MAP[ network ] == NETWORK_TYPE_ELDERLY && params->lockdown_elderly_on ) )
				params->daily_fraction_work_used[network] = params->daily_fraction_work;


		params->relative_transmission_used[HOUSEHOLD] = params->relative_transmission[HOUSEHOLD];
	}
	else
		return FALSE;

	params->lockdown_on = value;
	set_up_infectious_curves( model );

	for( pdx = 0; pdx < params->n_total; pdx++ )
		update_random_interactions( &(model->population[pdx]), params );

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

	if( value == TRUE )
	{
		for( network = 0; network < N_WORK_NETWORKS; network++ )
			if( NETWORK_TYPE_MAP[ network ] == NETWORK_TYPE_ELDERLY )
				params->daily_fraction_work_used[ network ] = params->daily_fraction_work *
															  params->lockdown_work_network_multiplier;
	}
	else
	if( value == FALSE )
	{
		if( !params->lockdown_elderly_on )
			return TRUE;

		if( !params->lockdown_on )
		{
			for( network = 0; network < N_WORK_NETWORKS; network++ )
				params->daily_fraction_work_used[ network ] = params->daily_fraction_work;
		}

	}else
		return FALSE;

	params->lockdown_elderly_on = value;
	set_up_infectious_curves( model );

	for( pdx = 0; pdx < params->n_total; pdx++ )
	{
		indiv = &(model->population[pdx]);
		if( indiv->age_type == AGE_TYPE_ELDERLY )
			update_random_interactions( indiv, params );
	}

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
*  Name:        set_model_param_lockdown_work_network_multiplier
*  Description: Sets the value of parameter
******************************************************************************************/
int set_model_param_lockdown_work_network_multiplier( model *model, double value )
{
	model->params->lockdown_work_network_multiplier = value;

	if( model->params->lockdown_on )
		return set_model_param_lockdown_on( model, TRUE );

	if( model->params->lockdown_elderly_on )
		return set_model_param_lockdown_elderly_on( model, TRUE );

	return TRUE;
}

/*****************************************************************************************
*  Name:		check_params
*  Description: Carries out checks on the input parameters
******************************************************************************************/
void check_params( parameters *params )
{
	int idx;

	if( params->days_of_interactions > MAX_DAILY_INTERACTIONS_KEPT )
    	print_exit( "BAD PARAM day_of_interaction - can't be greater than MAX_DAILY_INTERACTIONS " );

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
*  Name:		destroy_params
*  Description: Destroy the parameters
******************************************************************************************/
void destroy_params( parameters *params )
{
	int idx;
	for( idx = 0; idx < params->N_REFERENCE_HOUSEHOLDS; idx++ )
		free( params->REFERENCE_HOUSEHOLDS[idx] );
	free( params->REFERENCE_HOUSEHOLDS );
}
