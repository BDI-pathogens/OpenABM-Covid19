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
*  Name: 		get_param_test_on_symptoms
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_test_on_symptoms( model *model )
{
    return model->params->test_on_symptoms;
}

/*****************************************************************************************
*  Name:		get_param_test_on_traced
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_test_on_traced( model *model )
{
    return model->params->test_on_traced;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_on_traced
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_on_traced( model *model )
{
    return model->params->quarantine_on_traced;
}

/*****************************************************************************************
*  Name:		get_param_traceable_interaction_fraction
*  Description: Gets the value of an int parameter
******************************************************************************************/
double get_param_traceable_interaction_fraction( model *model )
{
    return model->params->traceable_interaction_fraction;
}

/*****************************************************************************************
*  Name:		get_param_tracing_network_depth
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_tracing_network_depth( model *model )
{
    return model->params->tracing_network_depth;
}

/*****************************************************************************************
*  Name:		get_param_allow_clinical_diagnosis
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_allow_clinical_diagnosis( model *model )
{
    return model->params->allow_clinical_diagnosis;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_household_on_positive(
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_household_on_positive( model *model )
{
    return model->params->quarantine_household_on_positive;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_household_on_symptoms
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_household_on_symptoms( model *model )
{
    return model->params->quarantine_household_on_symptoms;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_household_on_traced
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_household_on_traced( model *model )
{
    return model->params->quarantine_household_on_traced;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_household_contacts_on_positive
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_household_contacts_on_positive( model *model )
{
    return model->params->quarantine_household_contacts_on_positive;
}

/*****************************************************************************************
*  Name:		get_param_quarantine_days
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_quarantine_days( model *model )
{
    return model->params->quarantine_days;
}

/*****************************************************************************************
*  Name:		get_param_test_order_wait
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_test_order_wait( model *model )
{
    return model->params->test_order_wait;
}

/*****************************************************************************************
*  Name:		get_param_test_result_wait
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_test_result_wait( model *model )
{
    return model->params->test_result_wait;
}

/*****************************************************************************************
*  Name:		get_param_self_quarantine_fraction
*  Description: Gets the value of an int parameter
******************************************************************************************/
double get_param_self_quarantine_fraction( model *model )
{
    return model->params->self_quarantine_fraction;
}

/*****************************************************************************************
*  Name:		get_param_lockdown_on
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_lockdown_on( model *model )
{
    return model->params->lockdown_on;
}

/*****************************************************************************************
*  Name:		get_param_app_turned_on
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_app_turned_on( model *model )
{
    return model->params->app_turned_on;
}

/*****************************************************************************************
*  Name:		get_param_app_users_fraction
*  Description: Gets the value of double parameter
******************************************************************************************/
double get_param_app_users_fraction( model *model )
{
    return model->params->app_users_fraction;
}

/*****************************************************************************************
*  Name:        set_param_test_on_symptoms
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_test_on_symptoms( model *model, int value )
{
   model->params->test_on_symptoms = value;
   return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_test_on_traced
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_test_on_traced( model *model, int value )
{
    model->params->test_on_traced = value;
    return TRUE;
}

/*****************************************************************************************
*  Name: 		set_param_quarantine_on_traced
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_quarantine_on_traced( model *model, int value )
{
    model->params->quarantine_on_traced = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_traceable_interaction_fractio
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_traceable_interaction_fraction( model *model, double value )
{
    model->params->traceable_interaction_fraction = value;
    return TRUE;
}
/*****************************************************************************************
*  Name:		set_param_tracing_network_depth
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_tracing_network_depth( model *model, int value )
{
    model->params->tracing_network_depth = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_allow_clinical_diagnosis
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_allow_clinical_diagnosis( model *model, int value )
{
    model->params->allow_clinical_diagnosis = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_quarantine_household_on_positive
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_quarantine_household_on_positive( model *model, int value )
{
    model->params->quarantine_household_on_positive = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_quarantine_household_on_symptoms
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_quarantine_household_on_symptoms( model *model, int value )
{
    model->params->quarantine_household_on_symptoms = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_quarantine_household_on_traced
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_quarantine_household_on_traced( model *model, int value )
{
    model->params->quarantine_household_on_traced = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_quarantine_household_contacts_on_positive
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_quarantine_household_contacts_on_positive( model *model, int value )
{
    model->params->quarantine_household_contacts_on_positive = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_quarantine_days
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_quarantine_days( model *model, int value )
{
    model->params->quarantine_days = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_test_order_wait
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_test_order_wait( model *model, int value )
{
    model->params->test_order_wait = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_test_result_wait
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_test_result_wait( model *model, int value )
{
    model->params->test_result_wait = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_self_quarantine_fraction(model *model, double value)
{
    model->params->self_quarantine_fraction = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_lockdown_on
*  Description: Carries out checks on the input parameters
******************************************************************************************/
int set_param_lockdown_on( model *model, int value )
{
	int pdx;
	parameters *params = model->params;

	if( value == TRUE )
	{
		params->lockdown_on = TRUE;
		params->daily_fraction_work_used = params->daily_fraction_work * params->lockdown_work_network_multiplier;

		params->relative_transmission_by_type_used[HOUSEHOLD] = params->relative_transmission_by_type[HOUSEHOLD] *
																params->lockdown_house_interaction_multiplier;
		set_up_infectious_curves( model );

		for( pdx = 0; pdx < params->n_total; pdx++ )
			update_random_interactions( &(model->population[pdx]), params );
	}
	else
	if( value == FALSE )
	{
		params->lockdown_on = FALSE;
		params->daily_fraction_work_used = params->daily_fraction_work;

		params->relative_transmission_by_type_used[HOUSEHOLD] = params->relative_transmission_by_type[HOUSEHOLD];
		set_up_infectious_curves( model );

		for( pdx = 0; pdx < params->n_total; pdx++ )
			update_random_interactions( &(model->population[pdx]), params );
	}
	else
		return FALSE;

	return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_app_turned_on
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_app_turned_on( model *model, int value )
{
    model->params->app_turned_on = value;
    return TRUE;
}

/*****************************************************************************************
*  Name:		set_param_app_users_fraction
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_app_users_fraction( model *model, double value )
{
    if( value > 1 || value < model->params->app_users_fraction )
    	return FALSE;

    if( value == model->params->app_users_fraction )
    	return TRUE;

	model->params->app_users_fraction = value;
	set_up_app_users( model, value );
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
