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

/*****************************************************************************************
*  Name:        get_param_x
*  Description: Gets the value of an int parameter
******************************************************************************************/
int get_param_test_on_symptoms(model *model)
{
    return model->params->test_on_symptoms;
}

int get_param_test_on_traced(model *model)
{
    return model->params->test_on_traced;
}

int get_param_quarantine_on_traced(model *model)
{
    return model->params->quarantine_on_traced;
}

double get_param_traceable_interaction_fraction(model *model)
{
    return model->params->traceable_interaction_fraction;
}

int get_param_tracing_network_depth(model *model)
{
    return model->params->tracing_network_depth;
}

int get_param_allow_clinical_diagnosis(model *model)
{
    return model->params->allow_clinical_diagnosis;
}

int get_param_quarantine_household_on_positive(model *model)
{
    return model->params->quarantine_household_on_positive;
}

int get_param_quarantine_household_on_symptoms(model *model)
{
    return model->params->quarantine_household_on_symptoms;
}

int get_param_quarantine_household_on_traced(model *model)
{
    return model->params->quarantine_household_on_traced;
}

int get_param_quarantine_household_contacts_on_positive(model *model)
{
    return model->params->quarantine_household_contacts_on_positive;
}

int get_param_quarantine_days(model *model)
{
    return model->params->quarantine_days;
}

int get_param_test_order_wait(model *model)
{
    return model->params->test_order_wait;
}

int get_param_test_result_wait(model *model)
{
    return model->params->test_result_wait;
}

double get_param_self_quarantine_fraction(model *model)
{
    return model->params->self_quarantine_fraction;
}

/*****************************************************************************************
*  Name:        set_param_x
*  Description: Sets the value of x parameter
******************************************************************************************/
int set_param_test_on_symptoms(model *model, int value) {
   model->params->test_on_symptoms = value;
   return TRUE;
}

int set_param_test_on_traced(model *model, int value) {
    model->params->test_on_traced = value;
    return TRUE;
}

int set_param_quarantine_on_traced(model *model, int value) {
    model->params->quarantine_on_traced = value;
    return TRUE;
}

int set_param_traceable_interaction_fraction(model *model, double value) {
    model->params->traceable_interaction_fraction = value;
    return TRUE;
}

int set_param_tracing_network_depth(model *model, int value) {
    model->params->tracing_network_depth = value;
    return TRUE;
}

int set_param_allow_clinical_diagnosis(model *model, int value) {
    model->params->allow_clinical_diagnosis = value;
    return TRUE;
}

int set_param_quarantine_household_on_positive(model *model, int value) {
    model->params->quarantine_household_on_positive = value;
    return TRUE;
}

int set_param_quarantine_household_on_symptoms(model *model, int value) {
    model->params->quarantine_household_on_symptoms = value;
    return TRUE;
}

int set_param_quarantine_household_on_traced(model *model, int value) {
    model->params->quarantine_household_on_traced = value;
    return TRUE;
}

int set_param_quarantine_household_contacts_on_positive(model *model, int value) {
    model->params->quarantine_household_contacts_on_positive = value;
    return TRUE;
}

int set_param_quarantine_days(model *model, int value) {
    model->params->quarantine_days = value;
    return TRUE;
}

int set_param_test_order_wait(model *model, int value) {
    model->params->test_order_wait = value;
    return TRUE;
}

int set_param_test_result_wait(model *model, int value) {
    model->params->test_result_wait = value;
    return TRUE;
}

int set_param_self_quarantine_fraction(model *model, double value) {
    model->params->self_quarantine_fraction = value;
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

    if( params->social_distancing_time_on < 1 )
      	print_exit( "BAD PARAM social_distancing_time_on - can only be turned on at the first time step" );

    if( params->random_interaction_distribution != FIXED && params->random_interaction_distribution != NEGATIVE_BINOMIAL )
 	   print_exit( "BAR_PARAM - random_interaction_distribution - only fixed and negative-binomial distributions are supported" );

    if( params->random_interaction_distribution == NEGATIVE_BINOMIAL )
		for( idx = 0; idx < N_AGE_TYPES; idx++ )
			if( params->mean_random_interactions[idx] >= params->sd_random_interactions[idx] * params->sd_random_interactions[idx] )
				print_exit( "BAD_PARAM - sd_random_interations_xxxx - variance must be greater than the mean for (negative binomial distribution");
}

void destroy_params( parameters *params)
{
	int idx;
	for(idx=0; idx < params->N_REFERENCE_HOUSEHOLDS; idx++)
		free( params->REFERENCE_HOUSEHOLDS[idx] );
	free( params->REFERENCE_HOUSEHOLDS );
}
