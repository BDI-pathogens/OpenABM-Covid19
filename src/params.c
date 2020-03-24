/*
 * params.c
 *
 *  Created on: 7 Mar 2020
 *      Author: hinchr
 */

#include "params.h"
#include "constant.h"
#include "utilities.h"

/*****************************************************************************************
*  Name:		check_params
*  Description: Carries out checks on the input parameters
******************************************************************************************/
void check_params( parameters *params )
{
	if( params->days_of_interactions > MAX_DAILY_INTERACTIONS_KEPT )
    	print_exit( "asking for day_of_interaction to be greater than MAX_DAILY_INTERACTIONS " );

    if( params->end_time > MAX_TIME )
     	print_exit( "asking for end_time to be greater than MAX_TIME " );

    if( params->mean_time_to_hospital > 2 )
    	print_exit( "maximum time from symptoms to hospital is 2 days" );

    if( params->quarantine_days > params->days_of_interactions )
    	print_exit( "can only quarantine up to the number of days we store" );

    if( params->social_distancing_time_on < 1 )
      	print_exit( "social distancing can only be turned on at the first time step" );
}

void destroy_params( parameters *params)
{
	int idx;
	for(idx=0; idx < params->N_REFERENCE_HOUSEHOLDS; idx++)
		free( params->REFERENCE_HOUSEHOLDS[idx] );
	free( params->REFERENCE_HOUSEHOLDS );
}
