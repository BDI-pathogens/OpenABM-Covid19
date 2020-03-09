/*
 * input.c
 *
 *  Created on: 6 Mar 2020
 *      Author: p-robot
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "input.h"
#include "params.h"
#include "utilities.h"
#include "constant.h"

/*****************************************************************************************
*  Name:		read_command_line_args
*  Description: Read command-line arguments and attach to params struct
******************************************************************************************/
void read_command_line_args( parameters *params, int argc, char **argv )
{
	int param_line_number;
	char input_param_file[ INPUT_CHAR_LEN ];
	
	if(argc > 1)
	{
		strncpy(input_param_file, argv[1], INPUT_CHAR_LEN );
	}else{
		strncpy(input_param_file, "../tests/data/test_parameters.csv", INPUT_CHAR_LEN );
	}
	
	if(argc > 2)
	{
		param_line_number = (int) strtol(argv[2], NULL, 10);
		
		if(param_line_number <= 0)
			print_exit("Error Invalid line number, line number starts from 1");
	}else{
		param_line_number = 1;
	}
	
	// Attach to params struct, ensure string is null-terminated
	params->param_line_number = param_line_number;
	strncpy(params->input_param_file, input_param_file, sizeof(params->input_param_file) - 1);
	params->input_param_file[sizeof(params->input_param_file) - 1] = '\0';
}

/*****************************************************************************************
*  Name:		read_param_file
*  Description: Read line from parameter file (csv), attach parame values to params struct
******************************************************************************************/
void read_param_file( parameters *params)
{
	FILE *parameter_file;
	int i, check;
	
	parameter_file = fopen(params->input_param_file, "r");
	if(parameter_file == NULL)
		print_exit("Can't open parameter file");
	
	// Throw away header (and first `params->param_line_number` lines)
	for(i = 0; i < params->param_line_number; i++)
		fscanf(parameter_file, "%*[^\n]\n");
	
	// Read and attach parameter values to parameter structure
	check = fscanf(parameter_file, " %li ,", &(params->rng_seed));
	if( check < 1){ print_exit("Failed to read parameter rng_seed\n"); };
	
	check = fscanf(parameter_file, " %li ,", &(params->param_id));
	if( check < 1){ print_exit("Failed to read parameter param_id\n"); };
	
	check = fscanf(parameter_file, " %li ,", &(params->n_total));
	if( check < 1){ print_exit("Failed to read parameter n_total\n"); };
	
	check = fscanf(parameter_file, " %i ,",  &(params->mean_daily_interactions));
	if( check < 1){ print_exit("Failed to read parameter mean_daily_interactions\n"); };
	
	check = fscanf(parameter_file, " %i ,",  &(params->days_of_interactions));
	if( check < 1){ print_exit("Failed to read parameter days_of_interactions\n"); };
	
	check = fscanf(parameter_file, " %i ,",  &(params->end_time));
	if( check < 1){ print_exit("Failed to read parameter end_time\n"); };
	
	check = fscanf(parameter_file, " %i ,",  &(params->n_seed_infection));
	if( check < 1){ print_exit("Failed to read parameter n_seed_infection\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->mean_infectious_period));
	if( check < 1){ print_exit("Failed to read parameter mean_infectious_period\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->sd_infectious_period));
	if( check < 1){ print_exit("Failed to read parameter sd_infectious_period\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->infectious_rate));
	if( check < 1){ print_exit("Failed to read parameter infectious_rate\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->mean_time_to_symptoms));
	if( check < 1){ print_exit("Failed to read parameter mean_time_to_symptoms\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->sd_time_to_symptoms));
	if( check < 1){ print_exit("Failed to read parameter sd_time_to_symptoms\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->mean_time_to_hospital));
	if( check < 1){ print_exit("Failed to read parameter mean_time_to_hospital\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->mean_time_to_recover));
	if( check < 1){ print_exit("Failed to read parameter mean_time_to_recover\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->sd_time_to_recover));
	if( check < 1){ print_exit("Failed to read parameter sd_time_to_recover\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->mean_time_to_death));
	if( check < 1){ print_exit("Failed to read parameter mean_time_to_death\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->sd_time_to_death));
	if( check < 1){ print_exit("Failed to read parameter sd_time_to_death\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->cfr));
	if( check < 1){ print_exit("Failed to read parameter cfr\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->fraction_asymptomatic));
	if( check < 1){ print_exit("Failed to read parameter fraction_asymptomatic\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->asymptomatic_infectious_factor));
	if( check < 1){ print_exit("Failed to read parameter asymptomatic_infectious_factor\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->mean_asymptomatic_to_recovery));
	if( check < 1){ print_exit("Failed to read parameter mean_asymptomatic_to_recovery\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->sd_asymptomatic_to_recovery));
	if( check < 1){ print_exit("Failed to read parameter sd_asymptomatic_to_recovery\n"); };

	check = fscanf(parameter_file, " %i  ,", &(params->quarantined_daily_interactions));
	if( check < 1){ print_exit("Failed to read parameter quarantined_daily_interactions\n"); };
	
	check = fscanf(parameter_file, " %i  ,", &(params->quarantine_days));
	if( check < 1){ print_exit("Failed to read parameter quarantine_days\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->quarantine_fraction));
	if( check < 1){ print_exit("Failed to read parameter quarantine_fraction\n"); };
	
	check = fscanf(parameter_file, " %i  ,", &(params->hospitalised_daily_interactions));
	if( check < 1){ print_exit("Failed to read parameter hospitalised_daily_interactions\n"); };
	
	check = fscanf(parameter_file, " %i ",   &(params->test_insensititve_period));
	if( check < 1){ print_exit("Failed to read parameter test_insensititve_period\n"); };

	fclose(parameter_file);
}
