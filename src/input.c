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
#include "model.h"
#include "utilities.h"
#include "constant.h"

/*****************************************************************************************
*  Name:		read_command_line_args
*  Description: Read command-line arguments and attach to params struct
******************************************************************************************/
void read_command_line_args( parameters *params, int argc, char **argv )
{
	int param_line_number;
	char input_param_file[ INPUT_CHAR_LEN ], output_file_dir[ INPUT_CHAR_LEN ];
	
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
	
	if(argc > 3)
	{
		strncpy(output_file_dir, argv[3], INPUT_CHAR_LEN );
	}else{
		strncpy(output_file_dir, ".", INPUT_CHAR_LEN );
	}	
	
	// Attach to params struct, ensure string is null-terminated
	params->param_line_number = param_line_number;
	strncpy(params->input_param_file, input_param_file, sizeof(params->input_param_file) - 1);
	params->input_param_file[sizeof(params->input_param_file) - 1] = '\0';
	
	strncpy(params->output_file_dir, output_file_dir, sizeof(params->output_file_dir) - 1);
	params->output_file_dir[sizeof(params->output_file_dir) - 1] = '\0';
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
	check = fscanf(parameter_file, " %li ,", &(params->param_id));
	check = fscanf(parameter_file, " %li ,", &(params->n_total));
	check = fscanf(parameter_file, " %i ,",  &(params->mean_daily_interactions));
	check = fscanf(parameter_file, " %i ,",  &(params->days_of_interactions));
	check = fscanf(parameter_file, " %i ,",  &(params->end_time));
	check = fscanf(parameter_file, " %i ,",  &(params->n_seed_infection));
	check = fscanf(parameter_file, " %lf ,", &(params->mean_infectious_period));
	check = fscanf(parameter_file, " %lf ,", &(params->sd_infectious_period));
	check = fscanf(parameter_file, " %lf ,", &(params->infectious_rate));
	check = fscanf(parameter_file, " %lf ,", &(params->mean_time_to_symptoms));
	check = fscanf(parameter_file, " %lf ,", &(params->sd_time_to_symptoms));
	check = fscanf(parameter_file, " %lf ,", &(params->mean_time_to_hospital));
	check = fscanf(parameter_file, " %lf ,", &(params->mean_time_to_recover));
	check = fscanf(parameter_file, " %lf ,", &(params->sd_time_to_recover));
	check = fscanf(parameter_file, " %lf ,", &(params->mean_time_to_death));
	check = fscanf(parameter_file, " %lf ,", &(params->sd_time_to_death));
	check = fscanf(parameter_file, " %lf ,", &(params->cfr));
	check = fscanf(parameter_file, " %lf ,", &(params->fraction_asymptomatic));
	check = fscanf(parameter_file, " %lf ,", &(params->asymptomatic_infectious_factor));
	check = fscanf(parameter_file, " %lf ,", &(params->mean_asymptomatic_to_recovery));
	check = fscanf(parameter_file, " %lf ,", &(params->sd_asymptomatic_to_recovery));
	check = fscanf(parameter_file, " %i ," , &(params->quarantined_daily_interactions));
	check = fscanf(parameter_file, " %i ," , &(params->quarantine_days));
	check = fscanf(parameter_file, " %lf ,", &(params->quarantine_fraction));
	check = fscanf(parameter_file, " %i",    &(params->hospitalised_daily_interactions));

	fclose(parameter_file);
}

/*****************************************************************************************
*  Name:		write_individual_file
*  Description: Write (csv) file of individuals in simulation
******************************************************************************************/


void write_individual_file(model *model, parameters *params)
{
	
	char output_file[INPUT_CHAR_LEN];
	FILE *individual_output_file;
	int idx;
	long infector_id;
	
	char param_line_number[10];
	sprintf(param_line_number, "%d", params->param_line_number);
	
	// Concatenate file name
    strcpy(output_file, params->output_file_dir);
    strcat(output_file, "/individual_file_Run");
	strcat(output_file, param_line_number);
	strcat(output_file, ".csv");
	
	printf("%s", output_file);
	
	individual_output_file = fopen(output_file, "w");
	if(individual_output_file == NULL)
		print_exit("Can't open individual output file");
	
	fprintf(individual_output_file,"ID, ");
	fprintf(individual_output_file,"current_status, ");
	fprintf(individual_output_file,"quarantined, ");
	fprintf(individual_output_file,"hazard, ");
	fprintf(individual_output_file,"mean_interactions, ");
	fprintf(individual_output_file,"time_infected, ");
	fprintf(individual_output_file,"time_symptomatic, ");
	fprintf(individual_output_file,"time_asymptomatic, ");
	fprintf(individual_output_file,"time_hospitalised, ");
	fprintf(individual_output_file,"time_death, ");
	fprintf(individual_output_file,"time_recovered, ");
	fprintf(individual_output_file,"next_event_type, ");
	fprintf(individual_output_file,"ID_infector"); // no trailing comma on last entry
	fprintf(individual_output_file,"\n");
	
	// Loop through all individuals in the simulation
	for(idx = 0; idx < params->n_total; idx++)
	{
		
		/* Check the individual was infected during the simulation
		(otherwise the "infector" attribute does not point to another individual) */
		if(model->population[idx].status != UNINFECTED){
			infector_id = model->population[idx].infector->idx;
		}else{
			infector_id = UNKNOWN;
		}
		
		fprintf(individual_output_file, 
			"%li, %d, %d, %f, %d, %d, %d, %d, %d, %d, %d, %d, %li\n",
			model->population[idx].idx,
			model->population[idx].status,
			model->population[idx].quarantined,
			model->population[idx].hazard,
			model->population[idx].mean_interactions,
			model->population[idx].time_infected,
			model->population[idx].time_symptomatic,
			model->population[idx].time_asymptomatic,
			model->population[idx].time_hospitalised,
			model->population[idx].time_death,
			model->population[idx].time_recovered,
			model->population[idx].next_event_type,
			infector_id
			);
	}
	fclose(individual_output_file);
}
