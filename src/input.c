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
#include "demographics.h"
#include "interventions.h"

/*****************************************************************************************
*  Name:		read_command_line_args
*  Description: Read command-line arguments and attach to params struct
******************************************************************************************/
void read_command_line_args( parameters *params, int argc, char **argv )
{
	int param_line_number;
	char input_param_file[ INPUT_CHAR_LEN ];
	char input_household_file [INPUT_CHAR_LEN ];
	char output_file_dir[ INPUT_CHAR_LEN ];
	
	if(argc > 1)
	{
		strncpy(input_param_file, argv[1], INPUT_CHAR_LEN );
	}else{
		strncpy(input_param_file, "../tests/data/baseline_parameters.csv", INPUT_CHAR_LEN );
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
		params->sys_write_individual = TRUE;

	}else{
		strncpy(output_file_dir, ".", INPUT_CHAR_LEN );
		params->sys_write_individual = FALSE;
	}

	if(argc > 4)
	{
		strncpy(input_household_file, argv[4], INPUT_CHAR_LEN );
	}else{
		strncpy(input_household_file, "../tests/data/baseline_household_demographics.csv",
			INPUT_CHAR_LEN );
	}

	// Attach to params struct, ensure string is null-terminated
	params->param_line_number = param_line_number;
	
	strncpy(params->input_param_file, input_param_file, sizeof(params->input_param_file) - 1);
	params->input_param_file[sizeof(params->input_param_file) - 1] = '\0';
	
	strncpy(params->input_household_file, input_household_file, 
		sizeof(params->input_household_file) - 1);
	params->input_household_file[sizeof(params->input_household_file) - 1] = '\0';
	
	strncpy(params->output_file_dir, output_file_dir, sizeof(params->output_file_dir) - 1);
	params->output_file_dir[sizeof(params->output_file_dir) - 1] = '\0';
}

/*****************************************************************************************
*  Name:		load_parameters_file_to_buffer
*  Description: Return a buffer with the data from the parameters file
******************************************************************************************/
char *load_parameters_file_to_buffer( char *file_name )
{
	FILE *file = NULL;
	int ret = 0;
    long file_size = 0;
    char *buffer = NULL;

    file = fopen(file_name, "r");
	if(file == NULL)
		print_exit("Can't open parameter file");

    /* Go to the end of the file */
    ret = fseek(file, 0, SEEK_END);
    if (ret) print_exit("Can't get to end of file");

    /* Get the size */
    file_size = ftell(file);
    if (file_size < 0) print_exit("Can't get size of file");

    if (file_size >= INPUT_BUFFER_LEN) print_exit("Input buffer is too big");

   /* Go back to the start */
    ret = fseek(file, 0, SEEK_SET);
    if (ret) print_exit("Can't get to start of file");

    buffer = malloc(file_size + 1);
    if (buffer == NULL) print_exit("Unable to allocate memory");

    /* Read file to buffer */
    ret = fread(buffer, 1, file_size, file);
    buffer[file_size] = '\0';

    fclose(file);

    return buffer;
}

/*****************************************************************************************
*  Name:		load_household_demographics_file_to_buffer
*  Description: Return a buffer with the data from the household demographics file
******************************************************************************************/
char *load_household_demographics_file_to_buffer( parameters *params, char *file_name )
{
	FILE *file = NULL;
	int ret = 0;
    long lines = 0, file_size = 0;
    char *buffer = NULL;
    char ch;

    file = fopen(file_name, "r");
	if(file == NULL)
		print_exit("Can't open parameter file");

    /* Save file size for future memory allocation */
    while ((ch = fgetc(file)) != EOF)
        if (ch == '\n') lines++;

    params->N_REFERENCE_HOUSEHOLDS = lines - 1;
    
    /* Go to the end of the file */
    ret = fseek(file, 0, SEEK_END);
    if (ret) print_exit("Can't get to end of file");

    /* Get the size */
    file_size = ftell(file);
    if (file_size < 0) print_exit("Can't get size of file");

    if (file_size >= INPUT_BUFFER_LEN) print_exit("Input buffer is too big");

   /* Go back to the start */
    ret = fseek(file, 0, SEEK_SET);
    if (ret) print_exit("Can't get to start of file");

    buffer = malloc(file_size + 1);
    if (buffer == NULL) print_exit("Unable to allocate memory");

    /* Read file to buffer */
    ret = fread(buffer, 1, file_size, file);
    buffer[file_size] = '\0';

    fclose(file);

    return buffer;
}
/*****************************************************************************************
*  Name:		update_params_from_buffer
*  Description: Update parameters from csv formatted buffer
******************************************************************************************/
void update_params_from_buffer( parameters *params, char *buffer )
{
    int i = 0;
	int check = 0;
    char *line, *token;

    if (buffer == NULL) print_exit("Empty buffer");

	/* Throw away header (and first `params->param_line_number` lines) */
    line = strtok(buffer, "\n");
	for(i = 0; i < params->param_line_number; i++)
        line = strtok(NULL, "\n");

	// Read and attach parameter values to parameter structure
    token = strtok(line, ",");
	check = sscanf(token, "%li", &(params->rng_seed));
	if( check < 1){ print_exit("Failed to read parameter rng_seed\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%li", &(params->param_id));
	if( check < 1){ print_exit("Failed to read parameter param_id\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%li", &(params->n_total));
	if( check < 1){ print_exit("Failed to read parameter n_total\n"); };

	for( i = 0; i < N_WORK_NETWORK_TYPES; i++ )
	{
        token = strtok(NULL, ",");
		check = sscanf(token, "%lf",  &(params->mean_work_interactions[i]));
		if( check < 1){ print_exit("Failed to read parameter mean_work_interactions\n"); };
	}

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf",  &(params->daily_fraction_work));
	if( check < 1){ print_exit("Failed to read parameter daily_fraction_work\n"); };

	for( i = 0; i < N_AGE_TYPES; i++ )
	{
        token = strtok(NULL, ",");
		check = sscanf(token, "%lf",  &(params->mean_random_interactions[i]));
		if( check < 1){ print_exit("Failed to read parameter mean_daily_interactions\n"); };

        token = strtok(NULL, ",");
		check = sscanf(token, "%lf",  &(params->sd_random_interactions[i]));
		if( check < 1){ print_exit("Failed to read parameter sd_daily_interactions\n"); };
	}

    token = strtok(NULL, ",");
	check = sscanf(token, "%i",  &(params->random_interaction_distribution));
	if( check < 1){ print_exit("Failed to read parameter random_interaction_distribution\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf",  &(params->child_network_adults));
	if( check < 1){ print_exit("Failed to read parameter child_network_adults\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf",  &(params->elderly_network_adults));
	if( check < 1){ print_exit("Failed to read parameter elderly_network_adults\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i",  &(params->days_of_interactions));
	if( check < 1){ print_exit("Failed to read parameter days_of_interactions\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, " %i ,",  &(params->end_time));
	if( check < 1){ print_exit("Failed to read parameter end_time\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i",  &(params->n_seed_infection));
	if( check < 1){ print_exit("Failed to read parameter n_seed_infection\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->mean_infectious_period));
	if( check < 1){ print_exit("Failed to read parameter mean_infectious_period\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->sd_infectious_period));
	if( check < 1){ print_exit("Failed to read parameter sd_infectious_period\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->infectious_rate));
	if( check < 1){ print_exit("Failed to read parameter infectious_rate\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->mean_time_to_symptoms));
	if( check < 1){ print_exit("Failed to read parameter mean_time_to_symptoms\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->sd_time_to_symptoms));
	if( check < 1){ print_exit("Failed to read parameter sd_time_to_symptoms\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->mean_time_to_hospital));
	if( check < 1){ print_exit("Failed to read parameter mean_time_to_hospital\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->mean_time_to_critical));
	if( check < 1){ print_exit("Failed to read parameter mean_time_to_critical\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->mean_time_to_recover));
	if( check < 1){ print_exit("Failed to read parameter mean_time_to_recover\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->sd_time_to_recover));
	if( check < 1){ print_exit("Failed to read parameter sd_time_to_recover\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->mean_time_to_death));
	if( check < 1){ print_exit("Failed to read parameter mean_time_to_death\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->sd_time_to_death));
	if( check < 1){ print_exit("Failed to read parameter sd_time_to_death\n"); };

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
        token = strtok(NULL, ",");
		check = sscanf(token, "%lf", &(params->fraction_asymptomatic[i]));
		if( check < 1){ print_exit("Failed to read parameter fraction_asymptomatic\n"); };
	}

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->asymptomatic_infectious_factor));
	if( check < 1){ print_exit("Failed to read parameter asymptomatic_infectious_factor\n"); };

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
        token = strtok(NULL, ",");
		check = sscanf(token, "%lf", &(params->mild_fraction[i]));
		if( check < 1){ print_exit("Failed to read parameter mild_fraction\n"); };
	}

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->mild_infectious_factor));
	if( check < 1){ print_exit("Failed to read parameter mild_infectious_factor\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->mean_asymptomatic_to_recovery));
	if( check < 1){ print_exit("Failed to read parameter mean_asymptomatic_to_recovery\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->sd_asymptomatic_to_recovery));
	if( check < 1){ print_exit("Failed to read parameter sd_asymptomatic_to_recovery\n"); };


	for( i = 0; i < N_HOUSEHOLD_MAX; i++ )
	{
        token = strtok(NULL, ",");
		check = sscanf(token, "%lf", &(params->household_size[i]));
		if( check < 1){ print_exit("Failed to read parameter household_size_*\n"); };
	}

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
        token = strtok(NULL, ",");
		check = sscanf(token, "%lf", &(params->population[i]));
		if( check < 1){ print_exit("Failed to read parameter population\n"); };
	}

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->daily_non_cov_symptoms_rate));
	if( check < 1){ print_exit("Failed to read parameter seasonal_flu_rate\n"); };

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
        token = strtok(NULL, ",");
        check = sscanf(token, "%lf", &(params->relative_susceptibility[i]));
        if( check < 1){ print_exit("Failed to read parameter relative_susceptibility\n"); };
    }

	for( i = 0; i < N_INTERACTION_TYPES; i++ )
	{
        token = strtok(NULL, ",");
		check = sscanf(token, "%lf", &(params->relative_transmission[i]));
		if( check < 1){ print_exit("Failed to read parameter relative_transmission_**\n"); };
	}

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
        token = strtok(NULL, ",");
		check = sscanf(token, "%lf", &(params->hospitalised_fraction[i]));
		if( check < 1){ print_exit("Failed to read parameter hopsitalised_fraction_**\n"); };
	}

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
        token = strtok(NULL, ",");
		check = sscanf(token, "%lf", &(params->critical_fraction[i]));
		if( check < 1){ print_exit("Failed to read parameter critical_fraction_**\n"); };
	}

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
        token = strtok(NULL, ",");
		check = sscanf(token, "%lf", &(params->fatality_fraction[i]));
		if( check < 1){ print_exit("Failed to read parameter fatality_fraction\n"); };
	}

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->mean_time_hospitalised_recovery ));
	if( check < 1){ print_exit("Failed to read parameter mean_time_hospitalised_recovery\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->sd_time_hospitalised_recovery ));
	if( check < 1){ print_exit("Failed to read parameter sd_time_hospitalised_recovery\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->mean_time_critical_survive ));
	if( check < 1){ print_exit("Failed to read parameter mean_time_critical_survive\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->sd_time_critical_survive ));
	if( check < 1){ print_exit("Failed to read parameter sd_time_critical_survive\n"); };

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
        token = strtok(NULL, ",");
		check = sscanf(token, "%lf", &(params->icu_allocation[i]));
		if( check < 1){ print_exit("Failed to read parameter icu_allocation\n"); };
	}

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->quarantine_length_self));
	if( check < 1){ print_exit("Failed to read parameter quarantine_length_self\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->quarantine_length_traced));
	if( check < 1){ print_exit("Failed to read parameter quarantine_length_traced\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->quarantine_length_positive));
	if( check < 1){ print_exit("Failed to read parameter quarantine_length_positive\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->quarantine_dropout_self));
	if( check < 1){ print_exit("Failed to read parameter quarantine_dropout_self\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->quarantine_dropout_traced));
	if( check < 1){ print_exit("Failed to read parameter quarantine_dropout_traced\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->quarantine_dropout_positive));
	if( check < 1){ print_exit("Failed to read parameter quarantine_dropout_positive\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->test_on_symptoms));
	if( check < 1){ print_exit("Failed to read parameter test_on_symptoms\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->test_on_traced));
	if( check < 1){ print_exit("Failed to read parameter test_on_traced\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->trace_on_symptoms));
	if( check < 1){ print_exit("Failed to read parameter trace_on_symptoms\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->trace_on_positive));
	if( check < 1){ print_exit("Failed to read parameter trace_on_positive\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->quarantine_on_traced));
	if( check < 1){ print_exit("Failed to read parameter quarantine_on_traced\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->traceable_interaction_fraction));
	if( check < 1){ print_exit("Failed to read parameter traceable_interaction_fraction\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->tracing_network_depth));
	if( check < 1){ print_exit("Failed to read parameter tracing_network_depth\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, " %i ,", &(params->allow_clinical_diagnosis));
	if( check < 1){ print_exit("Failed to read parameter allow_clinical_diagnosis\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->quarantine_household_on_positive));
	if( check < 1){ print_exit("Failed to read parameter quarantine_household_on_positive\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->quarantine_household_on_symptoms));
	if( check < 1){ print_exit("Failed to read parameter quarantine_household_on_symptoms\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->quarantine_household_on_traced));
	if( check < 1){ print_exit("Failed to read parameter quarantine_household_on_traced\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->quarantine_household_contacts_on_positive));
	if( check < 1){ print_exit("Failed to read parameter quarantine_household_contacts_on_positive\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->quarantine_household_contacts_on_symptoms));
	if( check < 1){ print_exit("Failed to read parameter quarantine_household_contacts_on_symptoms\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->quarantined_daily_interactions));
	if( check < 1){ print_exit("Failed to read parameter quarantined_daily_interactions\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->quarantine_days));
	if( check < 1){ print_exit("Failed to read parameter quarantine_days\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->quarantine_smart_release_day));
	if( check < 1){ print_exit("Failed to read parameter quarantine_smart_release_day\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->hospitalised_daily_interactions));
	if( check < 1){ print_exit("Failed to read parameter hospitalised_daily_interactions\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i",   &(params->test_insensitive_period));
	if( check < 1){ print_exit("Failed to read parameter test_insensitive_period\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i",   &(params->test_order_wait));
	if( check < 1){ print_exit("Failed to read parameter test_order_wait\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i",   &(params->test_result_wait));
	if( check < 1){ print_exit("Failed to read parameter test_result_wait\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->self_quarantine_fraction));
	if( check < 1){ print_exit("Failed to read parameter self_quarantine_fraction\n"); };

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
        token = strtok(NULL, ",");
		check = sscanf(token, "%lf", &(params->app_users_fraction[i]));
		if( check < 1){ print_exit("Failed to read parameter app_users_fraction\n"); };
	}

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->app_turn_on_time));
	if( check < 1){ print_exit("Failed to read parameter app_turn_on_time)\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->lockdown_work_network_multiplier));
	if( check < 1){ print_exit("Failed to read parameter lockdown_work_network_multiplier)\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->lockdown_random_network_multiplier));
	if( check < 1){ print_exit("Failed to read parameter lockdown_random_network_multiplier)\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%lf", &(params->lockdown_house_interaction_multiplier));
	if( check < 1){ print_exit("Failed to read parameter lockdown_house_interaction_multiplier)\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->lockdown_time_on));
	if( check < 1){ print_exit("Failed to read parameter lockdown_time_on)\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->lockdown_time_off));
	if( check < 1){ print_exit("Failed to read parameter lockdown_time_off)\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->lockdown_elderly_time_on));
	if( check < 1){ print_exit("Failed to read parameter lockdown_elderly_time_on)\n"); };

    token = strtok(NULL, ",");
	check = sscanf(token, "%i", &(params->lockdown_elderly_time_off));
	if( check < 1){ print_exit("Failed to read parameter lockdown_elderly_time_off)\n"); };

    token = strtok(NULL, ",");
    check = sscanf(token, "%i", &(params->testing_symptoms_time_on));
    if( check < 1){ print_exit("Failed to read parameter testing_symptoms_time_on)\n"); };

    token = strtok(NULL, ",");
    check = sscanf(token, "%i", &(params->testing_symptoms_time_off));
    if( check < 1){ print_exit("Failed to read parameter testing_symptoms_time_off)\n"); };

    token = strtok(NULL, ",");
    check = sscanf(token, "%i", &(params->intervention_start_time));
    if( check < 1){ print_exit("Failed to read parameter intervention_start_time)\n"); };
}

/*****************************************************************************************
*  Name:        update_household_demographics_from_buffer
*  Description: Update the parameters structure from a csv formatted buffer
******************************************************************************************/
void update_household_demographics_from_buffer( parameters *params, char *buffer )
{
    int ret = 0;
    long hdx = 0;
    char *line;

    if (buffer == NULL) print_exit("Empty buffer");

    params->REFERENCE_HOUSEHOLDS = calloc(params->N_REFERENCE_HOUSEHOLDS, sizeof(int*));

    for(hdx = 0; hdx < params->N_REFERENCE_HOUSEHOLDS; hdx++)
        params->REFERENCE_HOUSEHOLDS[hdx] = calloc(N_AGE_GROUPS, sizeof(int));

    /* Throw away header */
    line = strtok(buffer, "\n");
    line = strtok(NULL, "\n");

    /* Update parameters from buffer */
    for(hdx = 0; hdx < params->N_REFERENCE_HOUSEHOLDS; hdx++){
        ret = sscanf(line,
                     "%d,%d,%d,%d,%d,%d,%d,%d,%d",
                     &params->REFERENCE_HOUSEHOLDS[hdx][AGE_0_9],
                     &params->REFERENCE_HOUSEHOLDS[hdx][AGE_10_19],
                     &params->REFERENCE_HOUSEHOLDS[hdx][AGE_20_29],
                     &params->REFERENCE_HOUSEHOLDS[hdx][AGE_30_39],
                     &params->REFERENCE_HOUSEHOLDS[hdx][AGE_40_49],
                     &params->REFERENCE_HOUSEHOLDS[hdx][AGE_50_59],
                     &params->REFERENCE_HOUSEHOLDS[hdx][AGE_60_69],
                     &params->REFERENCE_HOUSEHOLDS[hdx][AGE_70_79],
                     &params->REFERENCE_HOUSEHOLDS[hdx][AGE_80]);
        if (ret < 0) print_exit("Error parsing buffer");

        line = strtok(NULL, "\n");
    }
}

/*****************************************************************************************
*  Name:		read_param_file
*  Description: Read line from parameter file (csv), attach parame values to params struct
******************************************************************************************/
void read_param_file( parameters *params)
{
    char *buffer = NULL;

    buffer = load_parameters_file_to_buffer(params->input_param_file);
    update_params_from_buffer(params, buffer);
    free(buffer);
}

/*****************************************************************************************
*  Name:		write_output_files
*  Description: Write (csv) files of simulation output
******************************************************************************************/
void write_output_files(model *model, parameters *params)
{

	if(params->sys_write_individual == TRUE)
	{
		write_individual_file( model, params );
		write_interactions( model );
		write_transmissions( model );
		write_trace_tokens( model );
	}
}	

/*****************************************************************************************
*  Name:		write_individual_file
*  Description: Write (csv) file of individuals in simulation
******************************************************************************************/
void write_individual_file(model *model, parameters *params)
{
	
	char output_file[INPUT_CHAR_LEN];
	FILE *individual_output_file;
	individual *indiv;
	int infector_time_infected, infector_status;
	long idx, infector_id;
	
	char param_line_number[10];
	sprintf(param_line_number, "%d", params->param_line_number);
	
	// Concatenate file name
    strcpy(output_file, params->output_file_dir);
    strcat(output_file, "/individual_file_Run");
	strcat(output_file, param_line_number);
	strcat(output_file, ".csv");
	
	individual_output_file = fopen(output_file, "w");
	if(individual_output_file == NULL)
		print_exit("Can't open individual output file");
	
	fprintf(individual_output_file,"ID,");
	fprintf(individual_output_file,"current_status,");
	fprintf(individual_output_file,"age_group,");
	fprintf(individual_output_file,"work_network,");
	fprintf(individual_output_file,"house_no,");
	fprintf(individual_output_file,"quarantined,");
	fprintf(individual_output_file,"app_user,");
	fprintf(individual_output_file,"hazard,");
	fprintf(individual_output_file,"mean_interactions,");
	fprintf(individual_output_file,"time_infected,");
	fprintf(individual_output_file,"time_presymptomatic,");
	fprintf(individual_output_file,"time_presymptomatic_mild,");
	fprintf(individual_output_file,"time_presymptomatic_severe,");
	fprintf(individual_output_file,"time_symptomatic,");
	fprintf(individual_output_file,"time_symptomatic_mild,");
	fprintf(individual_output_file,"time_symptomatic_severe,");
	fprintf(individual_output_file,"time_asymptomatic,");
	fprintf(individual_output_file,"time_hospitalised,");
	fprintf(individual_output_file,"time_critical,");
	fprintf(individual_output_file,"time_hospitalised_recovering,");
	fprintf(individual_output_file,"time_death,");
	fprintf(individual_output_file,"time_recovered,");
	fprintf(individual_output_file,"time_quarantined,");
	fprintf(individual_output_file,"infector_ID,");
	fprintf(individual_output_file,"infector_time_infected,");
	fprintf(individual_output_file,"infector_status");
	fprintf(individual_output_file,"\n");
	
	// Loop through all individuals in the simulation
	for(idx = 0; idx < params->n_total; idx++)
	{
		indiv = &(model->population[idx]);
		
		/* Check the individual was infected during the simulation
		(otherwise the "infector" attribute does not point to another individual) */
		if(model->population[idx].status != UNINFECTED)
		{
			infector_id			   = indiv->infector->idx;
			infector_time_infected = time_infected( indiv->infector );
			infector_status        = indiv->infector_status;

		}
		else
		{
			infector_id            = UNKNOWN;
			infector_time_infected = UNKNOWN;
			infector_status        = UNKNOWN;
		}
		
		fprintf(individual_output_file, 
			"%li,%d,%d,%d,%li,%d,%d,%f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%li,%d,%d\n",
			indiv->idx,
			indiv->status,
			indiv->age_group,
			indiv->work_network,
			indiv->house_no,
			indiv->quarantined,
			indiv->app_user,
			indiv->hazard,
			indiv->random_interactions,
			time_infected(indiv),
			max( indiv->time_event[PRESYMPTOMATIC], indiv->time_event[PRESYMPTOMATIC_MILD] ),
			indiv->time_event[PRESYMPTOMATIC_MILD],
			indiv->time_event[PRESYMPTOMATIC],
			max( indiv->time_event[SYMPTOMATIC], indiv->time_event[SYMPTOMATIC_MILD] ),
			indiv->time_event[SYMPTOMATIC_MILD],
			indiv->time_event[SYMPTOMATIC],
			indiv->time_event[ASYMPTOMATIC],
			indiv->time_event[HOSPITALISED],
			indiv->time_event[CRITICAL],
			indiv->time_event[HOSPITALISED_RECOVERING],
			indiv->time_event[DEATH],
			indiv->time_event[RECOVERED],
			indiv->time_event[QUARANTINED],
			infector_id,
			infector_time_infected,
			infector_status
			);
	}
	fclose(individual_output_file);
}

/*****************************************************************************************
*  Name:		print_interactions_averages
*  Description: average interactions by type
******************************************************************************************/
void print_interactions_averages(model *model, int header)
{
	int day_idx, n_int, idx, jdx, cqh;
	long pdx;
	double int_tot = 0;
	double per_tot = 0;
	double  int_by_age[N_AGE_TYPES],per_by_age[N_AGE_TYPES];
	double int_by_cqh[3],per_by_cqh[3];
	double assort[N_AGE_TYPES][N_AGE_TYPES];
	individual *indiv;
	interaction *inter;

	for( idx = 0; idx < N_AGE_TYPES; idx++ )
	{
		 int_by_age[idx] = 0;
		 per_by_age[idx] = 0.00001;
		 for( jdx = 0; jdx < N_AGE_TYPES; jdx++ )
			 assort[idx][jdx] = 0;
	}

	for( idx = 0; idx < 3; idx++ )
	{
		 int_by_cqh[idx] = 0;
		 per_by_cqh[idx] = 0.00001;
	}

	day_idx = model->interaction_day_idx;
	ring_dec( day_idx, model->params->days_of_interactions );

	for( pdx = 0; pdx < model->params->n_total; pdx++ )
	{
		indiv = &(model->population[pdx]);
		if( indiv->status == DEATH )
			continue;

		n_int = indiv->n_interactions[day_idx];
		inter = indiv->interactions[day_idx];
		for( jdx = 0; jdx < n_int; jdx++ )
		{
			assort[ indiv->age_type][inter->individual->age_type]++;
			inter = inter->next;
		}

		int_tot += n_int;
		per_tot++;

		int_by_age[ indiv->age_type] += n_int;
		per_by_age[ indiv->age_type]++;

		cqh = ifelse( indiv->status == HOSPITALISED , 2, ifelse( indiv->quarantined && indiv->time_event[QUARANTINED] != model->time, 1, 0 ) );
		int_by_cqh[cqh] += n_int;
		per_by_cqh[cqh]++;
	}

	if( header )
		printf( "time,int,int_child,ind_adult,int_elderly,int_community,int_quarantined,int_hospital, assort_c_c, assort_c_a, assort_c_e, assort_a_c, assort_a_a, assort_a_e, assort_e_c, assort_e_a, assort_e_e\n" );

	printf( "%i %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf\n" ,
		model->time,
		1.0 * int_tot / per_tot,
		1.0 * int_by_age[0] / per_by_age[0],
		1.0 * int_by_age[1] / per_by_age[1],
		1.0 * int_by_age[2] / per_by_age[2],
		1.0 * int_by_cqh[0] / per_by_cqh[0],
		1.0 * int_by_cqh[1] / per_by_cqh[1],
		1.0 * int_by_cqh[2] / per_by_cqh[2],
		1.0 * assort[0][0]/ int_by_age[0],
		1.0 * assort[0][1]/ int_by_age[0],
		1.0 * assort[0][2]/ int_by_age[0],
		1.0 * assort[1][0]/ int_by_age[1],
		1.0 * assort[1][1]/ int_by_age[1],
		1.0 * assort[1][2]/ int_by_age[1],
		1.0 * assort[2][0]/ int_by_age[2],
		1.0 * assort[2][1]/ int_by_age[2],
		1.0 * assort[2][2]/ int_by_age[2]
	);
}

/*****************************************************************************************
*  Name:		read_household_demographics_file
*  Description: Read household demographics (csv), attach values to params struct
******************************************************************************************/
void read_household_demographics_file( parameters *params)
{
    char *buffer = NULL;

    buffer = load_household_demographics_file_to_buffer(params, params->input_household_file);
    update_household_demographics_from_buffer(params, buffer);
    free(buffer);
}


void set_up_reference_household_memory(parameters *params){
	long hdx;
	params->REFERENCE_HOUSEHOLDS = calloc(params->N_REFERENCE_HOUSEHOLDS, sizeof(int*));
	for(hdx = 0; hdx < params->N_REFERENCE_HOUSEHOLDS; hdx++){
		params->REFERENCE_HOUSEHOLDS[hdx] = calloc(N_AGE_GROUPS, sizeof(int));
	}
}

/*****************************************************************************************
*  Name:		write_interactions
*  Description: write interactions details
******************************************************************************************/
void write_interactions( model *model )
{
	char output_file_name[INPUT_CHAR_LEN];
	FILE *output_file;
	long pdx;
	int day, idx;
	individual *indiv;
	interaction *inter;

	char param_line_number[10];
	sprintf(param_line_number, "%d", model->params->param_line_number);

	// Concatenate file name
    strcpy(output_file_name, model->params->output_file_dir);
    strcat(output_file_name, "/interactions_Run");
	strcat(output_file_name, param_line_number);
	strcat(output_file_name, ".csv");

	output_file = fopen(output_file_name, "w");

	day = model->interaction_day_idx;
	ring_dec( day, model->params->days_of_interactions );

	fprintf(output_file ,"ID,age_group,house_no,work_network,type,ID_2,age_group_2,house_no_2,work_2\n");
	for( pdx = 0; pdx < model->params->n_total; pdx++ )
	{

		indiv = &(model->population[pdx]);

		if( indiv->n_interactions[day] > 0 )
		{
			inter = indiv->interactions[day];
			for( idx = 0; idx < indiv->n_interactions[day]; idx++ )
			{

				fprintf(output_file ,"%li,%i,%li,%i,%i,%li,%i,%li,%i\n",
					indiv->idx,
					indiv->age_group,
					indiv->house_no,
					indiv->work_network,
					inter->type,
					inter->individual->idx,
					inter->individual->age_group,
					inter->individual->house_no,
					inter->individual->work_network
				);
				inter = inter->next;
			}
		}
	}
	fclose(output_file);
}


/*****************************************************************************************
*  Name:		write_transmissions
*  Description: write_transmissions details
******************************************************************************************/
void write_transmissions( model *model )
{
	char output_file_name[INPUT_CHAR_LEN];
	FILE *output_file;
	long pdx;
	individual *indiv;

	char param_line_number[10];
	sprintf(param_line_number, "%d", model->params->param_line_number);

	// Concatenate file name
	strcpy(output_file_name, model->params->output_file_dir);
	strcat(output_file_name, "/transmission_Run");
	strcat(output_file_name, param_line_number);
	strcat(output_file_name, ".csv");

	output_file = fopen(output_file_name, "w");
	fprintf(output_file ,"time_infected,ID,age_group,house_no,work_network,infector_network,infector_infected_time,infector_status,ID_2,age_group_2,house_no_2,work_2\n");

	for( pdx = 0; pdx < model->params->n_total; pdx++ )
	{
		indiv = &(model->population[pdx]);
		if( indiv->status == UNINFECTED )
			continue;

		fprintf(output_file ,"%i,%li,%i,%li,%i,%i,%i,%i,%li,%i,%li,%i\n",
			time_infected(indiv),
			indiv->idx,
			indiv->age_group,
			indiv->house_no,
			indiv->work_network,
			indiv->infector_network,
			time_infected( indiv ) -time_infected( indiv->infector ),
			indiv->infector_status,
			indiv->infector->idx,
			indiv->infector->age_group,
			indiv->infector->house_no,
			indiv->infector->work_network
		);
	}
	fclose(output_file);
}

/*****************************************************************************************
*  Name:		write_trace_tokens
*  Description: write trace tokens details
******************************************************************************************/
void write_trace_tokens( model *model )
{
	char output_file_name[INPUT_CHAR_LEN];
	FILE *output_file;
	long idx, n_events;
	int day;
	individual *indiv;
	event *event, *next_event;
	trace_token *token;

	char param_line_number[10];
	sprintf(param_line_number, "%d", model->params->param_line_number);

	// Concatenate file name
    strcpy(output_file_name, model->params->output_file_dir);
    strcat(output_file_name, "/trace_tokens_Run");
	strcat(output_file_name, param_line_number);
	strcat(output_file_name, ".csv");

	output_file = fopen(output_file_name, "w");
	fprintf( output_file ,"time,days_since_index,index_ID,index_status,days_since_contact,traced_ID,traced_status,traced_infector_ID,traced_time_infected\n" );

	for( day = 1; day <= model->params->quarantine_length_traced; day++ )
	{
		n_events    = model->event_lists[TRACE_TOKEN_RELEASE].n_daily_current[ model->time + day ];
		next_event  = model->event_lists[TRACE_TOKEN_RELEASE].events[ model->time + day ];

		for( idx = 0; idx < n_events; idx++ )
		{
			event      = next_event;
			next_event = event->next;
			indiv      = event->individual;

			token = indiv->index_trace_token;
			if( token == NULL )
				continue;

			token = token->next_index;
			while( token != NULL )
			{
				fprintf( output_file, "%i,%i,%li,%i,%i,%li,%i,%li,%i\n",
					model->time + day - model->params->quarantine_length_traced,
					model->params->quarantine_length_traced - day,
					indiv->idx,
					indiv->status,
					token->days_since_contact,
					token->individual->idx,
					token->individual->status,
					ifelse( token->individual->status > 0, token->individual->infector->idx, -1 ),
					time_infected( token->individual )
				);
				token = token->next_index;
			}
		}
	}
	fclose(output_file);
}

/*****************************************************************************************
*  Name:		write_trace_tokens_ts
*  Description: write top level stats of trace_tokens
******************************************************************************************/
void write_trace_tokens_ts( model *model, int initialise )
{
	char output_file_name[INPUT_CHAR_LEN];
	FILE *output_file;
	long idx, n_events;
	int day, n_traced,n_symptoms,n_infected,n_infected_by_index, time_index;
	individual *indiv, *contact;
	event *event, *next_event;
	trace_token *token;

	char param_line_number[10];
	sprintf(param_line_number, "%d", model->params->param_line_number);

	// Concatenate file name
    strcpy(output_file_name, model->params->output_file_dir);
    strcat(output_file_name, "/trace_tokens_ts_Run");
	strcat(output_file_name, param_line_number);
	strcat(output_file_name, ".csv");


	if( initialise )
	{
		output_file = fopen(output_file_name, "w");
		fprintf( output_file ,"time,time_index,index_ID,n_traced,n_symptoms,n_infected,n_infected_by_index\n" );
		fclose(output_file);
		return;
	}
	else
		output_file = fopen(output_file_name, "a");

	for( day = 1; day <= model->params->quarantine_length_traced; day++ )
	{
		n_events    = model->event_lists[TRACE_TOKEN_RELEASE].n_daily_current[ model->time + day ];
		next_event  = model->event_lists[TRACE_TOKEN_RELEASE].events[ model->time + day ];

		for( idx = 0; idx < n_events; idx++ )
		{
			event      = next_event;
			next_event = event->next;
			indiv      = event->individual;
			time_index = model->time + day - model->params->quarantine_length_traced;

			n_traced   = 0;
			n_symptoms = 0;
			n_infected = 0;
			n_infected_by_index = 0;

			token = indiv->index_trace_token;
			if( token == NULL )
				continue;

			token = token->next_index;
			while( token != NULL )
			{
				contact = token->individual;
				n_traced++;
				if( contact->status > 0 )
					n_infected++;
				if( (contact->status >= SYMPTOMATIC) & (contact->time_event[ASYMPTOMATIC] == UNKNOWN)  &
					( (contact->time_event[RECOVERED] == UNKNOWN) | (contact->time_event[RECOVERED] > time_index) )
				)
					n_symptoms++;
				if( indiv == contact->infector )
					n_infected_by_index++;

				token = token->next_index;
			}
			fprintf( output_file, "%i,%i,%li,%i,%i,%i,%i\n",
				model->time,
				time_index,
				indiv->idx,
				n_traced,
				n_symptoms,
				n_infected,
				n_infected_by_index
			);
		}
	}
	fclose(output_file);
}

