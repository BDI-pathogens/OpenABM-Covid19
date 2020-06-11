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

	for( i = 0; i < N_OCCUPATION_NETWORK_TYPES; i++ )
	{
		check = fscanf(parameter_file, " %lf ,",  &(params->mean_work_interactions[i]));
		if( check < 1){ print_exit("Failed to read parameter mean_work_interactions\n"); };
	}

	check = fscanf(parameter_file, " %lf ,",  &(params->daily_fraction_work));
	if( check < 1){ print_exit("Failed to read parameter daily_fraction_work\n"); };

	check = fscanf(parameter_file, " %lf ,",  &(params->work_network_rewire));
	if( check < 1){ print_exit("Failed to read parameter work_network_rewire\n"); };

	for( i = 0; i < N_AGE_TYPES; i++ )
	{
		check = fscanf(parameter_file, " %lf ,",  &(params->mean_random_interactions[i]));
		if( check < 1){ print_exit("Failed to read parameter mean_daily_interactions\n"); };

		check = fscanf(parameter_file, " %lf ,",  &(params->sd_random_interactions[i]));
		if( check < 1){ print_exit("Failed to read parameter sd_daily_interactions\n"); };
	}

	check = fscanf(parameter_file, " %i ,",  &(params->random_interaction_distribution));
	if( check < 1){ print_exit("Failed to read parameter random_interaction_distribution\n"); };

	check = fscanf(parameter_file, " %lf ,",  &(params->child_network_adults));
	if( check < 1){ print_exit("Failed to read parameter child_network_adults\n"); };

	check = fscanf(parameter_file, " %lf ,",  &(params->elderly_network_adults));
	if( check < 1){ print_exit("Failed to read parameter elderly_network_adults\n"); };

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

	check = fscanf(parameter_file, " %lf ,", &(params->mean_time_to_critical));
	if( check < 1){ print_exit("Failed to read parameter mean_time_to_critical\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->sd_time_to_critical));
	if( check < 1){ print_exit("Failed to read parameter sd_time_to_critical\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->mean_time_to_recover));
	if( check < 1){ print_exit("Failed to read parameter mean_time_to_recover\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->sd_time_to_recover));
	if( check < 1){ print_exit("Failed to read parameter sd_time_to_recover\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->mean_time_to_death));
	if( check < 1){ print_exit("Failed to read parameter mean_time_to_death\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->sd_time_to_death));
	if( check < 1){ print_exit("Failed to read parameter sd_time_to_death\n"); };

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
		check = fscanf(parameter_file, " %lf ,", &(params->fraction_asymptomatic[i]));
		if( check < 1){ print_exit("Failed to read parameter fraction_asymptomatic\n"); };
	}

	check = fscanf(parameter_file, " %lf ,", &(params->asymptomatic_infectious_factor));
	if( check < 1){ print_exit("Failed to read parameter asymptomatic_infectious_factor\n"); };

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
		check = fscanf(parameter_file, " %lf ,", &(params->mild_fraction[i]));
		if( check < 1){ print_exit("Failed to read parameter mild_fraction\n"); };
	}

	check = fscanf(parameter_file, " %lf ,", &(params->mild_infectious_factor));
	if( check < 1){ print_exit("Failed to read parameter mild_infectious_factor\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->mean_asymptomatic_to_recovery));
	if( check < 1){ print_exit("Failed to read parameter mean_asymptomatic_to_recovery\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->sd_asymptomatic_to_recovery));
	if( check < 1){ print_exit("Failed to read parameter sd_asymptomatic_to_recovery\n"); };


	for( i = 0; i < N_HOUSEHOLD_MAX; i++ )
	{
		check = fscanf(parameter_file, " %lf ,", &(params->household_size[i]));
		if( check < 1){ print_exit("Failed to read parameter household_size_*\n"); };
	}

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
		check = fscanf(parameter_file, " %lf ,", &(params->population[i]));
		if( check < 1){ print_exit("Failed to read parameter population_**\n"); };
	}

	check = fscanf(parameter_file, " %lf ,", &(params->daily_non_cov_symptoms_rate));
	if( check < 1){ print_exit("Failed to read parameter daily_non_cov_symptoms_rate\n"); };

	for( i = 0; i < N_AGE_GROUPS; i++ )
		{
			check = fscanf(parameter_file, " %lf ,", &(params->relative_susceptibility[i]));
			if( check < 1){ print_exit("Failed to read parameter relative_susceptibility\n"); };
		}

	for( i = 0; i < N_INTERACTION_TYPES; i++ )
	{
		check = fscanf(parameter_file, " %lf ,", &(params->relative_transmission[i]));
		if( check < 1){ print_exit("Failed to read parameter relative_transmission_**\n"); };
	}

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
		check = fscanf(parameter_file, " %lf ,", &(params->hospitalised_fraction[i]));
		if( check < 1){ print_exit("Failed to read parameter hopsitalised_fraction_**\n"); };
	}

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
		check = fscanf(parameter_file, " %lf ,", &(params->critical_fraction[i]));
		if( check < 1){ print_exit("Failed to read parameter critical_fraction_**\n"); };
	}

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
		check = fscanf(parameter_file, " %lf ,", &(params->fatality_fraction[i]));
		if( check < 1){ print_exit("Failed to read parameter fatality_fraction\n"); };
	}

	check = fscanf(parameter_file, " %lf,", &(params->mean_time_hospitalised_recovery ));
	if( check < 1){ print_exit("Failed to read parameter mean_time_hospitalised_recovery\n"); };

	check = fscanf(parameter_file, " %lf,", &(params->sd_time_hospitalised_recovery ));
	if( check < 1){ print_exit("Failed to read parameter sd_time_hospitalised_recovery\n"); };

	check = fscanf(parameter_file, " %lf,", &(params->mean_time_critical_survive ));
	if( check < 1){ print_exit("Failed to read parameter mean_time_critical_survive\n"); };

	check = fscanf(parameter_file, " %lf,", &(params->sd_time_critical_survive ));
	if( check < 1){ print_exit("Failed to read parameter sd_time_critical_survive\n"); };

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
		check = fscanf(parameter_file, " %lf ,", &(params->location_death_icu[i]));
		if( check < 1){ print_exit("Failed to read parameter location_death_icu\n"); };
	}

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_length_self));
	if( check < 1){ print_exit("Failed to read parameter quarantine_length_self\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_length_traced_symptoms));
	if( check < 1){ print_exit("Failed to read parameter quarantine_length_traced_symptoms\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_length_traced_positive));
	if( check < 1){ print_exit("Failed to read parameter quarantine_length_traced_positive\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_length_positive));
	if( check < 1){ print_exit("Failed to read parameter quarantine_length_positive\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->quarantine_dropout_self));
	if( check < 1){ print_exit("Failed to read parameter quarantine_dropout_self\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->quarantine_dropout_traced_symptoms));
	if( check < 1){ print_exit("Failed to read parameter quarantine_dropout_traced\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->quarantine_dropout_traced_positive));
	if( check < 1){ print_exit("Failed to read parameter quarantine_dropout_traced\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->quarantine_dropout_positive));
	if( check < 1){ print_exit("Failed to read parameter quarantine_dropout_positive\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->quarantine_compliance_traced_symptoms));
	if( check < 1){ print_exit("Failed to read parameter quarantine_compliance_traced_symptoms\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->quarantine_compliance_traced_positive));
	if( check < 1){ print_exit("Failed to read parameter quarantine_compliance_traced_positive\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->test_on_symptoms));
	if( check < 1){ print_exit("Failed to read parameter test_on_symptoms\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->test_on_traced));
	if( check < 1){ print_exit("Failed to read parameter test_on_traced\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->trace_on_symptoms));
	if( check < 1){ print_exit("Failed to read parameter trace_on_symptoms\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->trace_on_positive));
	if( check < 1){ print_exit("Failed to read parameter trace_on_positive\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->retrace_on_positive));
	if( check < 1){ print_exit("Failed to read parameter retrace_on_positive\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_on_traced));
	if( check < 1){ print_exit("Failed to read parameter quarantine_on_traced\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->traceable_interaction_fraction));
	if( check < 1){ print_exit("Failed to read parameter traceable_interaction_fraction\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->tracing_network_depth));
	if( check < 1){ print_exit("Failed to read parameter tracing_network_depth\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->allow_clinical_diagnosis));
	if( check < 1){ print_exit("Failed to read parameter allow_clinical_diagnosis\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_household_on_positive));
	if( check < 1){ print_exit("Failed to read parameter quarantine_household_on_positive\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_household_on_symptoms));
	if( check < 1){ print_exit("Failed to read parameter quarantine_household_on_symptoms\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_household_on_traced_positive));
	if( check < 1){ print_exit("Failed to read parameter quarantine_household_on_traced_positive\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_household_on_traced_symptoms));
	if( check < 1){ print_exit("Failed to read parameter quarantine_household_on_traced_symptoms\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_household_contacts_on_positive));
	if( check < 1){ print_exit("Failed to read parameter quarantine_household_contacts_on_positive\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_household_contacts_on_symptoms));
	if( check < 1){ print_exit("Failed to read parameter quarantine_household_contacts_on_symptoms\n"); };

	check = fscanf(parameter_file, " %i  ,", &(params->quarantined_daily_interactions));
	if( check < 1){ print_exit("Failed to read parameter quarantined_daily_interactions\n"); };

	check = fscanf(parameter_file, " %i  ,", &(params->quarantine_days));
	if( check < 1){ print_exit("Failed to read parameter quarantine_days\n"); };

	check = fscanf(parameter_file, " %i  ,", &(params->quarantine_smart_release_day));
	if( check < 1){ print_exit("Failed to read parameter quarantine_smart_release_day\n"); };

	check = fscanf(parameter_file, " %i  ,", &(params->hospitalised_daily_interactions));
	if( check < 1){ print_exit("Failed to read parameter hospitalised_daily_interactions\n"); };

	check = fscanf(parameter_file, " %i , ",   &(params->test_insensitive_period));
	if( check < 1){ print_exit("Failed to read parameter test_insensitive_period\n"); };

	check = fscanf(parameter_file, " %i , ",   &(params->test_order_wait));
	if( check < 1){ print_exit("Failed to read parameter test_order_wait\n"); };

	check = fscanf(parameter_file, " %i , ",   &(params->test_result_wait));
	if( check < 1){ print_exit("Failed to read parameter test_result_wait\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->self_quarantine_fraction));
	if( check < 1){ print_exit("Failed to read parameter self_quarantine_fraction\n"); };

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
		check = fscanf(parameter_file, " %lf ,", &(params->app_users_fraction[i]));
		if( check < 1){ print_exit("Failed to read parameter app_users_fraction\n"); };
	}

	check = fscanf(parameter_file, " %i ,", &(params->app_turn_on_time));
	if( check < 1){ print_exit("Failed to read parameter app_turn_on_time)\n"); };

	for (i = 0; i<N_OCCUPATION_NETWORKS; i++){

		check = fscanf(parameter_file, " %lf ,", &(params->lockdown_occupation_multiplier[i]));
		if( check < 1){ print_exit("Failed to read parameter lockdown_occupation_multiplier)\n"); };

	}
	check = fscanf(parameter_file, " %lf ,", &(params->lockdown_random_network_multiplier));
	if( check < 1){ print_exit("Failed to read parameter lockdown_random_network_multiplier)\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->lockdown_house_interaction_multiplier));
	if( check < 1){ print_exit("Failed to read parameter lockdown_house_interaction_multiplier)\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->lockdown_time_on));
	if( check < 1){ print_exit("Failed to read parameter lockdown_time_on)\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->lockdown_time_off));
	if( check < 1){ print_exit("Failed to read parameter lockdown_time_off)\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->lockdown_elderly_time_on));
	if( check < 1){ print_exit("Failed to read parameter lockdown_elderly_time_on)\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->lockdown_elderly_time_off));
	if( check < 1){ print_exit("Failed to read parameter lockdown_elderly_time_off)\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->testing_symptoms_time_on));
	if( check < 1){ print_exit("Failed to read parameter testing_symptoms_time_on)\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->testing_symptoms_time_off));
	if( check < 1){ print_exit("Failed to read parameter testing_symptoms_time_off)\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->intervention_start_time));
	if( check < 1){ print_exit("Failed to read parameter intervention_start_time)\n"); };

	fclose(parameter_file);
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
*  Name:		write_quarantine_reasons
*  Description: Write (csv) files of reasons individuals are quarantined
******************************************************************************************/

void write_quarantine_reasons(model *model, parameters *params)
{
	char output_file_name[INPUT_CHAR_LEN];
	long idx, jdx;
	int quarantine_reasons[N_QUARANTINE_REASONS], quarantine_reason, n_reasons, i, n;
	int index_true_status, index_from_household;
	long index_id, index_house_no;
	
	individual *indiv;
	trace_token *index_token;
	long *members;

	char param_line_number[10];
	sprintf(param_line_number, "%d", model->params->param_line_number);

	// Concatenate file name
	strcpy(output_file_name, model->params->output_file_dir);
	strcat(output_file_name, "/quarantine_reasons_file_Run");
	strcat(output_file_name, param_line_number);
	strcat(output_file_name, ".csv");
	
	FILE *quarantine_reasons_output_file;
	quarantine_reasons_output_file = fopen(output_file_name, "w");
	if(quarantine_reasons_output_file == NULL)
		print_exit("Can't open quarantine_reasons output file");
	
	fprintf(quarantine_reasons_output_file,"time,");
	fprintf(quarantine_reasons_output_file,"ID,");
	fprintf(quarantine_reasons_output_file,"status,");
	fprintf(quarantine_reasons_output_file,"house_no,");
	fprintf(quarantine_reasons_output_file,"ID_index,");
	fprintf(quarantine_reasons_output_file,"status_index,");
	fprintf(quarantine_reasons_output_file,"house_no_index,");
	fprintf(quarantine_reasons_output_file,"quarantine_reason,");
	fprintf(quarantine_reasons_output_file,"n_reasons");
	fprintf(quarantine_reasons_output_file,"\n");
	
	for(idx = 0; idx < params->n_total; idx++)
	{
		indiv = &(model->population[idx]);
		
		if(indiv->quarantined == TRUE){
			
			for(i = 0; i < N_QUARANTINE_REASONS; i++)
				quarantine_reasons[i] = FALSE;
			
			index_true_status = UNKNOWN;
			index_id = UNKNOWN;
			index_house_no = UNKNOWN;
			
			// Check if this individual has non-NULL trace_tokens attribute
			if( indiv->index_trace_token != NULL ){
				
				// Quarantined from self-reported symptoms
				if(indiv->index_trace_token->index_status == SYMPTOMS_ONLY)
					quarantine_reasons[QR_SELF_SYMPTOMS] = TRUE;
				
				// Quarantined from self positive
				if(indiv->index_trace_token->index_status == POSITIVE_TEST)
					quarantine_reasons[QR_SELF_POSITIVE] = TRUE;
				
				index_true_status = indiv->status;
				index_id = indiv->idx;
				index_house_no = indiv->house_no;
			}
			
			if( indiv->trace_tokens != NULL ){
				// Find original index and check if the index was a household member
				n = model->household_directory->n_jdx[indiv->house_no];
				members = model->household_directory->val[indiv->house_no];
				
				index_token = indiv->trace_tokens;
				while( index_token->last_index != NULL )
					index_token = index_token->last_index;

				for(jdx = 0; jdx < n; jdx++){
					if( index_token->individual->idx == members[jdx] ){
						index_from_household = TRUE;
					}
				}
				
				if(index_from_household == TRUE){
					if(index_token->index_status == SYMPTOMS_ONLY)
						quarantine_reasons[QR_HOUSEHOLD_SYMPTOMS] = TRUE;
			
					if(index_token->index_status == POSITIVE_TEST)
						quarantine_reasons[QR_HOUSEHOLD_POSITIVE] = TRUE;
				}else{
					if(index_token->index_status == SYMPTOMS_ONLY)
						quarantine_reasons[QR_TRACE_SYMPTOMS] = TRUE;
					
					if(index_token->index_status == POSITIVE_TEST)
						quarantine_reasons[QR_TRACE_POSITIVE] = TRUE;
				}
				index_id = index_token->individual->idx;
				index_true_status = index_token->individual->status;
				index_house_no = index_token->individual->house_no;
			}
			
			// Resolve multiple reasons for quarantine into one reason
			quarantine_reason = resolve_quarantine_reasons(quarantine_reasons);
			
			n_reasons = 0;
			for(i = 0; i < N_QUARANTINE_REASONS; i++){
				if(quarantine_reasons[i] == TRUE)
					n_reasons += 1;
			}
			
			fprintf(quarantine_reasons_output_file, 
				"%d,%li,%d,%li,%li,%d,%li,%d,%d\n",
				model->time,
				indiv->idx,
				indiv->status,
				indiv->house_no,
				index_id, 
				index_true_status,
				index_house_no,
				quarantine_reason,
				n_reasons);
		}
	}
	fclose(quarantine_reasons_output_file);
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
	int infection_count;
	long idx;

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
	fprintf(individual_output_file,"occupation_network,");
	fprintf(individual_output_file,"house_no,");
	fprintf(individual_output_file,"quarantined,");
	fprintf(individual_output_file,"time_quarantined,");
	fprintf(individual_output_file,"app_user,");
	fprintf(individual_output_file,"mean_interactions,");
	fprintf(individual_output_file,"infection_count");
	fprintf(individual_output_file,"\n");

	// Loop through all individuals in the simulation
	for(idx = 0; idx < params->n_total; idx++)
	{
		indiv = &(model->population[idx]);

		/* Count the number of times an individual has been infected */
		infection_count = count_infection_events( indiv );

		fprintf(individual_output_file,
			"%li,%d,%d,%d,%li,%d,%d,%d,%d,%d\n",
			indiv->idx,
			indiv->status,
			indiv->age_group,
			indiv->occupation_network,
			indiv->house_no,
			indiv->quarantined,
			indiv->infection_events->times[QUARANTINED],
			indiv->app_user,
			indiv->random_interactions,
			infection_count
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

        cqh = ifelse( indiv->status == HOSPITALISED , 2, ifelse( indiv->quarantined && indiv->infection_events->times[QUARANTINED] != model->time, 1, 0 ) );
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
	FILE *hh_file;
	int check, value, adx;
	long hdx, fileSize;
	char lineBuffer[80];

	// get the length of the reference household file
	hh_file = fopen(params->input_household_file, "r");
	if(hh_file == NULL)
		print_exit("Can't open household demographics file");
	fileSize = 0;
	while( fgets(lineBuffer, 80, hh_file ) )
		fileSize++;
	fclose( hh_file );
	params->N_REFERENCE_HOUSEHOLDS = fileSize - 1;

	if( params->N_REFERENCE_HOUSEHOLDS < 100 )
		print_exit( "Reference household panel too small (<100) - will not be able to assign household structure");

	// allocate memory on the params object
	set_up_reference_household_memory(params);

	// read in the data (throw away the header line)
	hh_file = fopen(params->input_household_file, "r");
	fscanf(hh_file, "%*[^\n]\n");
	for(hdx = 0; hdx < params->N_REFERENCE_HOUSEHOLDS; hdx++){
		for(adx = 0; adx < N_AGE_GROUPS; adx++){
			// Read and attach parameter values to parameter structure
			check = fscanf(hh_file, " %d ,", &value);
			if( check < 1){ print_exit("Failed to read household demographics file\n"); };

			params->REFERENCE_HOUSEHOLDS[hdx][adx] = value;
		}
	}
	fclose(hh_file);
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

	fprintf(output_file ,"ID_1,age_group_1,house_no_1,occupation_network_1,type,ID_2,age_group_2,house_no_2,occupation_network_2\n");
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
					indiv->occupation_network,
					inter->type,
					inter->individual->idx,
					inter->individual->age_group,
					inter->individual->house_no,
					inter->individual->occupation_network
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
	infection_event *infection_event;

	char param_line_number[10];
	sprintf(param_line_number, "%d", model->params->param_line_number);

	// Concatenate file name
	strcpy(output_file_name, model->params->output_file_dir);
	strcat(output_file_name, "/transmission_Run");
	strcat(output_file_name, param_line_number);
	strcat(output_file_name, ".csv");

	output_file = fopen(output_file_name, "w");
	fprintf(output_file , "ID_recipient,");
	fprintf(output_file , "age_group_recipient,");
	fprintf(output_file , "house_no_recipient,");
	fprintf(output_file , "occupation_network_recipient,");
	fprintf(output_file , "infector_network,");
	fprintf(output_file , "generation_time,");
	fprintf(output_file , "ID_source,");
	fprintf(output_file , "age_group_source,");
	fprintf(output_file , "house_no_source,");
	fprintf(output_file , "occupation_network_source,");
	fprintf(output_file , "time_infected_source,");
	fprintf(output_file , "status_source,");
	fprintf(output_file , "time_infected,");
	fprintf(output_file , "time_presymptomatic,");
	fprintf(output_file , "time_presymptomatic_mild,");
	fprintf(output_file , "time_presymptomatic_severe,");
	fprintf(output_file , "time_symptomatic,");
	fprintf(output_file , "time_symptomatic_mild,");
	fprintf(output_file , "time_symptomatic_severe,");
	fprintf(output_file , "time_asymptomatic,");
	fprintf(output_file , "time_hospitalised,");
	fprintf(output_file , "time_critical,");
	fprintf(output_file , "time_hospitalised_recovering,");
	fprintf(output_file , "time_death,");
	fprintf(output_file , "time_recovered,");
	fprintf(output_file , "time_susceptible,");
	fprintf(output_file , "is_case\n");

	for( pdx = 0; pdx < model->params->n_total; pdx++ )
	{
		indiv = &(model->population[pdx]);
		infection_event = indiv->infection_events;

		while(infection_event != NULL)
		{
			if( time_infected_infection_event(infection_event) != UNKNOWN )
				fprintf(output_file ,"%li,%i,%li,%i,%i,%i,%li,%i,%li,%i,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
					indiv->idx,
					indiv->age_group,
					indiv->house_no,
					indiv->occupation_network,
					infection_event->infector_network,
					time_infected_infection_event( infection_event ) - infection_event->time_infected_infector,
					infection_event->infector->idx,
					infection_event->infector->age_group,
					infection_event->infector->house_no,
					infection_event->infector->occupation_network,
					infection_event->time_infected_infector,
					infection_event->infector_status,
					time_infected_infection_event( infection_event ),
					max( infection_event->times[PRESYMPTOMATIC], infection_event->times[PRESYMPTOMATIC_MILD] ),
					infection_event->times[PRESYMPTOMATIC_MILD],
					infection_event->times[PRESYMPTOMATIC],
					max(infection_event->times[SYMPTOMATIC], infection_event->times[SYMPTOMATIC_MILD]),
					infection_event->times[SYMPTOMATIC_MILD],
					infection_event->times[SYMPTOMATIC],
					infection_event->times[ASYMPTOMATIC],
					infection_event->times[HOSPITALISED],
					infection_event->times[CRITICAL],
					infection_event->times[HOSPITALISED_RECOVERING],
					infection_event->times[DEATH],
					infection_event->times[RECOVERED],
					infection_event->times[SUSCEPTIBLE],
					infection_event->is_case
				);
			infection_event = infection_event->next;
		}
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
	int day, index_time;
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
	fprintf( output_file ,"time,index_time,index_ID,index_reason,index_status,contact_time,traced_ID,traced_status,traced_infector_ID,traced_time_infected\n" );

	int max_quarantine_length = max( model->params->quarantine_length_traced_symptoms, model->params->quarantine_length_traced_positive );
	for( day = 1; day <= max_quarantine_length; day++ )
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

			index_time = token->contact_time;

			while( token != NULL )
			{
				fprintf( output_file, "%i,%i,%li,%i,%i,%i,%li,%i,%li,%i\n",
					model->time,
					index_time,
					indiv->idx,
					token->index_status,
					indiv->status,
					token->contact_time,
					token->individual->idx,
					token->individual->status,
					ifelse( token->individual->status > 0, token->individual->infection_events->infector->idx, -1 ),
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

	int max_quarantine_length = max( model->params->quarantine_length_traced_symptoms, model->params->quarantine_length_traced_positive );
	for( day = 1; day <=  max_quarantine_length; day++ )
	{
		n_events    = model->event_lists[TRACE_TOKEN_RELEASE].n_daily_current[ model->time + day ];
		next_event  = model->event_lists[TRACE_TOKEN_RELEASE].events[ model->time + day ];

		for( idx = 0; idx < n_events; idx++ )
		{
			event      = next_event;
			next_event = event->next;
			indiv      = event->individual;
			time_index = model->time + day -  max_quarantine_length;

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
				if( (contact->status >= SYMPTOMATIC) & (contact->infection_events->times[ASYMPTOMATIC] == UNKNOWN)  &
				+					( (contact->infection_events->times[RECOVERED] == UNKNOWN) | (contact->infection_events->times[RECOVERED] > time_index) )
				)
					n_symptoms++;
				if( indiv == contact->infection_events->infector )
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

/*****************************************************************************************
*  Name:		write_occupation_network
*  Description: Write (csv) file of occupation network
******************************************************************************************/
void write_occupation_network(model *model, parameters *params, int network_idx)
{

	if(network_idx < 0 || network_idx >= N_OCCUPATION_NETWORKS ){
		printf("Occupation network index outside range of 0, %d\n", N_OCCUPATION_NETWORKS-1);
		return;
	}

	char output_file[INPUT_CHAR_LEN];
	char param_line_number[10], network_idx_text[10];
	sprintf(param_line_number, "%d", params->param_line_number);

	sprintf(network_idx_text, "%d", network_idx);

	// Concatenate file name
	strcpy(output_file, params->output_file_dir);
	strcat(output_file, "/occupation_network");
	strcat(output_file, network_idx_text);
	strcat(output_file, "_Run");
	strcat(output_file, param_line_number);
	strcat(output_file, ".csv");

	write_network(output_file, model->occupation_network[network_idx]);

}

/*****************************************************************************************
*  Name:		write_household_network
*  Description: Write (csv) file of household network
******************************************************************************************/
void write_household_network(model *model, parameters *params)
{
	char output_file[INPUT_CHAR_LEN];
	char param_line_number[10];
	sprintf(param_line_number, "%d", params->param_line_number);

	// Concatenate file name
	strcpy(output_file, params->output_file_dir);
	strcat(output_file, "/household_network_Run");
	strcat(output_file, param_line_number);
	strcat(output_file, ".csv");

	write_network(output_file, model->household_network);
}

/*****************************************************************************************
*  Name:		write_random_network
*  Description: Write (csv) file of random network
******************************************************************************************/
void write_random_network(model *model, parameters *params)
{
	char output_file[INPUT_CHAR_LEN];

	char param_line_number[10];
	sprintf(param_line_number, "%d", params->param_line_number);

	// Concatenate file name
	strcpy(output_file, params->output_file_dir);
	strcat(output_file, "/random_network_Run");
	strcat(output_file, param_line_number);
	strcat(output_file, ".csv");

	write_network(output_file, model->random_network);
}

/*****************************************************************************************
*  Name:		write_random_network
*  Description: Write (csv) file of generic network
******************************************************************************************/

void write_network(char *output_file, network *network_ptr)
{
	long idx;
	FILE *network_file;

	network_file = fopen(output_file, "w");
	if(network_file == NULL)
		print_exit("Can't open network output file");

	fprintf(network_file,"ID1,");
	fprintf(network_file,"ID2");
	fprintf(network_file,"\n");

	// Loop through all edges in the network
	for(idx = 0; idx < network_ptr->n_edges; idx++)
	{
		fprintf(network_file, "%li,%li\n",
			network_ptr->edges[idx].id1,
			network_ptr->edges[idx].id2
			);
	}
	fclose(network_file);
}
