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
	
	for( i = 0; i < N_WORK_NETWORK_TYPES; i++ )
	{
		check = fscanf(parameter_file, " %i ,",  &(params->mean_work_interactions[i]));
		if( check < 1){ print_exit("Failed to read parameter mean_work_interactions\n"); };
	}

	check = fscanf(parameter_file, " %lf ,",  &(params->daily_fraction_work));
	if( check < 1){ print_exit("Failed to read parameter daily_fraction_work\n"); };

	for( i = 0; i < N_AGE_TYPES; i++ )
	{
		check = fscanf(parameter_file, " %i ,",  &(params->mean_random_interactions[i]));
		if( check < 1){ print_exit("Failed to read parameter mean_daily_interactions\n"); };

		check = fscanf(parameter_file, " %i ,",  &(params->sd_random_interactions[i]));
		if( check < 1){ print_exit("Failed to read parameter sd_daily_interactions\n"); };
	}

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

	check = fscanf(parameter_file, " %lf ,", &(params->mean_time_to_recover));
	if( check < 1){ print_exit("Failed to read parameter mean_time_to_recover\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->sd_time_to_recover));
	if( check < 1){ print_exit("Failed to read parameter sd_time_to_recover\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->mean_time_to_death));
	if( check < 1){ print_exit("Failed to read parameter mean_time_to_death\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->sd_time_to_death));
	if( check < 1){ print_exit("Failed to read parameter sd_time_to_death\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->fraction_asymptomatic));
	if( check < 1){ print_exit("Failed to read parameter fraction_asymptomatic\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->asymptomatic_infectious_factor));
	if( check < 1){ print_exit("Failed to read parameter asymptomatic_infectious_factor\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->mean_asymptomatic_to_recovery));
	if( check < 1){ print_exit("Failed to read parameter mean_asymptomatic_to_recovery\n"); };
	
	check = fscanf(parameter_file, " %lf ,", &(params->sd_asymptomatic_to_recovery));
	if( check < 1){ print_exit("Failed to read parameter sd_asymptomatic_to_recovery\n"); };

	
	for( i = 0; i < N_HOUSEHOLD_MAX; i++ )
	{
		check = fscanf(parameter_file, " %lf ,", &(params->household_size[i]));
		if( check < 1){ print_exit("Failed to read parameter household_size_*\n"); };
	}

	for( i = 0; i < N_AGE_TYPES; i++ )
	{
		check = fscanf(parameter_file, " %lf ,", &(params->population_type[i]));
		if( check < 1){ print_exit("Failed to read parameter population_type_**\n"); };
	}

	for( i = 0; i < N_AGE_GROUPS; i++ )
	{
		check = fscanf(parameter_file, " %lf ,", &(params->population_group[i]));
		if( check < 1){ print_exit("Failed to read parameter population_group_**\n"); };
	}

	check = fscanf(parameter_file, " %lf ,", &(params->seasonal_flu_rate));
	if( check < 1){ print_exit("Failed to read parameter seasonal_flu_rate\n"); };

	for( i = 0; i < N_AGE_GROUPS; i++ )
		{
			check = fscanf(parameter_file, " %lf ,", &(params->relative_susceptibility[i]));
			if( check < 1){ print_exit("Failed to read parameter relative_susceptibility\n"); };
		}

	for( i = 0; i < N_INTERACTION_TYPES; i++ )
	{
		check = fscanf(parameter_file, " %lf ,", &(params->relative_transmission_by_type[i]));
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

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_length_self));
	if( check < 1){ print_exit("Failed to read parameter quarantine_length_self\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_length_traced));
	if( check < 1){ print_exit("Failed to read parameter quarantine_length_traced\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_length_positive));
	if( check < 1){ print_exit("Failed to read parameter quarantine_length_positive\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->quarantine_dropout_self));
	if( check < 1){ print_exit("Failed to read parameter quarantine_dropout_self\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->quarantine_dropout_traced));
	if( check < 1){ print_exit("Failed to read parameter quarantine_dropout_traced\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->quarantine_dropout_positive));
	if( check < 1){ print_exit("Failed to read parameter quarantine_dropout_positive\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->test_on_symptoms));
	if( check < 1){ print_exit("Failed to read parameter test_on_symptoms\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->test_on_traced));
	if( check < 1){ print_exit("Failed to read parameter test_on_traced\n"); };

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

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_household_on_traced));
	if( check < 1){ print_exit("Failed to read parameter quarantine_household_on_traced\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->quarantine_household_contacts_on_positive));
	if( check < 1){ print_exit("Failed to read parameter quarantine_household_contacts_on_positive\n"); };

	check = fscanf(parameter_file, " %i  ,", &(params->quarantined_daily_interactions));
	if( check < 1){ print_exit("Failed to read parameter quarantined_daily_interactions\n"); };

	check = fscanf(parameter_file, " %i  ,", &(params->quarantine_days));
	if( check < 1){ print_exit("Failed to read parameter quarantine_days\n"); };

	check = fscanf(parameter_file, " %i  ,", &(params->hospitalised_daily_interactions));
	if( check < 1){ print_exit("Failed to read parameter hospitalised_daily_interactions\n"); };

	check = fscanf(parameter_file, " %i , ",   &(params->test_insensititve_period));
	if( check < 1){ print_exit("Failed to read parameter test_insensititve_period\n"); };

	check = fscanf(parameter_file, " %i , ",   &(params->test_order_wait));
	if( check < 1){ print_exit("Failed to read parameter test_order_wait\n"); };

	check = fscanf(parameter_file, " %i , ",   &(params->test_result_wait));
	if( check < 1){ print_exit("Failed to read parameter test_result_wait\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->self_quarantine_fraction));
	if( check < 1){ print_exit("Failed to read parameter self_quarantine_fraction\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->app_users_fraction));
	if( check < 1){ print_exit("Failed to read parameter app_users_fraction\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->app_turn_on_time));
	if( check < 1){ print_exit("Failed to read parameter app_turn_on_time)\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->social_distancing_work_network_multiplier));
	if( check < 1){ print_exit("Failed to read parameter social_distancing_work_network_multiplier)\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->social_distancing_random_network_multiplier));
	if( check < 1){ print_exit("Failed to read parameter social_distancing_random_network_multiplier)\n"); };

	check = fscanf(parameter_file, " %lf ,", &(params->social_distancing_house_interaction_multiplier));
	if( check < 1){ print_exit("Failed to read parameter social_distancing_house_interaction_multiplier)\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->social_distancing_time_on));
	if( check < 1){ print_exit("Failed to read parameter social_distancing_time_on)\n"); };

	check = fscanf(parameter_file, " %i ,", &(params->social_distancing_time_off));
	if( check < 1){ print_exit("Failed to read parameter social_distancing_time_off)\n"); };
	
    check = fscanf(parameter_file, " %i ,", &(params->testing_symptoms_time_on));
    if( check < 1){ print_exit("Failed to read parameter testing_symptoms_time_on)\n"); };

    check = fscanf(parameter_file, " %i ,", &(params->testing_symptoms_time_off));
    if( check < 1){ print_exit("Failed to read parameter testing_symptoms_time_off)\n"); };
	
	
	check = fscanf(parameter_file, " %li ,", &(params->N_REFERENCE_HOUSEHOLDS));
	if( check < 1){ print_exit("Failed to read parameter N_REFERENCE_HOUSEHOLDS)\n"); };
	
    //kelvin change
    //current sim  pop is 100,000. 66 million is uk pop. 150,000 doctors and 320,000 nurses + midwives in nhs according to https://www.nuffieldtrust.org.uk/resource/the-nhs-workforce-in-numbers
    params->n_total_doctors = 227;
    params->n_total_nurses  = 484;
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
	
	fprintf(individual_output_file,"ID, ");
	fprintf(individual_output_file,"current_status, ");
	fprintf(individual_output_file,"age_group, ");
	fprintf(individual_output_file,"work_network, ");
	fprintf(individual_output_file,"house_no, ");
	fprintf(individual_output_file,"quarantined, ");
	fprintf(individual_output_file,"hazard, ");
	fprintf(individual_output_file,"mean_interactions, ");
	fprintf(individual_output_file,"time_infected, ");
	fprintf(individual_output_file,"time_presypmtomatic, ");
	fprintf(individual_output_file,"time_symptomatic, ");
	fprintf(individual_output_file,"time_asymptomatic, ");
	fprintf(individual_output_file,"time_hospitalised, ");
	fprintf(individual_output_file,"time_critical, ");
	fprintf(individual_output_file,"time_death, ");
	fprintf(individual_output_file,"time_recovered, ");
	fprintf(individual_output_file,"infector_ID, ");
	fprintf(individual_output_file,"infector_time_infected, ");
	fprintf(individual_output_file,"infector_status, ");
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
			"%li, %d, %d, %d, %li, %d, %f, %d, %d, %d, %d, %d, %d, %d, %d, %d, %li, %d, %d\n",
			indiv->idx,
			indiv->status,
			indiv->age_group,
			indiv->work_network,
			indiv->house_no,
			indiv->quarantined,
			indiv->hazard,
			indiv->random_interactions,
			time_infected(indiv),
			indiv->time_event[PRESYMPTOMATIC],
			indiv->time_event[SYMPTOMATIC],
			indiv->time_event[ASYMPTOMATIC],
			indiv->time_event[HOSPITALISED],
			indiv->time_event[CRITICAL],
			indiv->time_event[DEATH],
			indiv->time_event[RECOVERED],
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
	FILE *hh_file;
	int check, value, adx;
	long hdx;
	
	params->REFERENCE_HOUSEHOLDS = calloc(params->N_REFERENCE_HOUSEHOLDS, sizeof(int*));
	
	for(hdx = 0; hdx < params->N_REFERENCE_HOUSEHOLDS; hdx++)
		params->REFERENCE_HOUSEHOLDS[hdx] = calloc(N_AGE_GROUPS, sizeof(int));
	
	hh_file = fopen(params->input_household_file, "r");
	if(hh_file == NULL)
		print_exit("Can't open household demographics file");
	
	// Throw away header
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
	print_exit( "# Output : write interactions end!");
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
	print_exit( "# Output : write transmissions end!");}


