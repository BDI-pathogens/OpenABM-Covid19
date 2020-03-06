/*
 * input.c
 *
 *  Created on: 6 Mar 2020
 *      Author: p-robot
 */

#include <stdio.h>
#include <stdlib.h>

#include "input.h"
#include "params.h"
#include "utilities.h"

#define FILENAME_PARAM "parameters.csv"


void read_param_file(){
	
	FILE *parameter_file;

	parameter_file = fopen(FILENAME_PARAM, "r");
	if(parameter_file == NULL){
		print_exit("Can't open parameter file");
	}
	printf("Found parameter file\n");
	
	// Throw away header
	fscanf(parameter_file, "%*[^\n]\n");
	
	int check;
	long param_id, n_total, mean_daily_interactions, days_of_interactions, end_time;
	
	check = fscanf(parameter_file, "%ld ", &param_id);
	check = fscanf(parameter_file, "%ld ", &n_total);
	check = fscanf(parameter_file, "%ld ", &(mean_daily_interactions));
	check = fscanf(parameter_file, "%ld ", &(days_of_interactions));
	check = fscanf(parameter_file, "%ld", &(end_time));
	
	
	printf("param_id: %ld\n", param_id);
	printf("n_total: %ld\n", n_total);
	printf("mean_daily_interactions: %ld\n", mean_daily_interactions);
	printf("days_of_interactions: %ld\n", days_of_interactions);
	printf("end_time: %ld\n", end_time);
	
	fclose(parameter_file);
}