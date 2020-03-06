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


/*****************************************************************************************
*  Name:		read_param_file
*  Description: Read parameter file (csv), attach parameter values to params structure
******************************************************************************************/

void read_param_file( parameters *params ){
	
	FILE *parameter_file;
	int check;
	
	parameter_file = fopen(FILENAME_PARAM, "r");
	if(parameter_file == NULL){
		print_exit("Can't open parameter file");
	}
	
	// Throw away header
	fscanf(parameter_file, "%*[^\n]\n");
	
	// Read and attach parameter values to parameter structure
	check = fscanf(parameter_file, " %li ,", &(params->param_id));
	check = fscanf(parameter_file, " %li ,", &(params->n_total));
	check = fscanf(parameter_file, " %i ,", &(params->mean_daily_interactions));
	check = fscanf(parameter_file, " %i ,", &(params->days_of_interactions));
	check = fscanf(parameter_file, " %i", &(params->end_time));
	fclose(parameter_file);
}