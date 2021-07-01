/*
 * input.h
 *
 *  Description: Top level model 'object' c
 *  Created on:  6 Mar 2020
 *      Author:  p-robot
 */

#ifndef INPUT_H_
#define INPUT_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/

#include "params.h"
#include "model.h"

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void read_command_line_args(parameters *, int, char **);
void read_param_file(parameters *);
void read_household_demographics_file(parameters *);
void read_hospital_param_file( parameters *);
void set_up_reference_household_memory(parameters *params);
void write_output_files(model *, parameters *);
void write_individual_file(model *, parameters *);
void write_interactions( model* );
void write_transmissions( model* );
void write_trace_tokens( model* );
void write_trace_tokens_ts( model*, int );
void write_quarantine_reasons( model*, parameters *);
void write_ward_data( model* );
int get_worker_ward_type( model *pmodel, int pdx );
void write_time_step_hospital_data( model *pmodel);
void write_hospital_interactions(model *pmodel);

void write_occupation_network(model *, parameters *, int );
void write_household_network(model *, parameters *);
void write_random_network(model *, parameters *);
void write_network(char *, network *);

void print_interactions_averages( model*, int );

long get_n_transmissions( model* );
void get_transmissions( model*,
	long*, int*,  long*, int*,  int*,  int*,  int*,  int*,  int*,
	long*,  int*, long*, int*, int*, int*,  int*, int*,  int*,
	int*,  int*,  int*,  int*,  int*,  int*,  int*,  int*,
	int*,  int*,  int*,  int*,  int*,  int*,  int*, float*, float*
);

#endif /* INPUT_H_ */
