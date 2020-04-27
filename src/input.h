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
void write_hcw_data( model* );
void write_ward_data( model* );

void print_interactions_averages( model*, int );

#endif /* INPUT_H_ */
