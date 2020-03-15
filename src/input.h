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
void write_output_files(model *, parameters *);
void write_individual_file(model *, parameters *);
void print_interactions_averages( model*, int );

#endif /* INPUT_H_ */
