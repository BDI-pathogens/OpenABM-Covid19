/*
 * input.h
 *
 *  Description: Top level model 'object' c
 *  Created on:  6 Mar 2020
 *      Author:  p-robot
 */

#ifndef INPUT_H_
#define INPUT_H_


#define FILENAME_PARAM "parameters.csv"

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/

#include "params.h"

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void read_command_line_args(parameters *, int, char **);
void read_param_file(parameters *);

#endif /* INPUT_H_ */
