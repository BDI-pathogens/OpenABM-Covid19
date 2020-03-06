/*
 * params.h
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#ifndef PARAMS_H_
#define PARAMS_H_

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct{
	int param_line_number;			// line number to be read from parameter file
	long param_id;					// id of the parameter set
	long n_total;  					// total number of people
	int mean_daily_interactions;    // mean number of daily interactions
	int days_of_interactions;		// the number of days of interactions to keep
	int end_time;				    // maximum end time
} parameters;

#endif /* PARAMS_H_ */
