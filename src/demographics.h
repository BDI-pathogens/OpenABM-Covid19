/*
 * demographics.h
 *
 *  Created on: 23 Mar 2020
 *      Author: hinchr
 */

#ifndef DEMOGRAPHICS_H_
#define DEMOGRAPHICS_H_

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct demographic_household_table demographic_household_table;

struct demographic_household_table{
	long n_households;
	long n_total;
	long *idx;
	int *age_group;
	long *house_no;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void generate_household_distribution( model* );
void assign_household_distribution( model*, demographic_household_table* );
void set_up_household_distribution( model* );
void set_up_allocate_work_places( model* );
void build_household_network_from_directroy( network*, directory* );
void add_reference_household( double *, long , int **);
	
#endif /* DEMOGRAPHICS_H_ */
