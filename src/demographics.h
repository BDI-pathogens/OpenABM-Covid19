/*
 * demographics.h
 *
 *  Created on: 23 Mar 2020
 *      Author: hinchr
 */

#ifndef DEMOGRAPHICS_H_
#define DEMOGRAPHICS_H_

void set_up_household_distribution( model* );
void set_up_allocate_work_places( model* );
void build_household_network_from_directroy( network*, directory* );

#endif /* DEMOGRAPHICS_H_ */
