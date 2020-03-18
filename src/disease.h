/*
 * disease.h
 *
 * Description: this file contains code relating to the dynamics of the disease
 *				within an individual and transmission events during interactions
 *  Created on: 18 Mar 2020
 *      Author: hinchr
 */

#ifndef DISEASE_H_
#define DISEASE_H_

// transmission of the virus
void transmit_virus( model* );
void transmit_virus_by_type( model*, int );

// progression of the disease
void new_infection( model*, individual*, individual* );
void transition_to_symptomatic( model* );
void transition_to_hospitalised( model* );
void transition_to_recovered( model* );
void transition_to_death( model* );

#endif /* DISEASE_H_ */
