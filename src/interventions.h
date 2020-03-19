/*
 * interventions.h
 *
 *  Created on: 18 Mar 2020
 *      Author: hinchr
 */

#ifndef INTERVENTIONS_H_
#define INTERVENTIONS_H_

#include "structure.h"
#include "individual.h"

void set_up_transition_times_intervention( model* );

void intervention_test_take( model*, individual*  );
void intervention_test_result( model*, individual*  );
void intervention_quarantine_release( model*, individual*  );
void intervention_quarantine_contacts( model*, individual* );

void intervention_on_symptoms( model*, individual*  );
void intervention_on_hospitalised( model*, individual*  );


#endif /* INTERVENTIONS_H_ */
