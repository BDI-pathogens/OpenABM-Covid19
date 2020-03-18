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

void intervention_test_take( model*, individual*  );
void intervention_test_result( model*, individual*  );
void intervention_quarantine_release( model*, individual*  );
void intervention_quarantine_contacts( model*, individual* );


#endif /* INTERVENTIONS_H_ */
