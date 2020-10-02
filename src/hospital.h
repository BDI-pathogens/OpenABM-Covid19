/*
 * hospital.h
 *
 *  Created on: 30 Mar 2020
 *      Author: vuurenk
 */

#ifndef HOSPITAL_H_
#define HOSPITAL_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/
#include "structure.h"
#include "utilities.h"
#include "constant.h"
#include "params.h"
#include "network.h"
#include "individual.h"
#include "ward.h"
#include "list.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct hospital hospital;

struct hospital
{
    int hospital_idx; 
    int n_workers[N_WORKER_TYPES];

    network *hospital_workplace_network;
    
    list *waiting_list[N_HOSPITAL_WARD_TYPES];
    int n_wards[N_HOSPITAL_WARD_TYPES];
    ward **wards;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialise_hospital( hospital*, parameters*, int );
void set_up_hospital_networks(model *pmodel);
void rebuild_healthcare_worker_patient_networks( model *pmodel, hospital *phospital );
void add_hospital_network_interactions(model *pmodel, hospital *phospital);
int healthcare_worker_working(individual* indiv);
void destroy_hospital( hospital* );

void transition_one_hospital_event( model *pmodel, individual *indiv, int from, int to, int edge );

void transition_to_waiting( model *pmodel, individual *indiv );
void transition_to_general( model *pmodel, individual *indiv );
void transition_to_icu( model *pmodel, individual *indiv );
void transition_to_mortuary( model *pmodel, individual *indiv );
void transition_to_discharged( model *pmodel, individual *indiv );

void add_healthcare_worker_to_hospital(hospital *phospital, individual *indiv, int type);
int  add_patient_to_hospital( model* pmodel, individual *indiv, int required_ward );
void release_patient_from_hospital( individual *indiv, hospital *phospital );
void add_patient_to_waiting_list( individual *indiv, hospital *phospital, int ward_type);

void hospital_waiting_list_transition_scheduler( model *pmodel, int disease_state );
void swap_waiting_general_and_icu_patients( model *pmodel );
void predict_patient_disease_progression(model *pmodel, individual *indiv, double patient_waiting_modifier, int type );

void remove_if_in_waiting_list( individual *indiv, hospital *phospital );
int hospital_available_beds( hospital *phospital, int ward_type );
int find_least_full_hospital(model* pmodel, int required_ward);

int individual_eligible_to_become_healthcare_worker( individual *indiv );

#endif /* HOSPITAL_H_ */
