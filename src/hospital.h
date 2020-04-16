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
#include "waiting_list.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct hospital hospital;

struct hospital
{
    int hospital_idx;               //hospital index number

    int n_workers[N_WORKER_TYPES];
    int n_patients_waiting; //TODO: have waiting list for general and icu
    int timestep_pending_admissions[N_HOSPITAL_WARD_TYPES];
    
    //TODO: need ventilator variables... when will a ventilator be needed? - question for rest of nhsx team
    //TODO: add non covid patients

    network *hospital_workplace_network;

    waiting_list *waiting_list;

    int n_wards[N_HOSPITAL_WARD_TYPES];
    ward **wards;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialise_hospital( hospital*, parameters*, int );
void set_up_hospital_networks(hospital* , int max_hcw_daily_interactions);
void build_hospital_networks( model *model, hospital *hospital );
int healthcare_worker_working(individual* indiv);
int assign_patient_to_hospital( model*, individual* );
void destroy_hospital( hospital* );

void transition_one_hospital_event( model *model, individual *indiv, int from, int to, int edge );

void transition_to_waiting( model *model, individual *indiv );
void transition_to_general( model *model, individual *indiv );
void transition_to_icu( model *model, individual *indiv );
void transition_to_mortuary( model *model, individual *indiv );
void transition_to_discharged( model *model, individual *indiv );

int assign_to_ward(individual *indiv, hospital *hospital, int ward_type );

void add_healthcare_worker_to_hospital(hospital *hospital, long pdx, int type);
int  add_patient_to_hospital( model* model, individual *indiv );
void release_patient_from_hospital( individual *indiv, hospital *hospital );
void add_patient_to_waiting_list( individual *indiv, hospital *hospital, int ward_type);
void remove_patient_from_waiting_list( individual *indiv, hospital *hospital, int ward_type );
long remove_first_patient_from_waiting_list( hospital *hospital, int ward_type );


int hospital_available_beds( hospital *hospital, int ward_type );


#endif /* HOSPITAL_H_ */
