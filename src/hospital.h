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

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct hospital hospital;

struct hospital
{
    int hospital_idx;               //hospital index number

    int n_total_beds;               //total beds at hospital
    int n_total_icus;               //total icus at hospital

    int n_total_doctors;            //total number of doctors
    int n_total_nurses;             //total number of nurses
    int n_total_general_patients;   //total number of general patients
    int n_total_icu_patients;       //total number of icu patients

    int n_patients_waiting;
    //TODO: need ventilator variables... when will a ventilator be needed? - question for rest of nhsx team
    //TODO: add non covid patients

    int n_covid_general_wards;
    int n_covid_icu_wards;

    network *hospital_workplace_network;

    int *n_wards;
    ward **wards;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialise_hospital( hospital*, parameters*, int );
void set_up_hospital_networks( hospital* );
void build_hospital_networks( model *model, hospital *hospital );
void add_healthcare_worker_to_hospital(hospital *hospital, long pdx, int type);
int healthcare_worker_working(individual* indiv);
void add_patient_to_hospital( model*, individual* );
void destroy_hospital( hospital* );

void transition_one_hospital_event( model *model, individual *indiv, int from, int to, int edge );

void transition_to_waiting( model *model, individual *indiv );
void transition_to_general( model *model, individual *indiv );
void transition_to_icu( model *model, individual *indiv );
void transition_to_mortuary( model *model, individual *indiv );
void transition_to_populace( model *model, individual *indiv );

int assign_to_ward(individual *indiv, hospital *hospital, int ward_type );

#endif /* HOSPITAL_H_ */
