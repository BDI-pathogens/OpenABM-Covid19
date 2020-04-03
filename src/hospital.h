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
#include "doctor.h"
#include "nurse.h"
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

    long *doctor_pdxs;              //stores the population ids of all doctors at the hospital
    long *nurse_pdxs;               //stores the population ids of all nurses at the hospital

    doctor *doctors;
    nurse  *nurses;

    long *general_patient_pdxs;     //stores the population ids of all general patients at the hospital
    long *icu_patient_pdxs;         //stores the population ids of all general patients at the hospital
    //TODO: at end of timestep there is an event list of people who have transitioned to admitted / hospitalised / critical / recovery

    int n_total_doctors;            //total number of doctors
    int n_total_nurses;             //total number of nurses
    int n_total_general_patients;   //total number of general patients
    int n_total_icu_patients;       //total number of icu patients
    //TODO: need ventilator variables... when will a ventilator be needed? - question for rest of nhsx team
    //TODO: add non covid patients

    int n_covid_general_wards;
    int n_covid_icu_wards;

    network *hospital_workplace_network;
    //ward related stuff
    int *n_wards;
    ward **wards;
    network **wards_networks;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialise_hospital( hospital*, parameters*, int );
void set_up_hospital_networks( hospital* );
void add_healthcare_worker_to_hospital(hospital *hospital, long pdx, int type);
int healthcare_worker_working(individual* indiv);
void add_patient_to_hospital(hospital *hospital, long pdx, int type);
void build_hcw_patient_network( model *model, network *network, long *patient_pdxs, long *worker_pdxs, int n_patients, int total_workers, int patient_required_interactions);
void destroy_hospital( hospital* );

#endif /* HOSPITAL_H_ */
