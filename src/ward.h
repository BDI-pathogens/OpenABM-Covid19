/*
 * ward.h
 *
 *  Created on: 03 Apr 2020
 *      Author: vuurenk
 */

#ifndef WARD_H_
#define WARD_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/
#include "network.h"
#include "doctor.h"
#include "nurse.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

#define NO_PATIENT -1

typedef struct ward ward;

struct ward
{
    int ward_idx;
    int type;

    int n_beds;

    int n_max_doctors; //TODO: maybe get rid of max variables as number of hcw is static
    int n_max_nurses;

    int n_doctors;
    int n_nurses;
    int n_patients;

    doctor *doctors;
    nurse  *nurses;
    long *patient_pdxs;

    network *doctor_patient_network;
    network *nurse_patient_network;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialise_ward(ward*, int, int , int n_beds);
void set_up_ward_networks( ward* ward );
void build_ward_networks(model *model, ward* ward );
void build_hcw_patient_network(ward* ward, network *network, long *hc_workers, int n_hcw_working, int n_patient_required_interactions, int max_hcw_daily_interactions );
int  add_patient_to_ward( ward *ward, long pdx );
void remove_patient_from_ward( ward* ward, long pdx);
void destroy_ward( ward* );

#endif /* WARD_H_ */
