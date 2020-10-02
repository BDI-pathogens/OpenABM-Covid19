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
#include "list.h"
#include "individual.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct ward ward;

struct ward
{
    int ward_idx;
    int type;

    int n_beds;
    int n_max_hcw[N_WORKER_TYPES];
    int n_worker[N_WORKER_TYPES];

    doctor *doctors;
    nurse  *nurses;
    list   *patients;

    network *doctor_patient_network;
    network *nurse_patient_network;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialise_ward(ward*, int, int , int n_beds, int n_max_doctors, int n_max_nurses);
void set_up_ward_networks(ward* ward , int max_hcw_daily_interactions);
void build_ward_networks(model *pmodel, ward* pward );
void build_hcw_patient_network(ward* ward, network *network, long *hc_workers, int n_hcw_working, int n_patient_required_interactions, int max_hcw_daily_interactions );
int  add_patient_to_ward( ward *ward, individual *indiv );
int  ward_available_beds( ward* ward);
void remove_patient_from_ward( ward* ward, individual *indiv );
void destroy_ward( ward* );

#endif /* WARD_H_ */
