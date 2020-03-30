/*
 * individual.h
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

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct hospital hospital;

struct hospital {
    int hospital_idx;

    network *hospital_network;

    int available_beds;
    int available_icus;

    long *doctor_pdxs;      //stores the population ids of all doctors at the hospital
    long *nurse_pdxs;       //stores the population ids of all nurses at the hospital

    int n_total_doctors;    //total number of doctors at the hospital
    int n_total_nurses;     //total number of nurses at the hospital
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialise_hospital( hospital*, parameters*, /*network*,*/ int );
void add_healthcare_worker_to_hospital(hospital *hospital, long pdx, int type);
void destroy_hospital( hospital* );

#endif /* HOSPITAL_H_ */
