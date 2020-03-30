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
#include "params.h"
#include "individual.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct hospital hospital;

struct hospital{
    int hospital_idx;

    int available_beds;
    int available_icus;

    long *doctor_pdxs;
    long *nurse_pdxs;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialise_hospital( hospital*, parameters*, int );
void add_healthcare_worker_to_hospital(hospital *hospital, int idx, long pdx, int type);
void destroy_hospital( hospital* );
#endif /* HOSPITAL_H_ */
