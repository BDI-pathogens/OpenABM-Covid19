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

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/



typedef struct ward ward;

struct ward
{
    int ward_idx;
    int type;

    int beds;

    int max_doctors;
    int max_nurses;

    int n_doctors;
    int n_nurses;
    int n_patients;

    network *doctor_patient_network;
    network *nurse_patient_network;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialise_ward( ward*, int, int );
void set_up_ward_networks( ward* ward );
void destroy_ward( ward* );

#endif /* WARD_H_ */
